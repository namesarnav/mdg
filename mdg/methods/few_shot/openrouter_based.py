from __future__ import annotations

from typing import List, Optional, Dict, Any, Tuple
import os, time, json, random
from tqdm.auto import tqdm
from jinja2 import Environment, BaseLoader

from mdg.methods.few_shot import FewShotModelWrapper
from mdg.datamodels import TimeDocument, TimeExpression, EventExpression
from mdg.registry import register, MODEL_WRAPPER

from mdg.methods.openrouter_utils import (
    call_openrouter_json_array,
    call_openrouter_json,
    call_openrouter_label_array,
    _timex_list_of_dicts_schema,
    _compositional_list_of_dicts_schema,

)

from mdg.methods.prompts.timex_reasoning_prompts import PROMPTS, REASONING_TYPES
from mdg.methods.prompts import TLINK_PROMPTS, TLINK_REASONING_TYPES

from mdg.methods.prompts.event_reasoning_prompts import (
    EVENT_PROMPTS,
    REASONING_TYPES as EVENT_REASONING_TYPES,
)

from mdg.methods.prompts.compositional_reasoning_prompts import (
    PROMPTS as COMP_PROMPTS,
    REASONING_TYPES as COMP_REASONING_TYPES,
)

from mdg.methods.prompts.causal_reasoning_prompts import (
    CAUSAL_PROMPTS,
    REASONING_TYPES as CAUSAL_REASONING_TYPES,
)

try:
    from requests.exceptions import ChunkedEncodingError as _ChunkedEncodingError
except ImportError:
    _ChunkedEncodingError = OSError

import re
import requests

# -----------------------------
# JSONL writer
# -----------------------------
def _append_jsonl(path: Optional[str], record: Dict[str, Any]) -> None:
    """
    Append a single JSON object as one line into a JSONL file.
    If path is None/empty -> do nothing.
    """
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# -----------------------------
# Abductive helpers
# -----------------------------
def _has_abductive(reasoning_type: str) -> bool:
    return "abductive" in (reasoning_type or "").lower()


def _safe_reasoning_steps(out_obj: Any, reasoning_type: str) -> Optional[List[str]]:
    """
    Only return reasoning_steps if reasoning_type contains 'abductive'.
    Supports outputs like:
      - [{"reasoning_steps":[...]}]
      - {"reasoning_steps":[...]}
    """
    if not _has_abductive(reasoning_type):
        return None
    steps: List[str] = []
    try:
        if isinstance(out_obj, list) and out_obj and isinstance(out_obj[0], dict):
            rs = out_obj[0].get("reasoning_steps", None)
            if isinstance(rs, list):
                steps = [s for s in rs if isinstance(s, str) and s.strip()]
        elif isinstance(out_obj, dict):
            rs = out_obj.get("reasoning_steps", None)
            if isinstance(rs, list):
                steps = [s for s in rs if isinstance(s, str) and s.strip()]
    except Exception:
        return None
    return steps if steps else None


def _postprocess_strings(xs: Any) -> List[str]:
    """
    Basic post-processing for predicted expression lists:
      - keep only strings
      - strip
      - drop empty
      - dedupe (preserve order)
    """
    if not xs or not isinstance(xs, list):
        return []
    out: List[str] = []
    seen = set()
    for x in xs:
        if not isinstance(x, str):
            continue
        s = x.strip()
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


# -----------------------------
# Few-shot example serialization
# -----------------------------
def _serialize_span_examples(examples: List[TimeDocument], span_extractor) -> List[Dict[str, Any]]:
    """
    Store the chosen few-shot examples into JSON-friendly form.
    Matches what's effectively rendered into the Jinja template (text + spans),
    but keeps spans as a real list instead of only spans_json.
    """
    out: List[Dict[str, Any]] = []
    for d in examples or []:
        try:
            spans = span_extractor(d)
        except Exception:
            spans = []
        out.append(
            {
                "doc_id": getattr(d, "doc_id", None),
                "dataset": getattr(d, "dataset", None),
                "text": getattr(d, "text", None),
                "spans": spans,
                "spans_json": json.dumps(spans, ensure_ascii=False),
            }
        )
    return out


def _serialize_tlink_examples(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ex in examples or []:
        out.append(
            {
                "id": ex.get("id"),
                "doc_id": ex.get("doc_id"),
                "text": ex.get("text"),
                "label": ex.get("label"),
            }
        )
    return out


# -----------------------------
# Utilities (TIMEX)
# -----------------------------
def _extract_timex_texts(doc: TimeDocument) -> List[str]:
    out: List[str] = []
    for te in (doc.time_expressions or []):
        try:
            txt = (te.text or "").strip()
        except Exception:
            txt = (te.get("text") or "").strip() if isinstance(te, dict) else ""
        if txt:
            out.append(txt)
    return out


# -----------------------------
# GOLD helpers (NEW)
# -----------------------------
def _gold_timex_texts(doc: TimeDocument) -> List[str]:
    # gold = what's already annotated on doc.time_expressions
    return _extract_timex_texts(doc)


def _gold_event_texts(doc: TimeDocument) -> List[str]:
    # defined later; keep wrapper here for symmetry
    return _extract_event_texts(doc)


def _gold_compositional_texts(doc: TimeDocument) -> List[str]:
    # union of gold timex + gold event (your compositional gold extractor already does that)
    return _extract_compositional_gold_texts(doc)


def _norm_label_tlink(s: str) -> str:
    if s is None:
        return ""
    return str(s).strip().upper().replace(" ", "_").replace("-", "_")


def _extract_first_label(text: str) -> str:
    """
    Accepts:
      - raw label: "YES"
      - JSON array: ["YES"]
      - JSON list-of-dicts: [{"label": "YES", ...}]
    Returns normalized label or "".
    """
    if not text:
        return ""
    raw = text.strip()

    # Try JSON parsing first
    try:
        obj = json.loads(raw)
        if isinstance(obj, list) and obj:
            first = obj[0]
            if isinstance(first, str):
                return _norm_label_tlink(first)
            if isinstance(first, dict):
                return _norm_label_tlink(first.get("label", ""))
        if isinstance(obj, dict):
            return _norm_label_tlink(obj.get("label", ""))
    except Exception:
        pass

    # Fallback: regex search
    m = re.search(r"\b(YES|NO)\b", raw.upper())
    return m.group(1) if m else ""


def _call_openrouter_text(
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    timeout_sec: int = 60,
    headers_extra: Optional[Dict[str, str]] = None,
) -> str:
    headers_extra = headers_extra or {}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        **headers_extra,
    }
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)

    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=timeout_sec,
    )
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]


def _extract_first_second_spans(text: str) -> tuple[str, str]:
    first = ""
    second = ""
    m_e1 = re.search(r"<e1>(.*?)</e1>", text)
    m_t1 = re.search(r"<t1>(.*?)</t1>", text)
    m_e2 = re.search(r"<e2>(.*?)</e2>", text)
    m_t2 = re.search(r"<t2>(.*?)</t2>", text)
    if m_e1:
        first = m_e1.group(1)
    elif m_t1:
        first = m_t1.group(1)
    if m_e2:
        second = m_e2.group(1)
    elif m_t2:
        second = m_t2.group(1)
    return first, second


def _choose_examples(
    train_data: Optional[List[TimeDocument]],
    k: int,
    selector: str = "random",
    seed: Optional[int] = None,
) -> List[TimeDocument]:
    if not train_data:
        return []
    candidates = [d for d in train_data if _extract_timex_texts(d)]
    if not candidates:
        return []
    k = max(0, int(k))
    if k == 0:
        return []
    if selector == "first_k":
        return candidates[:k]
    if seed is not None:
        random.seed(int(seed))
    return random.sample(candidates, min(k, len(candidates)))


def _attach_predictions(doc: TimeDocument, expressions: List[str]) -> TimeDocument:
    """
    Convert list[str] into TimeExpression list and attach to a new TimeDocument.
    Best-effort character offsets via case-insensitive search.
    """
    preds: List[TimeExpression] = []
    base_lower = (doc.text or "").lower()
    used_ranges = set()

    for i, raw in enumerate(expressions or []):
        text = (raw or "").strip()
        if not text:
            continue
        needle = text.lower()
        start_char = end_char = None
        idx = base_lower.find(needle)
        while idx >= 0 and (idx, idx + len(text)) in used_ranges:
            idx = base_lower.find(needle, idx + 1)
        if idx >= 0:
            start_char = idx
            end_char = idx + len(text)
            used_ranges.add((start_char, end_char))

        preds.append(
            TimeExpression(
                tid=f"p{i}",
                text=text,
                start_char=start_char,
                end_char=end_char,
                type="DATE",
                value="",
                temporal_function=False,
                function_in_document=None,
                anchor_time=None,
            )
        )

    return TimeDocument(
        doc_id=doc.doc_id,
        text=doc.text,
        dataset=doc.dataset,
        time_expressions=preds,
        event_expressions=None,
        signal_expressions=None,
        tlinks=None,
    )


# -----------------------------
# Few-shot base (OpenRouter)
# -----------------------------
class OpenRouterFewShotBase(FewShotModelWrapper):
    """
    Minimal constructor: API key + default model from init_params.
    Other knobs are passed at run-time via run(**kwargs).
    """

    TEMPLATE_DEFAULT = "llama-3.1-openrouter-few-shot-timex.jinja"

    def __init__(self, name: str, config: dict):
        super().__init__(config)
        self.name = name

        cfg = config or {}
        self.api_key_env: str = cfg.get("api_key_env", "OPENROUTER_API_KEY")
        self.api_key: Optional[str] = cfg.get("api_key") or os.getenv(self.api_key_env)
        if not self.api_key:
            raise RuntimeError(
                f"OpenRouter API key missing. Set env var {self.api_key_env} or pass init_params.api_key."
            )

        self.model: str = cfg.get("model", "meta-llama/llama-4-maverick")

    @staticmethod
    def _default_template_path() -> str:
        here = os.path.dirname(__file__)
        return os.path.join(here, "templates", OpenRouterFewShotBase.TEMPLATE_DEFAULT)

    @staticmethod
    def _render_prompt_from_file(
        template_path: str,
        examples: List[TimeDocument],
        query_text: str,
        span_extractor,
    ) -> str:
        if not os.path.isfile(template_path):
            raise FileNotFoundError(f"Few-shot template not found: {template_path}")
        with open(template_path, "r", encoding="utf-8") as f:
            src = f.read()

        env = Environment(
            loader=BaseLoader(),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )

        payload: List[Dict[str, Any]] = []
        for d in examples:
            spans = span_extractor(d)
            payload.append(
                {
                    "text": d.text,
                    "spans_json": json.dumps(spans, ensure_ascii=False),
                }
            )

        tmpl = env.from_string(src)
        return tmpl.render(examples=payload, query_text=query_text)

    def _infer_array(
        self,
        *,
        system_prompt: str,
        full_prompt: str,
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        retries: int,
        timeout_sec: int,
        sleep_ms: int,
        headers_extra: Optional[Dict[str, str]] = None,
    ) -> List[str]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_prompt},
        ]
        return call_openrouter_json_array(
            api_key=self.api_key,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            retries=retries,
            timeout_sec=timeout_sec,
            sleep_ms_between_calls=sleep_ms,
            headers_extra=headers_extra or {},
        )

    def run(self, test_data: List, **kwargs) -> List:
        raise NotImplementedError


# -----------------------------
# TIMEX wrapper
# -----------------------------
@register(_type=MODEL_WRAPPER, _name="llama-3.1-openrouter-few-shot-timex")
class Llama31OpenRouterFewShotTimex(OpenRouterFewShotBase):
    def __init__(self, config: dict):
        super().__init__(name="Llama 3.1 (few-shot TIMEX)", config=config)

    def run(self, test_data: List[TimeDocument], **kwargs) -> List[TimeDocument]:
        model = kwargs.get("model", self.model)

        k = int(kwargs.get("k", 5))
        selector = str(kwargs.get("example_selector", "random"))
        seed = kwargs.get("seed")
        template_path = kwargs.get("template_path") or self._default_template_path()

        temperature = float(kwargs.get("temperature", 0.0))
        max_tokens = kwargs.get("max_tokens")
        retries = int(kwargs.get("retries", 3))
        timeout_sec = int(kwargs.get("timeout_sec", 60))
        sleep_ms = int(kwargs.get("sleep_ms", 400))
        headers_extra = kwargs.get("headers_extra") or {}

        max_docs = kwargs.get("max_docs")
        show_progress = bool(kwargs.get("show_progress", True))

        # Output JSONL (model postprocessed outputs)
        model_output_path = kwargs.get("model_output_path")
        split = kwargs.get("split")  # optional

        reasoning_type = str(kwargs.get("reasoning_type", "inductive")).strip().lower()
        if reasoning_type not in PROMPTS:
            raise ValueError(f"Unknown reasoning_type='{reasoning_type}'. Allowed: {REASONING_TYPES}")

        spec = PROMPTS[reasoning_type]
        system_prompt = spec["system_prompt"]
        requires_examples = bool(spec["requires_examples"])
        output_kind = spec["output_kind"]

        train_data: Optional[List[TimeDocument]] = kwargs.get("train_data")

        if requires_examples and not train_data:
            print("[WARN] reasoning_type requires examples but no train_data provided; proceeding with ZERO examples.")

        iterable = test_data if max_docs is None else test_data[: int(max_docs)]
        outputs: List[TimeDocument] = []

        iterator = tqdm(
            iterable,
            total=len(iterable),
            desc=f"Predicting TIMEX ({reasoning_type})",
            unit="doc",
            disable=not show_progress,
        )

        for doc in iterator:
            chosen_examples: List[TimeDocument] = []

            # GOLD (NEW)
            gold_time_expressions = _gold_timex_texts(doc)

            # Build USER prompt
            if requires_examples and train_data:
                chosen_examples = _choose_examples(train_data, k=k, selector=selector, seed=seed)
                try:
                    full_prompt = self._render_prompt_from_file(
                        template_path,
                        chosen_examples,
                        query_text=doc.text,
                        span_extractor=_extract_timex_texts,
                    )
                except Exception as e:
                    print(f"[WARN] few-shot prompt render failed; doc_id={getattr(doc, 'doc_id', None)}: {e}")
                    examples_block = []
                    for i, ex in enumerate(chosen_examples, 1):
                        examples_block.append(
                            f"Example {i}:\n"
                            f"Input:\n{ex.text}\n\n"
                            f"Output:\n{json.dumps(_extract_timex_texts(ex), ensure_ascii=False)}"
                        )
                    full_prompt = (
                        "Annotated Examples:\n\n"
                        + "\n\n".join(examples_block)
                        + "\n\nInput:\n"
                        + (doc.text or "")
                    )
            else:
                if output_kind == "array_strings":
                    full_prompt = (
                        "Now extract all explicit time expressions from the next input.\n"
                        "Respond ONLY with a JSON array of strings\n\n"
                        f"Input:\n{doc.text}"
                    )
                elif output_kind == "list_dicts":
                    full_prompt = (
                        "Now extract all explicit time expressions from the next input.\n"
                        "Respond ONLY with a valid JSON list of dictionaries in the required format.\n\n"
                        f"Input:\n{doc.text}"
                    )
                else:
                    full_prompt = f"Input:\n{doc.text}"

            if kwargs.get("print_prompt_once", False) and not getattr(self, "_printed_prompt", False):
                flat_prompt = (
                    "\nFULL PROMPT (MODEL INPUT)\n"
                    f"[SYSTEM PROMPT — reasoning_type={reasoning_type}]\n"
                    f"{system_prompt}\n\n"
                    "[USER PROMPT]\n"
                    f"{full_prompt}\n"
                    "END PROMPT\n"
                )
                print(flat_prompt)
                self._printed_prompt = True

            pred_doc = _attach_predictions(doc, [])
            post_processed_time_expressions: List[str] = []
            abductive_reasoning_steps: Optional[List[str]] = None

            try:
                if output_kind == "array_strings":
                    spans = self._infer_array(
                        system_prompt=system_prompt,
                        full_prompt=full_prompt,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        retries=retries,
                        timeout_sec=timeout_sec,
                        sleep_ms=sleep_ms,
                        headers_extra=headers_extra,
                    )
                    spans_pp = _postprocess_strings(spans)
                    pred_doc = _attach_predictions(doc, spans_pp)
                    post_processed_time_expressions = spans_pp

                elif output_kind == "list_dicts":
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": full_prompt},
                    ]
                    out = call_openrouter_json(
                        api_key=self.api_key,
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        retries=retries,
                        timeout_sec=timeout_sec,
                        sleep_ms_between_calls=sleep_ms,
                        headers_extra=headers_extra,
                    )
                    time_exprs: List[str] = []
                    if isinstance(out, list) and out and isinstance(out[0], dict):
                        time_exprs = out[0].get("time_expressions", []) or []
                    spans_pp = _postprocess_strings(time_exprs)
                    pred_doc = _attach_predictions(doc, spans_pp)
                    post_processed_time_expressions = spans_pp
                    abductive_reasoning_steps = _safe_reasoning_steps(out, reasoning_type)
                else:
                    raise ValueError(f"Unknown output_kind='{output_kind}'")

            except Exception as e:
                print(f"[WARN] Prediction failed for doc_id={getattr(doc, 'doc_id', None)}: {e}")
                pred_doc = _attach_predictions(doc, [])
                post_processed_time_expressions = []
                abductive_reasoning_steps = None

            # ✅ write JSONL + few-shot + GOLD
            _append_jsonl(
                model_output_path,
                {
                    "task": "timex",
                    "split": split,
                    "reasoning_type": reasoning_type,
                    "model": model,
                    "doc_id": getattr(doc, "doc_id", None),
                    "dataset": getattr(doc, "dataset", None),
                    "text": getattr(doc, "text", None),

                    # GOLD
                    "gold_time_expressions": gold_time_expressions,

                    # PRED
                    "post_processed_time_expressions": post_processed_time_expressions,

                    **({"reasoning_steps": abductive_reasoning_steps} if abductive_reasoning_steps else {}),
                    **(
                        {"few_shot_examples": _serialize_span_examples(chosen_examples, _extract_timex_texts)}
                        if chosen_examples
                        else {}
                    ),
                },
            )

            outputs.append(pred_doc)
            if sleep_ms > 0:
                time.sleep(sleep_ms / 1000.0)

        if len(outputs) != len(iterable):
            print(f"[BUG] wrapper produced {len(outputs)} predictions for {len(iterable)} inputs")

        return outputs


# -----------------------------
# TLINK wrapper
# -----------------------------
ALLOWED_TLINK_LABELS = {"YES", "NO"}


def _group_by_label(examples: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    buckets: Dict[str, List[Dict[str, Any]]] = {lab: [] for lab in ALLOWED_TLINK_LABELS}
    for ex in examples or []:
        lab = _norm_label_tlink(ex.get("label", ""))
        if lab in buckets and ex.get("text"):
            buckets[lab].append(ex)
    return buckets


def _choose_examples_tlink_balanced(
    train_data: Optional[List[Dict[str, Any]]],
    *,
    per_label_k: int = 6,
    seed: Optional[int] = None,
    interleave: bool = True,
) -> List[Dict[str, Any]]:
    if not train_data or per_label_k <= 0:
        return []
    if seed is not None:
        random.seed(int(seed))

    buckets = _group_by_label(train_data)

    sampled: Dict[str, List[Dict[str, Any]]] = {}
    for lab, items in buckets.items():
        if not items:
            sampled[lab] = []
            continue
        items = items[:]
        random.shuffle(items)
        sampled[lab] = items[:per_label_k]

    if interleave:
        out: List[Dict[str, Any]] = []
        for i in range(per_label_k):
            for lab in ALLOWED_TLINK_LABELS:
                bucket = sampled.get(lab, [])
                if i < len(bucket):
                    out.append(bucket[i])
        return out
    else:
        out: List[Dict[str, Any]] = []
        for lab in ALLOWED_TLINK_LABELS:
            out.extend(sampled.get(lab, []))
        return out


def _render_tlink_prompt(template_path: str, examples: List[Dict[str, Any]], query_text: str) -> str:
    if not os.path.isfile(template_path):
        raise FileNotFoundError(f"Few-shot TLINK template not found: {template_path}")
    with open(template_path, "r", encoding="utf-8") as f:
        src = f.read()

    env = Environment(loader=BaseLoader(), autoescape=False, trim_blocks=True, lstrip_blocks=True)
    payload = [{"text": ex["text"], "label": _norm_label_tlink(ex["label"])} for ex in examples]
    tmpl = env.from_string(src)
    return tmpl.render(examples=payload, query_text=query_text)


@register(_type=MODEL_WRAPPER, _name="llama-3.1-openrouter-few-shot-tlink")
class Llama31OpenRouterFewShotTLink(FewShotModelWrapper):
    TEMPLATE_TLINK = "Llama-3.1-openrouter-few-shot-tlink.jinja"

    def __init__(self, config: dict):
        super().__init__(config)
        cfg = config or {}
        self.api_key_env: str = cfg.get("api_key_env", "OPENROUTER_API_KEY")
        self.api_key: Optional[str] = cfg.get("api_key") or os.getenv(self.api_key_env)
        if not self.api_key:
            raise RuntimeError(
                f"OpenRouter API key missing. Set env var {self.api_key_env} or pass init_params.api_key."
            )
        self.model: str = cfg.get("model", "meta-llama/llama-4-maverick")

    def _default_template_path_tlink(self) -> str:
        here = os.path.dirname(__file__)
        return os.path.join(here, "templates", self.TEMPLATE_TLINK)

    def run(self, test_data: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        model = kwargs.get("model", self.model)

        seed = kwargs.get("seed")
        template_path = kwargs.get("template_path") or self._default_template_path_tlink()

        temperature = float(kwargs.get("temperature", 0.0))
        max_tokens = kwargs.get("max_tokens")
        retries = int(kwargs.get("retries", 3))
        timeout_sec = int(kwargs.get("timeout_sec", 60))
        sleep_ms = int(kwargs.get("sleep_ms", 400))
        headers_extra = kwargs.get("headers_extra") or {}

        max_docs = kwargs.get("max_docs")
        show_progress = bool(kwargs.get("show_progress", True))

        model_output_path = kwargs.get("model_output_path")
        split = kwargs.get("split")  # optional

        reasoning_type = str(kwargs.get("reasoning_type", "deductive")).strip().lower()
        if reasoning_type not in TLINK_PROMPTS:
            raise ValueError(
                f"Unknown TLINK reasoning_type={reasoning_type}. Allowed: {', '.join(TLINK_REASONING_TYPES)}"
            )

        prompt_cfg = TLINK_PROMPTS[reasoning_type]
        system_prompt = prompt_cfg["system_prompt"]
        requires_examples = bool(prompt_cfg["requires_examples"])
        output_kind = str(prompt_cfg["output_kind"])  # "label" or "list_dicts"

        balance_by_label = bool(kwargs.get("balance_by_label", True))
        per_label_k = int(kwargs.get("per_label_k", 6))
        k = int(kwargs.get("k", 4))
        selector = str(kwargs.get("example_selector", "random"))

        train_data: Optional[List[Dict[str, Any]]] = kwargs.get("train_data")
        if requires_examples and not train_data:
            print(
                f"[WARN] TLINK reasoning_type={reasoning_type} requires examples but no train_data provided; proceeding zero-shot."
            )

        iterable = test_data if max_docs is None else test_data[: int(max_docs)]
        outputs: List[Dict[str, Any]] = []

        iterator = tqdm(
            iterable,
            total=len(iterable),
            desc=f"Classifying TLINK ({reasoning_type})",
            unit="ex",
            disable=not show_progress,
        )

        for ex in iterator:
            chosen_examples: List[Dict[str, Any]] = []

            # GOLD (NEW): if your test_data includes gold label, keep it; otherwise None
            gold_label = ex.get("label")

            if requires_examples and train_data:
                if balance_by_label:
                    chosen_examples = _choose_examples_tlink_balanced(
                        train_data, per_label_k=per_label_k, seed=seed, interleave=True
                    )
                else:
                    cands = [d for d in train_data if d.get("text") and d.get("label")]
                    if selector == "first_k":
                        chosen_examples = cands[:k]
                    else:
                        if seed is not None:
                            random.seed(int(seed))
                        chosen_examples = random.sample(cands, min(k, len(cands))) if cands else []
            else:
                chosen_examples = []

            query_text = ex.get("text", "") or ""

            try:
                examples_block = _render_tlink_prompt(template_path, chosen_examples, query_text="")
            except Exception as e:
                print(f"[WARN] TLINK prompt render failed; id={ex.get('id')}: {e}")
                examples_block = ""

            if examples_block.strip():
                examples_block = "Annotated Examples:\n" + examples_block

            task_prefix = "Now answer for the QUERY sentence only. Return ONLY YES or NO."
            user_content = (
                f"{examples_block}\n\n"
                f"{task_prefix}\n\n"
                "Input:\n"
                f"{query_text}\n"
            ).strip()

            print_once = bool(kwargs.get("print_flat_prompt_once", False))
            print_every = bool(kwargs.get("print_flat_prompt", False))
            should_print = print_every or (print_once and not getattr(self, "_printed_flat", False))
            if should_print:
                flat_prompt = (
                    "### SYSTEM INSTRUCTION ###\n"
                    f"{system_prompt}\n\n"
                    "### USER PROMPT ###\n"
                    f"{user_content}\n"
                )
                print("\n===== FLAT PROMPT (what the model sees) =====\n")
                print(flat_prompt)
                print("\n===== END FLAT PROMPT =====\n")
                if print_once:
                    self._printed_flat = True

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]

            raw = ""
            last_err = None
            for attempt in range(max(1, retries)):
                try:
                    raw = _call_openrouter_text(
                        api_key=self.api_key,
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout_sec=timeout_sec,
                        headers_extra=headers_extra,
                    )
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    if sleep_ms > 0:
                        time.sleep(sleep_ms / 1000.0)

            if last_err is not None:
                print(f"[WARN] TLINK prediction failed for id={ex.get('id')}: {last_err}")
                pred_label = "NO"
                abductive_reasoning_steps = None
            else:
                pred_label = _extract_first_label(raw)
                if pred_label not in ALLOWED_TLINK_LABELS:
                    pred_label = "NO"
                abductive_reasoning_steps = None
                if output_kind == "list_dicts" and raw:
                    try:
                        obj = json.loads(raw)
                        abductive_reasoning_steps = _safe_reasoning_steps(obj, reasoning_type)
                    except Exception:
                        abductive_reasoning_steps = None

            out = {
                "id": ex.get("id"),
                "doc_id": ex.get("doc_id"),
                "text": ex.get("text"),
                "label": pred_label,
            }
            if abductive_reasoning_steps:
                out["reasoning_steps"] = abductive_reasoning_steps

            # ✅ write JSONL + few-shot + GOLD
            _append_jsonl(
                model_output_path,
                {
                    "task": "tlink",
                    "split": split,
                    "reasoning_type": reasoning_type,
                    "model": model,
                    "id": ex.get("id"),
                    "doc_id": ex.get("doc_id"),
                    "text": ex.get("text"),

                    # GOLD + PRED
                    "gold_label": gold_label,
                    "predicted_label": pred_label,

                    **({"reasoning_steps": abductive_reasoning_steps} if abductive_reasoning_steps else {}),
                    **(
                        {"few_shot_examples": _serialize_tlink_examples(chosen_examples)}
                        if chosen_examples
                        else {}
                    ),
                },
            )

            outputs.append(out)
            if sleep_ms > 0:
                time.sleep(sleep_ms / 1000.0)

        if len(outputs) != len(iterable):
            print(f"[BUG] TLINK wrapper produced {len(outputs)} predictions for {len(iterable)} inputs")
        return outputs


# -----------------------------
# EVENT wrapper
# -----------------------------
def _extract_event_texts(doc: TimeDocument) -> List[str]:
    out: List[str] = []
    for ee in (doc.event_expressions or []):
        try:
            txt = (ee.text or "").strip()
        except Exception:
            txt = (ee.get("text") or "").strip() if isinstance(ee, dict) else ""
        if txt:
            out.append(txt)
    return out


def _attach_event_predictions(doc: TimeDocument, expressions: List[str]) -> TimeDocument:
    preds: List[EventExpression] = []
    base_text = doc.text or ""
    base_lower = base_text.lower()
    used_ranges = set()

    for i, raw in enumerate(expressions or []):
        text = (raw or "").strip()
        if not text:
            continue
        needle = text.lower()
        start_char = end_char = None
        idx = base_lower.find(needle)
        while idx >= 0 and (idx, idx + len(text)) in used_ranges:
            idx = base_lower.find(needle, idx + 1)
        if idx >= 0:
            start_char = idx
            end_char = idx + len(text)
            used_ranges.add((start_char, end_char))

        preds.append(
            EventExpression(
                eid=f"pe{i}",
                eiid=f"pei{i}",
                type="EVENT",
                text=text,
                start_char=start_char,
                end_char=end_char,
            )
        )

    return TimeDocument(
        doc_id=doc.doc_id,
        text=doc.text,
        dataset=doc.dataset,
        time_expressions=None,
        event_expressions=preds,
        signal_expressions=None,
        tlinks=None,
    )


def _choose_examples_event(
    train_data: Optional[List[TimeDocument]],
    k: int,
    selector: str = "random",
    seed: Optional[int] = None,
) -> List[TimeDocument]:
    if not train_data:
        return []
    candidates = [d for d in train_data if _extract_event_texts(d)]
    if not candidates:
        return []
    k = max(0, int(k))
    if k == 0:
        return []
    if selector == "first_k":
        return candidates[:k]
    if seed is not None:
        random.seed(int(seed))
    return random.sample(candidates, min(k, len(candidates)))


def _render_prompt_from_file_event(
    template_path: str,
    examples: List[TimeDocument],
    query_text: str,
) -> str:
    if not os.path.isfile(template_path):
        raise FileNotFoundError(f"Few-shot template not found: {template_path}")
    with open(template_path, "r", encoding="utf-8") as f:
        src = f.read()

    env = Environment(loader=BaseLoader(), autoescape=False, trim_blocks=True, lstrip_blocks=True)
    payload: List[Dict[str, Any]] = []
    for d in examples:
        spans = _extract_event_texts(d)
        payload.append({"text": d.text, "spans_json": json.dumps(spans, ensure_ascii=False)})
    tmpl = env.from_string(src)
    return tmpl.render(examples=payload, query_text=query_text)


@register(_type=MODEL_WRAPPER, _name="llama-3.1-openrouter-few-shot-event")
class Llama31OpenRouterFewShotEvent(OpenRouterFewShotBase):
    TEMPLATE_DEFAULT = "Llama-3.1-openrouter-few-shot-event.jinja"

    def __init__(self, config: dict):
        super().__init__(name="Llama 3.1 (few-shot EVENT)", config=config)

    @staticmethod
    def _default_template_path_event() -> str:
        here = os.path.dirname(__file__)
        return os.path.join(here, "templates", Llama31OpenRouterFewShotEvent.TEMPLATE_DEFAULT)

    def run(self, test_data: List[TimeDocument], **kwargs) -> List[TimeDocument]:
        model = kwargs.get("model", self.model)

        k = int(kwargs.get("k", 5))
        selector = str(kwargs.get("example_selector", "random")).strip().lower()
        seed = kwargs.get("seed")
        template_path = kwargs.get("template_path") or self._default_template_path_event()

        temperature = float(kwargs.get("temperature", 0.0))
        max_tokens = kwargs.get("max_tokens")
        retries = int(kwargs.get("retries", 3))
        timeout_sec = int(kwargs.get("timeout_sec", 60))
        sleep_ms = int(kwargs.get("sleep_ms", 400))
        headers_extra = kwargs.get("headers_extra") or {}

        max_docs = kwargs.get("max_docs")
        show_progress = bool(kwargs.get("show_progress", True))
        print_prompt_once = bool(kwargs.get("print_prompt_once", False))

        model_output_path = kwargs.get("model_output_path")
        split = kwargs.get("split")  # optional

        reasoning_type = str(kwargs.get("reasoning_type", "inductive")).strip().lower()
        if reasoning_type not in EVENT_PROMPTS:
            raise ValueError(f"Unknown reasoning_type='{reasoning_type}'. Allowed: {EVENT_REASONING_TYPES}")

        spec = EVENT_PROMPTS[reasoning_type]
        system_prompt = spec["system_prompt"]
        requires_examples = bool(spec["requires_examples"])
        output_kind = spec["output_kind"]

        train_data: Optional[List[TimeDocument]] = kwargs.get("train_data")

        iterable = test_data if max_docs is None else test_data[: int(max_docs)]
        outputs: List[TimeDocument] = []

        iterator = tqdm(
            iterable,
            total=len(iterable),
            desc=f"Predicting EVENT ({reasoning_type})",
            unit="doc",
            disable=not show_progress,
        )

        printed = False
        for doc in iterator:
            chosen_examples: List[TimeDocument] = []

            # GOLD (NEW)
            gold_event_expressions = _gold_event_texts(doc)

            if requires_examples and train_data:
                chosen_examples = _choose_examples_event(train_data, k=k, selector=selector, seed=seed)
                full_prompt = _render_prompt_from_file_event(template_path, chosen_examples, query_text=(doc.text or ""))
            else:
                full_prompt = "Now extract all explicit event expressions from the next input.\n\n" f"Input:\n{doc.text}"

            if print_prompt_once and not printed:
                print(
                    "\nFULL PROMPT (MODEL INPUT)\n"
                    f"[SYSTEM PROMPT — EVENT reasoning_type={reasoning_type}]\n"
                    f"{system_prompt}\n\n"
                    "[USER PROMPT]\n"
                    f"{full_prompt}\n"
                    "END PROMPT\n"
                )
                printed = True

            pred_doc = _attach_event_predictions(doc, [])
            post_processed_event_expressions: List[str] = []
            abductive_reasoning_steps: Optional[List[str]] = None

            try:
                if output_kind == "array_strings":
                    spans = self._infer_array(
                        system_prompt=system_prompt,
                        full_prompt=full_prompt,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        retries=retries,
                        timeout_sec=timeout_sec,
                        sleep_ms=sleep_ms,
                        headers_extra=headers_extra,
                    )
                    spans_pp = _postprocess_strings(spans)
                    pred_doc = _attach_event_predictions(doc, spans_pp)
                    post_processed_event_expressions = spans_pp

                elif output_kind == "list_dicts":
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": full_prompt},
                    ]
                    out = call_openrouter_json(
                        api_key=self.api_key,
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        retries=retries,
                        timeout_sec=timeout_sec,
                        sleep_ms_between_calls=sleep_ms,
                        headers_extra=headers_extra,
                    )
                    event_exprs: List[str] = []
                    if isinstance(out, list) and out:
                        d0 = out[0]
                        if isinstance(d0, dict):
                            event_exprs = d0.get("events") or d0.get("event_expressions") or d0.get("expressions") or []
                    elif isinstance(out, dict):
                        event_exprs = out.get("events") or out.get("event_expressions") or out.get("expressions") or []

                    spans_pp = _postprocess_strings(event_exprs)
                    pred_doc = _attach_event_predictions(doc, spans_pp)
                    post_processed_event_expressions = spans_pp
                    abductive_reasoning_steps = _safe_reasoning_steps(out, reasoning_type)
                else:
                    raise ValueError(f"Unknown output_kind='{output_kind}'")

            except Exception as e:
                print(f"[WARN] EVENT prediction failed for doc_id={getattr(doc, 'doc_id', None)}: {e}")
                pred_doc = _attach_event_predictions(doc, [])
                post_processed_event_expressions = []
                abductive_reasoning_steps = None

            _append_jsonl(
                model_output_path,
                {
                    "task": "event",
                    "split": split,
                    "reasoning_type": reasoning_type,
                    "model": model,
                    "doc_id": getattr(doc, "doc_id", None),
                    "dataset": getattr(doc, "dataset", None),
                    "text": getattr(doc, "text", None),

                    # GOLD
                    "gold_event_expressions": gold_event_expressions,

                    # PRED
                    "post_processed_event_expressions": post_processed_event_expressions,

                    **({"reasoning_steps": abductive_reasoning_steps} if abductive_reasoning_steps else {}),
                    **(
                        {"few_shot_examples": _serialize_span_examples(chosen_examples, _extract_event_texts)}
                        if chosen_examples
                        else {}
                    ),
                },
            )

            outputs.append(pred_doc)
            if sleep_ms > 0:
                time.sleep(sleep_ms / 1000.0)

        return outputs


# -----------------------------
# COMPOSITIONAL wrapper
# -----------------------------
COMMON_SUFFIX_COMPOSITIONAL = (
    "\n\nNow extract expressions for the following input.\n"
    "Return ONLY JSON in the required output format.\n\n"
    "Input:\n"
)


def _extract_compositional_gold_texts(doc: TimeDocument) -> List[str]:
    out: List[str] = []
    for te in (doc.time_expressions or []):
        try:
            txt = te.text
        except Exception:
            txt = te.get("text") if isinstance(te, dict) else ""
        if txt:
            out.append(str(txt))
    for ee in (getattr(doc, "event_expressions", None) or []):
        try:
            txt = ee.text
        except Exception:
            txt = ee.get("text") if isinstance(ee, dict) else ""
        if txt:
            out.append(str(txt))

    seen = set()
    merged: List[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            merged.append(x)
    return merged


def _choose_examples_compositional(
    train_data: Optional[List[TimeDocument]],
    k: int,
    selector: str = "random",
    seed: Optional[int] = None,
) -> List[TimeDocument]:
    if not train_data:
        return []
    candidates = [d for d in train_data if _extract_compositional_gold_texts(d)]
    if not candidates:
        return []
    k = max(0, int(k))
    if k == 0:
        return []
    if selector == "first_k":
        return candidates[:k]
    if seed is not None:
        random.seed(int(seed))
    return random.sample(candidates, min(k, len(candidates)))


def _render_prompt_from_file_compositional(
    template_path: str,
    examples: List[TimeDocument],
    query_text: str,
) -> str:
    if not os.path.isfile(template_path):
        raise FileNotFoundError(f"Few-shot template not found: {template_path}")
    with open(template_path, "r", encoding="utf-8") as f:
        src = f.read()
    env = Environment(loader=BaseLoader(), autoescape=False, trim_blocks=True, lstrip_blocks=True)

    payload: List[Dict[str, Any]] = []
    for d in examples:
        spans = _extract_compositional_gold_texts(d)
        payload.append({"text": d.text, "spans_json": json.dumps(spans, ensure_ascii=False)})

    tmpl = env.from_string(src)
    return tmpl.render(examples=payload, query_text=query_text)


def _attach_predictions_compositional(doc: TimeDocument, expressions: List[str]) -> TimeDocument:
    preds: List[TimeExpression] = []
    base_text = doc.text or ""
    base_lower = base_text.lower()
    used_ranges = set()
    seen = set()
    cleaned: List[str] = []

    for x in expressions or []:
        x = (x or "").strip()
        if not x:
            continue
        if x in seen:
            continue
        seen.add(x)
        cleaned.append(x)

    for i, text in enumerate(cleaned):
        needle = text.lower()
        start_char = end_char = None
        idx = base_lower.find(needle)
        while idx >= 0 and (idx, idx + len(text)) in used_ranges:
            idx = base_lower.find(needle, idx + 1)
        if idx >= 0:
            start_char = idx
            end_char = idx + len(text)
            used_ranges.add((start_char, end_char))

        preds.append(
            TimeExpression(
                tid=f"pc{i}",
                text=text,
                start_char=start_char,
                end_char=end_char,
                type="DATE",
                value="",
                temporal_function=False,
                function_in_document=None,
                anchor_time=None,
            )
        )

    return TimeDocument(
        doc_id=doc.doc_id,
        text=doc.text,
        dataset=doc.dataset,
        time_expressions=preds,
        event_expressions=None,
        signal_expressions=None,
        tlinks=None,
    )




@register(_type=MODEL_WRAPPER, _name="llama-3.1-openrouter-few-shot-compositional")
class Llama31OpenRouterFewShotCompositional(OpenRouterFewShotBase):
    
    TEMPLATE_DEFAULT = "Llama-3.1-openrouter-few-shot-compositional.jinja"

    def __init__(self, config: dict):
        super().__init__(name="Llama 3.1 (few-shot COMPOSITIONAL)", config=config)

    @staticmethod
    def _default_template_path_comp() -> str:
        here = os.path.dirname(__file__)
        return os.path.join(here, "templates", Llama31OpenRouterFewShotCompositional.TEMPLATE_DEFAULT)

    def run(self, test_data: List[TimeDocument], **kwargs) -> List[TimeDocument]:
        model = kwargs.get("model", self.model)

        # few-shot controls
        k = int(kwargs.get("k", 5))
        selector = str(kwargs.get("example_selector", "random")).strip().lower()
        seed = kwargs.get("seed")
        template_path = kwargs.get("template_path") or self._default_template_path_comp()

        # decoding / runtime
        temperature = float(kwargs.get("temperature", 0.0))
        max_tokens = kwargs.get("max_tokens")
        retries = int(kwargs.get("retries", 3))
        timeout_sec = int(kwargs.get("timeout_sec", 60))
        sleep_ms = int(kwargs.get("sleep_ms", 400))
        headers_extra = kwargs.get("headers_extra") or {}

        # run controls
        max_docs = kwargs.get("max_docs")
        show_progress = bool(kwargs.get("show_progress", True))
        print_prompt_once = bool(kwargs.get("print_prompt_once", False))

        # output JSONL
        model_output_path = kwargs.get("model_output_path")
        split = kwargs.get("split")  # optional

        # ✅ FIX 1: use COMP_PROMPTS (system_prompt + requires_examples + output_kind)
        reasoning_type = str(kwargs.get("reasoning_type", "inductive")).strip().lower()
        if reasoning_type not in COMP_PROMPTS:
            raise ValueError(
                f"Unknown compositional reasoning_type='{reasoning_type}'. "
                f"Allowed: {COMP_REASONING_TYPES}"
            )

        spec = COMP_PROMPTS[reasoning_type]
        system_prompt = spec["system_prompt"]
        requires_examples = bool(spec.get("requires_examples", False))
        output_kind = str(spec.get("output_kind", "array_strings")).strip().lower()

        train_data: Optional[List[TimeDocument]] = kwargs.get("train_data")

        if requires_examples and not train_data:
            print(
                f"[WARN] compositional reasoning_type='{reasoning_type}' requires examples "
                f"but no train_data was provided; proceeding ZERO-shot."
            )

        iterable = test_data if max_docs is None else test_data[: int(max_docs)]
        outputs: List[TimeDocument] = []

        iterator = tqdm(
            iterable,
            total=len(iterable),
            desc=f"Predicting COMPOSITIONAL ({reasoning_type})",
            unit="doc",
            disable=not show_progress,
        )

        printed = False
        for doc in iterator:
            chosen_examples: List[TimeDocument] = []

            # GOLD: union of gold timex + gold event (deduped)
            gold_expressions = _gold_compositional_texts(doc)

            
            if requires_examples and train_data:
                chosen_examples = _choose_examples_compositional(
                    train_data, k=k, selector=selector, seed=seed
                )
                
                try:
                    full_prompt = _render_prompt_from_file_compositional(
                        template_path=template_path,
                        examples=chosen_examples,
                        query_text=(doc.text or ""),
                    )
                except Exception as e:
                    # fallback prompt if template render fails
                    print(f"[WARN] compositional prompt render failed; doc_id={getattr(doc, 'doc_id', None)}: {e}")
                    examples_block = []
                    for i, ex in enumerate(chosen_examples, 1):
                        spans = _extract_compositional_gold_texts(ex)
                        examples_block.append(
                            f"Example {i}:\n"
                            f"Input:\n{ex.text}\n\n"
                            f"Output:\n{json.dumps(spans, ensure_ascii=False)}"
                        )
                    full_prompt = (
                        "Annotated Examples:\n\n"
                        + "\n\n".join(examples_block)
                        + "\n\nNow extract BOTH time expressions and event expressions from the next input.\n"
                        + "Respond ONLY with a JSON array of strings.\n\n"
                        + f"Input:\n{doc.text}"
                    )
            else:
                # zero-shot fallback (still uses system_prompt from COMP_PROMPTS)
                if output_kind == "array_strings":
                    full_prompt = (
                        "Now extract BOTH time expressions and event expressions from the next input.\n"
                        "Respond ONLY with a JSON array of strings.\n\n"
                        f"Input:\n{doc.text}"
                    )
                elif output_kind == "list_dicts":
                    full_prompt = (
                        "Now extract BOTH time expressions and event expressions from the next input.\n"
                        "Respond ONLY with valid JSON in the required dict/list format.\n\n"
                        f"Input:\n{doc.text}"
                    )
                else:
                    full_prompt = f"Input:\n{doc.text}"

            if print_prompt_once and not printed:
                print(
                    "\nFULL PROMPT (MODEL INPUT)\n"
                    f"[SYSTEM PROMPT — COMPOSITIONAL reasoning_type={reasoning_type}]\n"
                    f"{system_prompt}\n\n"
                    "[USER PROMPT]\n"
                    f"{full_prompt}\n"
                    "END PROMPT\n"
                )
                printed = True

            pred_doc = _attach_predictions_compositional(doc, [])
            post_processed_expressions: List[str] = []
            abductive_reasoning_steps: Optional[List[str]] = None

            try:
                if output_kind == "array_strings":
                    spans = self._infer_array(
                        system_prompt=system_prompt,
                        full_prompt=full_prompt,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        retries=retries,
                        timeout_sec=timeout_sec,
                        sleep_ms=sleep_ms,
                        headers_extra=headers_extra,
                    )
                    spans_pp = _postprocess_strings(spans)
                    pred_doc = _attach_predictions_compositional(doc, spans_pp)
                    post_processed_expressions = spans_pp

                elif output_kind == "list_dicts":
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": full_prompt},
                    ]
                    out = call_openrouter_json(
                        api_key=self.api_key,
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        retries=retries,
                        timeout_sec=timeout_sec,
                        sleep_ms_between_calls=sleep_ms,
                        headers_extra=headers_extra,
                        response_schema=_compositional_list_of_dicts_schema(),
                    )
                    """
                    print("\n[DEBUG RAW MODEL OUTPUT]")
                    print("doc_id:", getattr(doc, "doc_id", None))
                    print("reasoning_type:", reasoning_type)
                    print("output_kind:", output_kind)
                    print("type(out):", type(out))
                    print("out:", out)
                      """
                    # robust extraction of expression strings
                    exprs: List[str] = []

                    if out is None:
                        exprs = []

                    elif isinstance(out, list) and out:
                        if all(isinstance(x, str) for x in out):
                            exprs = out
                        elif isinstance(out[0], dict):
                            d0 = out[0]
                            raw_exprs = (
                                d0.get("expressions")
                                or d0.get("compositional_expressions")
                                or d0.get("outputs")
                                or []
                            )

                            if isinstance(raw_exprs, list):
                                if all(isinstance(x, str) for x in raw_exprs):
                                    exprs = raw_exprs
                                elif all(isinstance(x, dict) for x in raw_exprs):
                                    exprs = [
                                        (x.get("span") or x.get("text") or "").strip()
                                        for x in raw_exprs
                                        if isinstance(x, dict) and (x.get("span") or x.get("text"))
                                    ]

                    elif isinstance(out, dict):
                        raw_exprs = (
                            out.get("expressions")
                            or out.get("compositional_expressions")
                            or out.get("outputs")
                            or []
                        )

                        if isinstance(raw_exprs, list):
                            if all(isinstance(x, str) for x in raw_exprs):
                                exprs = raw_exprs
                            elif all(isinstance(x, dict) for x in raw_exprs):
                                exprs = [
                                    (x.get("span") or x.get("text") or "").strip()
                                    for x in raw_exprs
                                    if isinstance(x, dict) and (x.get("span") or x.get("text"))
                                ]

                    spans_pp = _postprocess_strings(exprs)
                    pred_doc = _attach_predictions_compositional(doc, spans_pp)
                    post_processed_expressions = spans_pp
                    abductive_reasoning_steps = _safe_reasoning_steps(out, reasoning_type)

                else:
                    raise ValueError(f"Unknown output_kind='{output_kind}'")

            except Exception as e:
                print(f"[WARN] COMPOSITIONAL prediction failed for doc_id={getattr(doc, 'doc_id', None)}: {e}")
                pred_doc = _attach_predictions_compositional(doc, [])
                post_processed_expressions = []
                abductive_reasoning_steps = None

            # ✅ JSONL write includes GOLD + PRED + FEW-SHOT examples
            _append_jsonl(
                model_output_path,
                {
                    "task": "compositional",
                    "split": split,
                    "reasoning_type": reasoning_type,
                    "model": model,
                    "doc_id": getattr(doc, "doc_id", None),
                    "dataset": getattr(doc, "dataset", None),
                    "text": getattr(doc, "text", None),

                    # GOLD
                    "gold_expressions": gold_expressions,

                    # PRED
                    "post_processed_expressions": post_processed_expressions,

                    **({"reasoning_steps": abductive_reasoning_steps} if abductive_reasoning_steps else {}),
                    **(
                        {"few_shot_examples": _serialize_span_examples(chosen_examples, _extract_compositional_gold_texts)}
                        if chosen_examples
                        else {}
                    ),
                },
            )

            outputs.append(pred_doc)
            if sleep_ms > 0:
                time.sleep(sleep_ms / 1000.0)

        if len(outputs) != len(iterable):
            print(f"[BUG] wrapper produced {len(outputs)} predictions for {len(iterable)} inputs")

        return outputs


# ─────────────────────────────────────────────────────────────────────────────
# CAUSAL wrapper
# ─────────────────────────────────────────────────────────────────────────────

def _choose_examples_causal_balanced(
    train_data: Optional[List[Dict[str, Any]]],
    label_space: List[str],
    *,
    per_label_k: int = 4,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Balanced sampler — per_label_k examples per label, interleaved."""
    if not train_data or per_label_k <= 0 or not label_space:
        return []
    if seed is not None:
        random.seed(int(seed))

    buckets: Dict[str, List[Dict[str, Any]]] = {lab: [] for lab in label_space}
    for ex in train_data:
        lab = (ex.get("label") or "").strip().upper()
        if lab in buckets and ex.get("text"):
            buckets[lab].append(ex)

    sampled: Dict[str, List] = {}
    for lab, items in buckets.items():
        items = items[:]
        random.shuffle(items)
        sampled[lab] = items[:per_label_k]

    out: List[Dict[str, Any]] = []
    for i in range(per_label_k):
        for lab in label_space:
            bucket = sampled.get(lab, [])
            if i < len(bucket):
                out.append(bucket[i])
    return out


def _render_causal_prompt(
    template_path: str,
    examples: List[Dict[str, Any]],
    query_text: str,
) -> str:
    if not os.path.isfile(template_path):
        raise FileNotFoundError(f"Causal few-shot template not found: {template_path}")
    with open(template_path, "r", encoding="utf-8") as f:
        src = f.read()
    env = Environment(loader=BaseLoader(), autoescape=False, trim_blocks=True, lstrip_blocks=True)
    payload = [{"text": ex["text"], "label": ex["label"]} for ex in examples]
    tmpl = env.from_string(src)
    return tmpl.render(examples=payload, query_text=query_text)


@register(_type=MODEL_WRAPPER, _name="openrouter-causal")
class OpenRouterFewShotCausal(FewShotModelWrapper):
    """
    Few-shot / zero-shot causal relation classification wrapper.

    Label space is loaded dynamically from the dataset records
    (each record carries label_space / label_space_display set by the causal
    dataset loader), so this wrapper works with any number of classes.

    Registered as: "openrouter-causal"
    """

    TEMPLATE_CAUSAL = "openrouter-causal-few-shot.jinja"

    def __init__(self, config: dict):
        super().__init__(config)
        cfg = config or {}
        self.api_key_env: str = cfg.get("api_key_env", "OPENROUTER_API_KEY")
        self.api_key: Optional[str] = cfg.get("api_key") or os.getenv(self.api_key_env)
        if not self.api_key:
            raise RuntimeError(
                f"OpenRouter API key missing. Set env var {self.api_key_env} or pass init_params.api_key."
            )
        self.model: str = cfg.get("model", "meta-llama/llama-4-maverick")

    def _default_template_path(self) -> str:
        here = os.path.dirname(__file__)
        return os.path.join(here, "templates", self.TEMPLATE_CAUSAL)

    def run(self, test_data: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        model = kwargs.get("model", self.model)

        seed = kwargs.get("seed")
        template_path = kwargs.get("template_path") or self._default_template_path()

        temperature = float(kwargs.get("temperature", 0.0))
        max_tokens = kwargs.get("max_tokens")
        retries = int(kwargs.get("retries", 3))
        timeout_sec = int(kwargs.get("timeout_sec", 60))
        sleep_ms = int(kwargs.get("sleep_ms", 400))
        headers_extra = kwargs.get("headers_extra") or {}

        max_docs = kwargs.get("max_docs")
        show_progress = bool(kwargs.get("show_progress", True))
        model_output_path = kwargs.get("model_output_path")
        split = kwargs.get("split")

        per_label_k = int(kwargs.get("per_label_k", 4))
        k = int(kwargs.get("k", 8))

        reasoning_type = str(kwargs.get("reasoning_type", "deductive")).strip().lower()
        if reasoning_type not in CAUSAL_PROMPTS:
            raise ValueError(
                f"Unknown causal reasoning_type='{reasoning_type}'. "
                f"Allowed: {CAUSAL_REASONING_TYPES}"
            )

        prompt_cfg = CAUSAL_PROMPTS[reasoning_type]
        requires_examples = bool(prompt_cfg["requires_examples"])
        output_kind = str(prompt_cfg["output_kind"])

        train_data: Optional[List[Dict[str, Any]]] = kwargs.get("train_data")
        if requires_examples and not train_data:
            print(
                f"[WARN] causal reasoning_type='{reasoning_type}' requires examples "
                "but no train_data provided; proceeding zero-shot."
            )

        iterable = test_data if max_docs is None else test_data[: int(max_docs)]
        outputs: List[Dict[str, Any]] = []

        # Resolve label space from the first record — the loader attaches it to every record.
        label_space: List[str] = []
        label_space_display: List[str] = []
        if iterable:
            label_space = list(iterable[0].get("label_space") or [])
            label_space_display = list(iterable[0].get("label_space_display") or label_space)
        if not label_space:
            raise RuntimeError(
                "Causal records are missing 'label_space'. "
                "Make sure the dataset loader attaches it (all causal_hf loaders do this)."
            )

        # Build system prompt once — format {label_defs} with the dataset's label space.
        label_defs_lines = []
        for lab, disp in zip(label_space, label_space_display if label_space_display else label_space):
            if disp and disp.upper() != lab.upper():
                label_defs_lines.append(f"- {lab} ({disp})")
            else:
                label_defs_lines.append(f"- {lab}")
        label_defs = "\n".join(label_defs_lines)
        system_prompt = prompt_cfg["system_prompt"].replace("{label_defs}", label_defs)

        iterator = tqdm(
            iterable,
            total=len(iterable),
            desc=f"Classifying CAUSAL ({reasoning_type})",
            unit="ex",
            disable=not show_progress,
        )

        printed = False
        for ex in iterator:
            gold_label = ex.get("label")
            query_text = ex.get("text", "") or ""

            # ── few-shot examples ────────────────────────────────────────────
            chosen_examples: List[Dict[str, Any]] = []
            if requires_examples and train_data:
                chosen_examples = _choose_examples_causal_balanced(
                    train_data, label_space, per_label_k=per_label_k, seed=seed
                )

            # ── user prompt ──────────────────────────────────────────────────
            if chosen_examples:
                try:
                    examples_block = _render_causal_prompt(template_path, chosen_examples, query_text="")
                    if examples_block.strip():
                        examples_block = "Annotated Examples:\n" + examples_block
                except Exception as e:
                    print(f"[WARN] causal prompt render failed; id={ex.get('id')}: {e}")
                    lines = []
                    for i, ce in enumerate(chosen_examples, 1):
                        lines.append(f"Example {i}:\nInput:\n{ce['text']}\n\nLabel:\n{ce['label']}\n")
                    examples_block = "Annotated Examples:\n" + "\n".join(lines)
            else:
                examples_block = ""

            label_list = " | ".join(label_space)
            task_suffix = (
                f"Now classify the QUERY below. Return ONLY one label from: {label_list}"
                if output_kind == "label_string"
                else "Now classify the QUERY below. Return ONLY a valid JSON list with one dict as described."
            )
            user_content = (
                f"{examples_block}\n\n{task_suffix}\n\nInput:\n{query_text}\n"
            ).strip()

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]

            if kwargs.get("print_flat_prompt_once", False) and not printed:
                print("\n===== CAUSAL FLAT PROMPT =====")
                print(f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user_content}")
                print("===== END PROMPT =====\n")
                printed = True

            # ── inference ────────────────────────────────────────────────────
            pred_label = ""
            abductive_steps: Optional[List[str]] = None

            for attempt in range(max(1, retries)):
                try:
                    if output_kind == "label_string":
                        raw = call_openrouter_label_array(
                            api_key=self.api_key,
                            model=model,
                            messages=messages,
                            allowed_labels=label_space,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            timeout_sec=timeout_sec,
                            sleep_ms_between_calls=sleep_ms,
                            headers_extra=headers_extra,
                        )
                        # call_openrouter_label_array returns a list; take the first element
                        if isinstance(raw, list) and raw:
                            pred_label = str(raw[0]).strip().upper()
                        elif isinstance(raw, str):
                            pred_label = raw.strip().upper()
                        if pred_label not in label_space:
                            pred_label = label_space[0] if label_space else ""
                    else:
                        out = call_openrouter_json(
                            api_key=self.api_key,
                            model=model,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            retries=1,
                            timeout_sec=timeout_sec,
                            sleep_ms_between_calls=sleep_ms,
                            headers_extra=headers_extra,
                        )
                        raw_label = ""
                        if isinstance(out, list) and out and isinstance(out[0], dict):
                            raw_label = out[0].get("label", "")
                            abductive_steps = _safe_reasoning_steps(out, reasoning_type)
                        elif isinstance(out, dict):
                            raw_label = out.get("label", "")
                            abductive_steps = _safe_reasoning_steps(out, reasoning_type)
                        pred_label = str(raw_label).strip().upper()
                        if pred_label not in label_space:
                            pred_label = label_space[0] if label_space else ""
                    break
                except (_ChunkedEncodingError, Exception) as e:
                    backoff = sleep_ms * (2 ** attempt) / 1000.0
                    print(f"[WARN] causal attempt {attempt+1}/{retries} failed for id={ex.get('id')}: {type(e).__name__}")
                    if attempt == retries - 1:
                        print(f"[WARN] causal giving up on id={ex.get('id')}: {e}")
                        pred_label = label_space[0] if label_space else ""
                    if sleep_ms > 0:
                        time.sleep(backoff)

            out_rec = {
                "id": ex.get("id"),
                "doc_id": ex.get("doc_id"),
                "text": query_text,
                "label": pred_label,
            }
            if abductive_steps:
                out_rec["reasoning_steps"] = abductive_steps

            _append_jsonl(
                model_output_path,
                {
                    "task": "causal",
                    "split": split,
                    "reasoning_type": reasoning_type,
                    "model": model,
                    "id": ex.get("id"),
                    "doc_id": ex.get("doc_id"),
                    "text": query_text,
                    "gold_label": gold_label,
                    "predicted_label": pred_label,
                    "label_space": label_space,
                    **({"reasoning_steps": abductive_steps} if abductive_steps else {}),
                    **(
                        {"few_shot_examples": _serialize_tlink_examples(chosen_examples)}
                        if chosen_examples
                        else {}
                    ),
                },
            )

            outputs.append(out_rec)
            if sleep_ms > 0:
                time.sleep(sleep_ms / 1000.0)

        if len(outputs) != len(iterable):
            print(f"[BUG] causal wrapper produced {len(outputs)} predictions for {len(iterable)} inputs")
        return outputs
    