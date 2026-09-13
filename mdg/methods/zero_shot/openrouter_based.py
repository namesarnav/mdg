from __future__ import annotations
from typing import List, Optional, Dict
import os
import time
from tqdm.auto import tqdm
from mdg.methods.zero_shot import ZeroShotModelWrapper
from mdg.methods.openrouter_utils import call_openrouter_json_array, call_openrouter_label_array
from mdg.datamodels import TimeDocument, TimeExpression
from mdg.registry import register, MODEL_WRAPPER
import re 


def _extract_first_second_spans(text: str) -> tuple[str, str]:
    """
    Extract FIRST span (inside <e1> or <t1>) and SECOND span (inside <e2> or <t2>)
    from the raw sentence string.
    """
    first = ""
    second = ""

    m_e1 = re.search(r"<e1>(.*?)</e1>", text)
    m_t1 = re.search(r"<t1>(.*?)</t1>", text)
    m_e2 = re.search(r"<e2>(.*?)</e2>", text)
    m_t2 = re.search(r"<t2>(.*?)</t2>", text)

    # FIRST: prefer e1, fallback to t1
    if m_e1:
        first = m_e1.group(1)
    elif m_t1:
        first = m_t1.group(1)

    # SECOND: prefer e2, fallback to t2
    if m_e2:
        second = m_e2.group(1)
    elif m_t2:
        second = m_t2.group(1)

    return first, second

class OpenRouterBasedWrapper(ZeroShotModelWrapper):
    """
    Base wrapper for OpenRouter-backed methods.
    Minimal constructor: only API key + default model from init_params.
    All other knobs are supplied at run-time via run(**kwargs).
    """

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

        # Default model family (can be overridden per run)
        self.model: str = cfg.get("model", "meta-llama/llama-4-maverick")

    # ---------- Prompt helpers ----------
    @staticmethod
    def _system_instruction() -> str:
        
        return (
            "Extract every explicit time expression that appears in the user text. "
            "Return ONLY a JSON array of strings, one string per time expression, "
            "no duplicates, in order of appearance."
        )

    def _build_messages(self, text: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": self._system_instruction()},
            {"role": "user", "content": text},
        ]

    # ---------- Post-processing ----------
    @staticmethod
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

            # Case-insensitive find, avoid reusing the exact same span.
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

        # CRITICAL: preserve doc_id and dataset so evaluators can align by ID
        return TimeDocument(
            doc_id=doc.doc_id,
            text=doc.text,
            dataset=doc.dataset,
            time_expressions=preds,
            event_expressions=None,
            signal_expressions=None,
            tlinks=None,
        )

    # Base class remains abstract
    def run(self, test_data: List, **kwargs) -> List:
        raise NotImplementedError("OpenRouterBasedWrapper is abstract; use a concrete subclass.")


@register(_type=MODEL_WRAPPER, _name="llama-3.1-openrouter-zero-shot-timex")
class Llama31OpenRouterWrapperTimex(OpenRouterBasedWrapper):
    """
    Zero-shot timex extractor via OpenRouter.
    Input : List[TimeDocument]
    Output: List[TimeDocument] with predicted time_expressions attached
    """

    def __init__(self, config: dict):
        super().__init__(name="Llama 3.1", config=config)

    def _infer_strings(
        self,
        text: str,
        *,
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        retries: int,
        timeout_sec: int,
        sleep_ms: int,
        headers_extra: Optional[Dict[str, str]] = None,
    ) -> List[str]:
        messages = self._build_messages(text)
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

    def run(self, test_data: List[TimeDocument], **kwargs) -> List[TimeDocument]:
        """
        Per-run overrides (all optional):
          - model: str (default = self.model)
          - temperature: float (default = 0.0)
          - max_tokens: Optional[int]
          - retries: int (default = 3)
          - timeout_sec: int (default = 60)
          - sleep_ms: int (default = 400)
          - headers_extra: Dict[str, str]
          - max_docs: int (evaluate only first N docs)
          - show_progress: bool (default = True)
        """
        model         = kwargs.get("model", self.model)
        temperature   = float(kwargs.get("temperature", 0.0))
        max_tokens    = kwargs.get("max_tokens")
        retries       = int(kwargs.get("retries", 3))
        timeout_sec   = int(kwargs.get("timeout_sec", 60))
        sleep_ms      = int(kwargs.get("sleep_ms", 400))
        headers_extra = kwargs.get("headers_extra") or {}
        max_docs      = kwargs.get("max_docs")
        show_progress = bool(kwargs.get("show_progress", True))

        iterable = test_data if max_docs is None else test_data[: int(max_docs)]
        outputs: List[TimeDocument] = []

        iterator = iterable
        if tqdm and show_progress:
            iterator = tqdm(iterable, total=len(iterable), desc="Predicting timex spans", unit="doc")

        for doc in iterator:
            try:
                exprs = self._infer_strings(
                    doc.text,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    retries=retries,
                    timeout_sec=timeout_sec,
                    sleep_ms=sleep_ms,
                    headers_extra=headers_extra,
                )
            except Exception as e:
                # On failure, return an empty prediction for this doc instead of aborting the loop
                print(f"Prediction failed for doc_id={getattr(doc, 'doc_id', None)}: {e}")
                exprs = []

            pred_doc = self._attach_predictions(doc, exprs)
            outputs.append(pred_doc)

            if sleep_ms > 0:
                time.sleep(sleep_ms / 1000.0)

        
        if len(outputs) != len(iterable):
            print(f"wrapper produced {len(outputs)} predictions for {len(iterable)} inputs")

        return outputs

ALLOWED_TLINK_LABELS = {
    "BEFORE", "AFTER"
}

def _norm_label(s: str) -> str:
    if s is None:
        return ""
    x = str(s).strip().upper().replace(" ", "_").replace("-", "_")
    mapping = {
        "ISINCLUDED": "IS_INCLUDED",
        "INCLUDED": "IS_INCLUDED",
        "NO_RELATION": "NONE",
        "NO_TEMPORAL_RELATION": "NONE",
    }
    return mapping.get(x, x)
"""
TLINK_SYSTEM_PROMPT = (
    "You are given a sentence with two marked spans.\n"
    "<e1> … </e1> and <e2> … </e2> mark EVENT spans.\n\n"
    "<t1> … </t1> and <t2> … </t2> mark TIME spans.\n\n"
    "Role mapping:\n"
    "The FIRST element is the span inside <e1>...</e1> or <t1>...</t1>.\n\n"
    "The SECOND element is the span inside <e2>...</e2> or <t2>...</t2>.\n\n"
    "Pair types can be: event–event, time–time, event–time, or time–event.\n"
    "TASK:\n"
    "Classify the temporal relation of the FIRST element WITH RESPECT TO the SECOND element (do not flip direction).\n"
    "Possible relation labels:\n"
    "BEFORE, AFTER, INCLUDES, IS_INCLUDED, SIMULTANEOUS, IDENTITY, DURING, NONE.\n\n"
    "BEFORE: The first element (inside <e1> or <t1>) happens earlier in time than the second element (inside <e2> or <t2>). The first is finished before the second begins (no overlap). Use BEFORE when the first is a prior action/decision/announcement and the second is a later report, reaction, or a future event that will occur after the first.\n\n"
    "AFTER:  The first element (inside <e1> or <t1>) happens later in time than the second element (inside <e2> or <t2>). The second element occurs first and the first element follows it or is a reaction or result of it.\n\n"
    "INCLUDES: The first element (inside <e1> or <t1>) fully contains the second element (inside <e2> or <t2>) within its time span or situation.The first element lasts longer in time and the second element happens completely inside it.\n\n"
    "IS_INCLUDED: The first element (inside <e1> or <t1>) happens completely within the time span or situation of the second element (inside <e2> or <t2>).The second element is a broader time period, process, or context that fully contains the first.\n\n"
    "SIMULTANEOUS: The first element (inside <e1> or <t1>) and the second element (inside <e2> or <t2>) happen at the same time or overlap in the same time window (neither clearly before the other, and neither fully contains the other).\n\n"
    "IDENTITY: The first element (inside <e1> or <t1>) and the second element (inside <e2> or <t2>) refer to the exact same event or the exact same time expression.They describe one identical situation or moment, not two separate actions or times.\n\n"
    "DURING: The first element (inside <e1> or <t1>) and the second element (inside <e2> or <t2>) happen in the same general time frame, but the first element occurs partly within the time span of the second, not completely inside it. They overlap in time, but one does not fully include the other.\n\n"
    "NONE: The first element (inside <e1> or <t1>) and the second element (inside <e2> or <t2>) do not have any clear or direct temporal relationship.\n\n"
    "Output Format:\n"
    "Output exactly one label ONLY (no punctuation, no explanation, no extra text) in UPPERCASE letters. "

    "IS_INCLUDED: The first element happens completely inside the time span or interval of the second.\n"
    "Example: \"The meeting was <e1>held</e1> on <t1>Thursday morning</t1>.\" → IS_INCLUDED.\n"
    "Clarification: The relation is IS_INCLUDED because the event <e1>held</e1> occurs entirely within the time span indicated by <t1>Thursday morning</t1>. The event starts and ends inside that time interval, with no part extending outside it.\n\n"

    "SIMULTANEOUS: Both elements happen at the same time or overlap with no before/after relation.\n"
    "Example: \"President Bush <e1>denounced</e1> Saddam's policies and <e2>said</e2> the U.S. would respond.\" → SIMULTANEOUS.\n"
    "Clarification: The relation is SIMULTANEOUS because denounced (e1) and said (e2) happen at the same time during a single, continuous reporting moment. Neither event clearly comes before or after the other—they overlap completely.\n\n"

    "NONE: No clear or direct temporal relationship is expressed.\n"
    "Example: \"It was unclear whether he was <e1>charged</e1> by <e2>summons</e2>.\" → NONE.\n"
    "Clarification: Use NONE when there is no explicit or inferable temporal link.\n\n"

)


TLINK_SYSTEM_PROMPT = (
    "You are given a sentence with two marked spans.\n"
    "<e1> … </e1> and <e2> … </e2> mark EVENT spans.\n\n"
    "<t1> … </t1> and <t2> … </t2> mark TIME spans.\n\n"
    "Role mapping:\n"
    "The FIRST element is the span inside <e1>...</e1> or <t1>...</t1>.\n\n"
    "The SECOND element is the span inside <e2>...</e2> or <t2>...</t2>.\n\n"
    "Pair types can be: event–event, time–time, event–time, or time–event.\n"
    "TASK:\n"
    "Classify the temporal relation of the FIRST element WITH RESPECT TO the SECOND element (do not flip direction).\n"
    "Possible relation labels:\n"
    "BEFORE , AFTER\n\n"

    "BEFORE: The first element happens earlier in time than the second. It finishes before the second begins.\n"
    "Example: \"Speaking to journalists after <e1>talks</e1> with Kofi Annan, Denktash <e2>said</e2> he was optimistic.\" → BEFORE.\n"
    "Clarification: The word “after” tells us the talks (e1) happened first, and the statement (e2) happened later. Therefore, e1 is BEFORE e2\n\n"

    "AFTER: The first element happens later in time than the second. The second occurs first, and the first follows it.\n"
    "Example: Leader of the strongest Macedonian opposition party VMRO-DPMNE, Nikola Gruevski, <e1>welcomed</e1> the decision, but called the government to <e2>increase</e2> efforts, for the good of all citizens.\" → AFTER.\n"
    "Clarification: The relation is AFTER because the call to increase efforts (e2) happens first, and the reaction — welcomed (e1) — comes later in response to it. Since e1 is a response to e2, e1 must occur after e2.\n\n"

    "Do NOT use the same label for most examples. Use whichever label best fits the sentence.\n"
    
    "Output Format:\n"
    "Output exactly one label ONLY (no punctuation, no explanation, no extra text) in UPPERCASE letters.\n "
)
"""

TLINK_SYSTEM_PROMPT = (
    "Task: Classify the temporal order of the first marked span (<e1>/<t1>) relative to the second marked span (<e2>/<t2>).\n\n"
    "Constraints: Output must be exactly one of the following labels: [BEFORE, AFTER]\n\n"

    "Definitions:\n"
    "BEFORE: The first span happens EARLIER than the second span.\n"
    "AFTER: The first span happens LATER than the second span.\n\n"

    "Examples:\n"
    "Input: \"Speaking to journalists after <e1>talks</e1> with Kofi Annan, Denktash <e2>said</e2> he was optimistic.\"\n"
    "Label: BEFORE\n\n"
    "Input: \"Leader of the strongest Macedonian opposition party VMRO-DPMNE, Nikola Gruevski, <e1>welcomed</e1> the decision, but called the government to <e2>increase</e2> efforts, for the good of all citizens.\"\n"
    "Label: AFTER\n\n"
    "Instructions: Based on the definitions and examples above, assign the correct label to the new input.\n"
    "Output Format:\n"
    "Output exactly one label ONLY (no punctuation, no explanation, no extra text) in UPPERCASE letters.\n "
    
)

@register(_type=MODEL_WRAPPER, _name="llama-3.1-openrouter-zero-shot-tlink")
class Llama31OpenRouterWrapperTLink(ZeroShotModelWrapper):
    """
    Zero-shot TLINK (temporal relation) classifier via OpenRouter.
    Input : List[dict] with keys: id, doc_id, text, label (gold)
    Output: List[dict] with predicted 'label' (normalized)
    """

    def __init__(self, config: dict):
        super().__init__(config)
        cfg = config or {}
        self.api_key_env: str = cfg.get("api_key_env", "OPENROUTER_API_KEY")
        self.api_key: Optional[str] = cfg.get("api_key") or os.getenv(self.api_key_env)
        if not self.api_key:
            raise RuntimeError(f"OpenRouter API key missing. Set env var {self.api_key_env} or pass init_params.api_key.")
        self.model: str = cfg.get("model", "meta-llama/llama-4-maverick")
    '''
    def _build_messages(self, text: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": TLINK_SYSTEM_PROMPT},
            {"role": "user", "content": f"{text}\n\nRespond ONLY with a JSON array containing exactly one string: the label."},
        ] '''
    
    def _build_messages(self, text: str) -> List[Dict[str, str]]:
        first_span, second_span = _extract_first_second_spans(text)

        user_content = (
            "Sentence:\n"
            f"{text}\n\n"
            "This is the FIRST span (from <e1> or <t1>):\n"
            f"{first_span}\n\n"
            "This is the SECOND span (from <e2> or <t2>):\n"
            f"{second_span}\n\n"
            "Question: What is the temporal relation of the FIRST span with respect to the SECOND span?\n"
            "Respond ONLY with a JSON array containing exactly one string: the relation label."
        )

        return [
            {"role": "system", "content": TLINK_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def _infer_label(
        self,
        text: str,
        *,
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        retries: int,
        timeout_sec: int,
        sleep_ms: int,
        headers_extra: Optional[Dict[str, str]] = None,
    ) -> str:
        # Strict schema: exactly one item from allowed enum
        arr = call_openrouter_label_array(
            api_key=self.api_key,
            model=model,
            messages=self._build_messages(text),
            allowed_labels=sorted(ALLOWED_TLINK_LABELS),
            temperature=temperature,
            max_tokens=max_tokens,
            retries=retries,
            timeout_sec=timeout_sec,
            sleep_ms_between_calls=sleep_ms,
            headers_extra=headers_extra or {},
        )

        print(f"[MODEL RAW OUTPUT] {arr}")
        # Post-process: pick the first valid label (extra safety)
        for x in arr:
            cand = _norm_label(x)
            if cand in ALLOWED_TLINK_LABELS:
                return cand
        return "NONE"

    def run(self, test_data: List[dict], **kwargs) -> List[dict]:
        model         = kwargs.get("model", self.model)
        temperature   = float(kwargs.get("temperature", 0.0))
        max_tokens    = kwargs.get("max_tokens")
        retries       = int(kwargs.get("retries", 3))
        timeout_sec   = int(kwargs.get("timeout_sec", 60))
        sleep_ms      = int(kwargs.get("sleep_ms", 400))
        headers_extra = kwargs.get("headers_extra") or {}
        max_docs      = kwargs.get("max_docs")
        show_progress = bool(kwargs.get("show_progress", True))

        iterable = test_data if max_docs is None else test_data[: int(max_docs)]
        outputs: List[dict] = []

        iterator = tqdm(iterable, total=len(iterable), desc="Classifying TLINK (zero-shot)", unit="ex", disable=not show_progress)
        for ex in iterator:
            try:
                pred = self._infer_label(
                    ex.get("text", ""),
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    retries=retries,
                    timeout_sec=timeout_sec,
                    sleep_ms=sleep_ms,
                    headers_extra=headers_extra,
                )
            except Exception as e:
                print(f"[WARN] TLINK prediction failed for id={ex.get('id')}: {e}")
                pred = ""

            outputs.append({
                "id": ex.get("id"),
                "doc_id": ex.get("doc_id"),
                "text": ex.get("text"),
                "label": pred,
            })

            if sleep_ms > 0:
                time.sleep(sleep_ms / 1000.0)

        if len(outputs) != len(iterable):
            print(f"[BUG] TLINK wrapper produced {len(outputs)} predictions for {len(iterable)} inputs")

        return outputs
    

    