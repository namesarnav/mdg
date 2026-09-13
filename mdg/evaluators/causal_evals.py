from __future__ import annotations
from typing import List, Tuple, Dict, Any

from mdg.evaluators import Evaluator, ExtractionEvaluator, ExtractionEvaluatorDoc
from mdg.datamodels import TimeDocument, TimeExpression, Result, CompositeResult
from mdg.registry import register, EVAL_METHOD
from mdg.metrics import evaluate_partial_and_match, evaluate_classification

def _spans_from_doc(doc: TimeDocument) -> List[str]:
    """
    Extract surface texts ONLY. Do not strip to match 'notebook' behavior
    (the evaluator itself lowercases, but does not strip/collapse).
    """
    spans: List[str] = []
    for te in (doc.time_expressions or []):
        try:
            txt = te.text  # no .strip()
        except Exception:
            txt = te.get("text") if isinstance(te, dict) else ""
        if txt:
            spans.append(str(txt))
    return spans


def _align_by_doc_id(
    gold_docs: List[TimeDocument],
    pred_docs: List[TimeDocument]
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Prefer alignment by doc_id. If not possible, fall back to index alignment.
    """
    try:
        gold_ids = [g.doc_id for g in gold_docs]
        pred_map = {p.doc_id: p for p in pred_docs if getattr(p, "doc_id", None) is not None}
        if len(pred_map) == len(pred_docs) and all(gid in pred_map for gid in gold_ids):
            gold_spans = [_spans_from_doc(g) for g in gold_docs]
            pred_spans = [_spans_from_doc(pred_map[gid]) for gid in gold_ids]
            return gold_spans, pred_spans
        else:
            # Helpful hint if IDs don't align
            missing = [gid for gid in gold_ids if gid not in pred_map]
            if missing:
                print(f"[WARN] Missing predictions for {len(missing)} gold doc_ids (e.g., {missing[:3]} ...). "
                      f"Falling back to index alignment; metrics may be low.")
    except Exception as e:
        print(f"[WARN] ID alignment failed ({e}); falling back to index alignment.")

    n = min(len(gold_docs), len(pred_docs))
    return (
        [_spans_from_doc(gold_docs[i]) for i in range(n)],
        [_spans_from_doc(pred_docs[i]) for i in range(n)],
    )




@register(_type=EVAL_METHOD, _name="causal_classification")
class CausalClassificationEvaluator(Evaluator):
    """
    Evaluator for causal relation classification.
    Works with any number of labels (2-way, 3-way, N-way) — the label space
    is inferred from the gold data itself, so no configuration is needed.

    Expects gold_data / predicted_data as lists of dicts with:
      - 'id' or 'doc_id'  (for alignment)
      - 'label'           (gold/pred string label, already normalised to UPPER)
    """

    def run(self, gold_data: List[Dict[str, Any]], predicted_data: List[Dict[str, Any]], **kwargs):
        pred_by_id: Dict[str, Any] = {}
        have_ids = True

        for p in predicted_data:
            pid = p.get("id") or p.get("doc_id")
            if pid is None:
                have_ids = False
                break
            pred_by_id[str(pid)] = p.get("label")

        gold_labels: List[str] = []
        pred_labels: List[str] = []

        if have_ids:
            for g in gold_data:
                gid = g.get("id") or g.get("doc_id")
                gold_labels.append(g.get("label"))
                pred_labels.append(pred_by_id.get(str(gid), None))
        else:
            n = min(len(gold_data), len(predicted_data))
            gold_labels = [gold_data[i].get("label") for i in range(n)]
            pred_labels = [predicted_data[i].get("label") for i in range(n)]

        # Report missing / empty predictions before scoring
        missing = sum(1 for p in pred_labels if p is None or str(p).strip() == "")
        if missing:
            print(f"[WARN] {missing}/{len(pred_labels)} predictions are missing or empty — will count as wrong.")

        return evaluate_classification(gold_labels, pred_labels)



def _event_spans_from_doc(doc: TimeDocument) -> List[str]:
    """
    Extract surface texts ONLY. Do not strip to match 'notebook' behavior
    (the evaluator itself lowercases, but does not strip/collapse).
    """
    spans: List[str] = []

    for ee in (getattr(doc, "event_expressions", None) or []):
        try:
            txt = ee.text  # no .strip()
        except Exception:
            txt = ee.get("text") if isinstance(ee, dict) else ""
        if txt:
            spans.append(str(txt))

    return spans


def _event_align_by_doc_id(
    gold_docs: List[TimeDocument],
    pred_docs: List[TimeDocument],
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Prefer alignment by doc_id. If not possible, fall back to index alignment.
    """
    try:
        gold_ids = [g.doc_id for g in gold_docs]
        pred_map = {p.doc_id: p for p in pred_docs if getattr(p, "doc_id", None) is not None}

        if len(pred_map) == len(pred_docs) and all(gid in pred_map for gid in gold_ids):
            gold_spans = [_event_spans_from_doc(g) for g in gold_docs]
            pred_spans = [_event_spans_from_doc(pred_map[gid]) for gid in gold_ids]
            return gold_spans, pred_spans
        else:
            missing = [gid for gid in gold_ids if gid not in pred_map]
            if missing:
                print(
                    f"[WARN] Missing predictions for {len(missing)} gold doc_ids (e.g., {missing[:3]} ...). "
                    f"Falling back to index alignment; metrics may be low."
                )
    except Exception as e:
        print(f"[WARN] ID alignment failed ({e}); falling back to index alignment.")

    n = min(len(gold_docs), len(pred_docs))
    return (
        [_event_spans_from_doc(gold_docs[i]) for i in range(n)],
        [_event_spans_from_doc(pred_docs[i]) for i in range(n)],
    )


@register(_type=EVAL_METHOD, _name="event_expression_partial")
class EventExtractionEvaluatorPartial(ExtractionEvaluatorDoc):
    """
    Returns notebook-style grouped JSON: {"dataset_name": ..., "groups": {...}}

    Same as TimeExtractionEvaluatorPartial, but uses doc.event_expressions.
    """

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        cfg = config or {}
        self.case_insensitive = bool(cfg.get("case_insensitive", True))
        self.strip = bool(cfg.get("strip", False))  # notebook ignores strip; keep flag for future

    def run(self, gold_data: List[TimeDocument], predicted_data: List[TimeDocument]):
        gold_spans, pred_spans = _event_align_by_doc_id(gold_data, predicted_data)

        # Debug hooks you can uncomment if scores look suspicious:
        # print(f"[DBG] gold first doc spans: {gold_spans[0] if gold_spans else None}")
        # print(f"[DBG] pred first doc spans: {pred_spans[0] if pred_spans else None}")

        m = evaluate_partial_and_match(
            gold_spans,
            pred_spans,
            case_insensitive=self.case_insensitive,
            strip=self.strip,
        )

        grouped = {
            "best_of": {
                "char": {"precision": m["char_precision"], "recall": m["char_recall"], "f1": m["char_f1"]},
                "word": {"precision": m["word_precision"], "recall": m["word_recall"], "f1": m["word_f1"]},
            },
            "hungarian": {
                "char": {
                    "precision": m["char_precision_match"],
                    "recall": m["char_recall_match"],
                    "f1": m["char_f1_match"],
                },
                "word": {
                    "precision": m["word_precision_match"],
                    "recall": m["word_recall_match"],
                    "f1": m["word_f1_match"],
                },
            },
        }

        dataset_name = gold_data[0].dataset if gold_data else "unknown"
        return {"dataset_name": dataset_name, "groups": grouped}
          

def _compositional_spans_from_doc(doc: TimeDocument) -> List[str]:
    """
    Merge TIME + EVENT surface texts. Do not strip/collapse whitespace
    to match existing evaluator behavior (lowercasing happens in metrics).
    """
    spans: List[str] = []

    # time spans
    for te in (doc.time_expressions or []):
        try:
            txt = te.text
        except Exception:
            txt = te.get("text") if isinstance(te, dict) else ""
        if txt:
            spans.append(str(txt))

    # event spans
    for ee in (getattr(doc, "event_expressions", None) or []):
        try:
            txt = ee.text
        except Exception:
            txt = ee.get("text") if isinstance(ee, dict) else ""
        if txt:
            spans.append(str(txt))

    return spans

def _compositional_align_by_doc_id(
    gold_docs: List[TimeDocument],
    pred_docs: List[TimeDocument],
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Prefer alignment by doc_id. If not possible, fall back to index alignment.
    """
    try:
        gold_ids = [g.doc_id for g in gold_docs]
        pred_map = {p.doc_id: p for p in pred_docs if getattr(p, "doc_id", None) is not None}

        if len(pred_map) == len(pred_docs) and all(gid in pred_map for gid in gold_ids):
            gold_spans = [_compositional_spans_from_doc(g) for g in gold_docs]
            pred_spans = [_compositional_spans_from_doc(pred_map[gid]) for gid in gold_ids]
            return gold_spans, pred_spans
        else:
            missing = [gid for gid in gold_ids if gid not in pred_map]
            if missing:
                print(
                    f"[WARN] Missing predictions for {len(missing)} gold doc_ids (e.g., {missing[:3]} ...). "
                    f"Falling back to index alignment; metrics may be low."
                )
    except Exception as e:
        print(f"[WARN] ID alignment failed ({e}); falling back to index alignment.")

    n = min(len(gold_docs), len(pred_docs))
    return (
        [_compositional_spans_from_doc(gold_docs[i]) for i in range(n)],
        [_compositional_spans_from_doc(pred_docs[i]) for i in range(n)],
    )

@register(_type=EVAL_METHOD, _name="compositional_expression_partial")
class CompositionalExtractionEvaluatorPartial(ExtractionEvaluatorDoc):
    """
    Compositional extraction evaluator:
      GOLD = time_expressions + event_expressions
      PRED = time_expressions + event_expressions

    Returns notebook-style grouped JSON: {"dataset_name": ..., "groups": {...}}
    Uses the SAME metric function: evaluate_partial_and_match
    """

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        cfg = config or {}
        self.case_insensitive = bool(cfg.get("case_insensitive", True))
        self.strip = bool(cfg.get("strip", False))  # keep consistent with others

    def run(self, gold_data: List[TimeDocument], predicted_data: List[TimeDocument]):
        gold_spans, pred_spans = _compositional_align_by_doc_id(gold_data, predicted_data)

        m = evaluate_partial_and_match(
            gold_spans,
            pred_spans,
            case_insensitive=self.case_insensitive,
            strip=self.strip,
        )

        grouped = {
            "best_of": {
                "char": {"precision": m["char_precision"], "recall": m["char_recall"], "f1": m["char_f1"]},
                "word": {"precision": m["word_precision"], "recall": m["word_recall"], "f1": m["word_f1"]},
            },
            "hungarian": {
                "char": {
                    "precision": m["char_precision_match"],
                    "recall": m["char_recall_match"],
                    "f1": m["char_f1_match"],
                },
                "word": {
                    "precision": m["word_precision_match"],
                    "recall": m["word_recall_match"],
                    "f1": m["word_f1_match"],
                },
            },
        }

        dataset_name = gold_data[0].dataset if gold_data else "unknown"
        return {"dataset_name": dataset_name, "groups": grouped}
