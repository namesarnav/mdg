from mdg.evaluators import (
    ExtractionEvaluator,
    ExtractionEvaluatorDoc,
    TrajectorExpression,
    LandmarkExpression,
    SpIndicatorExpression,
    Result,
    CompositeResult,
    SpatialDocument,
)
from typing import List, Tuple
from mdg.registry import register, EVAL_METHOD
from mdg.metrics import evaluate_partial_and_match

def _spans_from_doc(doc: SpatialDocument, expressions: str) -> List[str]:
    """
    Extract surface texts ONLY. Do not strip to match 'notebook' behavior
    (the evaluator itself lowercases, but does not strip/collapse).
    """
    spans: List[str] = []
    for exp in getattr(doc, expressions, []):
        try:
            txt = exp.text  # no .strip()
        except Exception:
            txt = exp.get("text") if isinstance(exp, dict) else ""
        if txt:
            spans.append(str(txt))
    return spans

def _align_by_doc_id(
    gold_docs: List[SpatialDocument],
    pred_docs: List[SpatialDocument],
    expressions: str
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
        expressions,
    )

@register(_type=EVAL_METHOD, _name="trajector_expression_single_doc")
class TrajectorExtractionEvaluator(ExtractionEvaluator):
    """Evaluator for trajector expressions in a single document."""

    def run(self, gold_data: List[TrajectorExpression], predicted_data: List[TrajectorExpression]) -> CompositeResult:
        raise NotImplementedError


@register(_type=EVAL_METHOD, _name="trajector_expression_multi_doc")
class TrajectorExtractionEvaluatorMultiDoc(ExtractionEvaluatorDoc):
    """Evaluator for trajector expressions in multiple documents."""

    def run(self, gold_data: List[SpatialDocument], predicted_data: List[SpatialDocument]) -> CompositeResult:
        gold_spans, pred_spans = _align_by_doc_id(gold_data, predicted_data, "trajector_expressions")
        m = evaluate_partial_and_match(gold_spans, pred_spans)
        results = [Result(metric_name=k, value=float(v)) for k, v in m.items()]
        dataset_name = gold_data[0].dataset if gold_data else "unknown"
        return CompositeResult(dataset_name=dataset_name, results=results)


@register(_type=EVAL_METHOD, _name="landmark_expression_single_doc")
class LandmarkExtractionEvaluator(ExtractionEvaluator):
    """Evaluator for landmark expressions in a single document."""

    def run(self, gold_data: List[LandmarkExpression], predicted_data: List[LandmarkExpression]) -> CompositeResult:
        raise NotImplementedError


@register(_type=EVAL_METHOD, _name="landmark_expression_multi_doc")
class LandmarkExtractionEvaluatorMultiDoc(ExtractionEvaluatorDoc):
    """Evaluator for trajector expressions in multiple documents."""

    def run(self, gold_data: List[SpatialDocument], predicted_data: List[SpatialDocument]) -> CompositeResult:
        gold_spans, pred_spans = _align_by_doc_id(gold_data, predicted_data, "landmark_expressions")
        m = evaluate_partial_and_match(gold_spans, pred_spans)
        results = [Result(metric_name=k, value=float(v)) for k, v in m.items()]
        dataset_name = gold_data[0].dataset if gold_data else "unknown"
        return CompositeResult(dataset_name=dataset_name, results=results)


@register(_type=EVAL_METHOD, _name="spindicator_expression_single_doc")
class SpatialIndicatorExtractionEvaluator(ExtractionEvaluator):
    """Evaluator for spatial indicator expressions in a single document."""

    def run(self, gold_data: List[SpIndicatorExpression], predicted_data: List[SpIndicatorExpression]) -> CompositeResult:
        raise NotImplementedError


@register(_type=EVAL_METHOD, _name="spindicator_expression_multi_doc")
class SpatialIndicatorExtractionEvaluatorMultiDoc(ExtractionEvaluatorDoc):
    """Evaluator for spatial indicator expressions in multiple documents."""

    def run(self, gold_data: List[SpatialDocument], predicted_data: List[SpatialDocument]) -> CompositeResult:
        gold_spans, pred_spans = _align_by_doc_id(gold_data, predicted_data, "spatial_indicator_expressions")
        m = evaluate_partial_and_match(gold_spans, pred_spans)
        results = [Result(metric_name=k, value=float(v)) for k, v in m.items()]
        dataset_name = gold_data[0].dataset if gold_data else "unknown"
        return CompositeResult(dataset_name=dataset_name, results=results)
