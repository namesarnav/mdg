`from __future__ import annotations
from typing import List, Dict, Tuple
from collections import Counter

def _normalize_label(label: str) -> str:
    if label is None:
        return ""
    # normalize like your evaluator expects
    return str(label).strip().upper().replace(" ", "_").replace("-", "_")

def evaluate_classification(gold: List[str], pred: List[str]) -> Dict[str, float | Dict]:
    """
    Single-label multiclass classification metrics:
      - accuracy
      - macro_precision, macro_recall, macro_f1   (EXCLUDES classes with support == 0)
      - micro_precision, micro_recall, micro_f1   (all labels including hallucinated ones)
      - weighted_f1                               (weighted by gold support)
      - abstentions                               (count of empty/None predictions)
      - per_class breakdown (precision/recall/f1/support)
    """
    gold = [_normalize_label(g) for g in gold]
    pred = [_normalize_label(p) for p in pred]

    if len(gold) != len(pred):
        print(
            f"[WARN] evaluate_classification: gold length {len(gold)} != pred length {len(pred)}. "
            f"Trimming to shorter."
        )
        n = min(len(gold), len(pred))
        gold, pred = gold[:n], pred[:n]

    if not gold:
        return {
            "accuracy": 0.0, "macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0,
            "micro_precision": 0.0, "micro_recall": 0.0, "micro_f1": 0.0,
            "weighted_f1": 0.0, "abstentions": 0, "per_class": {},
        }

    # Count abstentions (empty string after normalisation = failed/missing prediction)
    abstentions = sum(1 for p in pred if p == "")

    # Accuracy (abstentions count as wrong)
    correct = sum(g == p for g, p in zip(gold, pred))
    accuracy = correct / len(gold)

    # Per-class counts — over ALL labels (including hallucinated pred-only labels)
    labels = sorted(set(gold) | set(pred) - {""})
    tp, fp, fn, support = Counter(), Counter(), Counter(), Counter()
    for g, p in zip(gold, pred):
        support[g] += 1
        if g == p:
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1

    labels_in_gold = [c for c in labels if support[c] > 0]

    # Per-class metrics
    class_precisions, class_recalls, class_f1s = [], [], []
    per_class: Dict[str, Dict[str, float]] = {}
    for c in labels:
        p_c = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        r_c = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        f1_c = 2 * p_c * r_c / (p_c + r_c) if (p_c + r_c) else 0.0
        per_class[c] = {"support": support[c], "precision": p_c, "recall": r_c, "f1": f1_c}

        if support[c] > 0:
            class_precisions.append(p_c)
            class_recalls.append(r_c)
            class_f1s.append(f1_c)

    macro_precision = (sum(class_precisions) / len(class_precisions)) if class_precisions else 0.0
    macro_recall    = (sum(class_recalls)    / len(class_recalls))    if class_recalls    else 0.0
    macro_f1        = (sum(class_f1s)        / len(class_f1s))        if class_f1s        else 0.0

    # Micro metrics over ALL labels — hallucinated labels contribute as FP
    TP = sum(tp[c] for c in labels)
    FP = sum(fp[c] for c in labels)
    FN = sum(fn[c] for c in labels)

    micro_precision = TP / (TP + FP) if (TP + FP) else 0.0
    micro_recall    = TP / (TP + FN) if (TP + FN) else 0.0
    micro_f1        = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)
                       if (micro_precision + micro_recall) else 0.0)

    # Weighted-F1 (weights by gold support)
    total_support = sum(support[c] for c in labels_in_gold)
    weighted_f1 = (
        sum(per_class[c]["f1"] * support[c] for c in labels_in_gold) / total_support
        if total_support > 0 else 0.0
    )

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "weighted_f1": weighted_f1,
        "abstentions": abstentions,
        "per_class": per_class,
    }