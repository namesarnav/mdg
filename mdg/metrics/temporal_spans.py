from __future__ import annotations
from typing import List, Dict, Tuple, Callable
import numpy as np
from scipy.optimize import linear_sum_assignment

# --------- Normalization ---------
def _preprocess(text: str) -> str:
    return text.lower() if text else ""

# --------- LCS helpers ---------
def _lcs_len_char(a: str, b: str) -> int:
    """Longest contiguous common substring length (character-level)."""
    la, lb = len(a), len(b)
    dp = [0] * (lb + 1)
    best = 0
    for i in range(1, la + 1):
        new_dp = [0] * (lb + 1)
        ai = a[i - 1]
        for j in range(1, lb + 1):
            if ai == b[j - 1]:
                new_dp[j] = dp[j - 1] + 1
                if new_dp[j] > best:
                    best = new_dp[j]
        dp = new_dp
    return best

def _lcs_len_word(a: str, b: str) -> int:
    """Longest contiguous common subsequence length (word-level; space tokenization)."""
    ta, tb = a.split(), b.split()
    la, lb = len(ta), len(tb)
    dp = [0] * (lb + 1)
    best = 0
    for i in range(1, la + 1):
        new_dp = [0] * (lb + 1)
        ai = ta[i - 1]
        for j in range(1, lb + 1):
            if ai == tb[j - 1]:
                new_dp[j] = dp[j - 1] + 1
                if new_dp[j] > best:
                    best = new_dp[j]
        dp = new_dp
    return best

def _num_chars(x: str) -> int: return len(x)
def _num_words(x: str) -> int: return len(x.split())


def _partial_micro_metrics(
    data: List[Tuple[List[str], List[str]]],
    lcs_func: Callable[[str, str], int],
    length_func: Callable[[str], int],
) -> Tuple[float, float, float]:
    s_prec: List[float] = []
    s_rec: List[float] = []

    for gold_spans, pred_spans in data:
        # precision over predictions
        for p in pred_spans:
            pp = _preprocess(p)
            if not pp:
                s_prec.append(0.0); continue
            denom = length_func(pp)
            if denom <= 0 or not gold_spans:
                s_prec.append(0.0); continue
            best = 0.0
            for g in gold_spans:
                gg = _preprocess(g)
                if not gg: continue
                best = max(best, lcs_func(pp, gg) / denom)
            s_prec.append(best)

        # recall over golds
        for g in gold_spans:
            gg = _preprocess(g)
            if not gg:
                s_rec.append(0.0); continue
            denom = length_func(gg)
            if denom <= 0 or not pred_spans:
                s_rec.append(0.0); continue
            best = 0.0
            for p in pred_spans:
                pp = _preprocess(p)
                if not pp: continue
                best = max(best, lcs_func(gg, pp) / denom)
            s_rec.append(best)

    precision = sum(s_prec) / len(s_prec) if s_prec else 0.0
    recall    = sum(s_rec) / len(s_rec) if s_rec else 0.0
    f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def _matching_micro_metrics(
    data: List[Tuple[List[str], List[str]]],
    lcs_func: Callable[[str, str], int],
    pred_len_func: Callable[[str], int],
    gold_len_func: Callable[[str], int],
) -> Tuple[float, float, float]:
    total_preds = 0
    total_golds = 0
    sum_prec = 0.0
    sum_rec  = 0.0

    for gold_spans, pred_spans in data:
        nP, nG = len(pred_spans), len(gold_spans)
        total_preds += nP
        total_golds += nG
        if nP == 0 or nG == 0:
            continue

        P_texts = [_preprocess(p) for p in pred_spans]
        G_texts = [_preprocess(g) for g in gold_spans]

        P = np.zeros((nP, nG), dtype=float)
        for i, pp in enumerate(P_texts):
            denom_p = pred_len_func(pp) if pp else 0
            if denom_p <= 0: continue
            for j, gg in enumerate(G_texts):
                if not gg: continue
                num = lcs_func(pp, gg)
                if num > 0:
                    P[i, j] = num / denom_p

        R = np.zeros((nP, nG), dtype=float)
        for i, pp in enumerate(P_texts):
            for j, gg in enumerate(G_texts):
                denom_g = gold_len_func(gg) if gg else 0
                if denom_g <= 0: continue
                num = lcs_func(pp, gg)
                if num > 0:
                    R[i, j] = num / denom_g

        r, c = linear_sum_assignment(-P)
        sum_prec += float(P[r, c].sum())

        r, c = linear_sum_assignment(-R)
        sum_rec += float(R[r, c].sum())

    precision = (sum_prec / total_preds) if total_preds > 0 else 0.0
    recall    = (sum_rec  / total_golds) if total_golds > 0 else 0.0
    f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def evaluate_temporal_spans(gold: List[List[str]], pred: List[List[str]]) -> Dict[str, float]:
    """
    Compute char/word × partial/Hungarian micro-averaged precision/recall/F1.
    Returns a flat dict of metrics.
    """
    data = list(zip(gold, pred))

    # Best-of partial
    cP, cR, cF = _partial_micro_metrics(data, _lcs_len_char, _num_chars)
    wP, wR, wF = _partial_micro_metrics(data, _lcs_len_word, _num_words)

    # Hungarian 1:1
    mcP, mcR, mcF = _matching_micro_metrics(data, _lcs_len_char, _num_chars, _num_chars)
    mwP, mwR, mwF = _matching_micro_metrics(data, _lcs_len_word, _num_words, _num_words)

    return {
        "char_precision": cP,  "char_recall": cR,  "char_f1": cF,
        "word_precision": wP,  "word_recall": wR,  "word_f1": wF,
        "char_precision_match": mcP, "char_recall_match": mcR, "char_f1_match": mcF,
        "word_precision_match": mwP, "word_recall_match": mwR, "word_f1_match": mwF,
    }