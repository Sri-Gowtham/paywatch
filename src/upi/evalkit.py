"""Metrics, isotonic calibration and bootstrap intervals in plain numpy, so the pilot kit needs no scikit-learn."""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    """PR-AUC as average precision (the step-wise sum over ranked positives), ties broken by input order."""
    y = np.asarray(y, dtype=int)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-np.asarray(score), kind="mergesort")
    ys = y[order]
    tp = np.cumsum(ys)
    precision = tp / np.arange(1, len(ys) + 1)
    return float((precision * ys).sum() / ys.sum())


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = np.empty(len(score))
    order = np.argsort(score, kind="mergesort")
    s = np.asarray(score)[order]
    r = np.arange(1, len(s) + 1, dtype=float)
    i = 0
    while i < len(s):                                   # average ranks over ties
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        r[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    ranks[order] = r
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def recall_at_fpr(y: np.ndarray, score: np.ndarray, fpr: float = 0.01) -> float:
    """Share of frauds scored strictly above the (k+1)-th highest legitimate score, where k = floor(fpr * #legitimate): at most k
    legitimate transactions are flagged (ties are resolved conservatively)."""
    y = np.asarray(y, dtype=int)
    neg = np.sort(np.asarray(score)[y == 0])[::-1]
    if len(neg) == 0 or y.sum() == 0:
        return float("nan")
    k = int(np.floor(fpr * len(neg)))
    thr = neg[k] if k < len(neg) else -np.inf
    return float((np.asarray(score)[y == 1] > thr).mean())


def isotonic_fit(x: np.ndarray, y: np.ndarray) -> Tuple[List[float], List[float]]:
    """Pool-adjacent-violators: a non-decreasing step function mapping a raw score to the observed fraud rate.
    Returns (x_knots, y_knots) for `np.interp`; values are clipped to [0, 1]."""
    order = np.argsort(x, kind="mergesort")
    xs, ys = np.asarray(x, dtype=float)[order], np.asarray(y, dtype=float)[order]
    blocks: List[List[float]] = []                       # [sum_y, count, x_min, x_max]
    for xv, yv in zip(xs, ys):
        blocks.append([yv, 1.0, xv, xv])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            b = blocks.pop()
            blocks[-1] = [blocks[-1][0] + b[0], blocks[-1][1] + b[1], blocks[-1][2], b[3]]
    kx: List[float] = []
    ky: List[float] = []
    for s, c, lo, hi in blocks:
        v = float(min(max(s / c, 0.0), 1.0))
        kx.extend([float(lo), float(hi)])
        ky.extend([v, v])
    return kx, ky


def tier_thresholds(cal: np.ndarray, y: np.ndarray, targets: Sequence[float] = (0.5, 0.7, 0.9), n_min: int = 30) -> Dict[str, Optional[float]]:
    """Lowest calibrated-probability cut-off whose flagged set has precision >= target (and at least n_min rows), per target."""
    order = np.argsort(-cal, kind="mergesort")
    cs, ys = cal[order], np.asarray(y)[order]
    tp = np.cumsum(ys)
    count = np.arange(1, len(ys) + 1)
    prec = tp / count
    ends = np.r_[cs[1:] != cs[:-1], True]            # a cut-off flags EVERY row tied with it, so only tie-group ends are valid cut-offs
    out: Dict[str, Optional[float]] = {}
    for t in targets:
        ok = np.where(ends & (prec >= t) & (count >= n_min))[0]
        out[str(t)] = float(cs[ok.max()]) if len(ok) else None
    return out


def bootstrap_ci(y: np.ndarray, score: np.ndarray, fn, n: int = 200, seed: int = 0) -> List[float]:
    rng = np.random.default_rng(seed)
    y, score = np.asarray(y), np.asarray(score)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        v = fn(y[idx], score[idx])
        if v == v:                                       # skip NaN resamples without positives
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else [float("nan")] * 2
