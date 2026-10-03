"""Feature / score drift detection with PSI and the two-sample KS statistic (numpy only).

Evidently's PSI/KS drift tests were considered; a native implementation avoids a heavy dependency and
Evidently's column-order caching pitfall, and keeps the maths auditable. Conventions:
  PSI  < 0.1 stable, 0.1-0.2 moderate, > 0.2 significant (industry rule of thumb)
  KS   statistic D in [0, 1]; with thousands of rows p-values are always ~0, so the statistic itself
       is thresholded (default 0.2) instead of a p-value.
Missing values (the model's -999 fill) are treated as their own bin for PSI and excluded from KS.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np

MISSING = -999.0
EPS = 1e-4


def _props(values: np.ndarray, edges: np.ndarray, n_total: int) -> np.ndarray:
    idx = np.searchsorted(edges, values, side="right")
    return np.bincount(idx, minlength=len(edges) + 1) / n_total


def _psi(ref: np.ndarray, cur: np.ndarray) -> float:
    ref, cur = np.clip(ref, EPS, None), np.clip(cur, EPS, None)
    ref, cur = ref / ref.sum(), cur / cur.sum()
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def ks_statistic(a_sorted: np.ndarray, b: np.ndarray) -> float:
    if len(a_sorted) == 0 or len(b) == 0:
        return 0.0
    b_sorted = np.sort(b)
    allv = np.concatenate([a_sorted, b_sorted])
    cdf_a = np.searchsorted(a_sorted, allv, side="right") / len(a_sorted)
    cdf_b = np.searchsorted(b_sorted, allv, side="right") / len(b_sorted)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def psi_1d(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """PSI of a single continuous variable (e.g. the model score)."""
    reference, current = np.asarray(reference, float), np.asarray(current, float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, n_bins + 1)[1:-1]))
    return _psi(_props(reference, edges, len(reference)), _props(current, edges, len(current)))


class DriftDetector:
    def __init__(self, reference: np.ndarray, feature_names: Sequence[str], n_bins: int = 10,
                 ks_sample: int = 20000, seed: int = 0):
        ref = np.asarray(reference, dtype=float)
        self.names: List[str] = list(feature_names)
        rng = np.random.default_rng(seed)
        self.edges, self.ref_props, self.ref_sorted = [], [], []
        for j in range(ref.shape[1]):
            col = ref[:, j]
            miss = col == MISSING
            vals = col[~miss]
            edges = np.unique(np.quantile(vals, np.linspace(0, 1, n_bins + 1)[1:-1])) if len(vals) else np.array([])
            self.edges.append(edges)
            self.ref_props.append(np.append(_props(vals, edges, len(col)), miss.mean()))
            sample = vals if len(vals) <= ks_sample else rng.choice(vals, ks_sample, replace=False)
            self.ref_sorted.append(np.sort(sample))

    def _column_props(self, j: int, col: np.ndarray) -> np.ndarray:
        miss = col == MISSING
        return np.append(_props(col[~miss], self.edges[j], len(col)), miss.mean())

    def evaluate(self, current: np.ndarray, psi_threshold: float = 0.2, ks_threshold: float = 0.2,
                 moderate: float = 0.1, top_n: int = 10, severe_psi: float = 0.5, severe_ks: float = 0.3) -> Dict:
        cur = np.asarray(current, dtype=float)
        psi = np.array([_psi(self.ref_props[j], self._column_props(j, cur[:, j])) for j in range(cur.shape[1])])
        ks = np.array([ks_statistic(self.ref_sorted[j], cur[:, j][cur[:, j] != MISSING]) for j in range(cur.shape[1])])
        order = np.argsort(-psi)[:top_n]
        joint = (psi > psi_threshold) & (ks > ks_threshold)      # PSI and KS must agree
        severe = (psi > severe_psi) & (ks > severe_ks)
        return {
            "n_rows": int(len(cur)),
            "n_features": int(cur.shape[1]),
            "psi_mean": float(psi.mean()),
            "psi_max": float(psi.max()),
            "n_psi_significant": int((psi > psi_threshold).sum()),
            "n_psi_moderate": int(((psi > moderate) & (psi <= psi_threshold)).sum()),
            "share_psi_significant": float((psi > psi_threshold).mean()),
            "n_ks_significant": int((ks > ks_threshold).sum()),
            "n_joint_significant": int(joint.sum()),
            "n_joint_severe": int(severe.sum()),
            "share_joint_significant": float(joint.mean()),
            "joint_features": [self.names[i] for i in np.where(joint)[0]],
            "ks_max": float(ks.max()),
            "top_drifted": [{"feature": self.names[i], "psi": float(psi[i]), "ks": float(ks[i])} for i in order],
        }

    def distribution(self, feature: str, current_col: np.ndarray) -> Dict:
        """Reference vs current bin proportions for plotting (last element = missing bin)."""
        j = self.names.index(feature)
        edges = self.edges[j]
        labels = [f"<= {e:.4g}" for e in edges] + [f"> {edges[-1]:.4g}" if len(edges) else "all"] + ["missing"]
        return {"feature": feature, "bins": labels, "reference": self.ref_props[j].tolist(),
                "current": self._column_props(j, np.asarray(current_col, float)).tolist()}
