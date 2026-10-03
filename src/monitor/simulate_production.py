"""30-day production simulation with controlled drift injected on day 15.

Replays a time-ordered transaction stream through the real Predictor (warm state, 7-day delayed
labels), then monitors it day by day exactly as an operator would:
  - input drift (PSI/KS on the model's feature vectors vs the first-10-days reference)
  - score drift and alert-rate shift
  - label-delayed performance and calibration (only rows whose labels have arrived)
A no-drift control run measures the false-alarm rate of the same rules.
"""

import heapq
from typing import Callable, Dict, List, Optional

import numpy as np
from sklearn.metrics import average_precision_score

from .alerts import evaluate_window, retrain_recommended
from .drift_detector import DriftDetector, psi_1d

LAG_S = 7 * 86400.0
FLAGGED = ("SOFT_FLAG", "CHALLENGE", "HARD_BLOCK")


def clean(rec: dict) -> dict:
    return {k: (None if (v is None or (isinstance(v, float) and v != v)) else v) for k, v in rec.items()}


def covariate_drift(fields: dict, rng: np.random.Generator) -> dict:
    """Input drift: amounts inflate, a new e-mail domain appears, linked-card counts double."""
    f = dict(fields)
    f["TransactionAmt"] = float(f["TransactionAmt"]) * 1.8
    if rng.random() < 0.35:
        f["P_emaildomain"] = "newmail.xyz"
        f["R_emaildomain"] = "newmail.xyz"
    for c in ("C1", "C13", "C14"):
        if f.get(c) is not None:
            f[c] = float(f[c]) * 2.0
    return f


def replay(predictor, records: List[dict], ids: List[str], labels: np.ndarray, times: np.ndarray, pending_labels,
           drift_day: float = 15.0, drift_fn: Optional[Callable] = None, seed: int = 0) -> Dict:
    t0 = float(times[0])
    rng = np.random.default_rng(seed)
    heap = [tuple(p) for p in pending_labels]
    heapq.heapify(heap)
    X, score, cal, action = [], [], [], []
    for i, rec in enumerate(records):
        while heap and heap[0][0] <= times[i]:
            _, tid, lab = heapq.heappop(heap)
            predictor.feedback(tid, bool(lab))
        f = clean(rec)
        if drift_fn is not None and (times[i] - t0) / 86400.0 >= drift_day:
            f = drift_fn(f, rng)
        out = predictor.predict(ids[i], f, explain=False, include_vector=True)
        X.append(out["_vector"])
        score.append(out["score"])
        cal.append(out["calibrated_probability"])
        action.append(out["action"])
        heapq.heappush(heap, (float(times[i]) + LAG_S, ids[i], int(labels[i])))
    return {"day": (times - t0) / 86400.0, "y": np.asarray(labels), "X": np.asarray(X, dtype=np.float32),
            "score": np.asarray(score), "cal": np.asarray(cal), "action": np.asarray(action)}


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    return float(sum((idx == b).mean() * abs(p[idx == b].mean() - y[idx == b].mean()) for b in range(bins) if (idx == b).any()))


def _ap_sd(y: np.ndarray, s: np.ndarray, boot: int = 100, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() > 0:
            vals.append(average_precision_score(y[i], s[i]))
    return float(np.std(vals)) if vals else 0.0


def analyse(rep: Dict, feature_names: List[str], drift_day: Optional[float], ref_days: int = 10, window_days: int = 3,
            min_window_rows: int = 300, monitor_features: Optional[List[str]] = None, mature_days: int = 14) -> Dict:
    day, y, X, s, cal, action = rep["day"], rep["y"], rep["X"], rep["score"], rep["cal"], rep["action"]
    monitored = monitor_features or feature_names
    cols = [feature_names.index(f) for f in monitored]
    Xm = X[:, cols]
    ref = day < ref_days
    det = DriftDetector(Xm[ref], monitored)
    flagged = np.isin(action, FLAGGED)
    base_soft = float(flagged[ref].mean())
    base_pr = float(average_precision_score(y[ref], s[ref])) if y[ref].sum() >= 20 else None
    base_pr_sd = _ap_sd(y[ref], s[ref]) if base_pr is not None else None
    base_ece = _ece(y[ref], cal[ref])

    daily, alerts_all, last_window = [], [], None
    for d in range(ref_days + 1, int(np.ceil(day.max())) + 1):
        w = (day >= d - window_days) & (day < d)
        if w.sum() < min_window_rows:
            continue
        drift = det.evaluate(Xm[w])
        spsi = psi_1d(s[ref], s[w])
        mature = (day >= d - 7 - mature_days) & (day < d - 7)      # labels have arrived for these rows
        pr = float(average_precision_score(y[mature], s[mature])) if y[mature].sum() >= 30 else None
        pr_sd = _ap_sd(y[mature], s[mature]) if pr is not None else None
        ece_m = _ece(y[mature], cal[mature]) if mature.sum() >= 300 else None
        alerts = evaluate_window(d, drift, spsi, float(flagged[w].mean()), base_soft, pr, base_pr, pr_sd, base_pr_sd,
                                 ece_m, base_ece)
        alerts_all += alerts
        daily.append({
            "day": d, "rows": int(w.sum()), "psi_mean": drift["psi_mean"], "psi_max": drift["psi_max"],
            "n_joint_significant": drift["n_joint_significant"], "n_joint_severe": drift["n_joint_severe"],
            "share_joint_significant": drift["share_joint_significant"], "n_ks_significant": drift["n_ks_significant"],
            "score_psi": spsi, "flag_rate": float(flagged[w].mean()),
            "challenge_or_block_rate": float(np.isin(action[w], ("CHALLENGE", "HARD_BLOCK")).mean()),
            "mean_score": float(s[w].mean()), "pr_auc_mature_window": pr, "ece_mature_window": ece_m,
            "top_drifted": drift["top_drifted"][:5], "alerts": alerts, "retrain_recommended": retrain_recommended(alerts),
        })
        last_window = w

    first = {}
    for a in alerts_all:
        key = f"{a['signal']}:{a['severity']}"
        first.setdefault(key, a["day"])
    pre = [a for a in alerts_all if drift_day is not None and a["day"] <= drift_day]
    out = {
        "baseline": {"ref_days": ref_days, "flag_rate": base_soft, "pr_auc": base_pr, "pr_auc_sd": base_pr_sd, "ece": base_ece,
                     "rows": int(ref.sum()), "monitored_features": len(monitored), "window_days": window_days},
        "daily": daily, "alerts": alerts_all, "first_alert_day": first,
        "alerts_before_drift_day": len(pre) if drift_day is not None else None,
        "alert_counts": {sev: sum(1 for a in alerts_all if a["severity"] == sev) for sev in ("WARN", "CRITICAL")},
    }
    if drift_day is not None:
        out["detection_lag_days"] = {k: v - drift_day for k, v in first.items()}
    if last_window is not None and daily:
        top = [t["feature"] for t in daily[-1]["top_drifted"][:3]]
        out["final_window_distributions"] = [det.distribution(f, Xm[last_window][:, monitored.index(f)]) for f in top]
        edges = np.linspace(0, 1, 21)
        out["score_histogram"] = {"edges": edges.tolist(),
                                  "reference": (np.histogram(cal[ref], edges)[0] / ref.sum()).tolist(),
                                  "final_window": (np.histogram(cal[last_window], edges)[0] / last_window.sum()).tolist()}
    return out
