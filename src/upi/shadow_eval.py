"""Shadow-pilot evaluation: join the decision log to confirmed fraud labels and report honestly.

    python -m src.upi.shadow_eval --log "logs/decisions.jsonl*" --model-dir models/upi --criteria criteria.json --out report/

Rules that keep the result honest:
  * only decisions OLD ENOUGH for their label window to have closed are counted (default 7 days before the newest decision);
  * a mature decision with no fraud label is treated as legitimate; this is the usual assumption and it overstates precision if fraud
    reports arrive later than the window, so the report prints the window it used;
  * the criteria are read from a file that must be written BEFORE the pilot starts; the outcome is reported either way;
  * with fewer confirmed frauds than `min_frauds` the verdict is INCONCLUSIVE, not a pass or a fail.
"""

import argparse
import csv
import glob
import json
import os
from typing import Dict, List, Optional

import numpy as np

from . import evalkit
from .schema import pseudonymise

# cumulative state that grows by design as history accumulates: always "drifts", so it is left out of the feature-drift ranking
GROWS_BY_DESIGN = {"payee_age_s", "payee_seen", "uid_prior_tx_count"}


def grows_by_design(name: str) -> bool:
    return name in GROWS_BY_DESIGN or (name.startswith("h7_") and (name.endswith("_n") or name.endswith("_fraud_n")))


DEFAULT_CRITERIA = {"min_frauds": 100, "min_pr_auc": 0.0, "min_recall_at_fpr_1pct": 0.0, "min_hard_block_precision": 0.0}


def load_log(patterns: List[str]) -> List[dict]:
    events = []
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            with open(path, encoding="utf-8") as f:
                events.extend(json.loads(line) for line in f if line.strip())
    return events


def load_label_csv(path: str, salt: str) -> Dict[str, int]:
    """Labels exported from the partner's case system: transaction_id,is_fraud. Ids are hashed with the same salt as the log."""
    with open(path, newline="", encoding="utf-8") as f:
        return {pseudonymise(r["transaction_id"], salt): int(float(r["is_fraud"])) for r in csv.DictReader(f)}


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    a = np.histogram(ref, edges)[0] / max(len(ref), 1)
    b = np.histogram(cur, edges)[0] / max(len(cur), 1)
    a, b = np.clip(a, 1e-4, None), np.clip(b, 1e-4, None)
    return float(((b - a) * np.log(b / a)).sum())


def evaluate(events: List[dict], window_days: float = 7.0, tiers: Optional[Dict[str, Optional[float]]] = None,
             criteria: Optional[dict] = None, external_labels: Optional[Dict[str, int]] = None, feature_names: Optional[List[str]] = None,
             seed: int = 0) -> dict:
    crit = {**DEFAULT_CRITERIA, **(criteria or {})}
    decisions: Dict[str, dict] = {}
    labels: Dict[str, int] = dict(external_labels or {})
    for e in events:
        if e["event"] == "decision":
            decisions.setdefault(e["tx"], e)
        elif e["event"] == "label":
            labels[e["tx"]] = int(e["is_fraud"])
    if not decisions:
        return {"verdict": "NO DATA", "n_decisions": 0}
    as_of = max(d["ts"] for d in decisions.values())
    mature = [d for d in decisions.values() if d["ts"] <= as_of - window_days * 86400.0]
    report: dict = {"window_days": window_days, "n_decisions": len(decisions), "n_mature": len(mature), "as_of_ts": as_of,
                    "assumption": f"a decision older than {window_days:g} days with no fraud label is treated as legitimate",
                    "criteria": crit}
    if not mature:
        return {**report, "verdict": "INCONCLUSIVE", "reason": "no decision is older than the label window yet"}
    y = np.asarray([labels.get(d["tx"], 0) for d in mature])
    p = np.asarray([d["p"] for d in mature])
    s = np.asarray([d["score"] for d in mature])
    n_fraud = int(y.sum())
    report.update({"n_frauds": n_fraud, "fraud_rate": float(y.mean()), "unmatched_labels": int(sum(1 for k in labels if k not in decisions))})
    if n_fraud < crit["min_frauds"] or n_fraud == len(y):
        return {**report, "verdict": "INCONCLUSIVE", "reason": f"only {n_fraud} confirmed frauds among {len(y)} mature decisions "
                                                                f"(pre-registered minimum {crit['min_frauds']})"}
    metrics = {"pr_auc": evalkit.average_precision(y, s), "roc_auc": evalkit.roc_auc(y, s), "recall_at_fpr_1pct": evalkit.recall_at_fpr(y, s, 0.01),
               "pr_auc_ci95": evalkit.bootstrap_ci(y, s, evalkit.average_precision, n=200, seed=seed),
               "brier_probability": float(np.mean((p - y) ** 2)), "brier_constant_base_rate": float(np.mean((y.mean() - y) ** 2))}
    k = max(int(0.01 * len(y)), 1)
    top = np.argsort(-p, kind="mergesort")[:k]
    metrics["alert_budget_1pct"] = {"precision": float(y[top].mean()), "recall": float(y[top].sum() / max(n_fraud, 1))}
    tier_rows = {}
    for target, thr in (tiers or {}).items():
        if thr is None:
            continue
        flag = p >= thr
        tier_rows[target] = {"threshold": thr, "alert_rate": float(flag.mean()), "precision": float(y[flag].mean()) if flag.any() else None,
                             "recall": float(flag[y == 1].mean())}
    order = np.argsort(p, kind="mergesort")
    reliability = [{"mean_predicted": float(p[i].mean()), "observed": float(y[i].mean()), "n": int(len(i))} for i in np.array_split(order, 10) if len(i)]
    # drift: first week of decisions against the newest week (score, and the most-shifted features)
    ts_all = np.asarray([d["ts"] for d in decisions.values()])
    ref_mask, cur_mask = ts_all <= ts_all.min() + 7 * 86400, ts_all >= as_of - 7 * 86400
    all_dec = list(decisions.values())
    drift = {}
    if ref_mask.sum() >= 50 and cur_mask.sum() >= 50 and (ts_all.max() - ts_all.min()) > 8 * 86400:
        pa = np.asarray([d["p"] for d in all_dec])
        drift["probability_psi"] = psi(pa[ref_mask], pa[cur_mask])
        if all("x" in d for d in all_dec):                                   # feature vectors are only logged when explicitly enabled
            X = np.asarray([d["x"] for d in all_dec])
            per = [(psi(X[ref_mask, j], X[cur_mask, j]), j) for j in range(X.shape[1])
                   if not (feature_names and grows_by_design(feature_names[j]))]
            drift["top_shifted_features"] = [{"feature": (feature_names[j] if feature_names else f"x{j}"), "psi": v} for v, j in sorted(per, reverse=True)[:5]]
    checks = {"pr_auc": metrics["pr_auc"] >= crit["min_pr_auc"], "recall_at_fpr_1pct": metrics["recall_at_fpr_1pct"] >= crit["min_recall_at_fpr_1pct"]}
    hb = tier_rows.get("0.9")
    if crit["min_hard_block_precision"] > 0:
        checks["hard_block_precision"] = bool(hb and hb["precision"] is not None and hb["precision"] >= crit["min_hard_block_precision"])
    return {**report, "metrics": metrics, "tiers": tier_rows, "reliability_deciles": reliability, "drift": drift,
            "criteria_checks": checks, "verdict": "PASS" if all(checks.values()) else "FAIL"}


def to_markdown(r: dict) -> str:
    out = [f"# Shadow pilot report: {r['verdict']}", "",
           f"Decisions {r.get('n_decisions', 0):,}, mature {r.get('n_mature', 0):,}, confirmed frauds {r.get('n_frauds', 'n/a')}. "
           f"Label window {r.get('window_days', 'n/a')} days. {r.get('assumption', '')}"]
    if r["verdict"] in ("INCONCLUSIVE", "NO DATA"):
        out += ["", f"**Inconclusive:** {r.get('reason', 'no data')}. No pass or fail is claimed."]
        return "\n".join(out)
    m = r["metrics"]
    out += ["", "| Metric | Value |", "|---|---|", f"| PR-AUC | {m['pr_auc']:.3f} (95% CI {m['pr_auc_ci95'][0]:.3f} to {m['pr_auc_ci95'][1]:.3f}) |",
            f"| ROC-AUC | {m['roc_auc']:.3f} |", f"| Recall at 1% false-positive rate | {m['recall_at_fpr_1pct']:.3f} |",
            f"| Precision / recall at a 1% alert budget | {m['alert_budget_1pct']['precision']:.3f} / {m['alert_budget_1pct']['recall']:.3f} |",
            f"| Brier (model / constant base rate) | {m['brier_probability']:.4f} / {m['brier_constant_base_rate']:.4f} |"]
    if r["tiers"]:
        out += ["", "| Tier target | Threshold | Alert rate | Precision | Recall |", "|---|---|---|---|---|"]
        for t, v in r["tiers"].items():
            out.append(f"| {t} | {v['threshold']:.3f} | {v['alert_rate']:.2%} | {'n/a' if v['precision'] is None else format(v['precision'], '.3f')} | {v['recall']:.3f} |")
    out += ["", "Criteria (fixed before the pilot): " + ", ".join(f"{k} {'met' if v else 'NOT met'}" for k, v in r["criteria_checks"].items())]
    if r.get("drift"):
        out += ["", f"Drift: probability PSI {r['drift']['probability_psi']:.3f}" +
                ("; most shifted features: " + ", ".join(f"{d['feature']} ({d['psi']:.2f})" for d in r["drift"]["top_shifted_features"])
                 if r["drift"].get("top_shifted_features") else " (feature-level drift needs log_features enabled)")]
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True, nargs="+", help="decision log path(s) or glob(s)")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--criteria")
    ap.add_argument("--labels", help="optional CSV transaction_id,is_fraud from the case system (needs --salt)")
    ap.add_argument("--salt", help="the PAYWATCH_LOG_SALT used by the service")
    ap.add_argument("--window-days", type=float, default=7.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    spec = json.load(open(os.path.join(a.model_dir, "serving_spec.json")))
    tiers = {k: (v["calibrated_threshold"] if v else None) for k, v in spec["tiers"].items()}
    ext = load_label_csv(a.labels, a.salt) if a.labels else None
    res = evaluate(load_log(a.log), a.window_days, tiers, json.load(open(a.criteria)) if a.criteria else None, ext, spec["features"])
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "shadow_report.json"), "w"), indent=2)
    md = to_markdown(res)
    open(os.path.join(a.out, "shadow_report.md"), "w", encoding="utf-8").write(md)
    print(md)
