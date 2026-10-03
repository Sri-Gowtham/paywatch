"""Alert rules on top of the drift report, score drift, alert-rate shift and (label-delayed) performance.

Design notes (from the first 30-day simulation, where naive rules raised 32 false warnings on a no-drift run):
  - input drift is judged only on features the CLIENT sends (state that grows by design is excluded upstream)
  - a feature counts as drifted only when PSI and KS agree (a single sparse bin cannot trigger an alert)
  - performance alerts are statistical: the drop must exceed bootstrap noise (z-score), not just a % threshold
  - WARN thresholds sit just above the maxima seen in the no-drift control run (one 31-day stream with
    overlapping windows: a weak false-alarm estimate, to be re-validated on more traffic); CRITICAL levels are design values
"""

from typing import List, Optional

DEFAULTS = {
    "min_joint_warn": 5,              # features with PSI>0.2 AND KS>0.2 (control run never exceeded 4)
    "min_severe_warn": 1,             # any feature with PSI>0.5 AND KS>0.3 (control run: 0 in every window)
    "share_joint_warn": 0.10,
    "min_severe_critical": 3,         # features with PSI>0.5 AND KS>0.3
    "share_joint_critical": 0.25,
    "score_psi_warn": 0.15,
    "score_psi_critical": 0.25,
    "alert_rate_ratio_warn": (0.67, 1.5),      # flagged-rate vs baseline
    "alert_rate_ratio_critical": (0.5, 2.0),
    "perf_z_warn": 2.0, "perf_drop_warn": 0.15,        # control run's natural dip peaked at 12%
    "perf_z_critical": 3.0, "perf_drop_critical": 0.25,
    "ece_increase_warn": 0.02,
}


def _alert(day, severity, signal, value, threshold, message):
    return {"day": float(day), "severity": severity, "signal": signal, "value": float(value),
            "threshold": threshold, "message": message}


def evaluate_window(day: float, drift: dict, score_psi: float, flag_rate: float, baseline_flag_rate: float,
                    pr_auc: Optional[float] = None, baseline_pr_auc: Optional[float] = None,
                    pr_auc_sd: Optional[float] = None, baseline_pr_auc_sd: Optional[float] = None,
                    ece: Optional[float] = None, baseline_ece: Optional[float] = None,
                    cfg: Optional[dict] = None) -> List[dict]:
    c = {**DEFAULTS, **(cfg or {})}
    out = []

    n_joint, n_severe, share = drift["n_joint_significant"], drift["n_joint_severe"], drift["share_joint_significant"]
    top = ", ".join(drift.get("joint_features", [])[:4])
    if n_severe >= c["min_severe_critical"] or share >= c["share_joint_critical"]:
        out.append(_alert(day, "CRITICAL", "feature_drift", n_joint, c["min_severe_critical"],
                          f"{n_joint} input features drifted ({n_severe} severely): {top}"))
    elif n_joint >= c["min_joint_warn"] or n_severe >= c["min_severe_warn"] or share >= c["share_joint_warn"]:
        out.append(_alert(day, "WARN", "feature_drift", n_joint, c["min_joint_warn"],
                          f"{n_joint} input features drifted: {top}"))

    if score_psi >= c["score_psi_critical"]:
        out.append(_alert(day, "CRITICAL", "score_drift", score_psi, c["score_psi_critical"], "score distribution shifted"))
    elif score_psi >= c["score_psi_warn"]:
        out.append(_alert(day, "WARN", "score_drift", score_psi, c["score_psi_warn"], "score distribution moderately shifted"))

    if baseline_flag_rate > 0:
        ratio = flag_rate / baseline_flag_rate
        lo_c, hi_c = c["alert_rate_ratio_critical"]
        lo_w, hi_w = c["alert_rate_ratio_warn"]
        if ratio <= lo_c or ratio >= hi_c:
            out.append(_alert(day, "CRITICAL", "alert_rate", ratio, (lo_c, hi_c), f"flag rate is {ratio:.2f}x the baseline"))
        elif ratio <= lo_w or ratio >= hi_w:
            out.append(_alert(day, "WARN", "alert_rate", ratio, (lo_w, hi_w), f"flag rate is {ratio:.2f}x the baseline"))

    if pr_auc is not None and baseline_pr_auc:
        drop = (baseline_pr_auc - pr_auc) / baseline_pr_auc
        sd = ((pr_auc_sd or 0.0) ** 2 + (baseline_pr_auc_sd or 0.0) ** 2) ** 0.5
        z = (baseline_pr_auc - pr_auc) / sd if sd > 0 else 0.0
        if z >= c["perf_z_critical"] and drop >= c["perf_drop_critical"]:
            out.append(_alert(day, "CRITICAL", "performance", drop, c["perf_drop_critical"],
                              f"PR-AUC {pr_auc:.3f} is {drop:.0%} below baseline {baseline_pr_auc:.3f} (z={z:.1f}): retrain recommended"))
        elif z >= c["perf_z_warn"] and drop >= c["perf_drop_warn"]:
            out.append(_alert(day, "WARN", "performance", drop, c["perf_drop_warn"],
                              f"PR-AUC down {drop:.0%} vs baseline (z={z:.1f})"))

    if ece is not None and baseline_ece is not None and ece - baseline_ece >= c["ece_increase_warn"]:
        out.append(_alert(day, "WARN", "calibration", ece - baseline_ece, c["ece_increase_warn"],
                          f"ECE rose to {ece:.3f} (baseline {baseline_ece:.3f}): recalibrate"))
    return out


def retrain_recommended(alerts: List[dict]) -> bool:
    return any(a["severity"] == "CRITICAL" and a["signal"] in ("feature_drift", "score_drift", "performance") for a in alerts)
