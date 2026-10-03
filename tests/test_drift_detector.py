import numpy as np
import pytest

from src.monitor.alerts import evaluate_window, retrain_recommended
from src.monitor.drift_detector import MISSING, DriftDetector, ks_statistic, psi_1d


def _data(seed=0, n=6000):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.normal(0, 1, n), rng.integers(0, 2, n).astype(float), rng.gamma(2, 2, n)])


def test_identical_distribution_has_low_psi_and_ks():
    ref, cur = _data(0), _data(1)
    rep = DriftDetector(ref, ["a", "b", "c"]).evaluate(cur)
    assert rep["psi_max"] < 0.05 and rep["ks_max"] < 0.06 and rep["n_psi_significant"] == 0


def test_mean_shift_is_flagged():
    ref, cur = _data(0), _data(1)
    cur[:, 0] += 1.0
    rep = DriftDetector(ref, ["a", "b", "c"]).evaluate(cur)
    assert rep["top_drifted"][0]["feature"] == "a"
    assert rep["top_drifted"][0]["psi"] > 0.2 and rep["n_psi_significant"] == 1


def test_missing_rate_shift_is_flagged_by_psi():
    ref, cur = _data(0), _data(1)
    cur[::2, 2] = MISSING
    rep = DriftDetector(ref, ["a", "b", "c"]).evaluate(cur)
    assert rep["top_drifted"][0]["feature"] == "c" and rep["top_drifted"][0]["psi"] > 0.2


def test_binary_feature_prevalence_shift():
    ref, cur = _data(0), _data(1)
    cur[:, 1] = (np.random.default_rng(5).random(len(cur)) < 0.8).astype(float)
    assert DriftDetector(ref, ["a", "b", "c"]).evaluate(cur)["top_drifted"][0]["feature"] == "b"


def test_ks_and_score_psi_basics():
    a = np.sort(np.random.default_rng(0).normal(size=5000))
    assert ks_statistic(a, np.random.default_rng(1).normal(size=5000)) < 0.06
    assert ks_statistic(a, np.random.default_rng(1).normal(2, 1, size=5000)) > 0.5
    assert psi_1d(a, np.random.default_rng(2).normal(size=5000)) < 0.05
    assert psi_1d(a, np.random.default_rng(2).normal(1.5, 1, size=5000)) > 0.2


def _quiet_drift():
    return {"n_joint_significant": 0, "n_joint_severe": 0, "share_joint_significant": 0.0, "joint_features": []}


def test_joint_psi_and_ks_counts():
    ref, cur = _data(0), _data(1)
    cur[:, 0] += 2.0
    rep = DriftDetector(ref, ["a", "b", "c"]).evaluate(cur)
    assert rep["n_joint_significant"] == 1 and rep["n_joint_severe"] == 1 and rep["joint_features"] == ["a"]


def test_alert_rules():
    assert evaluate_window(12, _quiet_drift(), 0.01, 0.045, 0.045, 0.7, 0.7, 0.03, 0.03, 0.01, 0.01) == []
    loud = {"n_joint_significant": 6, "n_joint_severe": 4, "share_joint_significant": 0.07,
            "joint_features": ["C1", "C14", "TransactionAmt"]}
    alerts = evaluate_window(16, loud, 0.35, 0.09, 0.045, 0.5, 0.7, 0.03, 0.03)
    signals = {(a["signal"], a["severity"]) for a in alerts}
    assert {("feature_drift", "CRITICAL"), ("score_drift", "CRITICAL"), ("alert_rate", "CRITICAL"),
            ("performance", "CRITICAL")} <= signals
    assert retrain_recommended(alerts)


def test_moderate_signals_warn_but_do_not_recommend_retrain():
    warn = {"n_joint_significant": 5, "n_joint_severe": 0, "share_joint_significant": 0.06, "joint_features": ["a", "b", "c"]}
    alerts = evaluate_window(12, warn, 0.16, 0.045, 0.045)
    assert {(a["signal"], a["severity"]) for a in alerts} == {("feature_drift", "WARN"), ("score_drift", "WARN")}
    assert not retrain_recommended(alerts)


def test_noise_level_performance_drop_does_not_alert():
    # 8% drop with sd 0.04/0.03 is z = 1.2: inside the noise
    assert evaluate_window(20, _quiet_drift(), 0.01, 0.045, 0.045, 0.65, 0.71, 0.04, 0.03) == []
    # a 25.4% drop with the same noise is z = 3.6: real
    real = evaluate_window(20, _quiet_drift(), 0.01, 0.045, 0.045, 0.53, 0.71, 0.04, 0.03)
    assert [(a["signal"], a["severity"]) for a in real] == [("performance", "CRITICAL")]


def test_single_severe_feature_warns_but_below_control_levels_is_quiet():
    one_severe = {"n_joint_significant": 1, "n_joint_severe": 1, "share_joint_significant": 0.01, "joint_features": ["C1"]}
    assert [(a["signal"], a["severity"]) for a in evaluate_window(16, one_severe, 0.01, 0.045, 0.045)] == [("feature_drift", "WARN")]
    four_mild = {"n_joint_significant": 4, "n_joint_severe": 0, "share_joint_significant": 0.05, "joint_features": ["a"] * 4}
    assert evaluate_window(21, four_mild, 0.01, 0.045, 0.045) == []
