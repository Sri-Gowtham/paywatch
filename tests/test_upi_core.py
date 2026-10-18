"""UPI toolkit: schema privacy guard, shared feature path, synthetic traffic, numpy metrics and calibration."""

import numpy as np
import pytest

from src.upi import evalkit
from src.upi.features import FEATURE_NAMES, UpiFeaturizer, replay
from src.upi.schema import UpiTransaction, looks_raw, pseudonymise
from src.upi.synthetic import SYNTHETIC_NOTICE, generate

BASE = {"transaction_id": "t1", "timestamp": 1_700_000_000.0, "amount": 250.0, "payer_id": "p1", "payee_id": "q1", "device_id": "d1"}


def test_schema_accepts_pseudonymised_ids_and_rejects_raw_ones():
    assert UpiTransaction(**BASE).channel == "app"
    for bad in ("ravi@okhdfc", "9876543210", "+919876543210"):
        with pytest.raises(ValueError):
            UpiTransaction(**{**BASE, "payer_id": bad})
    with pytest.raises(ValueError):
        UpiTransaction(**{**BASE, "amount": -5})
    assert looks_raw("a@b") and not looks_raw("9f2c01aa77be3d10")


def test_pseudonymise_is_stable_keyed_and_not_the_input():
    a, b = pseudonymise("ravi@okhdfc", "salt-1"), pseudonymise("ravi@okhdfc", "salt-2")
    assert a == pseudonymise("ravi@okhdfc", "salt-1") and a != b and "ravi" not in a and len(a) == 16 and not looks_raw(a)
    with pytest.raises(ValueError):
        pseudonymise("x", "")


def test_synthetic_traffic_is_sorted_labelled_and_marked_synthetic():
    rows = generate(n_payers=300, n_payees=120, days=40, seed=1)
    ts = [r["timestamp"] for r in rows]
    assert ts == sorted(ts) and len({r["transaction_id"] for r in rows}) == len(rows)
    rate = np.mean([r["is_fraud"] for r in rows])
    assert 0.005 < rate < 0.05 and "SYNTHETIC" in SYNTHETIC_NOTICE
    assert not any(looks_raw(r["payer_id"]) or looks_raw(r["payee_id"]) for r in rows)


def test_featurizer_never_uses_the_current_transaction_or_its_label():
    fz = UpiFeaturizer()
    v1, _ = fz.featurize(BASE, commit=True, tx_id="t1")
    f = dict(zip(FEATURE_NAMES, v1))
    assert f["uid_prior_tx_count"] == 0 and f["pair_count"] == 0 and f["payee_seen"] == 0 and f["h7_payer_n"] == 0
    v2, _ = fz.featurize({**BASE, "transaction_id": "t2", "timestamp": BASE["timestamp"] + 600}, commit=True, tx_id="t2")
    g = dict(zip(FEATURE_NAMES, v2))
    assert g["uid_prior_tx_count"] == 1 and g["pair_count"] == 1 and g["payee_seen"] == 1 and g["is_new_merchant"] == 0
    assert g["h7_payer_n"] == 0                                   # t1's label has not arrived, so no fraud history yet
    assert fz.label("t1", True) and not fz.label("t1", True)      # a label joins exactly once
    v3, _ = fz.featurize({**BASE, "transaction_id": "t3", "timestamp": BASE["timestamp"] + 1200}, commit=False)
    assert dict(zip(FEATURE_NAMES, v3))["h7_payer_fraud_n"] == 1


def test_commit_false_changes_nothing():
    fz = UpiFeaturizer()
    fz.featurize(BASE, commit=False)
    assert len(fz.behavior) == 0 and not fz.link.pair and not fz.history.pending
    v, _ = fz.featurize(BASE, commit=False)
    assert dict(zip(FEATURE_NAMES, v))["uid_prior_tx_count"] == 0


def test_replay_applies_labels_only_after_the_lag():
    rows = [{**BASE, "transaction_id": "a", "timestamp": 1_700_000_000.0, "is_fraud": 1},
            {**BASE, "transaction_id": "b", "timestamp": 1_700_000_000.0 + 3 * 86400, "is_fraud": 0},
            {**BASE, "transaction_id": "c", "timestamp": 1_700_000_000.0 + 8 * 86400, "is_fraud": 0}]
    feats = {r["transaction_id"]: dict(zip(FEATURE_NAMES, v)) for v, _, r in replay(rows, label_lag_days=7)}
    assert feats["b"]["h7_payer_fraud_n"] == 0 and feats["c"]["h7_payer_fraud_n"] == 1     # a's fraud label is known only after 7 days
    with pytest.raises(ValueError):
        list(replay([rows[1], rows[0]]))


def test_metrics_match_hand_computed_values():
    y = np.array([1, 0, 1, 0, 0, 1])
    s = np.array([0.9, 0.8, 0.7, 0.4, 0.3, 0.2])
    assert evalkit.average_precision(y, s) == pytest.approx((1 / 1 + 2 / 3 + 3 / 6) / 3)
    assert evalkit.roc_auc(y, s) == pytest.approx(5 / 9)
    assert evalkit.roc_auc(np.array([1, 0]), np.array([0.5, 0.5])) == pytest.approx(0.5)
    assert evalkit.recall_at_fpr(y, s, 0.34) == pytest.approx(2 / 3)


def test_isotonic_is_monotone_and_tiers_respect_the_precision_target():
    rng = np.random.default_rng(0)
    raw = rng.random(4000)
    y = (rng.random(4000) < raw ** 3).astype(int)
    kx, ky = evalkit.isotonic_fit(raw, y)
    assert all(b >= a for a, b in zip(ky, ky[1:])) and 0 <= min(ky) and max(ky) <= 1
    cal = np.interp(raw, kx, ky)
    th = evalkit.tier_thresholds(cal, y, (0.3, 0.6), n_min=30)
    for t, thr in th.items():
        if thr is not None:
            assert y[cal >= thr].mean() >= float(t) - 1e-9
    lo, hi = evalkit.bootstrap_ci(y, cal, evalkit.average_precision, n=50)
    assert lo < evalkit.average_precision(y, cal) < hi
