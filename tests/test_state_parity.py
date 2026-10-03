"""Streaming state (src/api/state.py) must reproduce the batch features (src/features) exactly."""

import numpy as np
import pandas as pd
import pytest

from src.api.state import BASE_KEYS, BEHAVIOR_FEATURES, EntityHistoryStore, UserBehaviorState, entity_keys, uid_key

entity_history = pytest.importorskip("entity_history")
upi = pytest.importorskip("upi_fingerprint")

LAG_DAYS = 7


def synthetic_frame(n=800, seed=0):
    rng = np.random.default_rng(seed)
    n_cards = 25
    card = rng.integers(0, n_cards, n)
    ts = np.sort(rng.choice(np.arange(1, 90 * 86400, 37), n, replace=False)).astype(float)
    start = rng.integers(0, 40, n_cards)
    d1 = np.floor(ts / 86400.0 - start[card])
    df = pd.DataFrame({
        "TransactionDT": ts,
        "TransactionAmt": np.round(rng.gamma(2.0, 40.0, n) + 1, 2),
        "isFraud": (rng.random(n) < 0.05 + 0.4 * (card % 7 == 0)).astype(int),
        "card1": 1000 + card, "card2": 100.0 + (card % 5), "card3": 150.0, "card5": 226.0,
        "addr1": np.where(rng.random(n) < 0.1, np.nan, 300.0 + card % 9),
        "D1": np.where(rng.random(n) < 0.08, np.nan, d1),
        "P_emaildomain": np.where(rng.random(n) < 0.15, None, np.array(["gmail.com", "yahoo.com", "icloud.com"])[rng.integers(0, 3, n)]),
        "DeviceInfo": np.where(rng.random(n) < 0.5, None, np.array(["Windows", "iOS", "SM-G960F"])[rng.integers(0, 3, n)]),
        "ProductCD": np.array(["W", "C", "H"])[rng.integers(0, 3, n)],
    })
    df["P_emaildomain"] = df["P_emaildomain"].astype(object)
    df["DeviceInfo"] = df["DeviceInfo"].astype(object)
    df["addr2"] = np.nan
    for col in ("R_emaildomain", "id_31", "id_30"):
        df[col] = pd.Series([None] * n, dtype=object)
    return df


def row_fields(row):
    return {k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)) for k, v in row.items()}


def test_behavior_state_matches_batch():
    df = synthetic_frame()
    batch = upi.add_upi_fingerprint_features(df.copy())
    state = UserBehaviorState()
    rows = []
    for _, r in df.iterrows():
        f = row_fields(r)
        rows.append(state.observe(f, uid_key(f)))
    stream = pd.DataFrame(rows)
    for name in BEHAVIOR_FEATURES:
        np.testing.assert_allclose(stream[name].to_numpy(dtype=float), batch[name].to_numpy(dtype=float),
                                   rtol=1e-9, atol=1e-9, err_msg=f"behaviour feature {name} differs from batch")


def test_entity_history_matches_batch():
    df = synthetic_frame()
    batch = entity_history.add_lagged_fraud_history(upi.add_upi_fingerprint_features(df.copy()), lags=(LAG_DAYS,),
                                                    decay_lags=(), key_names=BASE_KEYS)
    store = EntityHistoryStore()
    ts = df["TransactionDT"].to_numpy()
    labels = df["isFraud"].to_numpy()
    all_keys = [entity_keys(row_fields(r)) for _, r in df.iterrows()]
    j, out = 0, []
    for i in range(len(df)):
        while j < len(df) and ts[j] <= ts[i] - LAG_DAYS * 86400.0:
            store.add_label_for_keys(all_keys[j], bool(labels[j]))
            j += 1
        out.append(store.features(all_keys[i]))
    stream = pd.DataFrame(out)
    for col in stream.columns:
        tol = {"rtol": 1e-5, "atol": 1e-6} if col.endswith("fraud_rate") else {"rtol": 0, "atol": 0}
        np.testing.assert_allclose(stream[col].to_numpy(dtype=float), batch[col].to_numpy(dtype=float),
                                   err_msg=f"entity history feature {col} differs from batch", **tol)


def test_missing_d1_gets_distinct_start_day():
    f = {"TransactionDT": 86400.0 * 10, "card1": 1, "card2": 2.0, "card3": 3.0, "card5": 4.0, "addr1": 5.0, "D1": None}
    assert uid_key(f).endswith("_-1")
