"""Snapshot round trip: a restored process must score exactly like the process that was snapshotted."""

import numpy as np
import pandas as pd
import pytest

from src.api.snapshot import export_state, import_state, load_state, save_state
from src.api.state import EntityHistoryStore, UserBehaviorState, entity_keys, uid_key

from test_state_parity import row_fields, synthetic_frame


def _run(df, behavior, history, upto):
    feats = []
    for _, r in df.iloc[:upto].iterrows():
        f = row_fields(r)
        keys = entity_keys(f)
        feats.append(behavior.observe(f, uid_key(f)))
        history.add_label_for_keys(keys, bool(f["isFraud"]))
    return feats


def test_state_round_trip_through_gzip(tmp_path):
    df = synthetic_frame(n=500, seed=3)
    b1, h1 = UserBehaviorState(), EntityHistoryStore()
    _run(df, b1, h1, 300)
    path = str(tmp_path / "state.json.gz")
    assert save_state(export_state(b1, h1), path) > 0

    b2, h2 = UserBehaviorState(), EntityHistoryStore()
    import_state(load_state(path), b2, h2)
    assert len(b2) == len(b1)

    for _, r in df.iloc[300:].iterrows():
        f = row_fields(r)
        keys = entity_keys(f)
        a, b = b1.observe(f, uid_key(f)), b2.observe(f, uid_key(f))
        for k in a:
            assert a[k] == pytest.approx(b[k], rel=1e-12, abs=1e-12), k
        ea, eb = h1.features(keys), h2.features(keys)
        assert ea == eb


def test_pruning_keeps_fraud_labelled_entities():
    df = synthetic_frame(n=400, seed=4)
    b, h = UserBehaviorState(), EntityHistoryStore()
    _run(df, b, h, 400)
    pruned = export_state(b, h, min_entity_n=5)
    for name, table in h.tables.items():
        for key, (n, fraud) in table.items():
            if fraud > 0:
                assert key in pruned["entity_tables"][name]
            elif n < 5:
                assert key not in pruned["entity_tables"][name]
