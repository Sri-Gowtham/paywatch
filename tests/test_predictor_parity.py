"""Encoders + model + calibrator in src/api/predictor.py must reproduce the batch pipeline scores."""

import json
import os

import numpy as np
import pytest

from conftest import FIXTURE, MODELS_DIR

pytestmark = pytest.mark.skipif(not os.path.exists(FIXTURE), reason="model assets / parity fixture not available")


@pytest.fixture(scope="module")
def predictor():
    from src.api.predictor import Predictor
    return Predictor(MODELS_DIR)


@pytest.fixture(scope="module")
def fixture():
    return json.load(open(FIXTURE))


def split_expected(predictor, expected):
    behavior, entity = {}, {}
    for f, v in zip(predictor.features, expected):
        if predictor.source[f] == "user_behavior_state":
            behavior[f] = v
        elif predictor.source[f] == "entity_history_state":
            entity[f] = v
    return behavior, entity


def test_feature_names_match_fixture(predictor, fixture):
    assert predictor.features == fixture["features"]


def test_stateless_features_match_batch(predictor, fixture):
    stateless = [i for i, f in enumerate(predictor.features)
                 if predictor.source[f] in ("raw_request_field", "anchored_D", "frequency_lookup")]
    bad = {}
    for row in fixture["rows"]:
        behavior, entity = split_expected(predictor, row["expected_features"])
        vec = predictor.build_vector(row["fields"], behavior, entity)
        exp = np.asarray(row["expected_features"], dtype=np.float32)
        for i in stateless:
            if not np.isclose(vec[i], exp[i], rtol=1e-6, atol=1e-6):
                bad.setdefault(predictor.features[i], []).append((row["transaction_id"], float(vec[i]), float(exp[i])))
    assert not bad, f"stateless features differ from batch on {len(bad)} columns: {dict(list(bad.items())[:5])}"


def test_scores_and_calibration_match_batch(predictor, fixture):
    for row in fixture["rows"]:
        vec = np.asarray(row["expected_features"], dtype=np.float32)
        score = predictor._score(vec)
        assert abs(score - row["expected_score"]) < 1e-5, row["transaction_id"]
        assert abs(predictor.calibrate(score) - row["expected_calibrated"]) < 1e-6


def test_action_tiers(predictor):
    thr = predictor.thresholds
    assert predictor.action_for(0.0) == "ALLOW"
    if "0.5" in thr:
        assert predictor.action_for(thr["0.5"] - 1e-6) == "ALLOW"
        assert predictor.action_for(thr["0.5"]) == "SOFT_FLAG"
    if "0.7" in thr:
        assert predictor.action_for(thr["0.7"]) == "CHALLENGE"
    if "0.9" in thr:
        assert predictor.action_for(thr["0.9"]) == "HARD_BLOCK"
        assert predictor.action_for(1.0) == "HARD_BLOCK"
