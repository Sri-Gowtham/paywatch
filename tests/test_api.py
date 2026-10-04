import gzip
import json
import os

import pytest

from conftest import FIXTURE, MODELS_DIR

pytestmark = pytest.mark.skipif(not os.path.exists(FIXTURE), reason="model assets / parity fixture not available")

fastapi_testclient = pytest.importorskip("fastapi.testclient")


@pytest.fixture()
def client():
    from src.api.main import create_app
    return fastapi_testclient.TestClient(create_app(MODELS_DIR))


@pytest.fixture(scope="module")
def sample_fields():
    return json.load(open(FIXTURE))["rows"][0]["fields"]


def test_health_and_model_info(client):
    assert client.get("/health").json()["status"] == "ok"
    info = client.get("/model-info").json()
    assert info["n_features"] > 0 and info["n_models"] >= 1


def test_predict_response_shape(client, sample_fields):
    r = client.post("/predict", json={"transaction_id": "t1", "fields": sample_fields, "top_k": 5})
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0.0 <= body["score"] <= 1.0 and 0.0 <= body["calibrated_probability"] <= 1.0
    assert body["action"] in {"ALLOW", "SOFT_FLAG", "CHALLENGE", "HARD_BLOCK"}
    assert len(body["reasons"]) == 5 and body["flags"]["new_user"] is True
    assert {"feature", "shap", "direction", "text"} <= set(body["reasons"][0])


def test_commit_false_does_not_change_state(client, sample_fields):
    before = client.get("/model-info").json()["users_tracked"]
    client.post("/predict", json={"transaction_id": "w1", "fields": sample_fields, "commit": False})
    assert client.get("/model-info").json()["users_tracked"] == before
    client.post("/predict", json={"transaction_id": "w2", "fields": sample_fields, "commit": True})
    assert client.get("/model-info").json()["users_tracked"] == before + 1


def test_missing_required_field_is_422(client, sample_fields):
    bad = {k: v for k, v in sample_fields.items() if k != "TransactionAmt"}
    assert client.post("/predict", json={"transaction_id": "x", "fields": bad}).status_code == 422


def test_feedback_updates_entity_history(client, sample_fields):
    assert client.post("/feedback", json={"transaction_id": "nope", "is_fraud": True}).status_code == 404
    first = client.post("/predict", json={"transaction_id": "a1", "fields": sample_fields, "explain": False}).json()
    assert first["flags"]["uid_has_labeled_history"] is False
    assert client.post("/feedback", json={"transaction_id": "a1", "is_fraud": True}).status_code == 200
    assert client.post("/feedback", json={"transaction_id": "a1", "is_fraud": True}).status_code == 404

    again = dict(sample_fields)
    again["TransactionDT"] = sample_fields["TransactionDT"] + 8 * 86400
    if again.get("D1") is not None:
        again["D1"] = sample_fields["D1"] + 8
    second = client.post("/predict", json={"transaction_id": "a2", "fields": again, "explain": False}).json()
    if sample_fields.get("D1") is not None:
        assert second["flags"]["uid_has_labeled_history"] is True and second["flags"]["uid_has_prior_fraud"] is True
        assert second["score"] > first["score"], "a confirmed prior fraud on the same user must raise the score"


def _state(client):
    info = client.get("/model-info").json()
    return info["users_tracked"], info["pending_labels"]


def test_non_numeric_required_field_is_422_and_changes_nothing(client, sample_fields):
    before = _state(client)
    bad = dict(sample_fields)
    bad["TransactionAmt"] = "abc"
    assert client.post("/predict", json={"transaction_id": "n1", "fields": bad}).status_code == 422
    assert _state(client) == before


def test_rejected_request_does_not_commit_state(client, sample_fields):
    before = _state(client)
    bad = dict(sample_fields)
    bad["D1"] = "xyz"                              # non-numeric optional field the state code uses
    assert client.post("/predict", json={"transaction_id": "n2", "fields": bad}).status_code == 422
    assert _state(client) == before


def test_retry_of_the_same_transaction_id_is_idempotent(client, sample_fields):
    body = {"transaction_id": "r1", "fields": sample_fields, "explain": False}
    first = client.post("/predict", json=body).json()
    state = _state(client)
    second = client.post("/predict", json=body).json()
    assert second["score"] == first["score"] and second["flags"] == first["flags"]
    assert _state(client) == state


def test_bad_snapshot_leaves_live_state_untouched(client, sample_fields, tmp_path):
    client.post("/predict", json={"transaction_id": "s1", "fields": sample_fields})
    before = _state(client)
    path = tmp_path / "bad.json.gz"
    with gzip.open(path, "wt") as f:
        json.dump({"version": 99, "users": {}, "entity_tables": {}, "pending": {}}, f)
    with pytest.raises(ValueError):
        client.app.state.predictor.load_state(str(path))
    assert _state(client) == before
