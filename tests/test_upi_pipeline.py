"""UPI toolkit end to end on SYNTHETIC data: training kit, shadow-mode API (auth, privacy, idempotency, state) and shadow evaluation."""

import json
import os

import numpy as np
import pytest

pytest.importorskip("lightgbm")
fastapi_testclient = pytest.importorskip("fastapi.testclient")

from src.upi import evalkit, shadow_eval  # noqa: E402
from src.upi.features import FEATURE_NAMES, UpiFeaturizer, replay  # noqa: E402
from src.upi.main import create_upi_app  # noqa: E402
from src.upi.synthetic import generate  # noqa: E402
from src.upi.train import load_rows, train  # noqa: E402

SMALL = {"seeds": [42, 1], "max_rounds": 120, "early_stopping_rounds": 30, "min_frauds_per_slice": 5, "synthetic": True,
         "lightgbm": {"num_leaves": 15, "min_data_in_leaf": 10, "learning_rate": 0.1}, "criteria": {"min_pr_auc": 0.05, "min_recall_at_fpr_1pct": 0.05}}
KEY = "test-key-123"
SALT = "test-salt"


@pytest.fixture(scope="module")
def rows():
    return generate(n_payers=600, n_payees=200, days=45, tx_per_payer_per_day=0.5, seed=4)


@pytest.fixture(scope="module")
def model_dir(rows, tmp_path_factory):
    out = tmp_path_factory.mktemp("upi_model")
    train(rows, str(out), SMALL, log=lambda *_: None)
    return str(out)


def test_training_kit_writes_a_loadable_model_and_reports_criteria(model_dir):
    for f in ("lgbm_seed42.txt", "lgbm_seed1.txt", "isotonic.json", "serving_spec.json", "results.json", "warm_state.json.gz"):
        assert os.path.exists(os.path.join(model_dir, f)), f
    spec = json.load(open(os.path.join(model_dir, "serving_spec.json")))
    res = json.load(open(os.path.join(model_dir, "results.json")))
    assert spec["kind"] == "upi" and spec["features"] == list(FEATURE_NAMES) and spec["trained_on"]["synthetic"] is True
    m = res["metrics_second_half_holdout"]
    assert m["pr_auc"] > 3 * m["fraud_rate_report_slice"]                # the injected patterns are learnable; this only proves the plumbing
    assert res["criteria_met"] is True and res["synthetic"] is True


def test_training_refuses_data_with_too_few_frauds(rows, tmp_path):
    few = [dict(r, is_fraud=0) for r in rows]
    for r in few[:3]:
        r["is_fraud"] = 1
    with pytest.raises(ValueError, match="confirmed frauds"):
        train(few, str(tmp_path), SMALL, log=lambda *_: None)


def test_loader_rejects_raw_identifiers(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("transaction_id,timestamp,amount,payer_id,payee_id,device_id,channel,is_fraud\nt1,1700000000,10,ravi@okhdfc,q1,d1,app,0\n")
    with pytest.raises(ValueError, match="raw identifier"):
        load_rows(str(p))


def test_live_features_equal_replay_features_when_labels_cannot_arrive(rows):
    sub = rows[:400]
    replayed = [v for v, _, _ in replay(sub, label_lag_days=1e9)]
    fz = UpiFeaturizer()
    live = [fz.featurize(r, commit=True, tx_id=r["transaction_id"])[0] for r in sub]
    assert np.array_equal(np.vstack(replayed), np.vstack(live))


def make_app(model_dir, tmp_path, **kw):
    cfg = {"api_keys": [KEY], "log_path": str(tmp_path / "decisions.jsonl"), "log_salt": SALT, "autosave_seconds": 0}
    cfg.update(kw)
    return create_upi_app(model_dir, **cfg)


TX = {"transaction_id": "tx-1", "timestamp": 1_700_100_000.0, "amount": 4321.5, "payer_id": "payerAAA", "payee_id": "payeeBBB", "device_id": "devCCC"}


def test_service_refuses_to_start_without_keys(model_dir):
    with pytest.raises(RuntimeError, match="authentication"):
        create_upi_app(model_dir, api_keys=[], allow_no_auth=False)


def test_auth_is_required_and_health_is_open(model_dir, tmp_path):
    with fastapi_testclient.TestClient(make_app(model_dir, tmp_path)) as c:
        assert c.get("/health").json()["shadow"] is True
        assert c.post("/score", json=TX).status_code == 401
        assert c.post("/score", json=TX, headers={"X-API-Key": "wrong"}).status_code == 401
        assert c.get("/model-info").status_code == 401
        assert c.post("/score", json=TX, headers={"X-API-Key": KEY}).status_code == 200


def test_validation_errors_never_echo_the_submitted_identifier(model_dir, tmp_path):
    with fastapi_testclient.TestClient(make_app(model_dir, tmp_path)) as c:
        r = c.post("/score", json={**TX, "payer_id": "ravi@okhdfc"}, headers={"X-API-Key": KEY})
        assert r.status_code == 422 and "ravi@okhdfc" not in r.text and "pseudonymised" in r.text
        big = c.post("/score", content=b"x" * 70_000, headers={"X-API-Key": KEY, "Content-Type": "application/json"})
        assert big.status_code == 413


def test_scores_are_advisory_idempotent_and_logged_without_raw_fields(model_dir, tmp_path):
    app = make_app(model_dir, tmp_path)
    with fastapi_testclient.TestClient(app) as c:
        a = c.post("/score", json=TX, headers={"X-API-Key": KEY}).json()
        b = c.post("/score", json=TX, headers={"X-API-Key": KEY}).json()
        assert a["shadow"] is True and "must not block" in a["advisory"] and a["synthetic_model"] is True
        assert 0.0 <= a["calibrated_probability"] <= 1.0 and a["action"] in {"ALLOW", "SOFT_FLAG", "CHALLENGE", "HARD_BLOCK"}
        assert b["calibrated_probability"] == a["calibrated_probability"]                    # a retry is not counted twice
        assert len(a["reasons"]) == 5 and all(r["text"] for r in a["reasons"])
        peek = c.post("/score", json={**TX, "transaction_id": "tx-peek", "commit": False}, headers={"X-API-Key": KEY})
        assert peek.status_code == 200
    lines = [json.loads(x) for x in open(tmp_path / "decisions.jsonl")]
    assert len(lines) == 1 and lines[0]["event"] == "decision" and "x" not in lines[0]      # features are not logged by default
    raw = open(tmp_path / "decisions.jsonl").read()
    for secret in ("tx-1", "payerAAA", "payeeBBB", "devCCC", "4321.5"):
        assert secret not in raw


def test_feature_logging_is_opt_in_and_still_holds_no_identifiers(model_dir, tmp_path):
    with fastapi_testclient.TestClient(make_app(model_dir, tmp_path, log_features=True)) as c:
        c.post("/score", json=TX, headers={"X-API-Key": KEY})
        assert c.get("/model-info", headers={"X-API-Key": KEY}).json()["logging_features"] is True
    raw = open(tmp_path / "decisions.jsonl").read()
    assert len(json.loads(raw)["x"]) == len(FEATURE_NAMES)
    for secret in ("tx-1", "payerAAA", "payeeBBB", "devCCC"):
        assert secret not in raw
    assert "4321.5" in raw                       # documented: the amount is one of the logged features, which is why this is opt-in


def test_feedback_joins_exactly_once_and_is_logged(model_dir, tmp_path):
    with fastapi_testclient.TestClient(make_app(model_dir, tmp_path)) as c:
        c.post("/score", json=TX, headers={"X-API-Key": KEY})
        assert c.post("/feedback", json={"transaction_id": "tx-1", "is_fraud": True}, headers={"X-API-Key": KEY}).status_code == 200
        assert c.post("/feedback", json={"transaction_id": "tx-1", "is_fraud": True}, headers={"X-API-Key": KEY}).status_code == 404
        later = {**TX, "transaction_id": "tx-2", "timestamp": TX["timestamp"] + 100}
        assert c.post("/score", json=later, headers={"X-API-Key": KEY, }).json()["reasons"] is not None
    events = [json.loads(x) for x in open(tmp_path / "decisions.jsonl")]
    assert [e["event"] for e in events] == ["decision", "label", "decision"] and events[1]["is_fraud"] == 1


def test_state_survives_a_restart(model_dir, tmp_path):
    state = str(tmp_path / "state.json.gz")
    with fastapi_testclient.TestClient(make_app(model_dir, tmp_path, state_path=state)) as c:
        c.post("/score", json=TX, headers={"X-API-Key": KEY})
    assert os.path.exists(state)                                                       # saved on graceful shutdown
    app2 = make_app(model_dir, tmp_path, state_path=state)
    fz = app2.state.predictor.fz
    vec, _ = fz.featurize({**TX, "transaction_id": "tx-9", "timestamp": TX["timestamp"] + 60}, commit=False)
    f = dict(zip(FEATURE_NAMES, vec))
    assert f["uid_prior_tx_count"] == 1 and f["pair_count"] == 1 and f["payee_seen"] == 1


def events_for(n_legit, n_fraud, days=20, seed=0, labelled=True):
    rng = np.random.default_rng(seed)
    ev = []
    for i in range(n_legit + n_fraud):
        fraud = i >= n_legit
        ts = 1_700_000_000.0 + rng.uniform(0, days) * 86400
        p = float(np.clip(rng.beta(6, 2) if fraud else rng.beta(1, 12), 0, 1))
        ev.append({"event": "decision", "tx": f"h{i}", "ts": ts, "p": p, "score": p, "tier": "ALLOW", "x": [float(rng.normal()), float(rng.normal())]})
        if fraud and labelled:
            ev.append({"event": "label", "tx": f"h{i}", "is_fraud": 1})
    return ev


def test_shadow_eval_passes_fails_and_is_inconclusive_honestly():
    tiers = {"0.5": 0.5, "0.9": 0.8}
    ok = shadow_eval.evaluate(events_for(3000, 200), 7.0, tiers, {"min_frauds": 50, "min_pr_auc": 0.3, "min_recall_at_fpr_1pct": 0.3}, feature_names=["a", "b"])
    assert ok["verdict"] == "PASS" and ok["metrics"]["pr_auc"] > 0.3 and ok["tiers"]["0.5"]["alert_rate"] > 0
    bad = shadow_eval.evaluate(events_for(3000, 200), 7.0, tiers, {"min_frauds": 50, "min_pr_auc": 1.01})
    assert bad["verdict"] == "FAIL" and bad["criteria_checks"]["pr_auc"] is False
    few = shadow_eval.evaluate(events_for(3000, 10), 7.0, tiers, {"min_frauds": 100})
    assert few["verdict"] == "INCONCLUSIVE" and "pre-registered minimum" in few["reason"]
    assert shadow_eval.evaluate([], 7.0)["verdict"] == "NO DATA"
    assert "INCONCLUSIVE" in shadow_eval.to_markdown(few) and "PASS" in shadow_eval.to_markdown(ok)


def test_shadow_eval_ignores_decisions_whose_label_window_is_still_open():
    ev = events_for(2000, 150, days=20)
    res = shadow_eval.evaluate(ev, 7.0, None, {"min_frauds": 10})
    newest = max(e["ts"] for e in ev if e["event"] == "decision")
    assert res["n_mature"] < res["n_decisions"]
    cutoff = newest - 7 * 86400
    assert res["n_mature"] == sum(1 for e in ev if e["event"] == "decision" and e["ts"] <= cutoff)


def test_numpy_metrics_agree_with_the_ranking_they_describe():
    y = np.array([0, 0, 1, 1, 0, 1])
    perfect = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.7])
    assert evalkit.average_precision(y, perfect) == 1.0 and evalkit.roc_auc(y, perfect) == 1.0


def test_cumulative_state_features_are_left_out_of_the_feature_drift_ranking():
    for name in ("payee_age_s", "payee_seen", "uid_prior_tx_count", "h7_payer_n", "h7_device_fraud_n"):
        assert shadow_eval.grows_by_design(name)
    for name in ("amount", "hour", "h7_payer_fraud_rate", "pair_count", "payee_distinct_payers"):
        assert not shadow_eval.grows_by_design(name)
    ev = events_for(2000, 100, days=20)
    res = shadow_eval.evaluate(ev, 7.0, None, {"min_frauds": 10}, feature_names=["payee_age_s", "amount"])
    assert [d["feature"] for d in res["drift"]["top_shifted_features"]] == ["amount"]
