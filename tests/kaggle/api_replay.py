"""PayWatch — API test kernel (Kaggle, CPU, internet ON only for `pip install`).

1) assemble repo layout + models/compact from the datasets / kernel outputs
2) run the pytest suite (state parity vs batch, predictor parity vs batch scores, API behaviour)
3) replay a 20k-row time-ordered stream through the API with 7-day delayed /feedback and report
   score quality, tier distribution and latency
The assembled models/compact folder is written to /kaggle/working so it can be downloaded into the repo.
"""

import glob
import heapq
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

T0 = time.time()
LAG_S = 7 * 86400.0


def log(msg):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {msg}", flush=True)


subprocess.run([sys.executable, "-m", "pip", "install", "-q", "fastapi", "httpx", "pytest"], check=True)
log("pip install done")

find = lambda p: sorted(glob.glob(p, recursive=True))  # noqa: E731
REPO, MODELS, OUT = "/kaggle/working/repo", "/kaggle/working/models/compact", "/kaggle/working"
for d in (f"{REPO}/src/api", f"{REPO}/tests", MODELS):
    os.makedirs(d, exist_ok=True)

state_py = [p for p in find("/kaggle/input/**/state.py") if os.path.exists(os.path.join(os.path.dirname(p), "predictor.py"))][0]
for f in os.listdir(os.path.dirname(state_py)):
    if f.endswith(".py"):
        shutil.copy(os.path.join(os.path.dirname(state_py), f), f"{REPO}/src/api/{f}")
open(f"{REPO}/src/__init__.py", "w").close()
tests_dir = os.path.dirname(find("/kaggle/input/**/test_api.py")[0])
for f in os.listdir(tests_dir):
    if f.endswith(".py"):
        shutil.copy(os.path.join(tests_dir, f), f"{REPO}/tests/{f}")
features_dir = os.path.dirname(find("/kaggle/input/**/upi_fingerprint.py")[0])

for p in find("/kaggle/input/**/prod_compact/xgb_seed*.json") + find("/kaggle/input/**/prod_compact/serving_spec.json"):
    shutil.copy(p, MODELS)
for name in ("categories.json", "freq_tables.json", "isotonic.json", "feature_meta.json"):
    shutil.copy(find(f"/kaggle/input/**/{name}")[0], MODELS)
# the repo ships a SYNTHETIC fixture (paywatch-synthfix); the raw-row one from the assets kernel is not used
fixtures = find("/kaggle/input/**/paywatch-synthfix/parity_fixture.json") or find("/kaggle/input/**/parity_fixture.json")
assert "synthfix" in fixtures[0], f"synthetic fixture not found among {fixtures}"
shutil.copy(fixtures[0], f"{MODELS}/parity_fixture.json")
stream_path = find("/kaggle/input/**/stream_sample.parquet")[0]
log(f"assembled: {sorted(os.listdir(MODELS))}")

results = {}

# ------------------------------------------------------------ 2) pytest
env = dict(os.environ, PAYWATCH_MODELS_DIR=MODELS, PAYWATCH_FIXTURE=f"{MODELS}/parity_fixture.json",
           PAYWATCH_FEATURES_DIR=features_dir, PYTHONPATH=REPO)
r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--tb=short", "tests"],
                   cwd=REPO, env=env, capture_output=True, text=True)
print(r.stdout[-8000:])
print(r.stderr[-2000:])
results["pytest"] = {"returncode": r.returncode, "tail": r.stdout.strip().splitlines()[-12:]}
log(f"pytest returncode={r.returncode}")

# ------------------------------------------------------------ 3) stream replay
sys.path.insert(0, REPO)
os.environ["PAYWATCH_MODELS_DIR"] = MODELS
from fastapi.testclient import TestClient  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from src.api.main import create_app  # noqa: E402

client = TestClient(create_app(MODELS))
df = pd.read_parquet(stream_path).sort_values("TransactionDT").reset_index(drop=True)
field_cols = [c for c in df.columns if c not in ("TransactionID", "isFraud")]
records = df[field_cols].to_dict("records")
ids = df["TransactionID"].astype(str).tolist()
labels = df["isFraud"].to_numpy()
times = df["TransactionDT"].to_numpy(dtype=float)


def clean(rec):
    return {k: (None if (v is None or (isinstance(v, float) and v != v)) else v) for k, v in rec.items()}


pending, scores, probs, actions, lat_client, lat_server = [], [], [], [], [], []
flags = {"new_user": 0, "uid_has_labeled_history": 0, "uid_has_prior_fraud": 0}
errors = 0
t_start = time.perf_counter()
for i, rec in enumerate(records):
    while pending and pending[0][0] <= times[i]:
        _, tid, lab = heapq.heappop(pending)
        client.post("/feedback", json={"transaction_id": tid, "is_fraud": bool(lab)})
    t0 = time.perf_counter()
    resp = client.post("/predict", json={"transaction_id": ids[i], "fields": clean(rec), "explain": False})
    lat_client.append((time.perf_counter() - t0) * 1000.0)
    if resp.status_code != 200:
        errors += 1
        scores.append(np.nan)
        probs.append(np.nan)
        actions.append("ERROR")
        continue
    body = resp.json()
    scores.append(body["score"])
    probs.append(body["calibrated_probability"])
    actions.append(body["action"])
    lat_server.append(body["latency_ms"])
    for k in flags:
        flags[k] += int(body["flags"][k])
    heapq.heappush(pending, (times[i] + LAG_S, ids[i], labels[i]))
    if (i + 1) % 5000 == 0:
        log(f"replayed {i + 1}/{len(records)}")
wall = time.perf_counter() - t_start

ok = ~np.isnan(scores)
y, p = labels[ok], np.asarray(scores)[ok]
act = np.asarray(actions)[ok]
pct = lambda a, q: float(np.percentile(a, q))  # noqa: E731
tiers = {}
for name in ("SOFT_FLAG", "CHALLENGE", "HARD_BLOCK"):
    thr_actions = {"SOFT_FLAG": {"SOFT_FLAG", "CHALLENGE", "HARD_BLOCK"}, "CHALLENGE": {"CHALLENGE", "HARD_BLOCK"},
                   "HARD_BLOCK": {"HARD_BLOCK"}}[name]
    m = np.isin(act, list(thr_actions))
    tiers[name] = {"alert_rate": float(m.mean()), "precision": float(y[m].mean()) if m.any() else None,
                   "recall": float(y[m].sum() / y.sum())}
results["stream_replay"] = {
    "requests": int(len(records)), "errors": errors, "wall_seconds": wall, "throughput_rps_with_feedback": len(records) / wall,
    "pr_auc": float(average_precision_score(y, p)), "roc_auc": float(roc_auc_score(y, p)),
    "fraud_rate": float(y.mean()), "tiers_cumulative": tiers,
    "flag_rates": {k: v / ok.sum() for k, v in flags.items()},
    "latency_server_ms": {"p50": pct(lat_server, 50), "p95": pct(lat_server, 95), "p99": pct(lat_server, 99)},
    "latency_client_ms_incl_testclient_overhead": {"p50": pct(lat_client, 50), "p95": pct(lat_client, 95), "p99": pct(lat_client, 99)},
}

# explain=True latency on 300 rows without touching state
lat_explain = []
for rec, tid in list(zip(records, ids))[-300:]:
    resp = client.post("/predict", json={"transaction_id": "x" + tid, "fields": clean(rec), "explain": True, "commit": False})
    lat_explain.append(resp.json()["latency_ms"])
results["latency_with_explanations_ms"] = {"p50": pct(lat_explain, 50), "p95": pct(lat_explain, 95), "p99": pct(lat_explain, 99)}
results["model_info"] = client.get("/model-info").json()
results["models_dir_files"] = {f: os.path.getsize(os.path.join(MODELS, f)) for f in sorted(os.listdir(MODELS))}

print("\n=== PYTEST ===")
print("\n".join(results["pytest"]["tail"]))
print("\n=== STREAM REPLAY ===")
print(json.dumps(results["stream_replay"], indent=1))
print("\n=== LATENCY WITH SHAP REASONS ===", results["latency_with_explanations_ms"])
print("\n=== MODEL INFO ===", results["model_info"])
print("\n=== models/compact files ===")
for f, s in results["models_dir_files"].items():
    print(f"{f:<24} {s:>12,}")

with open(f"{OUT}/results.json", "w") as f:
    json.dump(results, f, indent=1)
shutil.rmtree(REPO, ignore_errors=True)
log("done")
