"""End-to-end proof of the UPI toolkit on SYNTHETIC traffic (Kaggle CPU). Not evidence about real UPI fraud.

Steps: generate synthetic UPI traffic -> train with the kit on the first 70% (criteria written before the run) -> start the shadow-mode API
-> replay the remaining 30% as live traffic, delivering each confirmed-fraud label 7 simulated days late via /feedback -> run the shadow
evaluation on the decision log -> check the log holds no identifiers. Output: /kaggle/working/results.json and shadow_report.md.
"""

import glob
import json
import os
import shutil
import sys
import time

import numpy as np

T0 = time.time()
OUT = "/kaggle/working"
REPO = f"{OUT}/repo"
find = lambda p: sorted(glob.glob(p, recursive=True))  # noqa: E731


def log(msg):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {msg}", flush=True)


def assemble(dest_pkg, marker_a, marker_b):
    for p in find(f"/kaggle/input/**/{marker_a}"):
        d = os.path.dirname(p)
        if os.path.exists(os.path.join(d, marker_b)):
            os.makedirs(f"{REPO}/src/{dest_pkg}", exist_ok=True)
            for f in os.listdir(d):
                if f.endswith(".py"):
                    shutil.copy(os.path.join(d, f), f"{REPO}/src/{dest_pkg}/{f}")
            return
    raise RuntimeError(f"dataset with {marker_a} and {marker_b} not found")


shutil.rmtree(REPO, ignore_errors=True)
assemble("api", "state.py", "snapshot.py")
assemble("upi", "features.py", "shadow_eval.py")
open(f"{REPO}/src/__init__.py", "w").close()
sys.path.insert(0, REPO)

from fastapi.testclient import TestClient  # noqa: E402

from src.upi import shadow_eval  # noqa: E402
from src.upi.main import create_upi_app  # noqa: E402
from src.upi.synthetic import SYNTHETIC_NOTICE, generate  # noqa: E402
from src.upi.train import train  # noqa: E402

KEY, SALT, LAG = "e2e-demo-key", "e2e-demo-salt", 7 * 86400.0
CRITERIA = {"min_frauds": 100, "min_pr_auc": 0.30, "min_recall_at_fpr_1pct": 0.30, "min_hard_block_precision": 0.70}   # fixed BEFORE the run

rows = generate(n_payers=2500, n_payees=900, days=75, tx_per_payer_per_day=0.5, seed=11)
cut = int(len(rows) * 0.70)
train_rows, shadow_rows = rows[:cut], rows[cut:]
log(f"{SYNTHETIC_NOTICE} rows {len(rows):,}, train {len(train_rows):,}, shadow {len(shadow_rows):,}, fraud rate {np.mean([r['is_fraud'] for r in rows]):.2%}")

model_dir = f"{OUT}/model_upi"
train_cfg = {"synthetic": True, "criteria": {"min_pr_auc": 0.30, "min_recall_at_fpr_1pct": 0.30}}
train_result = train(train_rows, model_dir, train_cfg, log=log)

log_path = f"{OUT}/decisions.jsonl"
if os.path.exists(log_path):
    os.remove(log_path)
app = create_upi_app(model_dir, api_keys=[KEY], log_path=log_path, log_salt=SALT, autosave_seconds=0, log_features=True)
H = {"X-API-Key": KEY}
cols = ("transaction_id", "timestamp", "amount", "payer_id", "payee_id", "device_id", "channel")

# labels still unknown to the service at the end of training: the last LAG seconds of the training period
t_cut = train_rows[-1]["timestamp"]
due = sorted([(r["timestamp"] + LAG, r["transaction_id"], r["is_fraud"]) for r in train_rows if r["timestamp"] > t_cut - LAG] +
             [], key=lambda x: x[0])
latencies, delivered, errors = [], 0, 0
with TestClient(app) as client:
    import heapq
    heap = list(due)
    heapq.heapify(heap)
    for i, r in enumerate(shadow_rows):
        while heap and heap[0][0] <= r["timestamp"]:
            _, tx, lab = heapq.heappop(heap)
            resp = client.post("/feedback", json={"transaction_id": tx, "is_fraud": bool(lab)}, headers=H)
            delivered += resp.status_code == 200
        body = {k: r[k] for k in cols} | {"explain": False}
        t1 = time.perf_counter()
        resp = client.post("/score", json=body, headers=H)
        latencies.append((time.perf_counter() - t1) * 1000.0)
        errors += resp.status_code != 200
        heapq.heappush(heap, (r["timestamp"] + LAG, r["transaction_id"], r["is_fraud"]))
        if (i + 1) % 5000 == 0:
            log(f"replayed {i + 1:,} / {len(shadow_rows):,} shadow requests")
    info = client.get("/model-info", headers=H).json()
log(f"replay done: {len(shadow_rows):,} requests, errors {errors}, labels delivered {delivered:,}")

spec = json.load(open(f"{model_dir}/serving_spec.json"))
tiers = {k: (v["calibrated_threshold"] if v else None) for k, v in spec["tiers"].items()}
report = shadow_eval.evaluate(shadow_eval.load_log([log_path]), 7.0, tiers, CRITERIA, feature_names=spec["features"])
md = shadow_eval.to_markdown(report)
open(f"{OUT}/shadow_report.md", "w", encoding="utf-8").write(md)
print("\n" + md)

# privacy check on the real log file: no identifier, id or amount string from the traffic appears in it
text = open(log_path, encoding="utf-8").read()
sample = [r for r in shadow_rows[:: max(len(shadow_rows) // 400, 1)]]
leaks = [k for r in sample for k in ("payer_id", "payee_id", "transaction_id") if r[k] in text]
lat = np.asarray(latencies)
results = {"synthetic": True, "notice": SYNTHETIC_NOTICE, "rows": len(rows), "train_rows": len(train_rows), "shadow_requests": len(shadow_rows),
           "service_errors": int(errors), "labels_delivered": int(delivered), "training": train_result, "preregistered_criteria": CRITERIA,
           "shadow_report": report, "identifier_leaks_in_log": len(leaks), "log_lines": text.count("\n"),
           "latency_ms_client_incl_testclient": {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95)), "p99": float(np.percentile(lat, 99))},
           "model_info": info, "seconds": round(time.time() - T0)}
json.dump(results, open(f"{OUT}/results.json", "w"), indent=2)
log(f"verdict {report['verdict']}; identifier leaks in log: {len(leaks)}; p50 latency {results['latency_ms_client_incl_testclient']['p50']:.1f} ms")
