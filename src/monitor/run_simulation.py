"""PayWatch — 30-day production simulation with injected drift (Kaggle, CPU).

Runs two scenarios on the same warm-started API state and stream:
  control : no drift (measures false alarms of the monitoring rules)
  drift   : covariate drift injected from day 15
Outputs sim_results.json (consumed by the dashboard) and a readable summary.
"""

import glob
import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

T0 = time.time()
find = lambda p: sorted(glob.glob(p, recursive=True))  # noqa: E731
REPO, MODELS, OUT = "/kaggle/working/repo", "/kaggle/working/models/compact", "/kaggle/working"


def log(msg):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {msg}", flush=True)


for d in (f"{REPO}/src/api", f"{REPO}/src/monitor", MODELS):
    os.makedirs(d, exist_ok=True)
for marker, dest in (("snapshot.py", "api"), ("simulate_production.py", "monitor")):
    p = [x for x in find(f"/kaggle/input/**/{marker}")][0]
    for f in os.listdir(os.path.dirname(p)):
        if f.endswith(".py"):
            shutil.copy(os.path.join(os.path.dirname(p), f), f"{REPO}/src/{dest}/{f}")
open(f"{REPO}/src/__init__.py", "w").close()
models_src = os.path.dirname(find("/kaggle/input/**/xgb_seed42.json")[0])
for f in os.listdir(models_src):
    shutil.copy(os.path.join(models_src, f), MODELS)
sys.path.insert(0, REPO)

from src.api.predictor import Predictor  # noqa: E402
from src.monitor.simulate_production import analyse, covariate_drift, replay  # noqa: E402

warm_path = find("/kaggle/input/**/warm_state.json.gz")[0]
pending = json.load(open(find("/kaggle/input/**/pending_labels.json")[0]))
stream = pd.read_parquet(find("/kaggle/input/**/stream_sample.parquet")[0]).sort_values("TransactionDT").reset_index(drop=True)
fields = [c for c in stream.columns if c not in ("TransactionID", "isFraud")]
records = stream[fields].to_dict("records")
ids = stream["TransactionID"].astype(str).tolist()
labels = stream["isFraud"].to_numpy()
times = stream["TransactionDT"].to_numpy(dtype=float)
log(f"stream rows={len(records)} span={(times[-1] - times[0]) / 86400:.1f} days; pending labels={len(pending)}")

pred = Predictor(MODELS)
meta = json.load(open(f"{MODELS}/feature_meta.json"))
# input-drift monitoring covers what the CLIENT sends (+ fixed lookup tables). State that grows by design
# (entity counters, user counters, anchored D = absolute day) is excluded: it drifts without anything being wrong.
monitored = [f for f in pred.features if meta["feature_source"][f] in ("raw_request_field", "frequency_lookup")]
DRIFT_DAY = 15.0
WINDOWS = (1, 2, 3)
results = {"config": {"drift_day": DRIFT_DAY, "reference_days": 10, "windows_tested": list(WINDOWS), "label_lag_days": 7,
                      "drift": "amount x1.8; 35% of rows get unseen e-mail domain; C1/C13/C14 x2",
                      "stream_rows": len(records), "warm_state": os.path.basename(warm_path),
                      "monitored_features": len(monitored), "total_features": len(pred.features)}}
reps = {}
for name, fn in (("control", None), ("drift", covariate_drift)):
    pred.load_state(warm_path)
    reps[name] = replay(pred, records, ids, labels, times, [tuple(p) for p in pending], drift_day=DRIFT_DAY,
                        drift_fn=fn, seed=0)
    log(f"{name}: replay done")
for name, fn in (("control", None), ("drift", covariate_drift)):
    results[name] = {}
    for w in WINDOWS:
        results[name][f"window_{w}"] = analyse(reps[name], pred.features, DRIFT_DAY if fn else None, window_days=w,
                                               monitor_features=monitored)
    log(f"{name}: analysed windows {WINDOWS}")

# window choice uses the CONTROL run only: smallest window with zero alerts
control_alerts = {w: len(results["control"][f"window_{w}"]["alerts"]) for w in WINDOWS}
chosen = next((w for w in WINDOWS if control_alerts[w] == 0), min(WINDOWS, key=lambda w: control_alerts[w]))
results["chosen_window_days"] = chosen
results["control_alert_counts_by_window"] = control_alerts

# ---------------------------------------------------------------- calibration curves for the dashboard (control run, aggregates only)
rep = reps["control"]
amt = stream["TransactionAmt"].to_numpy(dtype=float)
second = rep["day"] >= (rep["day"].max() / 2.0)      # calibrator was fitted on the FIRST half of the test period


def curve(mask):
    yy, cc, aa = rep["y"][mask], rep["cal"][mask], amt[mask]
    pts = []
    for t in np.round(np.arange(0.02, 0.96, 0.02), 2):
        pred = cc >= t
        tp = int((pred & (yy == 1)).sum())
        pts.append({"threshold": float(t), "alert_rate": float(pred.mean()), "precision": float(tp / max(pred.sum(), 1)),
                    "recall": float(tp / yy.sum()), "amount_weighted_recall": float(aa[pred & (yy == 1)].sum() / aa[yy == 1].sum())})
    return {"rows": int(mask.sum()), "fraud_rate": float(yy.mean()), "points": pts}


edges = np.quantile(rep["cal"][second], np.linspace(0, 1, 11))
bin_idx = np.clip(np.searchsorted(edges, rep["cal"][second], side="right") - 1, 0, 9)
reliability = [{"bin": b, "n": int((bin_idx == b).sum()), "mean_predicted": float(rep["cal"][second][bin_idx == b].mean()),
                "observed_fraud_rate": float(rep["y"][second][bin_idx == b].mean())} for b in range(10) if (bin_idx == b).any()]
results["calibration"] = {"curve_second_half": curve(second), "curve_all": curve(np.ones(len(second), bool)),
                          "reliability_second_half": reliability, "note": "control replay (no drift); second half is out-of-sample for the calibrator"}

with open(f"{OUT}/sim_results.json", "w") as f:
    json.dump(results, f, indent=1)

print(f"\nmonitored features: {len(monitored)} of {len(pred.features)}")
print("control alerts by window:", control_alerts, "-> chosen window:", chosen, "day(s) (selected on control only)")
for w in WINDOWS:
    for name in ("control", "drift"):
        r = results[name][f"window_{w}"]
        print(f"window={w} {name:<8} alerts={r['alert_counts']} before/at day15={r['alerts_before_drift_day']} "
              f"first={r['first_alert_day']} lag={r.get('detection_lag_days')}")
for name in ("control", "drift"):
    r = results[name][f"window_{chosen}"]
    print(f"\n=== {name.upper()} (window {chosen}d) baseline: {r['baseline']} ===")
    print(f"{'day':>4}{'rows':>6}{'psi_mean':>9}{'joint':>6}{'severe':>7}{'score_psi':>10}{'flag_rate':>10}{'PR-AUC(mat)':>12}  alerts")
    for d in r["daily"]:
        pr = d["pr_auc_mature_window"]
        print(f"{d['day']:>4}{d['rows']:>6}{d['psi_mean']:>9.3f}{d['n_joint_significant']:>6}{d['n_joint_severe']:>7}"
              f"{d['score_psi']:>10.3f}{d['flag_rate']:>10.3f}{(f'{pr:.3f}' if pr is not None else '   n/a'):>12}  "
              f"{','.join(a['signal'][:5] + ':' + a['severity'][0] for a in d['alerts'])}")
print("\nTop drifted monitored features, final window (drift run):")
for t in results["drift"][f"window_{chosen}"]["daily"][-1]["top_drifted"]:
    print(t)
log("done")
