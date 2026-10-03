"""PayWatch — warm-start state snapshot (Kaggle, CPU).

Builds the API state (user behaviour + entity fraud history) from all train+val transactions the way
production would have it at deployment time T0 (labels only after the 7-day delay; the last 7 days are
'pending' and their labels arrive during the replay). Compares cold start vs warm start at several
pruning levels on the 20k-row stream sample and exports the smallest level that keeps accuracy.
"""

import glob
import heapq
import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

T0_WALL = time.time()
LAG_S = 7 * 86400.0
OUT = "/kaggle/working"
REPO, MODELS = "/kaggle/working/repo", "/kaggle/working/models/compact"
find = lambda p: sorted(glob.glob(p, recursive=True))  # noqa: E731


def log(msg):
    print(f"[{(time.time() - T0_WALL) / 60:5.1f} min] {msg}", flush=True)


# ------------------------------------------------------------ assemble repo layout
for d in (f"{REPO}/src/api", MODELS):
    os.makedirs(d, exist_ok=True)
state_py = [p for p in find("/kaggle/input/**/state.py") if os.path.exists(os.path.join(os.path.dirname(p), "snapshot.py"))][0]
for f in os.listdir(os.path.dirname(state_py)):
    if f.endswith(".py"):
        shutil.copy(os.path.join(os.path.dirname(state_py), f), f"{REPO}/src/api/{f}")
open(f"{REPO}/src/__init__.py", "w").close()
models_src = os.path.dirname(find("/kaggle/input/**/xgb_seed42.json")[0])
for f in os.listdir(models_src):
    shutil.copy(os.path.join(models_src, f), MODELS)
sys.path.insert(0, REPO)

from src.api.predictor import Predictor  # noqa: E402
from src.api.snapshot import export_state, import_state, load_state, save_state  # noqa: E402
from src.api.state import EntityHistoryStore, UserBehaviorState, entity_keys  # noqa: E402

# ------------------------------------------------------------ build the state at T0
RAW = os.path.dirname(find("/kaggle/input/**/train_transaction.csv")[0])
use = ["TransactionID", "TransactionDT", "TransactionAmt", "ProductCD", "card1", "card2", "card3", "card5",
       "addr1", "D1", "P_emaildomain", "isFraud"]
df = pd.read_csv(f"{RAW}/train_transaction.csv", usecols=use).merge(
    pd.read_csv(f"{RAW}/train_identity.csv", usecols=["TransactionID", "DeviceInfo"]), on="TransactionID", how="left")
df = df.sort_values("TransactionDT", kind="stable").reset_index(drop=True)
n = len(df)
n_hist = int(n * 0.85)
ts = df["TransactionDT"].to_numpy(dtype=float)
y = df["isFraud"].to_numpy()
ids = df["TransactionID"].astype(str).tolist()
field_cols = ["TransactionDT", "TransactionAmt", "ProductCD", "card1", "card2", "card3", "card5", "addr1", "D1",
              "P_emaildomain", "DeviceInfo"]
cols = {c: df[c].astype(object).where(df[c].notna(), None).tolist() for c in field_cols}
T0 = float(ts[n_hist])
log(f"rows={n} history rows (train+val)={n_hist} deployment time T0 = day {T0 / 86400:.1f}")

behavior, history = UserBehaviorState(), EntityHistoryStore()
all_keys, j = [], 0
for i in range(n_hist):
    f = {c: cols[c][i] for c in field_cols}
    keys = entity_keys(f)
    all_keys.append(keys)
    behavior.observe(f, keys["uid"], commit=True)
    while j < i and ts[j] <= ts[i] - LAG_S:
        history.add_label_for_keys(all_keys[j], bool(y[j]))
        j += 1
while j < n_hist and ts[j] <= T0 - LAG_S:
    history.add_label_for_keys(all_keys[j], bool(y[j]))
    j += 1
pending_labels = []
for k in range(j, n_hist):
    history.register(ids[k], all_keys[k])
    pending_labels.append((float(ts[k] + LAG_S), ids[k], int(y[k])))
log(f"state built: users={len(behavior)} entity keys={ {k: len(v) for k, v in history.tables.items()} } pending={len(pending_labels)}")
del cols, all_keys

# ------------------------------------------------------------ pruning levels
levels = {
    "A_full": dict(window_days=None, min_entity_n=1),
    "B_active30d": dict(window_days=30, min_entity_n=1),
    "C_active30d_min2": dict(window_days=30, min_entity_n=2),
    "D_active14d_min3": dict(window_days=14, min_entity_n=3),
}
paths = {}
for name, cfg in levels.items():
    active = None
    if cfg["window_days"] is not None:
        active = {u for u, s in behavior._s.items() if s["last_ts"] >= T0 - cfg["window_days"] * 86400.0}
    obj = export_state(behavior, history, active_uids=active, min_entity_n=cfg["min_entity_n"])
    paths[name] = f"{OUT}/state_{name}.json.gz"
    size = save_state(obj, paths[name])
    levels[name].update({"gz_bytes": size, "users": len(obj["users"]),
                         "entity_keys": {k: len(v) for k, v in obj["entity_tables"].items()}})
    log(f"level {name}: {size / 1e6:.2f} MB gz, users={len(obj['users'])}")
del behavior, history

# ------------------------------------------------------------ replay evaluation
stream = pd.read_parquet(find("/kaggle/input/**/stream_sample.parquet")[0]).sort_values("TransactionDT").reset_index(drop=True)
field_names = [c for c in stream.columns if c not in ("TransactionID", "isFraud")]
records = stream[field_names].to_dict("records")
stream_ids = stream["TransactionID"].astype(str).tolist()
stream_y = stream["isFraud"].to_numpy()
stream_t = stream["TransactionDT"].to_numpy(dtype=float)
assert stream_t.min() >= T0 - 1, f"stream starts before deployment time ({stream_t.min()} < {T0})"
clean = lambda r: {k: (None if (v is None or (isinstance(v, float) and v != v)) else v) for k, v in r.items()}  # noqa: E731

pred = Predictor(MODELS)


def replay(label, state_path, pending):
    pred.reset_state()
    if state_path:
        import_state(load_state(state_path), pred.behavior, pred.history)
    heap = list(pending)
    heapq.heapify(heap)
    scores, new_user, hist = [], 0, 0
    for i, rec in enumerate(records):
        while heap and heap[0][0] <= stream_t[i]:
            _, tid, lab = heapq.heappop(heap)
            pred.feedback(tid, bool(lab))
        out = pred.predict(stream_ids[i], clean(rec), explain=False)
        scores.append(out["score"])
        new_user += int(out["flags"]["new_user"])
        hist += int(out["flags"]["uid_has_labeled_history"])
        heapq.heappush(heap, (stream_t[i] + LAG_S, stream_ids[i], int(stream_y[i])))
    s = np.asarray(scores)
    fpr, tpr, _ = roc_curve(stream_y, s)
    res = {"pr_auc": float(average_precision_score(stream_y, s)), "roc_auc": float(roc_auc_score(stream_y, s)),
           "recall_at_fpr_1pct": float(np.interp(0.01, fpr, tpr)), "new_user_rate": new_user / len(s),
           "uid_has_labeled_history_rate": hist / len(s)}
    log(f"replay {label:<18} PR-AUC={res['pr_auc']:.4f} ROC={res['roc_auc']:.4f} recall@1%FPR={res['recall_at_fpr_1pct']:.4f} "
        f"new_user={res['new_user_rate']:.3f} has_history={res['uid_has_labeled_history_rate']:.3f}")
    return res


results = {"T0_day": T0 / 86400.0, "stream_rows": len(records), "levels": levels, "replay": {}}
results["replay"]["cold"] = replay("cold (empty)", None, [])
for name in levels:
    results["replay"][name] = replay(name, paths[name], pending_labels)
    results["levels"][name]["replay"] = results["replay"][name]

best = max(r["pr_auc"] for k, r in results["replay"].items() if k != "cold")
eligible = [k for k in levels if results["replay"][k]["pr_auc"] >= best - 0.003]
chosen = min(eligible, key=lambda k: levels[k]["gz_bytes"])
results["chosen_level"] = chosen
shutil.copy(paths[chosen], f"{OUT}/warm_state.json.gz")
with open(f"{OUT}/pending_labels.json", "w") as f:
    json.dump(pending_labels, f, separators=(",", ":"))
for name, p in paths.items():
    os.remove(p)
with open(f"{OUT}/results.json", "w") as f:
    json.dump(results, f, indent=1)

print("\n=== WARM START RESULTS ===")
print(f"{'scenario':<20}{'gz MB':>8}{'PR-AUC':>9}{'ROC':>8}{'rec@1%FPR':>11}{'new_user':>10}{'has_hist':>10}")
c = results["replay"]["cold"]
print(f"{'cold':<20}{0:>8.2f}{c['pr_auc']:>9.4f}{c['roc_auc']:>8.4f}{c['recall_at_fpr_1pct']:>11.4f}{c['new_user_rate']:>10.3f}{c['uid_has_labeled_history_rate']:>10.3f}")
for name in levels:
    r = results["replay"][name]
    print(f"{name:<20}{levels[name]['gz_bytes'] / 1e6:>8.2f}{r['pr_auc']:>9.4f}{r['roc_auc']:>8.4f}{r['recall_at_fpr_1pct']:>11.4f}{r['new_user_rate']:>10.3f}{r['uid_has_labeled_history_rate']:>10.3f}")
print(f"\nchosen level: {chosen} ({levels[chosen]['gz_bytes'] / 1e6:.2f} MB) -> warm_state.json.gz")
print("offline reference (full test, full state): PR-AUC 0.706, ROC 0.948, recall@1%FPR 0.637")
log("done")
