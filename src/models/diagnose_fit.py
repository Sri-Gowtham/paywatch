"""PayWatch: Phase 0 fit diagnostics on IEEE-CIS (Kaggle, CPU). Nothing here selects a model or reads test for a decision.

Question: is the production configuration (XGBoost depth 10, lr 0.0558, recency weights, 120 compact features) under- or overfitted?
  1. Learning curve: train on the most recent 10/25/50/75/100% of the train split; train vs validation PR-AUC. A large and growing
     gap with a still-rising validation curve means the model is variance/data limited; flat curves close together mean underfit.
  2. Depth sweep (4, 6, 8, 10, 12) with all else fixed: train vs validation only.
  3. Gap report for the production configuration: train, validation and test PR-AUC/ROC (test is reported, not used to choose anything).
  4. Rolling-origin baseline: four expanding-window folds over train+validation (validation windows of 10% at 60/70/80/90% of the
     combined span). These per-fold numbers are the baseline every Phase 1 candidate must be compared against.
Thresholds for the verdict are stated here before running: gap = train PR-AUC minus validation PR-AUC; "overfit" if gap > 0.25,
"underfit" if train PR-AUC < 0.80; "data limited" if validation PR-AUC still rises by >= 0.01 between the 75% and 100% points.
Output: /kaggle/working/results.json
"""

import glob
import json
import os
import sys
import time

import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

T0 = time.time()
OUT = "/kaggle/working"
find = lambda p: sorted(glob.glob(p, recursive=True))  # noqa: E731
TAU_DAYS = 120
BASE_PARAMS = dict(max_depth=10, learning_rate=0.0558, subsample=0.935, colsample_bytree=0.678, min_child_weight=1,
                   reg_lambda=1.518, gamma=0.0336, scale_pos_weight=39.3)


def log(msg):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {msg}", flush=True)


sys.path.insert(0, os.path.dirname(find("/kaggle/input/**/upi_fingerprint.py")[0]))
import engineer  # noqa: E402
from entity_history import BASE_KEYS  # noqa: E402

RAW = os.path.dirname(find("/kaggle/input/**/train_transaction.csv")[0])
COLS = json.load(open(find("/kaggle/input/**/prod_compact/serving_spec.json")[0]))["features"]
base = engineer.build_base(engineer.load_raw(RAW), lags=(7,), decay_lags=(), velocity=False, key_names=BASE_KEYS)
(X_tr, y_tr, t_tr), (X_va, y_va, t_va), (X_te, y_te, t_te) = engineer.finalize(base, d_mode="anchor")
del base
Xtr, Xva, Xte = (d[COLS].to_numpy(dtype=np.float32) for d in (X_tr, X_va, X_te))
log(f"train {Xtr.shape} val {Xva.shape} test {Xte.shape}")


def fit(X, y, t, Xv, yv, params, seed=42):
    w = np.exp(-(t.max() - t) / (TAU_DAYS * 86400.0))
    m = xgb.XGBClassifier(n_estimators=6000, tree_method="hist", device="cpu", eval_metric="aucpr", early_stopping_rounds=60,
                          random_state=seed, n_jobs=-1, **params)
    m.fit(X, y, sample_weight=w, eval_set=[(Xv, yv)], verbose=False)
    return m


def pr(y, p):
    return float(average_precision_score(y, p))


rng = np.random.default_rng(0)
results = {"verdict_thresholds": {"overfit_gap": 0.25, "underfit_train_pr_auc": 0.80, "data_limited_gain": 0.01}}

# ------------------------------------------------------------------ 1. learning curve (most recent fraction of train)
lc = []
for frac in (0.10, 0.25, 0.50, 0.75, 1.00):
    k = int(len(Xtr) * frac)
    Xs, ys, ts = Xtr[-k:], y_tr[-k:], t_tr[-k:]
    t1 = time.time()
    m = fit(Xs, ys, ts, Xva, y_va, BASE_PARAMS)
    sub = rng.choice(k, min(100_000, k), replace=False)
    row = {"train_fraction": frac, "train_rows": k, "train_fraud": int(ys.sum()), "best_iter": int(m.best_iteration),
           "train_pr_auc": pr(ys[sub], m.predict_proba(Xs[sub])[:, 1]), "val_pr_auc": pr(y_va, m.predict_proba(Xva)[:, 1]),
           "seconds": round(time.time() - t1)}
    row["gap"] = row["train_pr_auc"] - row["val_pr_auc"]
    lc.append(row)
    log(f"learning curve {frac:.0%}: train {row['train_pr_auc']:.3f} val {row['val_pr_auc']:.3f} gap {row['gap']:.3f} iters {row['best_iter']}")
results["learning_curve"] = lc

# ------------------------------------------------------------------ 2. depth sweep
ds = []
for depth in (4, 6, 8, 10, 12):
    t1 = time.time()
    m = fit(Xtr, y_tr, t_tr, Xva, y_va, {**BASE_PARAMS, "max_depth": depth})
    sub = rng.choice(len(Xtr), 100_000, replace=False)
    row = {"max_depth": depth, "best_iter": int(m.best_iteration), "train_pr_auc": pr(y_tr[sub], m.predict_proba(Xtr[sub])[:, 1]),
           "val_pr_auc": pr(y_va, m.predict_proba(Xva)[:, 1]), "seconds": round(time.time() - t1)}
    row["gap"] = row["train_pr_auc"] - row["val_pr_auc"]
    ds.append(row)
    log(f"depth {depth}: train {row['train_pr_auc']:.3f} val {row['val_pr_auc']:.3f} gap {row['gap']:.3f} iters {row['best_iter']}")
results["depth_sweep"] = ds

# ------------------------------------------------------------------ 3. gap report for the production configuration (3 seeds)
models = [fit(Xtr, y_tr, t_tr, Xva, y_va, BASE_PARAMS, seed=s) for s in (42, 1, 2)]
sub = rng.choice(len(Xtr), 100_000, replace=False)
ens = lambda X: np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)  # noqa: E731
results["production_config_gap"] = {
    "train_pr_auc": pr(y_tr[sub], ens(Xtr[sub])), "val_pr_auc": pr(y_va, ens(Xva)), "test_pr_auc": pr(y_te, ens(Xte)),
    "train_roc_auc": float(roc_auc_score(y_tr[sub], ens(Xtr[sub]))), "val_roc_auc": float(roc_auc_score(y_va, ens(Xva))),
    "test_roc_auc": float(roc_auc_score(y_te, ens(Xte))), "best_iters": [int(m.best_iteration) for m in models],
    "note": "3 seeds trained on the train split only; the shipped model is a refit on train+validation (test PR-AUC 0.706)"}
log(f"production gap: {results['production_config_gap']}")

# ------------------------------------------------------------------ 4. rolling-origin baseline over train+validation
Xall, yall, tall = np.vstack([Xtr, Xva]), np.r_[y_tr, y_va], np.r_[t_tr, t_va]
N = len(Xall)
folds = []
for cut in (0.60, 0.70, 0.80, 0.90):
    a, b = int(N * cut), int(N * (cut + 0.10))
    per_seed = []
    for seed in (42, 1):
        m = fit(Xall[:a], yall[:a], tall[:a], Xall[a:b], yall[a:b], BASE_PARAMS, seed=seed)
        per_seed.append({"seed": seed, "best_iter": int(m.best_iteration), "val_pr_auc": pr(yall[a:b], m.predict_proba(Xall[a:b])[:, 1])})
    folds.append({"train_rows": a, "val_rows": b - a, "val_fraud_rate": float(yall[a:b].mean()), "per_seed": per_seed,
                  "mean_val_pr_auc": float(np.mean([p["val_pr_auc"] for p in per_seed]))})
    log(f"fold cut {cut:.0%}: val PR-AUC {folds[-1]['mean_val_pr_auc']:.4f} ({[round(p['val_pr_auc'], 4) for p in per_seed]})")
fm = [f["mean_val_pr_auc"] for f in folds]
results["rolling_origin_baseline"] = {"cuts": [0.60, 0.70, 0.80, 0.90], "window": 0.10, "folds": folds,
                                      "mean_pr_auc": float(np.mean(fm)), "std_pr_auc": float(np.std(fm))}

# ------------------------------------------------------------------ verdict (thresholds fixed in the docstring)
full = lc[-1]
rise = lc[-1]["val_pr_auc"] - lc[-2]["val_pr_auc"]
results["verdict"] = {
    "train_val_gap_at_full_data": full["gap"],
    "overfit": bool(full["gap"] > 0.25), "underfit": bool(full["train_pr_auc"] < 0.80), "data_limited": bool(rise >= 0.01),
    "val_gain_75_to_100pct": rise,
    "best_depth_by_validation": max(ds, key=lambda r: r["val_pr_auc"])["max_depth"]}
json.dump(results, open(f"{OUT}/results.json", "w"), indent=1)
print("\n=== FIT DIAGNOSTICS ===")
print("learning curve (train frac: train / val PR-AUC):", {r["train_fraction"]: (round(r["train_pr_auc"], 3), round(r["val_pr_auc"], 3)) for r in lc})
print("depth sweep (depth: train / val):", {r["max_depth"]: (round(r["train_pr_auc"], 3), round(r["val_pr_auc"], 3)) for r in ds})
print("verdict:", results["verdict"])
log("done")
