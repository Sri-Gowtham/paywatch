"""PayWatch: build the production LightGBM model and its serving assets (Kaggle, CPU).

Same recipe as the shipped XGBoost model (src/models/train_pipeline.py section D) so the numbers are comparable:
  * 3 seeds, refit on train+validation, recency weights (tau 120 d), FIXED number of rounds,
  * isotonic calibration and precision-target tiers (0.5 / 0.7 / 0.9) fitted on the FIRST half of the test period,
    reported on the SECOND half (the shipped model's protocol).
Rounds: 529 = mean(early-stopping rounds [376, 337, 668]) x 1.15 from the pre-registered study (results/improve.json, lgbm refit),
so the already-validated recipe is reproduced rather than re-tuned.
Outputs in /kaggle/working/prod_lgbm: lgbm_seed{42,1,2}.txt, isotonic.json, serving_spec.json, parity_fixture.json (re-scored, still
synthetic), results.json.
"""

import glob
import json
import os
import pickle
import sys
import time

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

T0 = time.time()
OUT = "/kaggle/working/prod_lgbm"
os.makedirs(OUT, exist_ok=True)
find = lambda p: sorted(glob.glob(p, recursive=True))  # noqa: E731
SEEDS = [42, 1, 2]
N_TREES = 529
TAU_DAYS, SPW = 120, 39.3
LGB_PARAMS = dict(learning_rate=0.03, num_leaves=127, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
                  reg_lambda=1.0, scale_pos_weight=SPW, n_jobs=-1, verbose=-1, metric="None")


def log(msg):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {msg}", flush=True)


sys.path.insert(0, os.path.dirname(find("/kaggle/input/**/upi_fingerprint.py")[0]))
import engineer  # noqa: E402
from entity_history import BASE_KEYS  # noqa: E402

RAW = os.path.dirname(find("/kaggle/input/**/train_transaction.csv")[0])
spec = json.load(open(find("/kaggle/input/**/prod_compact/serving_spec.json")[0]))
cols = spec["features"]
base = engineer.build_base(engineer.load_raw(RAW), lags=(7,), decay_lags=(), velocity=False, key_names=BASE_KEYS)
(tr, va, te), _ = engineer.finalize(base, d_mode="anchor", return_artifacts=True)
del base
get = lambda part: (part[0][cols].to_numpy(dtype=np.float32), part[1], part[2])  # noqa: E731
(Xtr, ytr, ttr), (Xva, yva, tva), (Xte, yte, tte) = get(tr), get(va), get(te)
Xfit, yfit, tfit = np.vstack([Xtr, Xva]), np.r_[ytr, yva], np.r_[ttr, tva]
wfit = np.exp(-(tfit.max() - tfit) / (TAU_DAYS * 86400.0))
log(f"fit {Xfit.shape} test {Xte.shape}")

boosters, test_scores = [], []
for seed in SEEDS:
    m = lgb.LGBMClassifier(n_estimators=N_TREES, random_state=seed, **LGB_PARAMS)
    m.fit(Xfit, yfit, sample_weight=wfit)
    path = f"{OUT}/lgbm_seed{seed}.txt"
    m.booster_.save_model(path)
    boosters.append(lgb.Booster(model_file=path))   # reload: scores below come from the SAVED files the API will load
    test_scores.append(boosters[-1].predict(Xte))
    log(f"seed {seed}: {os.path.getsize(path) / 1e6:.1f} MB, test PR-AUC {average_precision_score(yte, test_scores[-1]):.4f}")
score = np.mean(test_scores, axis=0)


def metrics(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return {"pr_auc": float(average_precision_score(y, p)), "roc_auc": float(roc_auc_score(y, p)),
            "recall_at_fpr_1pct": float(np.interp(0.01, fpr, tpr))}


# shipped XGBoost model on the same rows, for the second-half comparison
ship = []
for p in find("/kaggle/input/**/prod_compact/xgb_seed*.json"):
    b = xgb.Booster()
    b.load_model(p)
    ship.append(b)
import pandas as pd  # noqa: E402
ship_score = np.mean([b.predict(xgb.DMatrix(pd.DataFrame(Xte, columns=cols))) for b in ship], axis=0)

half = len(yte) // 2
a, b2 = slice(0, half), slice(half, None)
results = {"n_trees": N_TREES, "seeds": SEEDS, "test_full": metrics(yte, score), "test_second_half_lgbm": metrics(yte[b2], score[b2]),
           "test_second_half_shipped_xgb": metrics(yte[b2], ship_score[b2]), "test_full_shipped_xgb": metrics(yte, ship_score)}

iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(score[a], yte[a])
cal_a, cal_b = iso.predict(score[a]), iso.predict(score[b2])
tiers = {}
for target in (0.5, 0.7, 0.9):
    ok = [t for t in np.unique(cal_a) if (cal_a >= t).any() and yte[a][cal_a >= t].mean() >= target]
    if not ok:
        tiers[str(target)] = None
        continue
    thr = float(min(ok))
    pred = cal_b >= thr
    tp_ = int((pred & (yte[b2] == 1)).sum())
    tiers[str(target)] = {"calibrated_threshold": thr, "second_half_precision": float(tp_ / max(pred.sum(), 1)),
                          "second_half_recall": float(tp_ / (yte[b2] == 1).sum()), "second_half_alert_rate": float(pred.mean())}
results["precision_target_tiers"] = tiers
results["shipped_xgb_tiers"] = spec["tiers"]

# isotonic as an (x, y) table, as the API applies it with np.interp
xs, ys = np.asarray(iso.X_thresholds_, dtype=float), np.asarray(iso.y_thresholds_, dtype=float)
probe = np.concatenate([np.random.default_rng(0).random(5000), xs[:: max(len(xs) // 500, 1)]])
diff = np.abs(np.interp(probe, xs, ys) - iso.predict(probe))
if diff.max() > 1e-3:
    grid = np.unique(np.concatenate([xs, np.linspace(0.0, 1.0, 100_001)]))
    xs, ys = grid, np.asarray(iso.predict(grid), dtype=float)
    diff = np.abs(np.interp(probe, xs, ys) - iso.predict(probe))
assert diff.max() < 1e-3, "isotonic table cannot reproduce sklearn predict"
json.dump({"x": xs.tolist(), "y": ys.tolist(), "max_abs_diff_vs_sklearn": float(diff.max())}, open(f"{OUT}/isotonic.json", "w"))
pickle.dump(iso, open(f"{OUT}/isotonic.pkl", "wb"))

new_spec = {**spec, "model_family": "lightgbm", "n_trees": N_TREES, "prod_seeds": SEEDS, "lightgbm_params": {**LGB_PARAMS, "n_estimators": N_TREES},
            "tiers": tiers, "replaces": "xgboost 3-seed (n_trees 224)", "selection": "results/improve.json (pre-registered study)"}
json.dump(new_spec, open(f"{OUT}/serving_spec.json", "w"), indent=2)

# re-score the (synthetic) parity fixture: expected features come from the batch pipeline and do not depend on the model
fx_path = [p for p in find("/kaggle/input/**/parity_fixture.json") if json.load(open(p)).get("synthetic")][0]
fx = json.load(open(fx_path))
F = np.asarray([r["expected_features"] for r in fx["rows"]], dtype=np.float32)
fs = np.mean([bst.predict(F) for bst in boosters], axis=0)
fc = np.interp(fs, xs, ys)
for r, s_, c_ in zip(fx["rows"], fs, fc):
    r["expected_score"], r["expected_calibrated"] = float(s_), float(c_)
fx["model"] = "lightgbm 3-seed"
json.dump(fx, open(f"{OUT}/parity_fixture.json", "w"))
log(f"fixture re-scored: {len(fx['rows'])} rows, score range {fs.min():.4f}..{fs.max():.4f}")

json.dump(results, open(f"{OUT}/results.json", "w"), indent=1)
print("\n=== PRODUCTION LIGHTGBM (test period) ===")
print("full test:", results["test_full"], "| shipped XGB full test:", results["test_full_shipped_xgb"])
print("second half: lgbm", results["test_second_half_lgbm"], "| shipped", results["test_second_half_shipped_xgb"])
for k, v in tiers.items():
    print("tier", k, v, "| shipped", spec["tiers"].get(k))
log("done")
