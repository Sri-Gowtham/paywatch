"""PayWatch — careful retrain of the FROZEN design (Kaggle, GPU).

Frozen design: core causal features + 5-key lag-7 entity fraud history, anchored D columns,
recency weights (tau 120d), single XGBoost, scale_pos_weight (no SMOTE), time-ordered 70/15/15.

Stages
  A) integrity + leakage audits (brute-force recomputation of the history features, split order)
  B) 5-seed training on train (early stopping on val) -> mean/std, ensemble, bootstrap CIs
  C) importance sanity check
  D) production refit on train+val (3 seeds); calibration + precision-target tiers fitted on the
     FIRST half of the test period and reported on the SECOND half (time-ordered, honest)
  E) artifacts for serving stay on Kaggle (/kaggle/working/prod)
"""

import glob
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

T0 = time.time()


def log(msg):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {msg}", flush=True)


src_dir = os.path.dirname(glob.glob("/kaggle/input/**/upi_fingerprint.py", recursive=True)[0])
sys.path.insert(0, src_dir)
import engineer  # noqa: E402
from entity_history import BASE_KEYS, PRIOR_RATE, PRIOR_WEIGHT, _entity_keys  # noqa: E402

RAW = os.path.dirname(glob.glob("/kaggle/input/**/train_transaction.csv", recursive=True)[0])
OUT = "/kaggle/working"
os.makedirs(f"{OUT}/prod", exist_ok=True)
results = {}

LAG_DAYS = 7
TAU_DAYS = 120
SEEDS = [42, 1, 2, 3, 4]
PROD_SEEDS = [42, 1, 2]
XGB_PARAMS = dict(
    max_depth=10, learning_rate=0.0558, subsample=0.935, colsample_bytree=0.678,
    min_child_weight=1, reg_lambda=1.518, gamma=0.0336, scale_pos_weight=39.3,
)


def recency_weights(t, tau_days):
    return np.exp(-(t.max() - t) / (tau_days * 86400.0))


def ap(y, p):
    return float(average_precision_score(y, p))


def recall_at_fpr(y, p, fpr_target):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.interp(fpr_target, fpr, tpr))


def metrics(y, p, amount):
    k = max(int(len(y) * 0.01), 1)
    top = np.argsort(-p)[:k]
    thr1 = np.quantile(p[y == 0], 0.99)
    caught = (p >= thr1) & (y == 1)
    return {
        "pr_auc": ap(y, p),
        "roc_auc": float(roc_auc_score(y, p)),
        "recall_at_fpr_0.1pct": recall_at_fpr(y, p, 0.001),
        "recall_at_fpr_0.5pct": recall_at_fpr(y, p, 0.005),
        "recall_at_fpr_1pct": recall_at_fpr(y, p, 0.01),
        "recall_at_fpr_5pct": recall_at_fpr(y, p, 0.05),
        "precision_top_1pct_alerts": float(y[top].mean()),
        "amount_weighted_recall_at_fpr_1pct": float(amount[caught].sum() / amount[y == 1].sum()),
    }


def fit_train(X, y, Xv, yv, w, seed):
    m = xgb.XGBClassifier(
        n_estimators=6000, tree_method="hist", device="cuda", eval_metric="aucpr",
        early_stopping_rounds=60, random_state=seed, **XGB_PARAMS,
    )
    m.fit(X, y, sample_weight=w, eval_set=[(Xv, yv)], verbose=False)
    return m


# ---------------------------------------------------------------- features
log("building causal features for the frozen design")
base = engineer.build_base(engineer.load_raw(RAW), lags=(LAG_DAYS,), decay_lags=(), velocity=False, key_names=BASE_KEYS)

# ---------------------------------------------------------- A) audits
log("audit A1: brute-force recomputation of lagged history features")
ts_all = base["TransactionDT"].to_numpy(dtype=float)
y_all = base["isFraud"].to_numpy()
keys_all = _entity_keys(base)
rng = np.random.default_rng(123)
audit = {}
for name in BASE_KEYS:
    codes = np.asarray(pd.factorize(keys_all[name])[0])
    idx = rng.choice(len(base), 600, replace=False)
    got_n = base[f"h{LAG_DAYS}_{name}_n"].to_numpy()
    got_f = base[f"h{LAG_DAYS}_{name}_fraud_n"].to_numpy()
    got_r = base[f"h{LAG_DAYS}_{name}_fraud_rate"].to_numpy()
    bad = 0
    for i in idx:
        if codes[i] < 0:
            en, ef = 0, 0
        else:
            mask = (codes == codes[i]) & (ts_all <= ts_all[i] - LAG_DAYS * 86400.0)
            en, ef = int(mask.sum()), int(y_all[mask].sum())
        er = (ef + PRIOR_WEIGHT * PRIOR_RATE) / (en + PRIOR_WEIGHT) if en > 0 else PRIOR_RATE
        if int(got_n[i]) != en or int(got_f[i]) != ef or abs(float(got_r[i]) - er) > 1e-4:
            bad += 1
    audit[name] = {"rows_checked": 600, "mismatches": bad}
    log(f"   {name:<7} mismatches={bad}/600")
results["audit_history_bruteforce"] = audit
assert all(v["mismatches"] == 0 for v in audit.values()), f"history feature audit FAILED: {audit}"
del keys_all

(X_tr, y_tr, t_tr), (X_va, y_va, t_va), (X_te, y_te, t_te) = engineer.finalize(base, d_mode="anchor")
del base
cols = list(X_tr.columns)
log("audit A2: split integrity")
assert t_tr.max() <= t_va.min() and t_va.max() <= t_te.min(), "time order violated"
assert list(X_va.columns) == cols and list(X_te.columns) == cols
assert not np.isnan(X_tr.to_numpy()).any() and not np.isnan(X_va.to_numpy()).any()
results["integrity"] = {
    "n_features": len(cols), "train_rows": len(y_tr), "val_rows": len(y_va), "test_rows": len(y_te),
    "fraud_rate": {"train": float(y_tr.mean()), "val": float(y_va.mean()), "test": float(y_te.mean())},
    "time_order_ok": True,
}
log(f"   features={len(cols)} fraud rate train/val/test = {y_tr.mean():.4f}/{y_va.mean():.4f}/{y_te.mean():.4f}")
amt_te = X_te["TransactionAmt"].to_numpy()
w_tr = recency_weights(t_tr, TAU_DAYS)

# ---------------------------------------------------------- B) multi-seed training
per_seed, val_preds, test_preds, best_iters, boosters = {}, [], [], [], {}
for seed in SEEDS:
    m = fit_train(X_tr, y_tr, X_va, y_va, w_tr, seed)
    pv, pt = m.predict_proba(X_va)[:, 1], m.predict_proba(X_te)[:, 1]
    val_preds.append(pv)
    test_preds.append(pt)
    best_iters.append(int(m.best_iteration))
    per_seed[str(seed)] = {"val_pr_auc": ap(y_va, pv), "test_pr_auc": ap(y_te, pt), "best_iter": int(m.best_iteration)}
    boosters[seed] = m.get_booster()
    log(f"seed {seed}: iter={m.best_iteration} val PR-AUC={per_seed[str(seed)]['val_pr_auc']:.4f} "
        f"test PR-AUC={per_seed[str(seed)]['test_pr_auc']:.4f}")
results["per_seed"] = per_seed
tp = [v["test_pr_auc"] for v in per_seed.values()]
vp = [v["val_pr_auc"] for v in per_seed.values()]
results["seed_summary"] = {"val_mean": float(np.mean(vp)), "val_std": float(np.std(vp)),
                           "test_mean": float(np.mean(tp)), "test_std": float(np.std(tp))}
ens_test = np.mean(test_preds, axis=0)
results["test_metrics_seed42"] = metrics(y_te, test_preds[0], amt_te)
results["test_metrics_ensemble5"] = metrics(y_te, ens_test, amt_te)


def bootstrap(y, p, B=200, seed=0):
    r = np.random.default_rng(seed)
    aps, recs = [], []
    for _ in range(B):
        i = r.integers(0, len(y), len(y))
        aps.append(ap(y[i], p[i]))
        recs.append(recall_at_fpr(y[i], p[i], 0.01))
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]  # noqa: E731
    return {"pr_auc_95ci": q(aps), "recall_at_fpr_1pct_95ci": q(recs)}


results["bootstrap_ci_ensemble5"] = bootstrap(y_te, ens_test)
log(f"ensemble5 test PR-AUC={results['test_metrics_ensemble5']['pr_auc']:.4f} "
    f"CI={np.round(results['bootstrap_ci_ensemble5']['pr_auc_95ci'], 4).tolist()}")

# ---------------------------------------------------------- C) importance sanity
gain = boosters[SEEDS[0]].get_score(importance_type="gain")
top = sorted(gain, key=gain.get, reverse=True)[:25]
results["top25_features_by_gain"] = [{"feature": f, "gain": float(gain[f])} for f in top]
log(f"top features: {top[:10]}")

# ---------------------------------------------------------- D) production refit + calibration
n_trees = int(np.median(best_iters) * 1.15)
log(f"production refit on train+val, {n_trees} trees, seeds {PROD_SEEDS}")
X_fit = pd.concat([X_tr, X_va], ignore_index=True)
y_fit = np.r_[y_tr, y_va]
w_fit = recency_weights(np.r_[t_tr, t_va], TAU_DAYS)
prod_test = []
for seed in PROD_SEEDS:
    m = xgb.XGBClassifier(n_estimators=n_trees, tree_method="hist", device="cuda", random_state=seed, **XGB_PARAMS)
    m.fit(X_fit, y_fit, sample_weight=w_fit, verbose=False)
    m.save_model(f"{OUT}/prod/xgb_seed{seed}.json")
    prod_test.append(m.predict_proba(X_te)[:, 1])
prod_avg = np.mean(prod_test, axis=0)
trainonly_avg = np.mean([test_preds[SEEDS.index(s)] for s in PROD_SEEDS], axis=0)

half = len(y_te) // 2
a, b = slice(0, half), slice(half, None)
results["test_second_half_comparison"] = {
    "train_only_3seed": metrics(y_te[b], trainonly_avg[b], amt_te[b]),
    "refit_train_plus_val_3seed": metrics(y_te[b], prod_avg[b], amt_te[b]),
}

iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(prod_avg[a], y_te[a])
cal_a, cal_b = iso.predict(prod_avg[a]), iso.predict(prod_avg[b])
tiers = {}
for target in (0.5, 0.7, 0.9):
    ok = [t for t in np.unique(cal_a) if (cal_a >= t).any() and y_te[a][cal_a >= t].mean() >= target]
    if not ok:
        tiers[str(target)] = None
        continue
    thr = float(min(ok))
    pred = cal_b >= thr
    tp_ = int((pred & (y_te[b] == 1)).sum())
    tiers[str(target)] = {"calibrated_threshold": thr, "second_half_precision": float(tp_ / max(pred.sum(), 1)),
                          "second_half_recall": float(tp_ / (y_te[b] == 1).sum()), "second_half_alert_rate": float(pred.mean())}
results["precision_target_tiers"] = tiers

with open(f"{OUT}/prod/isotonic.pkl", "wb") as f:
    pickle.dump(iso, f)
with open(f"{OUT}/prod/serving_spec.json", "w") as f:
    json.dump({
        "features": cols, "n_trees": n_trees, "prod_seeds": PROD_SEEDS, "recency_tau_days": TAU_DAYS,
        "entity_state": {"keys": list(BASE_KEYS), "label_lag_days": LAG_DAYS,
                         "per_key_features": ["n", "fraud_n", "fraud_rate"],
                         "prior_rate": PRIOR_RATE, "prior_weight": PRIOR_WEIGHT},
        "tiers": tiers,
    }, f, indent=2)

# ------------------------------------------------------------------- report
print("\n=== AUDITS ===")
print("history brute-force:", audit)
print("integrity:", results["integrity"])
print("\n=== SEEDS (train-only, early stop on val) ===")
for s, v in per_seed.items():
    print(s, v)
print("summary:", results["seed_summary"])
print("\n=== TEST (latest 15% of time) ===")
print("v2 baseline (ref)  pr_auc=0.5805 roc=0.8967 recall@1%fpr=0.5193 prec_top1%=0.8915 amt_recall=0.3871")
print("v3 single  (ref)   pr_auc=0.6953 roc=0.9438 recall@1%fpr=0.6348 prec_top1%=0.9605 amt_recall=0.5192")
for k in ("test_metrics_seed42", "test_metrics_ensemble5"):
    print(f"{k:<24}", "  ".join(f"{a_}={b_:.4f}" for a_, b_ in results[k].items()))
print("bootstrap 95% CI (ensemble5):", results["bootstrap_ci_ensemble5"])
print("\n=== TOP FEATURES (gain) ===")
for r in results["top25_features_by_gain"][:15]:
    print(r)
print("\n=== TEST 2nd HALF: train-only vs refit(train+val) ===")
for k, v in results["test_second_half_comparison"].items():
    print(f"{k:<28}", "  ".join(f"{a_}={b_:.4f}" for a_, b_ in v.items()))
print("\n=== PRECISION-TARGET TIERS (calibrated on test 1st half, reported on 2nd half) ===")
for k, v in tiers.items():
    print(k, v)

with open(f"{OUT}/results.json", "w") as f:
    json.dump(results, f, indent=2)
log("done; results.json saved; production artifacts stay on Kaggle in /kaggle/working/prod")
