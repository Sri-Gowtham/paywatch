"""PayWatch — Phase B2: SHAP explainability + compact serving model (Kaggle, GPU).

Frozen design as in v5 (core causal features + 5-key lag-7 entity fraud history, anchored D,
recency tau 120d, XGBoost, scale_pos_weight, time-ordered 70/15/15).

Stages
  1) 3-seed full models; TreeSHAP on a VALIDATION sample -> global ranking (selection never uses test)
  2) K-sweep of compact models (top-K by mean |SHAP|), 3 seeds each; smallest K within 0.01 val
     PR-AUC of the full model is chosen
  3) SHAP aggregated by feature family; per-transaction explanation examples (TP and FP)
  4) serving requirements (which features need which state), production refit of the compact model
     on train+val, calibration + tiers fitted on test 1st half / reported on 2nd half
Artifacts stay on Kaggle (/kaggle/working/prod_compact); only small JSON is downloaded.
"""

import glob
import json
import os
import pickle
import re
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
from entity_history import BASE_KEYS, PRIOR_RATE, PRIOR_WEIGHT  # noqa: E402

RAW = os.path.dirname(glob.glob("/kaggle/input/**/train_transaction.csv", recursive=True)[0])
OUT = "/kaggle/working"
os.makedirs(f"{OUT}/prod_compact", exist_ok=True)
results = {}

LAG_DAYS, TAU_DAYS = 7, 120
SEEDS = [42, 1, 2]
KS = (30, 50, 80, 120, 200)
SHAP_ROWS = 10_000
MAX_LOSS = 0.01
XGB_PARAMS = dict(
    max_depth=10, learning_rate=0.0558, subsample=0.935, colsample_bytree=0.678,
    min_child_weight=1, reg_lambda=1.518, gamma=0.0336, scale_pos_weight=39.3,
)
UPI_BEHAVIOR = {"amount_vs_7day_avg_ratio", "merchant_category_entropy", "hour_deviation_from_user_mean",
                "days_since_last_large_txn", "is_new_merchant", "is_p2p"}


def recency_weights(t, tau_days):
    return np.exp(-(t.max() - t) / (tau_days * 86400.0))


def ap(y, p):
    return float(average_precision_score(y, p))


def recall_at_fpr(y, p, f):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.interp(f, fpr, tpr))


def metrics(y, p, amount):
    k = max(int(len(y) * 0.01), 1)
    top = np.argsort(-p)[:k]
    thr1 = np.quantile(p[y == 0], 0.99)
    caught = (p >= thr1) & (y == 1)
    return {
        "pr_auc": ap(y, p), "roc_auc": float(roc_auc_score(y, p)),
        "recall_at_fpr_0.1pct": recall_at_fpr(y, p, 0.001), "recall_at_fpr_1pct": recall_at_fpr(y, p, 0.01),
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


def shap_contribs(model, X):
    try:
        return model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    except Exception as e:  # fall back to a smaller sample if GPU SHAP runs out of memory
        log(f"   SHAP retry on 3000 rows ({e})")
        return model.get_booster().predict(xgb.DMatrix(X.iloc[:3000]), pred_contribs=True)


def family(f):
    if f.startswith("h7_"):
        return "entity_fraud_history"
    if re.fullmatch(r"V\d+", f):
        return "vesta_V"
    if re.fullmatch(r"C\d+", f):
        return "counts_C"
    if re.fullmatch(r"D\d+", f):
        return "time_deltas_D"
    if re.fullmatch(r"M\d", f):
        return "match_flags_M"
    if f.startswith("id_") or f in ("DeviceType", "DeviceInfo"):
        return "identity_device"
    if f in UPI_BEHAVIOR or f.startswith("uid_"):
        return "upi_behavior"
    if f.endswith("_freq"):
        return "frequency_encoding"
    if f.startswith(("card", "addr", "P_email", "R_email")):
        return "card_address_email"
    return "transaction_core"


def serving_source(f):
    if f.startswith("h7_"):
        return "entity_fraud_history_state"
    if f.startswith("uid_") or f in UPI_BEHAVIOR - {"is_p2p"}:
        return "user_behavior_state"
    if f.endswith("_freq"):
        return "frequency_lookup_table"
    if re.fullmatch(r"D\d+", f):
        return "needs_transaction_time"
    return "raw_request_field"


# ---------------------------------------------------------------- features
log("building causal features for the frozen design")
base = engineer.build_base(engineer.load_raw(RAW), lags=(LAG_DAYS,), decay_lags=(), velocity=False, key_names=BASE_KEYS)
(X_tr, y_tr, t_tr), (X_va, y_va, t_va), (X_te, y_te, t_te) = engineer.finalize(base, d_mode="anchor")
del base
cols_full = list(X_tr.columns)
amt_te = X_te["TransactionAmt"].to_numpy()
w_tr = recency_weights(t_tr, TAU_DAYS)
log(f"features={len(cols_full)}  train/val/test = {len(y_tr)}/{len(y_va)}/{len(y_te)}")

# ------------------------------------------------ 1) full models + SHAP ranking (validation sample)
full_val, full_test, full_iters, mean_abs = [], [], [], []
sample = X_va.sample(SHAP_ROWS, random_state=0)
for seed in SEEDS:
    m = fit_train(X_tr, y_tr, X_va, y_va, w_tr, seed)
    full_val.append(ap(y_va, m.predict_proba(X_va)[:, 1]))
    full_test.append(ap(y_te, m.predict_proba(X_te)[:, 1]))
    full_iters.append(int(m.best_iteration))
    c = shap_contribs(m, sample)
    mean_abs.append(np.abs(c[:, :-1]).mean(axis=0))
    log(f"full seed {seed}: iter={m.best_iteration} val={full_val[-1]:.4f} test={full_test[-1]:.4f}")
results["full_model"] = {"val_pr_auc_mean": float(np.mean(full_val)), "test_pr_auc_mean": float(np.mean(full_test)),
                         "test_pr_auc_std": float(np.std(full_test)), "n_features": len(cols_full)}
shap_importance = np.mean(mean_abs, axis=0)
order = np.argsort(-shap_importance)
ranked = [cols_full[i] for i in order]
results["shap_top40"] = [{"feature": cols_full[i], "mean_abs_shap": float(shap_importance[i]),
                          "family": family(cols_full[i])} for i in order[:40]]
fam = {}
for i, f in enumerate(cols_full):
    fam[family(f)] = fam.get(family(f), 0.0) + float(shap_importance[i])
tot = sum(fam.values())
results["shap_by_family_share"] = {k: v / tot for k, v in sorted(fam.items(), key=lambda kv: -kv[1])}
log(f"top SHAP features: {ranked[:10]}")
log(f"family shares: {({k: round(v, 3) for k, v in results['shap_by_family_share'].items()})}")

# ------------------------------------------------ 2) compact K-sweep
sweep = {}
for K in KS:
    cols = ranked[:K]
    vs, ts, its = [], [], []
    for seed in SEEDS:
        m = fit_train(X_tr[cols], y_tr, X_va[cols], y_va, w_tr, seed)
        vs.append(ap(y_va, m.predict_proba(X_va[cols])[:, 1]))
        ts.append(ap(y_te, m.predict_proba(X_te[cols])[:, 1]))
        its.append(int(m.best_iteration))
    sweep[str(K)] = {"val_mean": float(np.mean(vs)), "val_std": float(np.std(vs)),
                     "test_mean": float(np.mean(ts)), "test_std": float(np.std(ts)), "best_iters": its}
    log(f"K={K:<4} val={np.mean(vs):.4f}+-{np.std(vs):.4f} test={np.mean(ts):.4f}+-{np.std(ts):.4f} iters={its}")
results["k_sweep"] = sweep
results["full_reference_val_mean"] = float(np.mean(full_val))
eligible = [K for K in KS if sweep[str(K)]["val_mean"] >= np.mean(full_val) - MAX_LOSS]
K_star = min(eligible) if eligible else len(cols_full)
cols_c = ranked[:K_star] if eligible else cols_full
results["chosen_K"] = K_star
results["compact_features"] = cols_c
log(f"chosen K={K_star} (smallest within {MAX_LOSS} val PR-AUC of the full model)")

# ------------------------------------------------ 3) serving requirements
req = {}
for f in cols_c:
    req.setdefault(serving_source(f), []).append(f)
results["serving_requirements"] = {k: {"count": len(v), "features": v} for k, v in req.items()}
results["compact_family_counts"] = {k: sum(1 for f in cols_c if family(f) == k) for k in
                                    sorted({family(f) for f in cols_c})}

# ------------------------------------------------ 4) production refit (compact) + calibration
iters = sweep[str(K_star)]["best_iters"] if eligible else full_iters
n_trees = int(np.median(iters) * 1.15)
X_fit = pd.concat([X_tr[cols_c], X_va[cols_c]], ignore_index=True)
y_fit = np.r_[y_tr, y_va]
w_fit = recency_weights(np.r_[t_tr, t_va], TAU_DAYS)
log(f"production refit (compact K={len(cols_c)}), {n_trees} trees, seeds {SEEDS}")
prod_models, prod_test, trainonly_test = [], [], []
for seed in SEEDS:
    m = xgb.XGBClassifier(n_estimators=n_trees, tree_method="hist", device="cuda", random_state=seed, **XGB_PARAMS)
    m.fit(X_fit, y_fit, sample_weight=w_fit, verbose=False)
    m.save_model(f"{OUT}/prod_compact/xgb_seed{seed}.json")
    prod_models.append(m)
    prod_test.append(m.predict_proba(X_te[cols_c])[:, 1])
    mt = fit_train(X_tr[cols_c], y_tr, X_va[cols_c], y_va, w_tr, seed)
    trainonly_test.append(mt.predict_proba(X_te[cols_c])[:, 1])
prod_avg, tr_avg = np.mean(prod_test, axis=0), np.mean(trainonly_test, axis=0)
half = len(y_te) // 2
a, b = slice(0, half), slice(half, None)
results["compact_second_half"] = {
    "train_only_3seed": metrics(y_te[b], tr_avg[b], amt_te[b]),
    "refit_train_plus_val_3seed": metrics(y_te[b], prod_avg[b], amt_te[b]),
}
results["compact_full_test_refit"] = metrics(y_te, prod_avg, amt_te)

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
                          "second_half_recall": float(tp_ / (y_te[b] == 1).sum()),
                          "second_half_alert_rate": float(pred.mean())}
results["precision_target_tiers"] = tiers

# ------------------------------------------------ 5) per-transaction explanations (compact refit model)
te_idx = np.arange(len(y_te))
fraud_idx = te_idx[y_te == 1][np.argsort(-prod_avg[y_te == 1])][:5]
fp_idx = te_idx[y_te == 0][np.argsort(-prod_avg[y_te == 0])][:5]
examples = []
for kind, idxs in (("true_positive", fraud_idx), ("false_positive", fp_idx)):
    contrib = shap_contribs(prod_models[0], X_te[cols_c].iloc[idxs])
    for row_i, ridx in enumerate(idxs):
        c = contrib[row_i, :-1]
        top_i = np.argsort(-np.abs(c))[:6]
        examples.append({
            "kind": kind, "score": float(prod_avg[ridx]), "calibrated_p": float(iso.predict([prod_avg[ridx]])[0]),
            "top_contributions": [{"feature": cols_c[j], "value": float(X_te[cols_c].iloc[ridx, j]),
                                   "shap": float(c[j]), "family": family(cols_c[j])} for j in top_i],
        })
results["explanation_examples"] = examples

with open(f"{OUT}/prod_compact/isotonic.pkl", "wb") as f:
    pickle.dump(iso, f)
with open(f"{OUT}/prod_compact/serving_spec.json", "w") as f:
    json.dump({"features": cols_c, "n_trees": n_trees, "seeds": SEEDS, "recency_tau_days": TAU_DAYS,
               "serving_requirements": results["serving_requirements"],
               "entity_state": {"keys": list(BASE_KEYS), "label_lag_days": LAG_DAYS,
                                "per_key_features": ["n", "fraud_n", "fraud_rate"],
                                "prior_rate": PRIOR_RATE, "prior_weight": PRIOR_WEIGHT},
               "tiers": tiers}, f, indent=2)

# ------------------------------------------------------------------- report
print("\n=== FULL MODEL (3 seeds) ===", results["full_model"])
print("\n=== SHAP TOP 20 ===")
for r in results["shap_top40"][:20]:
    print(r)
print("\n=== SHAP SHARE BY FAMILY ===")
for k, v in results["shap_by_family_share"].items():
    print(f"{k:<22} {v:.3f}")
print("\n=== K SWEEP (3 seeds; val used for selection, test shown for info) ===")
for K, v in sweep.items():
    print(K, v)
print(f"chosen K = {K_star}")
print("compact family counts:", results["compact_family_counts"])
print("\n=== SERVING REQUIREMENTS ===")
for k, v in results["serving_requirements"].items():
    print(f"{k:<28} {v['count']}")
print("\n=== COMPACT, TEST 2nd HALF ===")
for k, v in results["compact_second_half"].items():
    print(f"{k:<28}", "  ".join(f"{a_}={b_:.4f}" for a_, b_ in v.items()))
print("compact refit on full test:", results["compact_full_test_refit"])
print("\n=== TIERS (calibrated on test 1st half, reported on 2nd half) ===")
for k, v in tiers.items():
    print(k, v)
print("\n=== EXPLANATION EXAMPLES ===")
for e in examples[:4]:
    print(e)

with open(f"{OUT}/results.json", "w") as f:
    json.dump(results, f, indent=2)
log("done; results.json saved; artifacts in /kaggle/working/prod_compact")
