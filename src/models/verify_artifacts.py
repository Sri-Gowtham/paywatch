"""PayWatch — independent verification of the SAVED production artifacts (Kaggle, CPU).

Loads the v5 full model (prod/) and the B2 compact model (prod_compact/) from kernel outputs,
rebuilds the frozen-design features from raw data, re-scores the held-out test period and checks:
  1) round trip: reproduced PR-AUC / tiers match what the training kernels reported
  2) full score table (ranking, alert-budget, calibration)
  3) slice analysis: entity history known vs new, amount bands, weekly stability
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
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score, roc_curve

T0 = time.time()


def log(msg):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {msg}", flush=True)


src_dir = os.path.dirname(glob.glob("/kaggle/input/**/upi_fingerprint.py", recursive=True)[0])
sys.path.insert(0, src_dir)
import engineer  # noqa: E402
from entity_history import BASE_KEYS  # noqa: E402

RAW = os.path.dirname(glob.glob("/kaggle/input/**/train_transaction.csv", recursive=True)[0])
OUT = "/kaggle/working"
results = {}
checks = {}


def find(pattern):
    return sorted(glob.glob(pattern, recursive=True))


def ap(y, p):
    return float(average_precision_score(y, p))


def recall_at_fpr(y, p, f):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.interp(f, fpr, tpr))


def score_table(y, p, amount):
    out = {"pr_auc": ap(y, p), "roc_auc": float(roc_auc_score(y, p))}
    for f in (0.001, 0.005, 0.01, 0.05):
        out[f"recall_at_fpr_{f * 100:g}pct"] = recall_at_fpr(y, p, f)
    order = np.argsort(-p)
    for rate in (0.005, 0.01, 0.02, 0.05):
        k = int(len(y) * rate)
        top = order[:k]
        out[f"precision_top_{rate * 100:g}pct_alerts"] = float(y[top].mean())
        out[f"recall_top_{rate * 100:g}pct_alerts"] = float(y[top].sum() / y.sum())
    thr1 = np.quantile(p[y == 0], 0.99)
    caught = (p >= thr1) & (y == 1)
    out["amount_weighted_recall_at_fpr_1pct"] = float(amount[caught].sum() / amount[y == 1].sum())
    best_f1 = 0.0
    for t in np.unique(np.quantile(p, np.linspace(0.8, 0.999, 200))):
        pred = p >= t
        tp = (pred & (y == 1)).sum()
        pr, rc = tp / max(pred.sum(), 1), tp / y.sum()
        best_f1 = max(best_f1, 2 * pr * rc / (pr + rc + 1e-12))
    out["best_f1_over_thresholds"] = float(best_f1)
    return out


def calibration_table(y, p_cal, bins=10):
    edges = np.quantile(p_cal, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p_cal, side="right") - 1, 0, bins - 1)
    rows, ece = [], 0.0
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append({"bin": b, "n": int(m.sum()), "mean_pred": float(p_cal[m].mean()), "observed_rate": float(y[m].mean())})
        ece += m.mean() * abs(p_cal[m].mean() - y[m].mean())
    return float(ece), rows


def booster_predict(paths, X, cols):
    preds = []
    for p in paths:
        bst = xgb.Booster()
        bst.load_model(p)
        preds.append(bst.predict(xgb.DMatrix(X[cols])))
    return np.mean(preds, axis=0)


# ---------------------------------------------------------------- features
log("building frozen-design features from raw data")
base = engineer.build_base(engineer.load_raw(RAW), lags=(7,), decay_lags=(), velocity=False, key_names=BASE_KEYS)
(_, _, _), (_, _, _), (X_te, y_te, t_te) = engineer.finalize(base, d_mode="anchor")
del base
amt = X_te["TransactionAmt"].to_numpy()
half = len(y_te) // 2
a, b = slice(0, half), slice(half, None)
log(f"test rows={len(y_te)} fraud={int(y_te.sum())} ({y_te.mean():.4f})")

# ---------------------------------------------------------------- load artifacts
art = {}
for name, folder in (("full_v5", "prod"), ("compact_b2", "prod_compact")):
    models = [m for m in find(f"/kaggle/input/**/{folder}/xgb_seed*.json")]
    spec = json.load(open(find(f"/kaggle/input/**/{folder}/serving_spec.json")[0]))
    iso = pickle.load(open(find(f"/kaggle/input/**/{folder}/isotonic.pkl")[0], "rb"))
    art[name] = {"models": models, "spec": spec, "iso": iso}
    log(f"loaded {name}: {len(models)} models, {len(spec['features'])} features")
    missing = [c for c in spec["features"] if c not in X_te.columns]
    assert not missing, f"{name} spec features missing from rebuilt features: {missing[:5]}"

# ---------------------------------------------------------------- score + round trip
expected = {"full_v5": {"second_half_pr_auc": 0.7121}, "compact_b2": {"full_test_pr_auc": 0.7061559, "second_half_pr_auc": 0.7045}}
scores = {}
for name, d in art.items():
    cols = d["spec"]["features"]
    p = booster_predict(d["models"], X_te, cols)
    scores[name] = p
    cal = d["iso"].predict(p)
    results[name] = {
        "n_features": len(cols),
        "full_test": score_table(y_te, p, amt),
        "second_half": score_table(y_te[b], p[b], amt[b]),
        "first_half": score_table(y_te[a], p[a], amt[a]),
    }
    # calibration is fitted on the FIRST half of test; honest evaluation is on the SECOND half
    p_cal_b = cal[b]
    ece, rel = calibration_table(y_te[b], p_cal_b)
    results[name]["calibration_second_half"] = {
        "brier": float(brier_score_loss(y_te[b], p_cal_b)),
        "log_loss": float(log_loss(y_te[b], np.clip(p_cal_b, 1e-6, 1 - 1e-6))),
        "ece_10_equal_mass_bins": ece, "reliability": rel,
        "brier_of_constant_base_rate": float(brier_score_loss(y_te[b], np.full(len(y_te[b]), y_te[a].mean()))),
    }
    tier_rows = {}
    for target, t in d["spec"]["tiers"].items():
        if t is None:
            continue
        pred = cal[b] >= t["calibrated_threshold"]
        tp = int((pred & (y_te[b] == 1)).sum())
        tier_rows[target] = {"threshold": t["calibrated_threshold"], "precision": float(tp / max(pred.sum(), 1)),
                             "recall": float(tp / (y_te[b] == 1).sum()), "alert_rate": float(pred.mean()),
                             "reported_precision": t["second_half_precision"]}
    results[name]["tiers_second_half"] = tier_rows
    for k, exp in expected[name].items():
        got = results[name]["full_test" if "full_test" in k else "second_half"]["pr_auc"]
        checks[f"{name}.{k}"] = {"expected": exp, "got": got, "pass": bool(abs(got - exp) < 5e-4)}
    checks[f"{name}.tiers_match_reported"] = {
        "pass": all(abs(v["precision"] - v["reported_precision"]) < 1e-3 for v in tier_rows.values())}
    log(f"{name}: test PR-AUC={results[name]['full_test']['pr_auc']:.4f} 2nd-half={results[name]['second_half']['pr_auc']:.4f}")
results["round_trip_checks"] = checks

# ---------------------------------------------------------------- slices (compact model)
p = scores["compact_b2"]
thr1 = np.quantile(p[y_te == 0], 0.99)
uid_known = X_te["h7_uid_n"].to_numpy() > 0
any_fraud_hist = ((X_te["h7_uid_fraud_n"] > 0) | (X_te["h7_card_fraud_n"] > 0) | (X_te["h7_addr_fraud_n"] > 0)
                  | (X_te["h7_email_fraud_n"] > 0) | (X_te["h7_device_fraud_n"] > 0)).to_numpy()
uid_fraud_hist = (X_te["h7_uid_fraud_n"] > 0).to_numpy()
days = (t_te - t_te.min()) / 86400.0


def slice_row(mask):
    y_s, p_s = y_te[mask], p[mask]
    row = {"rows": int(mask.sum()), "fraud": int(y_s.sum()), "fraud_rate": float(y_s.mean()) if mask.any() else None,
           "share_of_all_fraud": float(y_s.sum() / y_te.sum())}
    if y_s.sum() > 0 and (y_s == 0).sum() > 0:
        pred = p_s >= thr1
        tp = int((pred & (y_s == 1)).sum())
        row.update({"pr_auc": ap(y_s, p_s), "recall_at_global_1pct_fpr_threshold": float(tp / y_s.sum()),
                    "precision_at_global_threshold": float(tp / max(pred.sum(), 1))})
    return row


slices = {
    "uid_has_labeled_history": slice_row(uid_known),
    "uid_new_or_unlabeled_history": slice_row(~uid_known),
    "uid_has_prior_fraud": slice_row(uid_fraud_hist),
    "no_fraud_history_on_any_entity": slice_row(~any_fraud_hist),
    "any_entity_has_prior_fraud": slice_row(any_fraud_hist),
    "amount_lt_50": slice_row(amt < 50),
    "amount_50_200": slice_row((amt >= 50) & (amt < 200)),
    "amount_200_1000": slice_row((amt >= 200) & (amt < 1000)),
    "amount_ge_1000": slice_row(amt >= 1000),
}
for w in range(int(days.max() // 7) + 1):
    slices[f"test_week_{w + 1}"] = slice_row((days >= 7 * w) & (days < 7 * (w + 1)))
results["slices_compact_model_global_threshold"] = {"global_1pct_fpr_threshold": float(thr1), "slices": slices}

# ------------------------------------------------------------------- report
for name in ("full_v5", "compact_b2"):
    r = results[name]
    print(f"\n=== {name} ({r['n_features']} features) ===")
    for part in ("full_test", "second_half"):
        print(f"[{part}]", "  ".join(f"{k}={v:.4f}" for k, v in r[part].items()))
    c = r["calibration_second_half"]
    print(f"[calibration 2nd half] brier={c['brier']:.5f} (constant base-rate brier={c['brier_of_constant_base_rate']:.5f}) "
          f"log_loss={c['log_loss']:.4f} ECE={c['ece_10_equal_mass_bins']:.4f}")
    for k, v in r["tiers_second_half"].items():
        print(f"[tier target {k}]", v)
print("\n=== ROUND-TRIP CHECKS ===")
for k, v in checks.items():
    print(k, v)
print("\n=== SLICES (compact model, threshold = global 1% FPR) ===")
for k, v in slices.items():
    print(f"{k:<34}", v)

with open(f"{OUT}/results.json", "w") as f:
    json.dump(results, f, indent=2)
log("done; results.json saved")
