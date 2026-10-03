"""PayWatch — Layer 2 test: Isolation Forest + score fusion with XGBoost, all on Kaggle.

XGBoost is refit with the v9 best params. Isolation Forest is fit on the top-K most important
features (raw 438 mixed-scale columns make it noisy). Fusion weight and threshold are chosen on
validation only; test is used once to compare XGBoost alone vs fused.
"""

import glob
import json
import os
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_curve

TARGET = "isFraud"
TOP_K = 40

DATA_DIR = os.path.dirname(glob.glob("/kaggle/input/**/train.parquet", recursive=True)[0])
train_df = pd.read_parquet(f"{DATA_DIR}/train.parquet")
val_df = pd.read_parquet(f"{DATA_DIR}/val.parquet")
test_df = pd.read_parquet(f"{DATA_DIR}/test.parquet")

X_train, y_train = train_df.drop(columns=[TARGET]), train_df[TARGET]
X_val, y_val = val_df.drop(columns=[TARGET]), val_df[TARGET]
X_test, y_test = test_df.drop(columns=[TARGET]), test_df[TARGET]

params = {
    "max_depth": 10, "learning_rate": 0.05578380588081771, "subsample": 0.9353340184422343,
    "colsample_bytree": 0.6783519848540355, "min_child_weight": 1,
    "reg_lambda": 1.5184210034614605, "gamma": 0.03360032509079314,
    "scale_pos_weight": 39.34532117125249,
}
model = xgb.XGBClassifier(
    n_estimators=2000, tree_method="hist", device="cuda", eval_metric="aucpr",
    early_stopping_rounds=50, random_state=42, **params,
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

xgb_val = model.predict_proba(X_val)[:, 1]
xgb_test = model.predict_proba(X_test)[:, 1]
print(f"XGB alone  val PR-AUC {average_precision_score(y_val, xgb_val):.4f}  "
      f"test PR-AUC {average_precision_score(y_test, xgb_test):.4f}")

gain = model.get_booster().get_score(importance_type="gain")
top_features = sorted(gain, key=gain.get, reverse=True)[:TOP_K]

iforest = IsolationForest(n_estimators=300, max_samples=2048, random_state=42, n_jobs=-1)
iforest.fit(X_train[top_features])

train_raw = -iforest.score_samples(X_train[top_features])
sorted_train = np.sort(train_raw)


def if_score(X):
    raw = -iforest.score_samples(X[top_features])
    return np.searchsorted(sorted_train, raw) / len(sorted_train)  # percentile vs train, 0..1


if_val, if_test = if_score(X_val), if_score(X_test)
print(f"IForest alone  val PR-AUC {average_precision_score(y_val, if_val):.4f}  "
      f"test PR-AUC {average_precision_score(y_test, if_test):.4f}")


def best_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
    i = f1.argmax()
    return float(thr[i]), float(prec[i]), float(rec[i]), float(f1[i])


def report(name, p_val, p_test):
    thr, _, _, _ = best_threshold(y_val, p_val)
    pred = p_test >= thr
    tp = int((pred & (y_test == 1)).sum())
    prec = tp / max(int(pred.sum()), 1)
    rec = tp / int((y_test == 1).sum())
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    print(f"{name:<22} test PR-AUC {average_precision_score(y_test, p_test):.4f}  "
          f"thr {thr:.3f}  precision {prec:.4f}  recall {rec:.4f}  F1 {f1:.4f}")
    return {"pr_auc": float(average_precision_score(y_test, p_test)), "threshold": thr,
            "precision": prec, "recall": rec, "f1": f1}


results = {"xgb_alone": report("XGB alone", xgb_val, xgb_test)}

best_w, best_ap = 1.0, -1.0
for w in np.arange(0.0, 1.0001, 0.05):
    ap = average_precision_score(y_val, w * xgb_val + (1 - w) * if_val)
    if ap > best_ap:
        best_w, best_ap = float(w), float(ap)
print(f"\nBest fusion weight on VAL: w_xgb={best_w:.2f} (val PR-AUC {best_ap:.4f})")

results["fused_best_w"] = report(f"Fused w={best_w:.2f}", best_w * xgb_val + (1 - best_w) * if_val,
                                 best_w * xgb_test + (1 - best_w) * if_test)
results["fused_roadmap_0.7_0.3"] = report("Fused 0.7/0.3 (roadmap)", 0.7 * xgb_val + 0.3 * if_val,
                                          0.7 * xgb_test + 0.3 * if_test)
results["best_weight"] = best_w
results["top_features"] = top_features

with open("/kaggle/working/fusion_results.json", "w") as f:
    json.dump(results, f, indent=2)
with open("/kaggle/working/iforest.pkl", "wb") as f:
    pickle.dump({"iforest": iforest, "sorted_train": sorted_train, "top_features": top_features,
                 "w_xgb": best_w, "xgb_threshold": results["fused_best_w"]["threshold"]}, f)
print("\nSaved fusion_results.json and iforest.pkl")
