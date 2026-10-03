"""PayWatch — tuned fraud classifier (XGBoost vs LightGBM) on Kaggle GPU.

Imbalance handled via scale_pos_weight (no SMOTE). Optuna tunes each model on
validation PR-AUC; the winner is refit and evaluated once on the held-out test set.
"""

import glob
import os
import pickle
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "optuna"], check=False)

import lightgbm as lgb
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
    recall_score,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

candidates = glob.glob("/kaggle/input/**/train.parquet", recursive=True)
if not candidates:
    print("Contents of /kaggle/input:", os.listdir("/kaggle/input"))
    raise FileNotFoundError("Could not locate train.parquet under /kaggle/input")
DATA_DIR = os.path.dirname(candidates[0])
print(f"Resolved DATA_DIR: {DATA_DIR}")
TARGET = "isFraud"

train_df = pd.read_parquet(f"{DATA_DIR}/train.parquet")
val_df = pd.read_parquet(f"{DATA_DIR}/val.parquet")
test_df = pd.read_parquet(f"{DATA_DIR}/test.parquet")

X_train, y_train = train_df.drop(columns=[TARGET]), train_df[TARGET]
X_val, y_val = val_df.drop(columns=[TARGET]), val_df[TARGET]
X_test, y_test = test_df.drop(columns=[TARGET]), test_df[TARGET]

spw = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight = {spw:.2f}")

N_TRIALS = 20


def xgb_params(trial):
    return dict(
        n_estimators=2000,
        max_depth=trial.suggest_int("max_depth", 4, 12),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 20, log=True),
        gamma=trial.suggest_float("gamma", 0, 5),
        scale_pos_weight=trial.suggest_float("scale_pos_weight", 1, spw * 1.5),
        tree_method="hist",
        device="cuda",
        eval_metric="aucpr",
        early_stopping_rounds=50,
        random_state=42,
    )


def lgb_params(trial):
    return dict(
        n_estimators=2000,
        num_leaves=trial.suggest_int("num_leaves", 31, 400),
        max_depth=trial.suggest_int("max_depth", -1, 14),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        subsample_freq=1,
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 20, log=True),
        scale_pos_weight=trial.suggest_float("scale_pos_weight", 1, spw * 1.5),
        random_state=42,
        verbose=-1,
    )


def objective_xgb(trial):
    m = xgb.XGBClassifier(**xgb_params(trial))
    m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return average_precision_score(y_val, m.predict_proba(X_val)[:, 1])


def objective_lgb(trial):
    m = lgb.LGBMClassifier(**lgb_params(trial))
    m.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return average_precision_score(y_val, m.predict_proba(X_val)[:, 1])


results = {}
for name, objective, make_params, builder in [
    ("xgboost", objective_xgb, xgb_params, xgb.XGBClassifier),
    ("lightgbm", objective_lgb, lgb_params, lgb.LGBMClassifier),
]:
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS)
    print(f"[{name}] best val PR-AUC = {study.best_value:.4f}")
    print(f"[{name}] best params = {study.best_params}")
    results[name] = (study.best_value, study.best_params, make_params, builder)

winner = max(results, key=lambda k: results[k][0])
best_val, best_params, make_params, builder = results[winner]
print(f"\n=== WINNER: {winner} (val PR-AUC {best_val:.4f}) ===")


class FixedTrial:
    def __init__(self, params):
        self.params = params

    def suggest_int(self, name, *a, **k):
        return self.params[name]

    suggest_float = suggest_int


final_params = make_params(FixedTrial(best_params))
model = builder(**final_params)
if winner == "xgboost":
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
else:
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

val_proba = model.predict_proba(X_val)[:, 1]
test_proba = model.predict_proba(X_test)[:, 1]

# Threshold chosen on validation only, then applied to test (no test leakage)
prec, rec, thr = precision_recall_curve(y_val, val_proba)
f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
best_thr = float(thr[f1.argmax()])

print(f"\nValidation PR-AUC: {average_precision_score(y_val, val_proba):.4f}")
print(f"F1-optimal threshold (from val): {best_thr:.4f}")
print("\n=== Test (threshold from val) ===")
print(f"PR-AUC: {average_precision_score(y_test, test_proba):.4f}")
print(f"Recall @{best_thr:.3f}: {recall_score(y_test, test_proba >= best_thr):.4f}")
print(classification_report(y_test, test_proba >= best_thr, digits=4))
print("=== Test @0.5 for reference ===")
print(f"Recall @0.5: {recall_score(y_test, test_proba >= 0.5):.4f}")

with open("/kaggle/working/model.pkl", "wb") as f:
    pickle.dump({"model": model, "winner": winner, "threshold": best_thr, "params": best_params}, f)
if winner == "xgboost":
    model.save_model("/kaggle/working/model.json")
print("\nSaved model.pkl to /kaggle/working/")
