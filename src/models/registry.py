"""MLflow experiment history + model registry for PayWatch (executed on Kaggle, store committed to the repo).

Runs
  - HISTORY: earlier experiments recorded from session logs (their kernel outputs were overwritten);
    tagged source=session_log, `leaky=true` where the split was random (invalid for fraud).
  - kernel results.json files (audited retrain, SHAP/compact, verification, API test): flattened metrics.
Registry
  - model `paywatch-fraud-xgb`: v1 full 452-feature refit (stage Staging, alias challenger),
    v2 compact 120-feature refit (stage Production, alias champion).
  Weights are NOT stored in MLflow (28 MB of JSON live in the repo under models/compact); the registered
  artifact bundle holds the serving spec + SHA-256 hashes of the weight files.
"""

import hashlib
import json
import os
import sqlite3
from typing import Dict, List, Optional

import mlflow
from mlflow import MlflowClient

EXPERIMENT = "paywatch-fraud"
MODEL_NAME = "paywatch-fraud-xgb"
SESSION = {"source": "session_log"}

HISTORY: List[dict] = [
    {"name": "h01-smote-xgb-random-split", "tags": {"leaky": "true", "design": "SMOTE, random split, 43 feats"},
     "metrics": {"test_pr_auc": 0.6979, "test_recall": 0.5224, "test_precision": 0.8517}},
    {"name": "h02-lgbm-optuna-random-split", "tags": {"leaky": "true", "design": "scale_pos_weight, random split, global user stats"},
     "metrics": {"test_pr_auc": 0.8455, "test_recall": 0.7396, "test_precision": 0.8860}},
    {"name": "h03-time-split-causal-42-features", "tags": {"leaky": "false", "design": "leak fixed: causal features + time split"},
     "metrics": {"test_pr_auc": 0.5187, "test_recall": 0.4693, "test_precision": 0.5332}},
    {"name": "h04-wide-432-features", "tags": {"leaky": "false", "design": "all C/D/M/V/id columns"},
     "metrics": {"test_pr_auc": 0.5529, "test_recall": 0.4963, "test_precision": 0.5835}},
    {"name": "h05-uid-reconstruction", "tags": {"leaky": "false", "design": "uid + per-uid causal aggregates"},
     "metrics": {"test_pr_auc": 0.5563, "test_recall": 0.4781, "test_precision": 0.6423}},
    {"name": "h06-isolation-forest-fusion", "tags": {"leaky": "false", "verdict": "rejected"},
     "metrics": {"iforest_alone_test_pr_auc": 0.0974, "xgb_alone_test_pr_auc": 0.5563,
                 "fused_val_chosen_test_pr_auc": 0.5563, "fused_0.7_0.3_test_pr_auc": 0.5033, "val_chosen_xgb_weight": 1.0}},
    {"name": "h07-feature-variants-and-boosters", "tags": {"leaky": "false", "verdict": "anchor_D adopted; LGB/CatBoost blend rejected"},
     "metrics": {"val_pr_auc_raw_D": 0.6580, "val_pr_auc_drop_D": 0.6490, "val_pr_auc_anchor_D": 0.6806,
                 "val_pr_auc_drop_D_drop_V": 0.6337, "test_pr_auc_xgb": 0.5802, "test_pr_auc_lgb": 0.5493, "test_pr_auc_cat": 0.5258}},
    {"name": "h08-adversarial-pruning-and-learning-curve", "tags": {"leaky": "false", "verdict": "pruning rejected; more rows add ~nothing"},
     "metrics": {"adv_auc_unpruned": 1.0, "adv_auc_after_80_features_dropped": 0.805, "val_pr_auc_unpruned": 0.679,
                 "val_pr_auc_after_80_features_dropped": 0.642, "test_pr_auc": 0.5805, "lc_test_pr_auc_25pct": 0.5227,
                 "lc_test_pr_auc_50pct": 0.5659, "lc_test_pr_auc_75pct": 0.5798, "lc_test_pr_auc_100pct": 0.5805,
                 "refit_train_plus_val_test_pr_auc": 0.6683}},
    {"name": "h09-phaseA-lagged-entity-history-recency", "tags": {"leaky": "false", "design": "hist7 5 keys + velocity, tau 120d"},
     "metrics": {"test_pr_auc": 0.6953, "test_roc_auc": 0.9438, "test_recall_at_fpr_1pct": 0.6348,
                 "test_precision_top_1pct": 0.9605, "refit_gap0_test_pr_auc": 0.7216, "refit_gap14_test_pr_auc": 0.7058,
                 "refit_gap30_test_pr_auc": 0.6949}},
    {"name": "h10-phaseA2-12-entity-keys", "tags": {"leaky": "false", "verdict": "tied within seed noise; not adopted"},
     "metrics": {"test_pr_auc": 0.6876, "label_lag7_test_pr_auc": 0.6876, "label_lag14_test_pr_auc": 0.6495,
                 "label_lag30_test_pr_auc": 0.6063}},
]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def flatten(d: dict, prefix: str = "") -> Dict[str, float]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            out[key] = float(v)
        elif isinstance(v, dict):
            out.update(flatten(v, f"{key}."))
    return out


def log_run(name: str, metrics: Optional[dict] = None, params: Optional[dict] = None, tags: Optional[dict] = None,
            artifacts: Optional[List[str]] = None) -> str:
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags(tags or {})
        mlflow.log_params({k: str(v)[:250] for k, v in (params or {}).items()})
        for k, v in (metrics or {}).items():
            mlflow.log_metric(k, float(v))
        for a in artifacts or []:
            mlflow.log_artifact(a)
        return run.info.run_id


def log_history() -> int:
    for h in HISTORY:
        log_run(h["name"], metrics=h["metrics"], tags={**SESSION, **h["tags"]})
    return len(HISTORY)


def log_results(name: str, results: dict, tags: dict, artifacts: Optional[List[str]] = None) -> str:
    return log_run(name, metrics=flatten(results), tags=tags, artifacts=artifacts)


def log_k_sweep(b2_results: dict) -> None:
    for k, v in b2_results["k_sweep"].items():
        log_run(f"b2-compact-top{k}-features", metrics={"val_mean": v["val_mean"], "val_std": v["val_std"],
                                                         "test_mean": v["test_mean"], "test_std": v["test_std"]},
                params={"K": k, "seeds": 3}, tags={"source": "kernel:paywatch-b2", "phase": "compact_sweep"})


def register_version(client: MlflowClient, run_name: str, files: List[str], weight_files: List[str], params: dict,
                     metrics: dict, stage: str, alias: str, description: str) -> str:
    hashes = {f"sha256.{os.path.basename(p)}": sha256(p) for p in weight_files}
    sizes = {f"bytes.{os.path.basename(p)}": os.path.getsize(p) for p in weight_files}
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({"stage": stage, "weights_location": "models/compact in the repo (not stored in MLflow)"})
        mlflow.log_params({**{k: str(v)[:250] for k, v in params.items()}, **hashes})
        for k, v in {**metrics, **sizes}.items():
            mlflow.log_metric(k, float(v))
        for f in files:
            mlflow.log_artifact(f, artifact_path="model")
        run_id, uri = run.info.run_id, run.info.artifact_uri
    try:
        client.create_registered_model(MODEL_NAME, description="PayWatch XGBoost fraud scorer (seed-averaged, calibrated)")
    except mlflow.exceptions.MlflowException:
        pass
    mv = client.create_model_version(MODEL_NAME, source=f"{uri}/model", run_id=run_id, description=description)
    client.set_registered_model_alias(MODEL_NAME, alias, mv.version)
    client.set_model_version_tag(MODEL_NAME, mv.version, "stage", stage)
    return mv.version


def rebase_artifact_paths(db_path: str, old_prefix: str, new_prefix: str) -> Dict[str, int]:
    """Rewrite absolute Kaggle artifact paths in the sqlite store so the repo copy is portable."""
    con = sqlite3.connect(db_path)
    changed = {}
    for table, col in (("experiments", "artifact_location"), ("runs", "artifact_uri"), ("model_versions", "source")):
        cur = con.execute(f"UPDATE {table} SET {col} = REPLACE({col}, ?, ?) WHERE {col} LIKE ?",
                          (old_prefix, new_prefix, f"%{old_prefix}%"))
        changed[table] = cur.rowcount
    con.commit()
    con.close()
    return changed
