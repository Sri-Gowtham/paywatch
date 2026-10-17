"""PayWatch — build the MLflow store (Kaggle, CPU, internet ON only for `pip install mlflow`).

Writes /kaggle/working/mlflow/mlflow.db (sqlite) + small artifacts. Weights are only hashed, not stored.
"""

import glob
import json
import os
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "mlflow"], check=True)

import mlflow  # noqa: E402
from mlflow import MlflowClient  # noqa: E402

find = lambda p: sorted(glob.glob(p, recursive=True))  # noqa: E731

reg_dir = os.path.dirname(find("/kaggle/input/**/registry.py")[0])
sys.path.insert(0, reg_dir)
import registry  # noqa: E402

OUT = "/kaggle/working/mlflow"
os.makedirs(f"{OUT}/artifacts", exist_ok=True)
mlflow.set_tracking_uri(f"sqlite:///{OUT}/mlflow.db")
if mlflow.get_experiment_by_name(registry.EXPERIMENT) is None:
    mlflow.create_experiment(registry.EXPERIMENT, artifact_location=f"{OUT}/artifacts")
mlflow.set_experiment(registry.EXPERIMENT)
client = MlflowClient()


def results_of(slug):
    paths = [p for p in find("/kaggle/input/**/results.json") if slug in p]
    return json.load(open(paths[0])), paths[0]


n = registry.log_history()
print(f"logged {n} historical runs")

v5, _ = results_of("paywatch-pipeline")
b2, _ = results_of("paywatch-b2")
verify, _ = results_of("paywatch-verify")
apitest, _ = results_of("paywatch-apitest")
registry.log_results("v5-audited-retrain-frozen-design", {k: v for k, v in v5.items() if k in (
    "seed_summary", "per_seed", "test_metrics_seed42", "test_metrics_ensemble5", "bootstrap_ci_ensemble5",
    "test_second_half_comparison", "precision_target_tiers", "integrity")},
    {"source": "kernel:paywatch-pipeline v5", "phase": "audited_retrain", "leaky": "false"})
registry.log_results("b2-shap-and-compact-model", {k: v for k, v in b2.items() if k in (
    "full_model", "k_sweep", "chosen_K", "shap_by_family_share", "compact_second_half", "compact_full_test_refit",
    "precision_target_tiers")},
    {"source": "kernel:paywatch-b2", "phase": "compact", "leaky": "false"})
registry.log_k_sweep(b2)
verify_metrics = {name: {k: v for k, v in verify[name].items() if k != "calibration_second_half"}
                  for name in ("full_v5", "compact_b2")}
registry.log_results("verify-saved-artifacts", verify_metrics,
                     {"source": "kernel:paywatch-verify", "phase": "verification"})
registry.log_results("apitest-stream-replay", {"stream_replay": apitest["stream_replay"],
                                              "latency_with_explanations_ms": apitest["latency_with_explanations_ms"]},
                     {"source": "kernel:paywatch-apitest", "phase": "serving"})

# ---------------------------------------------------------------- registry
full_dir = os.path.dirname(find("/kaggle/input/**/prod/serving_spec.json")[0])
cmp_dir = os.path.dirname(find("/kaggle/input/**/prod_compact/serving_spec.json")[0])
assets = {n: find(f"/kaggle/input/**/{n}")[0] for n in ("feature_meta.json", "isotonic.json")}
bundles = {
    "full": ([f"{full_dir}/serving_spec.json"], sorted(glob.glob(f"{full_dir}/xgb_seed*.json"))),
    "compact": ([f"{cmp_dir}/serving_spec.json", assets["feature_meta.json"], assets["isotonic.json"]],
                sorted(glob.glob(f"{cmp_dir}/xgb_seed*.json"))),
}
v_full = registry.register_version(
    client, "register-full-452f-refit", *bundles["full"],
    params={"features": 452, "seeds": "42,1,2", "train_data": "train+val", "recency_tau_days": 120, "label_lag_days": 7},
    metrics={"test_pr_auc": verify["full_v5"]["full_test"]["pr_auc"], "test_roc_auc": verify["full_v5"]["full_test"]["roc_auc"],
             "test_recall_at_fpr_1pct": verify["full_v5"]["full_test"]["recall_at_fpr_1pct"]},
    stage="Staging", alias="challenger", description="Full 452-feature refit; slightly higher PR-AUC, heavier to serve")
prod_lgbm, _ = results_of("paywatch-prod-lgbm")
lgbm_dir = os.path.dirname(find("/kaggle/input/**/paywatch-prod-lgbm/**/lgbm_seed42.txt")[0])
v_cmp = registry.register_version(
    client, "register-compact-120f-refit", *bundles["compact"],
    params={"features": 120, "seeds": "42,1,2", "train_data": "train+val", "recency_tau_days": 120, "label_lag_days": 7},
    metrics={"test_pr_auc": verify["compact_b2"]["full_test"]["pr_auc"], "test_roc_auc": verify["compact_b2"]["full_test"]["roc_auc"],
             "test_recall_at_fpr_1pct": verify["compact_b2"]["full_test"]["recall_at_fpr_1pct"]},
    stage="Archived", alias="previous_champion", description="Compact 120-feature XGBoost refit, served by the API until the LightGBM switch")
v_lgbm = registry.register_version(
    client, "register-compact-120f-lightgbm-refit",
    [f"{lgbm_dir}/serving_spec.json", f"{lgbm_dir}/isotonic.json", assets["feature_meta.json"]],
    sorted(glob.glob(f"{lgbm_dir}/lgbm_seed*.txt")),
    params={"features": 120, "seeds": "42,1,2", "train_data": "train+val", "recency_tau_days": 120, "label_lag_days": 7,
            "model_family": "lightgbm", "n_trees": prod_lgbm["n_trees"], "selected_by": "pre-registered study (results/improve.json), LightGBM alone chosen post hoc"},
    metrics={"test_pr_auc": prod_lgbm["test_full"]["pr_auc"], "test_roc_auc": prod_lgbm["test_full"]["roc_auc"],
             "test_recall_at_fpr_1pct": prod_lgbm["test_full"]["recall_at_fpr_1pct"],
             "second_half_pr_auc": prod_lgbm["test_second_half_lgbm"]["pr_auc"], "second_half_pr_auc_previous_xgb": prod_lgbm["test_second_half_shipped_xgb"]["pr_auc"],
             "hard_block_precision": prod_lgbm["precision_target_tiers"]["0.9"]["second_half_precision"],
             "hard_block_recall": prod_lgbm["precision_target_tiers"]["0.9"]["second_half_recall"]},
    stage="Production", alias="champion", description="Compact 120-feature LightGBM 3-seed refit served by the API (models/compact)")
print("registered versions:", v_full, v_cmp, v_lgbm)

changed = registry.rebase_artifact_paths(f"{OUT}/mlflow.db", OUT, "mlflow")
print("artifact paths rebased to repo-relative 'mlflow/':", changed)

# ---------------------------------------------------------------- report
runs = client.search_runs([mlflow.get_experiment_by_name(registry.EXPERIMENT).experiment_id], max_results=200)
print(f"\nruns in experiment: {len(runs)}")
for r in sorted(runs, key=lambda r: r.info.start_time):
    m = r.data.metrics
    key = next((k for k in ("test_pr_auc", "test_mean", "pr_auc") if k in m), None)
    print(f"{r.info.run_name:<48} {r.data.tags.get('source', ''):<28} {key or '':<12} {m.get(key, float('nan')) if key else ''}")
print("\nregistered model versions:")
for mv in client.search_model_versions(f"name='{registry.MODEL_NAME}'"):
    aliases = [a for a, v in client.get_registered_model(registry.MODEL_NAME).aliases.items() if v == mv.version]
    print(f"v{mv.version} stage_tag={mv.tags.get('stage')} aliases={aliases} source={mv.source}")
print("\nfiles:")
for p in sorted(glob.glob(f"{OUT}/**/*", recursive=True)):
    if os.path.isfile(p):
        print(f"{os.path.relpath(p, OUT):<70} {os.path.getsize(p):>10,}")
