"""Bring-your-own-data training kit for UPI traffic.

Run it where the data lives; no data is copied into this repo and nothing leaves the machine.

    python -m src.upi.train --data transactions.csv --out models/upi --config my_config.json

CSV columns (map your own names in the config): transaction_id, timestamp (Unix seconds), amount, payer_id, payee_id, device_id
(optional), channel (optional: app|qr|collect|intent) and is_fraud (0/1, the confirmed label). Identifiers MUST already be
pseudonymised (see src/upi/schema.py); the loader refuses raw-looking identifiers.

Method (the one validated on IEEE-CIS in this project):
  * the file is replayed in time order through the SAME feature code the live service uses; a transaction's label only becomes visible
    `label_lag_days` after it happened (default 7);
  * time-ordered split: train 60% | validation 15% | holdout 25% (first half of the holdout calibrates, the second half is reported);
  * 3 LightGBM seeds, recency weights, rounds chosen on validation (early stopping) then FIXED for a refit on train + validation;
  * isotonic calibration and precision-target tiers (0.5 / 0.7 / 0.9) fitted on the first half of the holdout, reported on the second;
  * the success criteria in the config are written BEFORE the run and the outcome is reported either way.
"""

import argparse
import csv
import json
import os
import time
from typing import Dict, List, Optional

import numpy as np

from . import evalkit
from .features import FEATURE_NAMES, KEY_NAMES, UpiFeaturizer, replay
from .schema import looks_raw

DEFAULT_CONFIG: Dict = {
    "columns": {"transaction_id": "transaction_id", "timestamp": "timestamp", "amount": "amount", "payer_id": "payer_id",
                "payee_id": "payee_id", "device_id": "device_id", "channel": "channel", "is_fraud": "is_fraud"},
    "label_lag_days": 7.0, "recency_tau_days": 120.0, "seeds": [42, 1, 2],
    "split": {"train": 0.60, "valid": 0.15},
    "lightgbm": {"learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 20, "feature_fraction": 0.8, "bagging_fraction": 0.8,
                 "bagging_freq": 1, "lambda_l2": 1.0},
    "max_rounds": 2000, "early_stopping_rounds": 100, "min_frauds_per_slice": 20,
    # WRITE THESE BEFORE RUNNING and do not change them after seeing results:
    "criteria": {"min_pr_auc": 0.0, "min_recall_at_fpr_1pct": 0.0},
    "synthetic": False,
}


def merge_config(user: Optional[dict]) -> dict:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    for k, v in (user or {}).items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    return cfg


def load_rows(path: str, columns: Optional[Dict[str, str]] = None) -> List[dict]:
    cols = {**DEFAULT_CONFIG["columns"], **(columns or {})}
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for rec in csv.DictReader(f):
            row = {"transaction_id": rec[cols["transaction_id"]], "timestamp": float(rec[cols["timestamp"]]),
                   "amount": float(rec[cols["amount"]]), "payer_id": rec[cols["payer_id"]], "payee_id": rec[cols["payee_id"]],
                   "device_id": (rec.get(cols["device_id"]) or None), "channel": (rec.get(cols["channel"]) or "app"),
                   "is_fraud": int(float(rec[cols["is_fraud"]]))}
            for k in ("payer_id", "payee_id", "device_id"):
                if row[k] and looks_raw(row[k]):
                    raise ValueError(f"row {row['transaction_id']}: '{k}' looks like a raw identifier; pseudonymise before training")
            rows.append(row)
    return rows


def _check_slice(name: str, y: np.ndarray, minimum: int) -> None:
    if int(y.sum()) < minimum:
        raise ValueError(f"the {name} slice has only {int(y.sum())} confirmed frauds (need at least {minimum}); use more data or a longer period")


def train(rows: List[dict], out_dir: str, config: Optional[dict] = None, log=print) -> dict:
    import lightgbm as lgb

    cfg = merge_config(config)
    t0 = time.time()
    rows = sorted(rows, key=lambda r: r["timestamp"])
    fz = UpiFeaturizer()
    X, y, ts = [], [], []
    for vec, label, row in replay(rows, cfg["label_lag_days"], featurizer=fz):
        X.append(vec), y.append(label), ts.append(row["timestamp"])
    X, y, ts = np.vstack(X), np.asarray(y), np.asarray(ts)
    n = len(y)
    a, b = int(n * cfg["split"]["train"]), int(n * (cfg["split"]["train"] + cfg["split"]["valid"]))
    mid = b + (n - b) // 2
    sl = {"train": slice(0, a), "valid": slice(a, b), "cal": slice(b, mid), "report": slice(mid, n)}
    for name, s in sl.items():
        _check_slice(name, y[s], cfg["min_frauds_per_slice"])
    log(f"rows {n:,}, fraud {int(y.sum()):,} ({y.mean():.2%}); features {X.shape[1]}; replay {time.time() - t0:.0f}s")

    spw = float(np.clip((y[sl["train"]] == 0).sum() / max(y[sl["train"]].sum(), 1), 1.0, 100.0))
    base = {"objective": "binary", "metric": "average_precision", "scale_pos_weight": spw, "verbosity": -1, **cfg["lightgbm"]}
    tau = cfg["recency_tau_days"] * 86400.0
    weight = lambda t: np.exp(-(t.max() - t) / tau)            # noqa: E731

    iters = []
    for seed in cfg["seeds"]:
        dtr = lgb.Dataset(X[sl["train"]], y[sl["train"]], weight=weight(ts[sl["train"]]))
        dva = lgb.Dataset(X[sl["valid"]], y[sl["valid"]], reference=dtr)
        bst = lgb.train({**base, "seed": seed}, dtr, num_boost_round=cfg["max_rounds"], valid_sets=[dva],
                        callbacks=[lgb.early_stopping(cfg["early_stopping_rounds"], verbose=False)])
        iters.append(int(bst.best_iteration))
    rounds = max(int(np.mean(iters) * 1.15), 50)

    fit = slice(0, b)
    os.makedirs(out_dir, exist_ok=True)
    scores = []
    for seed in cfg["seeds"]:
        bst = lgb.train({**base, "seed": seed}, lgb.Dataset(X[fit], y[fit], weight=weight(ts[fit])), num_boost_round=rounds)
        bst.save_model(os.path.join(out_dir, f"lgbm_seed{seed}.txt"))
        scores.append(bst.predict(X[b:]))
    score = np.mean(scores, axis=0)
    s_cal, s_rep = score[: mid - b], score[mid - b:]
    y_cal, y_rep = y[sl["cal"]], y[sl["report"]]

    kx, ky = evalkit.isotonic_fit(s_cal, y_cal)
    cal_cal, cal_rep = np.interp(s_cal, kx, ky), np.interp(s_rep, kx, ky)
    thr = evalkit.tier_thresholds(cal_cal, y_cal)
    tiers = {}
    for target, th in thr.items():
        if th is None:
            tiers[target] = None
            continue
        flag = cal_rep >= th
        tiers[target] = {"calibrated_threshold": th, "second_half_precision": float(y_rep[flag].mean()) if flag.any() else None,
                         "second_half_recall": float(flag[y_rep == 1].mean()), "second_half_alert_rate": float(flag.mean())}
    metrics = {"pr_auc": evalkit.average_precision(y_rep, s_rep), "roc_auc": evalkit.roc_auc(y_rep, s_rep),
               "recall_at_fpr_1pct": evalkit.recall_at_fpr(y_rep, s_rep, 0.01),
               "pr_auc_ci95": evalkit.bootstrap_ci(y_rep, s_rep, evalkit.average_precision, n=200),
               "brier_calibrated": float(np.mean((cal_rep - y_rep) ** 2)),
               "brier_constant_base_rate": float(np.mean((y_cal.mean() - y_rep) ** 2)), "fraud_rate_report_slice": float(y_rep.mean())}
    crit = cfg["criteria"]
    met = bool(metrics["pr_auc"] >= crit["min_pr_auc"] and metrics["recall_at_fpr_1pct"] >= crit["min_recall_at_fpr_1pct"])

    json.dump({"x": [float(v) for v in kx], "y": [float(v) for v in ky]}, open(os.path.join(out_dir, "isotonic.json"), "w"))
    spec = {"kind": "upi", "model_family": "lightgbm", "features": list(FEATURE_NAMES), "key_names": list(KEY_NAMES),
            "label_lag_days": cfg["label_lag_days"], "n_trees": rounds, "prod_seeds": cfg["seeds"], "tiers": tiers,
            "trained_on": {"rows": n, "frauds": int(y.sum()), "t_start": float(ts[0]), "t_end": float(ts[-1]), "synthetic": bool(cfg["synthetic"])},
            "criteria": crit, "criteria_met": met}
    json.dump(spec, open(os.path.join(out_dir, "serving_spec.json"), "w"), indent=2)
    from .state_io import save_featurizer
    save_featurizer(fz, os.path.join(out_dir, "warm_state.json.gz"))
    result = {"metrics_second_half_holdout": metrics, "tiers": tiers, "rounds": rounds, "validation_best_iterations": iters,
              "criteria": crit, "criteria_met": met, "synthetic": bool(cfg["synthetic"]), "seconds": round(time.time() - t0)}
    json.dump(result, open(os.path.join(out_dir, "results.json"), "w"), indent=2)
    log(f"PR-AUC {metrics['pr_auc']:.3f} (CI {metrics['pr_auc_ci95'][0]:.3f}-{metrics['pr_auc_ci95'][1]:.3f}), "
        f"recall@1%FPR {metrics['recall_at_fpr_1pct']:.3f}, criteria met: {met}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config")
    args = ap.parse_args()
    user_cfg = json.load(open(args.config)) if args.config else {}
    cfg = merge_config(user_cfg)
    train(load_rows(args.data, cfg["columns"]), args.out, cfg)
