"""PayWatch — adversarial training against fresh-identity evasion (Kaggle, GPU).

Idea: the model leans on entity-history features, so a fresh identity evades it. Add rotated copies of known
training frauds (identity features reset to what a brand-new identity looks like) so the model learns that fraud
signals survive without history. Variants: baseline | aug_all | aug_half | aug_all_w0.5 (copies weighted 0.5).

PRE-REGISTERED success criterion (set before any run), on the held-out TEST period:
  identity-rotation evasion at 1 probe < 20%  AND  test PR-AUC >= baseline - 0.01
  AND  recall @1% FPR >= baseline - 0.02.
Variant selection uses VALIDATION only (lowest validation identity-evasion among variants whose validation PR-AUC is
within 0.01 of baseline). Caveat: the attack is the same transformation as the augmentation, so identity-evasion
is an optimistic robustness number; real first-time-fraud recall (new users in the test period) is reported as the
independent check. Operating point per model: deepest validation threshold with precision >= 0.60 (SOFT-tier-like).
"""

import glob
import json
import os
import sys
import time

import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

T0 = time.time()
find = lambda p: sorted(glob.glob(p, recursive=True))  # noqa: E731
OUT = "/kaggle/working"
SEEDS = [42, 1, 2]
TAU_DAYS, PRECISION_TARGET, N_MIN = 120, 0.60, 100
BUDGETS = (1, 5, 20)
XGB_PARAMS = dict(max_depth=10, learning_rate=0.0558, subsample=0.935, colsample_bytree=0.678, min_child_weight=1,
                  reg_lambda=1.518, gamma=0.0336, scale_pos_weight=39.3)


def log(msg):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {msg}", flush=True)


sys.path.insert(0, os.path.dirname(find("/kaggle/input/**/upi_fingerprint.py")[0]))
import engineer  # noqa: E402

sys.path.insert(0, os.path.dirname(find("/kaggle/input/**/evasion.py")[0]))
import evasion  # noqa: E402
from entity_history import BASE_KEYS  # noqa: E402

RAW = os.path.dirname(find("/kaggle/input/**/train_transaction.csv")[0])
cols = json.load(open(find("/kaggle/input/**/prod_compact/serving_spec.json")[0]))["features"]
base = engineer.build_base(engineer.load_raw(RAW), lags=(7,), decay_lags=(), velocity=False, key_names=BASE_KEYS)
(X_tr, y_tr, t_tr), (X_va, y_va, _), (X_te, y_te, _) = engineer.finalize(base, d_mode="anchor")
del base
Xtr, Xva, Xte = (d[cols].to_numpy(dtype=np.float32) for d in (X_tr, X_va, X_te))
uid_new_te = (X_te["h7_uid_n"].to_numpy() == 0)
w_tr = np.exp(-(t_tr.max() - t_tr) / (TAU_DAYS * 86400.0))
GROUPS = evasion.build_groups(cols)
rng = np.random.default_rng(0)
legit_pool = Xtr[y_tr == 0]
donors_all = legit_pool[rng.choice(len(legit_pool), 20000, replace=False)]
fraud_idx = np.where(y_tr == 1)[0]
log(f"train {Xtr.shape} fraud {len(fraud_idx)}; identity_rotation sets {len(GROUPS['identity_rotation']['set'])} features")


def rotated(idx):
    donors = donors_all[rng.integers(0, len(donors_all), len(idx))]
    return evasion.apply_moves(Xtr[idx], cols, GROUPS, ["identity_rotation"], donors)


aug_all = rotated(fraud_idx)
half = rng.choice(fraud_idx, len(fraud_idx) // 2, replace=False)
aug_half = rotated(half)
variants = {
    "baseline": (Xtr, y_tr, w_tr),
    "aug_all": (np.vstack([Xtr, aug_all]), np.r_[y_tr, np.ones(len(fraud_idx), int)], np.r_[w_tr, w_tr[fraud_idx]]),
    "aug_half": (np.vstack([Xtr, aug_half]), np.r_[y_tr, np.ones(len(half), int)], np.r_[w_tr, w_tr[half]]),
    "aug_all_w0.5": (np.vstack([Xtr, aug_all]), np.r_[y_tr, np.ones(len(fraud_idx), int)], np.r_[w_tr, 0.5 * w_tr[fraud_idx]]),
}


def fit(X, y, w, seed):
    m = xgb.XGBClassifier(n_estimators=6000, tree_method="hist", device="cuda", eval_metric="aucpr",
                          early_stopping_rounds=60, random_state=seed, **XGB_PARAMS)
    m.fit(X, y, sample_weight=w, eval_set=[(Xva, y_va)], verbose=False)
    return m


def op_threshold(score, y):
    order = np.argsort(-score)
    tp = np.cumsum(y[order])
    n = np.arange(1, len(y) + 1)
    ok = np.where((tp / n >= PRECISION_TARGET) & (n >= N_MIN))[0]
    return float(score[order][ok.max()])


def recall_at_fpr(y, p, f=0.01):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.interp(f, fpr, tpr))


def evasion_rates(scorer, X, y, thr, seed):
    r = np.random.default_rng(seed)
    flagged = np.where((y == 1) & (scorer(X) >= thr))[0]
    targets = r.choice(flagged, min(1500, len(flagged)), replace=False)
    out = {"flagged_frauds": int(len(flagged))}
    for name, names in (("identity_rotation", ["identity_rotation"]), ("email_rotation", ["email_rotation"]),
                        ("all_moves", list(GROUPS))):
        best = evasion.attack(scorer, X[targets], cols, GROUPS, names, donors_all, r, BUDGETS)
        out[name] = {f"budget_{q}": float((b < thr).mean()) for q, b in best.items()}
    return out


results = {"criterion": {"identity_evasion_budget1_below": 0.20, "pr_auc_loss_max": 0.01, "recall_fpr1_loss_max": 0.02},
           "variants": {}}
for name, (X, y, w) in variants.items():
    models = [fit(X, y, w, s) for s in SEEDS]
    scorer = lambda Z, ms=models: np.mean([m.predict_proba(Z)[:, 1] for m in ms], axis=0)  # noqa: E731
    sv, st = scorer(Xva), scorer(Xte)
    thr = op_threshold(sv, y_va)
    flagged = st >= thr
    entry = {
        "best_iters": [int(m.best_iteration) for m in models], "threshold": thr,
        "val_pr_auc": float(average_precision_score(y_va, sv)),
        "test_pr_auc": float(average_precision_score(y_te, st)), "test_roc_auc": float(roc_auc_score(y_te, st)),
        "test_recall_at_fpr_1pct": recall_at_fpr(y_te, st),
        "test_flagged_precision": float(y_te[flagged].mean()), "test_flagged_recall": float(flagged[y_te == 1].mean()),
        "test_recall_first_time_fraud": float(flagged[(y_te == 1) & uid_new_te].mean()),
        "test_recall_known_entity_fraud": float(flagged[(y_te == 1) & ~uid_new_te].mean()),
        "evasion_val": evasion_rates(scorer, Xva, y_va, thr, 11),
        "evasion_test": evasion_rates(scorer, Xte, y_te, thr, 22),
    }
    results["variants"][name] = entry
    log(f"{name}: test PR-AUC {entry['test_pr_auc']:.4f} recall@1%FPR {entry['test_recall_at_fpr_1pct']:.3f} "
        f"first-time recall {entry['test_recall_first_time_fraud']:.3f} | evasion identity@1 test "
        f"{entry['evasion_test']['identity_rotation']['budget_1']:.1%}")

b = results["variants"]["baseline"]
eligible = {k: v for k, v in results["variants"].items()
            if k != "baseline" and v["val_pr_auc"] >= b["val_pr_auc"] - 0.01}
chosen = min(eligible, key=lambda k: eligible[k]["evasion_val"]["identity_rotation"]["budget_1"]) if eligible else None
results["chosen_on_validation"] = chosen
if chosen:
    c = results["variants"][chosen]
    results["criterion_met"] = bool(
        c["evasion_test"]["identity_rotation"]["budget_1"] < 0.20
        and c["test_pr_auc"] >= b["test_pr_auc"] - 0.01 and c["test_recall_at_fpr_1pct"] >= b["test_recall_at_fpr_1pct"] - 0.02)
json.dump(results, open(f"{OUT}/results.json", "w"), indent=1)

print("\n=== ADVERSARIAL TRAINING (held-out test period) ===")
print(f"{'variant':<14}{'valPR':>7}{'testPR':>8}{'ROC':>7}{'rec@1%':>8}{'flag prec':>10}{'flag rec':>9}{'1st-time':>9}{'known':>7}"
      f"{'id@1':>7}{'id@20':>7}{'email@20':>9}{'all@20':>8}")
for k, v in results["variants"].items():
    e = v["evasion_test"]
    print(f"{k:<14}{v['val_pr_auc']:>7.4f}{v['test_pr_auc']:>8.4f}{v['test_roc_auc']:>7.4f}{v['test_recall_at_fpr_1pct']:>8.3f}"
          f"{v['test_flagged_precision']:>10.3f}{v['test_flagged_recall']:>9.3f}{v['test_recall_first_time_fraud']:>9.3f}"
          f"{v['test_recall_known_entity_fraud']:>7.3f}{e['identity_rotation']['budget_1']:>7.1%}{e['identity_rotation']['budget_20']:>7.1%}"
          f"{e['email_rotation']['budget_20']:>9.1%}{e['all_moves']['budget_20']:>8.1%}")
print("chosen on validation:", chosen, "| pre-registered criterion met:", results.get("criterion_met"))
log("done")
