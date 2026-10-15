"""PayWatch Phase 1: pool, select and evaluate the pre-registered criterion ONCE (Kaggle, CPU, reads the candidate kernels' outputs).

PRE-REGISTERED (fixed in the plan before any candidate ran):
  Pool: xgb_v1, xgb_v2, lgbm, cat, xgb_link, blend_all (mean logit of all five), blend_top2 (mean logit of the two best single
  candidates by rolling-fold mean).  SELECTION uses ONLY the rolling-origin fold scores (mean PR-AUC over the four folds).
  Criterion, evaluated on the untouched test period for the selected candidate only, using its REFIT scores (fixed rounds,
  train+validation, the same recipe as the shipped model) against the shipped model's test scores:
     PR-AUC diff >= +0.01  AND  recall@1%FPR diff >= +0.01  AND  paired-bootstrap 95% CI of the PR-AUC diff excludes 0.
  Failing it is reported as a null result and the shipped model stays. Test numbers of the other candidates are descriptive only.
Output: /kaggle/working/results.json
"""

import glob
import json
import os

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

OUT = "/kaggle/working"
find = lambda p: sorted(glob.glob(p, recursive=True))  # noqa: E731
CUTS = (0.60, 0.70, 0.80, 0.90)
SINGLES = ["xgb_v1", "xgb_v2", "lgbm", "cat", "xgb_link"]
BASELINE_FOLD_MEAN = 0.6957   # results/diagnose.json rolling_origin_baseline (production config, patience 60)


def pr(y, p):
    return float(average_precision_score(y, p))


def rec(y, p, f=0.01):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.interp(f, fpr, tpr))


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def one(pattern):
    hits = find(pattern)
    assert hits, f"missing {pattern}"
    return hits[0]


y_all = np.load(one("/kaggle/input/**/y_all.npy")).astype(int)
y_te = np.load(one("/kaggle/input/**/y_te.npy")).astype(int)
shipped = np.load(one("/kaggle/input/**/shipped_test.npy"))
N = len(y_all)
windows = [(int(N * c), int(N * (c + 0.10))) for c in CUTS]

have = [s for s in SINGLES if find(f"/kaggle/input/**/{s}_test_refit.npy") and find(f"/kaggle/input/**/{s}_fold3.npy")]
missing = sorted(set(SINGLES) - set(have))
print("candidates available:", have, "| missing:", missing)
folds = {s: [np.load(one(f"/kaggle/input/**/{s}_fold{i}.npy")) for i in range(4)] for s in have}
refit = {s: np.load(one(f"/kaggle/input/**/{s}_test_refit.npy")) for s in have}
trainonly = {s: np.load(one(f"/kaggle/input/**/{s}_test_trainonly.npy")) for s in have}


def fold_means(fold_scores):
    return [pr(y_all[a:b], fold_scores[i]) for i, (a, b) in enumerate(windows)]


pool_folds, pool_refit, pool_trainonly, members = {}, {}, {}, {}
for s in have:
    pool_folds[s], pool_refit[s], pool_trainonly[s], members[s] = folds[s], refit[s], trainonly[s], [s]


def add_blend(name, names):
    pool_folds[name] = [np.mean([logit(folds[s][i]) for s in names], axis=0) for i in range(4)]
    pool_refit[name] = np.mean([logit(refit[s]) for s in names], axis=0)
    pool_trainonly[name] = np.mean([logit(trainonly[s]) for s in names], axis=0)
    members[name] = list(names)


single_fold_mean = {s: float(np.mean(fold_means(folds[s]))) for s in have}
if len(have) >= 2:
    add_blend("blend_all", have)
    top2 = sorted(have, key=lambda s: -single_fold_mean[s])[:2]
    add_blend("blend_top2", top2)

table = {}
for name in pool_folds:
    fm = fold_means(pool_folds[name])
    table[name] = {"members": members[name], "fold_pr_auc": fm, "fold_mean_pr_auc": float(np.mean(fm)), "fold_std_pr_auc": float(np.std(fm)),
                   "test_pr_auc_trainonly": pr(y_te, pool_trainonly[name]),
                   "test_pr_auc_refit": pr(y_te, pool_refit[name]), "test_roc_auc_refit": float(roc_auc_score(y_te, pool_refit[name])),
                   "test_recall_at_fpr_1pct_refit": rec(y_te, pool_refit[name])}
chosen = max(table, key=lambda k: table[k]["fold_mean_pr_auc"])
print("SELECTION (rolling-fold mean PR-AUC only):", {k: round(v["fold_mean_pr_auc"], 4) for k, v in table.items()}, "-> chosen", chosen)

# ------------------------------------------------------------------ criterion on test, once
rng = np.random.default_rng(0)
cs, ss = pool_refit[chosen], shipped
d_pr, d_rec = [], []
for _ in range(300):
    idx = rng.integers(0, len(y_te), len(y_te))
    d_pr.append(pr(y_te[idx], cs[idx]) - pr(y_te[idx], ss[idx]))
    d_rec.append(rec(y_te[idx], cs[idx]) - rec(y_te[idx], ss[idx]))
ci_pr, ci_rec = [float(np.percentile(d_pr, q)) for q in (2.5, 97.5)], [float(np.percentile(d_rec, q)) for q in (2.5, 97.5)]
diff_pr, diff_rec = table[chosen]["test_pr_auc_refit"] - pr(y_te, ss), table[chosen]["test_recall_at_fpr_1pct_refit"] - rec(y_te, ss)
passed = bool(diff_pr >= 0.01 and diff_rec >= 0.01 and ci_pr[0] > 0)
criterion = {"chosen": chosen, "members": members[chosen],
             "shipped_test_pr_auc": pr(y_te, ss), "shipped_test_recall_at_fpr_1pct": rec(y_te, ss),
             "chosen_test_pr_auc": table[chosen]["test_pr_auc_refit"], "chosen_test_recall_at_fpr_1pct": table[chosen]["test_recall_at_fpr_1pct_refit"],
             "pr_auc_diff": diff_pr, "pr_auc_diff_ci95": ci_pr, "recall_diff": diff_rec, "recall_diff_ci95": ci_rec,
             "required": {"pr_auc_diff": 0.01, "recall_diff": 0.01, "ci_excludes_zero": True}, "criterion_met": passed}

# ------------------------------------------------------------------ E2 linkage verdict (descriptive, same recipe both sides)
e2 = None
if "xgb_link" in have and "xgb_v1" in have:
    a = np.array([table["xgb_link"]["fold_pr_auc"], table["xgb_v1"]["fold_pr_auc"]])
    e2 = {"fold_mean_link": table["xgb_link"]["fold_mean_pr_auc"], "fold_mean_without": table["xgb_v1"]["fold_mean_pr_auc"],
          "folds_where_link_wins": int((a[0] > a[1]).sum()), "test_pr_auc_refit_link": table["xgb_link"]["test_pr_auc_refit"],
          "test_pr_auc_refit_without": table["xgb_v1"]["test_pr_auc_refit"]}

results = {"baseline_fold_mean_pr_auc_patience60": BASELINE_FOLD_MEAN, "available": have, "missing": missing, "pool": table,
           "chosen_by_rolling_folds": chosen, "criterion": criterion, "e2_linkage": e2}
json.dump(results, open(f"{OUT}/results.json", "w"), indent=1)
print("\n=== PHASE 1 RESULT ===")
print(f"baseline rolling-fold mean (patience 60): {BASELINE_FOLD_MEAN}")
for k, v in table.items():
    print(f"{k:<11} fold mean {v['fold_mean_pr_auc']:.4f} | test PR-AUC train-only {v['test_pr_auc_trainonly']:.4f} refit {v['test_pr_auc_refit']:.4f} "
          f"recall@1%FPR {v['test_recall_at_fpr_1pct_refit']:.3f}")
print(f"shipped test PR-AUC {criterion['shipped_test_pr_auc']:.4f}, recall@1%FPR {criterion['shipped_test_recall_at_fpr_1pct']:.3f}")
print(f"CHOSEN {chosen}: PR-AUC diff {diff_pr:+.4f} CI {ci_pr}, recall diff {diff_rec:+.4f} CI {ci_rec} -> criterion met: {passed}")
print("E2 linkage:", e2)
