"""PayWatch: build and EXECUTE the roadmap notebooks 02, 02b, 03, 04 and 04b on Kaggle (GPU, internet for nbclient).

Each notebook is generated with nbformat and executed with nbclient on the raw IEEE-CIS data, so every number and
plot is computed when the kernel runs. Notebooks show aggregates only: no raw transaction rows are ever displayed.
Output: /kaggle/working/{02_features,02b_upi_features,03_model_xgb,04_shap,04b_adversarial_test}.ipynb + notebooks_run.json
"""

import json
import os
import subprocess
import sys
import time
import traceback

try:
    import nbclient  # noqa: F401
    import nbformat  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "nbformat", "nbclient", "ipykernel"], check=False)

import nbformat  # noqa: E402
from nbclient import NotebookClient  # noqa: E402

OUT = "/kaggle/working"
md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell

PRELUDE = '''import glob, json, os, re, shutil, sys, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, precision_recall_curve
warnings.filterwarnings("ignore")
%matplotlib inline
pd.set_option("display.width", 140)
T0 = time.time()
DEVICE = "cuda" if shutil.which("nvidia-smi") else "cpu"
find = lambda p: sorted(glob.glob(p, recursive=True))
sys.path.insert(0, os.path.dirname(find("/kaggle/input/**/upi_fingerprint.py")[0]))
sys.path.insert(0, os.path.dirname(find("/kaggle/input/**/evasion.py")[0]))
import engineer
from entity_history import BASE_KEYS
RAW = os.path.dirname(find("/kaggle/input/**/train_transaction.csv")[0])
SPEC = json.load(open(find("/kaggle/input/**/prod_compact/serving_spec.json")[0]))
COLS = SPEC["features"]
SHIPPED = find("/kaggle/input/**/prod_compact/xgb_seed*.json")
raw = engineer.load_raw(RAW)
base = engineer.build_base(raw.copy(), lags=(7,), decay_lags=(), velocity=False, key_names=BASE_KEYS)
(tr, va, te), art = engineer.finalize(base, d_mode="anchor", return_artifacts=True)
(X_tr, y_tr, t_tr), (X_va, y_va, t_va), (X_te, y_te, t_te) = tr, va, te
UPI_B = {"amount_vs_7day_avg_ratio", "merchant_category_entropy", "hour_deviation_from_user_mean",
         "days_since_last_large_txn", "is_new_merchant", "is_p2p"}


def family(f):
    if f.startswith("h7_"):
        return "entity fraud history"
    if f.startswith("uid_") or f in UPI_B:
        return "UPI behaviour"
    if f.endswith("_freq"):
        return "frequency encoding"
    if re.fullmatch(r"D\\d+", f):
        return "time deltas (D)"
    if re.fullmatch(r"V\\d+", f):
        return "Vesta V"
    if f.startswith("id_") or f.startswith("Device"):
        return "identity / device"
    if re.fullmatch(r"[CM]\\d+", f):
        return "counts / match flags (C, M)"
    return "raw transaction"


def recall_at_fpr(y, p, f):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.interp(f, fpr, tpr))


def op_threshold(score, y, target, n_min=100):
    order = np.argsort(-score)
    tp = np.cumsum(y[order])
    n = np.arange(1, len(y) + 1)
    ok = np.where((tp / n >= target) & (n >= n_min))[0]
    return float(score[order][ok.max()]) if len(ok) else None


def shipped_boosters():
    out = []
    for p in SHIPPED:
        b = xgb.Booster()
        b.load_model(p)
        out.append(b)
    return out


def ensemble_score(boosters, X):
    d = xgb.DMatrix(pd.DataFrame(np.asarray(X, dtype=np.float32), columns=COLS))
    return np.mean([b.predict(d) for b in boosters], axis=0)


print(f"device: {DEVICE} | pipeline built in {(time.time() - T0) / 60:.1f} min | train {X_tr.shape}, val {X_va.shape}, test {X_te.shape} | compact features: {len(COLS)}")'''

XGB_PARAMS = '''XGB_PARAMS = dict(max_depth=10, learning_rate=0.0558, subsample=0.935, colsample_bytree=0.678, min_child_weight=1,
                  reg_lambda=1.518, gamma=0.0336, scale_pos_weight=39.3)
TAU_DAYS = 120'''

NOTEBOOKS = {}

# ------------------------------------------------------------------------------------------------ 02_features
NOTEBOOKS["02_features"] = [
    md("# PayWatch: feature engineering and leakage checks\n\nThe causal pipeline in `src/features/` turns raw IEEE-CIS rows into model features. "
       "This notebook runs it on the real data and then tries to break it: time-ordered splits, train-only encodings and an "
       "independent recomputation of the lagged fraud-history features. Only aggregates are shown, never raw rows."),
    code(PRELUDE),
    md("## 1. What the pipeline produces"),
    code('''all_cols = art["feature_columns"]
fam = pd.DataFrame({"feature": all_cols, "family": [family(c) for c in all_cols]})
fam["in_compact_model"] = fam.feature.isin(COLS)
summary = fam.groupby("family").agg(features=("feature", "size"), in_compact=("in_compact_model", "sum")).sort_values("features", ascending=False)
display(summary)
print(f"raw columns: {raw.shape[1]} -> kept {len(all_cols)} model features (full) -> {len(COLS)} (compact, chosen by SHAP rank on validation)")
summary.features.plot(kind="barh", figsize=(7, 3.2), title="features per family (full set)"); plt.gca().invert_yaxis(); plt.tight_layout(); plt.show()'''),
    md("## 2. Split: strictly time-ordered 70 / 15 / 15\n\nA random split would let the model see the future of a card. Rows are sorted by `TransactionDT` and cut by position."),
    code('''spl = pd.DataFrame({"rows": [len(y_tr), len(y_va), len(y_te)], "fraud_rate": [y_tr.mean(), y_va.mean(), y_te.mean()],
                    "start_day": [t_tr.min() / 86400, t_va.min() / 86400, t_te.min() / 86400],
                    "end_day": [t_tr.max() / 86400, t_va.max() / 86400, t_te.max() / 86400]}, index=["train", "validation", "test"])
display(spl.round(4))
assert t_tr.max() <= t_va.min() and t_va.max() <= t_te.min(), "splits overlap in time"
print("check passed: every validation row is later than every train row, every test row later than every validation row")'''),
    md("## 3. Encodings fitted on train only\n\nFrequency encodings use train counts. Values never seen in train get frequency 0 on val/test instead of leaking their own counts."),
    code('''rows = []
for col in art["freq_tables"]:
    for name, key in (("validation", X_va), ("test", X_te)):
        zero = float((key[f"{col}_freq"] == 0).mean())
        rows.append({"column": col, "split": name, "share_with_frequency_0": round(zero, 4)})
display(pd.DataFrame(rows).pivot(index="column", columns="split", values="share_with_frequency_0"))
print("train frequency tables sum to 1 per column:", {c: round(sum(t.values()), 6) for c, t in list(art["freq_tables"].items())[:4]})'''),
    md("## 4. Lagged fraud history: independent recomputation\n\n"
       "For a row at time t the history features may only use rows with `TransactionDT <= t - 7 days` (the label has had time to arrive) and never the row's own label. "
       "Below, 400 random rows are recomputed from scratch with plain pandas and compared with the pipeline's `h7_card_n` and `h7_card_fraud_n`."),
    code('''from upi_fingerprint import make_card_key
b = base.sort_values("TransactionDT").reset_index(drop=True)
b["card_key"] = make_card_key(b)
rng = np.random.default_rng(0)
cand = np.where(b["h7_card_n"].to_numpy() > 0)[0]
pick = rng.choice(cand, 400, replace=False)
groups = {k: (g["TransactionDT"].to_numpy(), g["isFraud"].to_numpy()) for k, g in b[b.card_key.isin(set(b.card_key.iloc[pick]))].groupby("card_key")}
bad_n = bad_f = 0
for i in pick:
    ts, lab = groups[b.card_key.iloc[i]]
    m = ts <= b.TransactionDT.iloc[i] - 7 * 86400
    bad_n += int(m.sum() != b.h7_card_n.iloc[i])
    bad_f += int(lab[m].sum() != b.h7_card_fraud_n.iloc[i])
print(f"mismatches vs independent recomputation: count {bad_n}/400, fraud count {bad_f}/400")
assert bad_n == 0 and bad_f == 0
print("pipeline and recomputation agree; a row's own label and the last 7 days of labels are never used")'''),
    md("## 5. How informative is each feature on its own? (validation ROC-AUC, no model)"),
    code('''auc = {}
for c in COLS:
    v = X_va[c].to_numpy()
    if np.nanstd(v) > 0:
        a = roc_auc_score(y_va, v)
        auc[c] = max(a, 1 - a)
s = pd.Series(auc).sort_values(ascending=False)
top = s.head(20)
colors = ["#4cc2a6" if family(f) in ("entity fraud history", "UPI behaviour") else "#6c8ebf" for f in top.index]
top[::-1].plot(kind="barh", figsize=(7, 6), color=colors[::-1], title="top 20 single-feature AUCs (green = engineered history / UPI)")
plt.xlim(0.5, top.max() + 0.02); plt.tight_layout(); plt.show()
print(pd.DataFrame({"family": [family(f) for f in s.index], "auc": s.values}).groupby("family").auc.agg(["count", "max", "mean"]).round(3))'''),
    md("## Takeaways\n\n- Splits are time-ordered and the encodings are train-only; the label-lag logic reproduces exactly under an independent recomputation.\n"
       "- Engineered entity-history and behaviour features carry real single-feature signal, but no feature alone is close to the final model: the gain is in combining them.\n"
       "- The user id is a reconstructed proxy (card fields + address + account-age anchor), not a real UPI id."),
]

# ------------------------------------------------------------------------------------------------ 02b_upi_features
NOTEBOOKS["02b_upi_features"] = [
    md("# PayWatch: UPI behaviour features\n\nIEEE-CIS has no UPI ids. `src/features/upi_fingerprint.py` rebuilds a user proxy (`uid`) and computes causal behaviour features "
       "from that user's earlier transactions only. This notebook inspects them and measures what they add. Aggregates only."),
    code(PRELUDE),
    code(XGB_PARAMS),
    md("## 1. The user proxy"),
    code('''from upi_fingerprint import make_card_key, make_uid
n_card = len(set(make_card_key(base)))
n_uid = len(set(make_uid(base)))
print(f"transactions: {len(base):,} | distinct card composites: {n_card:,} | distinct uid proxies: {n_uid:,}")
print(f"first-ever transaction of its uid proxy (no behaviour history yet): train {(X_tr.uid_prior_tx_count == 0).mean():.1%}, test {(X_te.uid_prior_tx_count == 0).mean():.1%}")
print("caveat: uid is a reconstruction, so some real users are split across several uids and some uids merge users")'''),
    md("## 2. The behaviour features"),
    code('''UPI = [c for c in ["amount_vs_7day_avg_ratio", "merchant_category_entropy", "hour_deviation_from_user_mean",
                   "days_since_last_large_txn", "is_new_merchant", "is_p2p", "uid_amt_zscore", "uid_prior_tx_count", "uid_secs_since_prev_tx"] if c in X_va]
rows = []
for c in UPI:
    v = X_va[c].to_numpy()
    a = roc_auc_score(y_va, v) if np.nanstd(v) > 0 else np.nan
    rows.append({"feature": c, "val_auc": round(a, 3), "direction": "higher = riskier" if a >= 0.5 else "lower = riskier",
                 "median_legit": round(float(np.median(v[y_va == 0])), 3), "median_fraud": round(float(np.median(v[y_va == 1])), 3)})
display(pd.DataFrame(rows))
fig, ax = plt.subplots(1, 3, figsize=(15, 3.5))
for a, c in zip(ax, ["amount_vs_7day_avg_ratio", "uid_amt_zscore", "hour_deviation_from_user_mean"]):
    v = X_va[c].to_numpy()
    lo, hi = np.percentile(v, [1, 99])
    for lab, name in ((0, "legit"), (1, "fraud")):
        a.hist(np.clip(v[y_va == lab], lo, hi), bins=50, alpha=0.5, density=True, label=name)
    a.set_title(c); a.legend()
plt.tight_layout(); plt.show()'''),
    md("## 3. First-time versus known users\n\nBehaviour features need history. For a user's first transaction they carry no information, which is the hardest case for the whole system."),
    code('''first = (X_te.uid_prior_tx_count == 0).to_numpy()
t = pd.DataFrame({"share_of_test_rows": [first.mean(), 1 - first.mean()], "fraud_rate": [y_te[first].mean(), y_te[~first].mean()],
                  "share_of_all_fraud": [y_te[first].sum() / y_te.sum(), y_te[~first].sum() / y_te.sum()]}, index=["first-time uid", "known uid"])
display(t.round(4))'''),
    md("## 4. What do they add? A small ablation\n\nOne seed each, validation early stopping, test PR-AUC. This is indicative, not the production comparison (see `results/pipeline.json` for the 5-seed study)."),
    code('''def fit_eval(cols, seed=42):
    w = np.exp(-(t_tr.max() - t_tr) / (TAU_DAYS * 86400.0))
    params = dict(XGB_PARAMS) if DEVICE == "cuda" else {**XGB_PARAMS, "max_depth": 8, "learning_rate": 0.1}
    m = xgb.XGBClassifier(n_estimators=3000 if DEVICE == "cuda" else 600, tree_method="hist", device=DEVICE, eval_metric="aucpr",
                          early_stopping_rounds=50, random_state=seed, n_jobs=-1, **params)
    m.fit(X_tr[cols], y_tr, sample_weight=w, eval_set=[(X_va[cols], y_va)], verbose=False)
    p = m.predict_proba(X_te[cols])[:, 1]
    return {"features": len(cols), "best_iter": int(m.best_iteration), "test_pr_auc": round(float(average_precision_score(y_te, p)), 4),
            "test_roc_auc": round(float(roc_auc_score(y_te, p)), 4), "recall@1%FPR": round(recall_at_fpr(y_te, p, 0.01), 3)}


no_upi = [c for c in COLS if family(c) != "UPI behaviour"]
no_hist = [c for c in COLS if family(c) != "entity fraud history"]
no_either = [c for c in COLS if family(c) not in ("UPI behaviour", "entity fraud history")]
res = {"compact (all)": fit_eval(COLS), "without UPI behaviour": fit_eval(no_upi), "without entity history": fit_eval(no_hist),
       "without both": fit_eval(no_either)}
abl = pd.DataFrame(res).T
display(abl)
abl.test_pr_auc.plot(kind="barh", figsize=(7, 2.8), title="test PR-AUC by feature set (1 seed)"); plt.tight_layout(); plt.show()'''),
    code('''full = abl.test_pr_auc["compact (all)"]
print(f"removing UPI behaviour features: PR-AUC {abl.test_pr_auc['without UPI behaviour'] - full:+.4f}")
print(f"removing entity fraud history:   PR-AUC {abl.test_pr_auc['without entity history'] - full:+.4f}")
print(f"removing both:                   PR-AUC {abl.test_pr_auc['without both'] - full:+.4f}")
print("single seed, so differences of a few thousandths are noise; the size of each effect is what matters")'''),
    md("## Reading the results\n\nThe behaviour features are causal and reconstructed from a proxy id, and the ablation above measures what they add instead of assuming it. "
       "First-time users, where none of them can help, are a large share of test fraud, which is why recall on new identities is the known weak spot."),
]

# ------------------------------------------------------------------------------------------------ 03_model_xgb
NOTEBOOKS["03_model_xgb"] = [
    md("# PayWatch: the XGBoost model, calibration and tiers\n\nTrains three seeds from scratch on the TRAIN split only with the production settings, so every threshold and calibration map below "
       "comes from validation scores of models that never trained on validation rows. Test is only read for the reported numbers. "
       "The shipped production model was refit on train + validation, so it is compared on test only and is not used to pick thresholds here."),
    code(PRELUDE),
    code(XGB_PARAMS),
    md("## 1. Train three seeds (recency weights, early stopping on validation)"),
    code('''w_tr = np.exp(-(t_tr.max() - t_tr) / (TAU_DAYS * 86400.0))
models = []
for seed in (42, 1, 2):
    m = xgb.XGBClassifier(n_estimators=6000, tree_method="hist", device=DEVICE, eval_metric="aucpr", early_stopping_rounds=60,
                          random_state=seed, n_jobs=-1, **XGB_PARAMS)
    t1 = time.time()
    m.fit(X_tr[COLS], y_tr, sample_weight=w_tr, eval_set=[(X_va[COLS], y_va)], verbose=False)
    models.append(m)
    print(f"seed {seed}: {time.time() - t1:.0f}s, best iteration {m.best_iteration}")
print(f"scale_pos_weight {XGB_PARAMS['scale_pos_weight']} (class weights instead of SMOTE), device {DEVICE}")
pv = [m.predict_proba(X_va[COLS])[:, 1] for m in models]
pt = [m.predict_proba(X_te[COLS])[:, 1] for m in models]
s_va, s_te = np.mean(pv, axis=0), np.mean(pt, axis=0)
ship = shipped_boosters()
ship_te = ensemble_score(ship, X_te[COLS])
rows = []
for name, p in [(f"seed {sd} alone", q) for sd, q in zip((42, 1, 2), pt)] + [("3-seed ensemble (train only)", s_te),
                                                                                ("shipped ensemble (refit on train+val)", ship_te)]:
    rows.append({"model": name, "test_pr_auc": round(float(average_precision_score(y_te, p)), 4), "test_roc_auc": round(float(roc_auc_score(y_te, p)), 4),
                 "recall@0.1%FPR": round(recall_at_fpr(y_te, p, 0.001), 3), "recall@1%FPR": round(recall_at_fpr(y_te, p, 0.01), 3),
                 "recall@5%FPR": round(recall_at_fpr(y_te, p, 0.05), 3)})
res = pd.DataFrame(rows).set_index("model")
display(res)
singles = res.test_pr_auc.iloc[:3]
print(f"single seeds: PR-AUC {singles.min():.3f} to {singles.max():.3f}; train-only ensemble {res.test_pr_auc.iloc[3]:.3f}; "
      f"shipped refit ensemble {res.test_pr_auc.iloc[4]:.3f} (refit adds the validation period as training data)")
print(f"reference: a random scorer has PR-AUC = fraud rate = {y_te.mean():.4f}")'''),
    md("## 2. Curves (train-only ensemble versus the shipped refit)"),
    code('''fig, ax = plt.subplots(1, 2, figsize=(13, 4))
for name, p in (("train-only 3-seed ensemble", s_te), ("shipped refit ensemble", ship_te)):
    pr, rc, _ = precision_recall_curve(y_te, p)
    ax[0].plot(rc, pr, label=f"{name} (PR-AUC {average_precision_score(y_te, p):.3f})")
    fpr, tpr, _ = roc_curve(y_te, p)
    ax[1].plot(fpr, tpr, label=name)
ax[0].set_xlabel("recall"); ax[0].set_ylabel("precision"); ax[0].set_title("precision-recall, test period"); ax[0].legend()
ax[1].set_xscale("log"); ax[1].set_xlim(1e-4, 1); ax[1].set_xlabel("false-positive rate (log)"); ax[1].set_ylabel("recall"); ax[1].set_title("ROC, test period"); ax[1].legend()
plt.tight_layout(); plt.show()'''),
    md("## 3. Calibration (fitted on validation scores of the train-only ensemble)\n\nRaw scores are not probabilities (class weights inflate them). An isotonic map turns them into calibrated probabilities."),
    code('''from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
iso = IsotonicRegression(out_of_bounds="clip").fit(s_va, y_va)
c_te = iso.predict(s_te)
print(f"Brier score on test: raw {brier_score_loss(y_te, s_te):.4f}  ->  calibrated {brier_score_loss(y_te, c_te):.4f}")
bins = pd.qcut(pd.Series(c_te).rank(method="first"), 10, labels=False)
rel = pd.DataFrame({"predicted": pd.Series(c_te).groupby(bins).mean(), "observed": pd.Series(y_te).groupby(bins).mean()})
rel_raw = pd.DataFrame({"predicted": pd.Series(s_te).groupby(bins).mean(), "observed": pd.Series(y_te).groupby(bins).mean()})
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for a, r, t in ((ax[0], rel_raw, "raw score"), (ax[1], rel, "isotonic-calibrated")):
    a.plot([0, 1], [0, 1], "--", c="grey"); a.plot(r.predicted, r.observed, "o-"); a.set_title(f"reliability by score decile: {t}")
    a.set_xlabel("predicted"); a.set_ylabel("observed fraud rate")
plt.tight_layout(); plt.show()'''),
    md("## 4. Action tiers by precision target (thresholds from validation, applied to test)\n\n"
       "Production picks thresholds on the first half of the test period and reports the second half (`models/compact/serving_spec.json`); this notebook uses the simpler validation-to-test protocol."),
    code('''rows = []
for tier, target in (("SOFT_FLAG", 0.5), ("CHALLENGE", 0.7), ("HARD_BLOCK", 0.9)):
    thr = op_threshold(s_va, y_va, target)
    flag = s_te >= thr
    rows.append({"tier": tier, "val_precision_target": target, "alert_rate_test": round(float(flag.mean()), 4),
                 "test_precision": round(float(y_te[flag].mean()), 3), "test_recall": round(float(flag[y_te == 1].mean()), 3)})
tiers = pd.DataFrame(rows).set_index("tier")
display(tiers)
gap = (tiers.val_precision_target - tiers.test_precision).round(3)
print("target minus achieved test precision per tier:", gap.to_dict())
print("a positive gap means the validation period was easier than the test period; that gap is the honest estimate of what operators will see")'''),
    md("## Reading the results\n\nEverything above is computed: compare the single-seed rows, the train-only ensemble and the shipped refit model in section 1 to see what seed averaging and the extra training period are each worth. "
       "Precision of 0.99 on all alerts is not reachable on this data; only the very top slice is that precise (see the README)."),
]

# ------------------------------------------------------------------------------------------------ 04_shap
NOTEBOOKS["04_shap"] = [
    md("# PayWatch: SHAP explanations\n\nThe API returns the top reasons for each decision from XGBoost's exact tree SHAP values (`pred_contribs`). "
       "This notebook checks that the contributions are exact and shows which features and families drive the shipped model. "
       "Individual rows are never shown, only feature names and contribution sizes."),
    code(PRELUDE),
    md("## 1. Exactness: contributions must sum to the model output"),
    code('''ship = shipped_boosters()
rng = np.random.default_rng(0)
idx = np.sort(rng.choice(len(X_te), 5000, replace=False))
Xs = pd.DataFrame(X_te[COLS].to_numpy(dtype=np.float32)[idx], columns=COLS)
d = xgb.DMatrix(Xs)
contribs = [b.predict(d, pred_contribs=True) for b in ship]
margins = [b.predict(d, output_margin=True) for b in ship]
err = max(float(np.abs(c.sum(axis=1) - m_).max()) for c, m_ in zip(contribs, margins))
print(f"max |sum of SHAP values + bias - model margin| over 3 models x 5000 rows: {err:.2e}")
assert err < 1e-3
shap = np.mean([c[:, :-1] for c in contribs], axis=0)'''),
    md("## 2. What drives the model"),
    code('''imp = pd.Series(np.abs(shap).mean(axis=0), index=COLS).sort_values(ascending=False)
colors = ["#4cc2a6" if family(f) in ("entity fraud history", "UPI behaviour") else "#6c8ebf" for f in imp.head(20).index]
imp.head(20)[::-1].plot(kind="barh", figsize=(7, 6), color=colors[::-1], title="mean |SHAP| (log-odds), top 20 (green = engineered)")
plt.tight_layout(); plt.show()
fam = imp.groupby([family(f) for f in imp.index]).sum().sort_values(ascending=False)
(fam / fam.sum()).plot(kind="barh", figsize=(7, 3.2), title="share of total |SHAP| by feature family"); plt.gca().invert_yaxis(); plt.tight_layout(); plt.show()
display((fam / fam.sum()).round(3).to_frame("share"))'''),
    md("## 3. Direction of effect\n\nCorrelation between a feature's value and its SHAP value tells whether higher values push towards fraud (+) or away (-) on this sample."),
    code('''vals = Xs.to_numpy()
rows = []
for c in imp.head(15).index:
    j = COLS.index(c)
    v, s = vals[:, j], shap[:, j]
    r = np.corrcoef(v, s)[0, 1] if v.std() > 0 and s.std() > 0 else np.nan
    rows.append({"feature": c, "family": family(c), "mean_abs_shap": round(float(np.abs(s).mean()), 4), "corr(value, shap)": round(float(r), 2)})
display(pd.DataFrame(rows))'''),
    md("## 4. Reason lists the way the API shows them\n\nFor three confidently flagged test frauds, the top positive contributions (feature name and SHAP size only)."),
    code('''score = ensemble_score(ship, Xs)
flagged = np.where((y_te[idx] == 1) & (score >= np.percentile(score, 98)))[0][:3]
for k, i in enumerate(flagged, 1):
    order = np.argsort(-shap[i])[:5]
    print(f"flagged fraud {k}: " + ", ".join(f"{COLS[j]} (+{shap[i, j]:.2f})" for j in order))
print("\\nthe API additionally turns these into plain-language text per feature family")'''),
    md("## Takeaways\n\n- SHAP values add up exactly to the model output, so the reasons shown to users are the model's real arithmetic, not an approximation.\n"
       "- Engineered history and behaviour features account for a large share of total attribution; they are also the features a fresh identity resets, which is the evasion risk tested in `04b`."),
]

# ------------------------------------------------------------------------------------------------ 04b_adversarial_test
NOTEBOOKS["04b_adversarial_test"] = [
    md("# PayWatch: adversarial stress test\n\nBlack-box evasion: an attacker who can query the scorer repeatedly modifies only the fields they control "
       "(rotate identity, rotate email, mimic amount or behaviour) and keeps the lowest score. We measure how often a transaction that was flagged "
       "drops below the threshold. The attack code is `src/models/evasion.py`; the adversarial-training counter-measure is in `results/adv_training.json`."),
    code(PRELUDE),
    md("## 1. Setup: shipped scorer, production SOFT_FLAG threshold, flagged test frauds"),
    code('''import evasion
ship = shipped_boosters()
Xtr = X_tr[COLS].to_numpy(dtype=np.float32)
Xva = X_va[COLS].to_numpy(dtype=np.float32)
Xte = X_te[COLS].to_numpy(dtype=np.float32)
scorer = lambda Z: ensemble_score(ship, Z)
iso = json.load(open(find("/kaggle/input/**/isotonic.json")[0]))
cal = lambda raw_score: np.interp(raw_score, iso["x"], iso["y"])
thr_cal = SPEC["tiers"]["0.5"]["calibrated_threshold"]   # production SOFT_FLAG, chosen on the first half of the test period
sc_te = scorer(Xte)
flagged = np.where((y_te == 1) & (cal(sc_te) >= thr_cal))[0]
thr = float(thr_cal)
scorer_cal = lambda Z: cal(scorer(Z))
print(f"flag rule: calibrated probability >= {thr_cal:.4f} (the production SOFT_FLAG tier, as in results/stress_test.json)")
print(f"test frauds flagged by that rule: {len(flagged)} of {int(y_te.sum())} ({len(flagged) / y_te.sum():.1%})")
GROUPS = evasion.build_groups(COLS)
rng = np.random.default_rng(0)
legit = Xtr[y_tr == 0]
donors = legit[rng.choice(len(legit), 20000, replace=False)]
targets = rng.choice(flagged, min(800, len(flagged)), replace=False)
print({k: len(v["copy"]) + len(v["set"]) for k, v in GROUPS.items()}, "<- features touched per move")'''),
    md("## 2. Attack\n\nBudget = number of queries per transaction (each draws a random donor profile). `all_moves` combines every move."),
    code('''BUDGETS = (1, 5, 20)
res = {}
for name, moves in (("amount_mimic", ["amount_mimic"]), ("behavior_mimic", ["behavior_mimic"]), ("email_rotation", ["email_rotation"]),
                    ("identity_rotation", ["identity_rotation"]), ("all_moves", list(GROUPS))):
    best = evasion.attack(scorer_cal, Xte[targets], COLS, GROUPS, moves, donors, np.random.default_rng(1), BUDGETS)
    res[name] = {q: float((b < thr).mean()) for q, b in best.items()}
df = pd.DataFrame(res).T
df.columns = [f"{q} queries" for q in df.columns]
display((df * 100).round(1).astype(str) + " %")
(df * 100).plot(kind="bar", figsize=(9, 4), title="share of flagged frauds pushed below the flag threshold"); plt.ylabel("%"); plt.xticks(rotation=20)
plt.tight_layout(); plt.show()'''),
    md("## 3. Reading the result\n\n- Moves that only change what an attacker cannot easily fake (amount, hour) barely help them; moves that reset identity-linked history are what evade the model.\n"
       "- This is a threat model for the *scoring layer* only: real systems add velocity limits, device binding and manual review on top.\n"
       "- Adversarial training (rotated fraud copies added to the training set) was tested separately and lowered identity-rotation evasion with almost no accuracy loss; the numbers and the caveat that the attack equals the augmentation are in the README."),
]


def run_all():
    status = {}
    for name, cells in NOTEBOOKS.items():
        t0 = time.time()
        nb = nbformat.v4.new_notebook()
        nb.cells = cells
        nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
        try:
            NotebookClient(nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": OUT}}).execute()
            status[name] = {"ok": True, "minutes": round((time.time() - t0) / 60, 1)}
        except Exception as e:  # keep going so one failure does not lose the others
            traceback.print_exc()
            status[name] = {"ok": False, "error": str(e)[-1500:], "minutes": round((time.time() - t0) / 60, 1)}
        nbformat.write(nb, f"{OUT}/{name}.ipynb")
        print(f"{name}: {status[name]} size {os.path.getsize(f'{OUT}/{name}.ipynb'):,} bytes", flush=True)
        json.dump(status, open(f"{OUT}/notebooks_run.json", "w"), indent=1)
    return status


if __name__ == "__main__":
    st = run_all()
    if not all(v["ok"] for v in st.values()):
        sys.exit(1)
