"""PayWatch: second-dataset study on the Sparkov simulated credit-card data (Kaggle, CPU).

Question: IEEE-CIS has no real user ids, so the UPI behaviour features there are built on a reconstructed proxy and added
almost nothing (PR-AUC -0.003 when removed, notebook 02b). Sparkov has a real card id per transaction. Do causal per-user
behaviour features, and lagged fraud history, help once the user id is real?

CAVEATS (stated up front, repeated in the README): the data is SIMULATED (1,000 customers, 800 merchants, fraud injected by
the generator's rules), so absolute scores are not comparable to real traffic and say little by themselves. Only the paired
ablation below is informative. Profile attributes (age, gender, city population, home-to-merchant distance) are constant per
user and identical in every feature set, so differences between sets isolate the history features; with only 1,000 users a
model can memorise users from them, which inflates all three sets equally.

Splits (time ordered, nothing random): fraudTrain is cut by time into train (first 85%) and validation (last 15%);
fraudTest (the dataset's own later period) is the held-out test and is read only for the numbers reported.
Feature sets:  A = transaction + profile fields (no history) | B = A + causal per-user behaviour state |
               C = B + lagged (7-day label delay) fraud history per card and per merchant.

PRE-REGISTERED criteria (written before any run; evaluated on test with a paired bootstrap of the test rows, 200 resamples):
  C1  B beats A by >= 0.01 PR-AUC AND the 95% CI of (B - A) excludes 0   -> "behaviour features help with real user ids".
  C2  C beats B by >= 0.01 PR-AUC AND the 95% CI of (C - B) excludes 0   -> "lagged fraud history helps".
Failing a criterion is reported as such; no criterion or feature set is changed after seeing test results.
Models: XGBoost, identical settings for all sets, 3 seeds averaged, early stopping on validation only.
"""

import glob
import json
import os
import time
from collections import deque

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

T0 = time.time()
OUT = "/kaggle/working"
SEEDS = [42, 1, 2]
N_BOOT = 200
find = lambda p: sorted(glob.glob(p, recursive=True))  # noqa: E731


def log(msg):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {msg}", flush=True)


DATA = os.path.dirname(find("/kaggle/input/**/fraudTrain.csv")[0])
raw = pd.concat([pd.read_csv(f"{DATA}/fraudTrain.csv").assign(is_test=0), pd.read_csv(f"{DATA}/fraudTest.csv").assign(is_test=1)],
                ignore_index=True)
raw = raw.sort_values(["unix_time", "trans_num"]).reset_index(drop=True)
n = len(raw)
test_mask = raw["is_test"].to_numpy() == 1
pre_pos = np.where(~test_mask)[0]
cut = pre_pos[int(len(pre_pos) * 0.85)]
split = np.where(test_mask, 2, np.where(np.arange(n) < cut, 0, 1))  # 0 train, 1 validation, 2 test
y = raw["is_fraud"].to_numpy().astype(int)
t = raw["unix_time"].to_numpy(dtype=float)
assert t[split == 0].max() <= t[split == 1].min() and t[split == 1].max() <= t[split == 2].min(), "splits overlap in time"
log(f"rows {n:,} | train {int((split == 0).sum()):,} val {int((split == 1).sum()):,} test {int((split == 2).sum()):,} | "
    f"fraud rate train {y[split == 0].mean():.4f} val {y[split == 1].mean():.4f} test {y[split == 2].mean():.4f} | "
    f"cards {raw['cc_num'].nunique()} merchants {raw['merchant'].nunique()}")

# ------------------------------------------------------------------ static (set A) features
dt = pd.to_datetime(raw["trans_date_trans_time"])
hour = dt.dt.hour.to_numpy()
dow = dt.dt.dayofweek.to_numpy()
age = (t - (pd.to_datetime(raw["dob"]) - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy()) / (365.25 * 86400)
lat1, lon1, lat2, lon2 = (np.radians(raw[c].to_numpy(dtype=float)) for c in ("lat", "long", "merch_lat", "merch_long"))
a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
dist_km = 6371.0 * 2 * np.arcsin(np.sqrt(a))
cat_code = pd.factorize(raw["category"])[0]
merch_code = pd.factorize(raw["merchant"])[0]
card_code = pd.factorize(raw["cc_num"])[0]
A = {"amt": raw["amt"].to_numpy(dtype=float), "hour": hour, "dow": dow, "category": cat_code, "dist_km": dist_km, "age": age,
     "gender": (raw["gender"] == "M").to_numpy().astype(int), "city_pop": raw["city_pop"].to_numpy(dtype=float)}


# ------------------------------------------------------------------ causal per-user behaviour state (set B)
def behaviour_features():
    amt = A["amt"]
    out = {k: np.full(n, -1.0) for k in ("prior_count", "amt_mean", "amt_std", "amt_z", "secs_since_prev", "amt_vs_7d", "new_merchant",
                                          "cat_entropy", "hour_dev", "new_category", "tx_24h")}
    state = {}
    for i in range(n):
        c, ti, ai, h, m, k = card_code[i], t[i], amt[i], hour[i], merch_code[i], cat_code[i]
        s = state.get(c)
        if s is None:
            s = {"n": 0, "mean": 0.0, "m2": 0.0, "last": None, "hsum": 0.0, "cats": {}, "merch": set(), "w7": deque(), "s7": 0.0, "w1": deque()}
            state[c] = s
        out["prior_count"][i] = s["n"]
        if s["n"] > 0:
            sd = (s["m2"] / s["n"]) ** 0.5 if s["n"] > 1 else 0.0
            out["amt_mean"][i], out["amt_std"][i] = s["mean"], sd
            out["amt_z"][i] = (ai - s["mean"]) / sd if sd > 0 else 0.0
            out["secs_since_prev"][i] = ti - s["last"]
            out["hour_dev"][i] = abs(h - s["hsum"] / s["n"])
            tot = sum(s["cats"].values())
            p = np.fromiter(s["cats"].values(), dtype=float) / tot
            out["cat_entropy"][i] = float(-(p * np.log2(p)).sum())
            out["new_merchant"][i] = 0.0 if m in s["merch"] else 1.0
            out["new_category"][i] = 0.0 if k in s["cats"] else 1.0
        while s["w7"] and ti - s["w7"][0][0] > 7 * 86400:
            s["s7"] -= s["w7"].popleft()[1]
        while s["w1"] and ti - s["w1"][0] > 86400:
            s["w1"].popleft()
        out["amt_vs_7d"][i] = ai / (s["s7"] / len(s["w7"])) if s["w7"] and s["s7"] > 0 else 1.0
        out["tx_24h"][i] = len(s["w1"])
        # update with the current transaction AFTER its features are fixed
        s["n"] += 1
        d = ai - s["mean"]
        s["mean"] += d / s["n"]
        s["m2"] += d * (ai - s["mean"])
        s["last"] = ti
        s["hsum"] += h
        s["cats"][k] = s["cats"].get(k, 0) + 1
        s["merch"].add(m)
        s["w7"].append((ti, ai))
        s["s7"] += ai
        s["w1"].append(ti)
    return out


# ------------------------------------------------------------------ lagged fraud history (set C)
PRIOR_RATE, PRIOR_WEIGHT, LAG = float(y[split == 0].mean()), 20.0, 7 * 86400.0


def lagged_history(keys):
    seen, frauds = {}, {}
    n_out, f_out = np.zeros(n), np.zeros(n)
    r_out = np.full(n, PRIOR_RATE)
    j = 0
    for i in range(n):
        while j < n and t[j] <= t[i] - LAG:
            kj = keys[j]
            seen[kj] = seen.get(kj, 0) + 1
            frauds[kj] = frauds.get(kj, 0) + y[j]
            j += 1
        ki = keys[i]
        if ki in seen:
            n_out[i], f_out[i] = seen[ki], frauds[ki]
            r_out[i] = (frauds[ki] + PRIOR_WEIGHT * PRIOR_RATE) / (seen[ki] + PRIOR_WEIGHT)
    return n_out, f_out, r_out


B = behaviour_features()
log("behaviour state built")
C = {}
for name, keys in (("card", card_code), ("merchant", merch_code)):
    C[f"h7_{name}_n"], C[f"h7_{name}_fraud_n"], C[f"h7_{name}_rate"] = lagged_history(keys)
log("lagged fraud history built")

SETS = {"A_profile_plus_transaction": A, "B_plus_behaviour": {**A, **B}, "C_plus_fraud_history": {**A, **B, **C}}
mats = {k: np.column_stack([np.asarray(v, dtype=np.float32) for v in d.values()]) for k, d in SETS.items()}
names = {k: list(d) for k, d in SETS.items()}
log("feature sets: " + ", ".join(f"{k} {m.shape[1]}" for k, m in mats.items()))

# ------------------------------------------------------------------ train / evaluate
tr, va, te = split == 0, split == 1, split == 2
spw = float((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1))


def recall_at_fpr(yt, p, f):
    fpr, tpr, _ = roc_curve(yt, p)
    return float(np.interp(f, fpr, tpr))


def op_threshold(score, yy, target, n_min=50):
    order = np.argsort(-score)
    tp = np.cumsum(yy[order])
    k = np.arange(1, len(yy) + 1)
    ok = np.where((tp / k >= target) & (k >= n_min))[0]
    return float(score[order][ok.max()]) if len(ok) else None


scores, results = {}, {}
for key, X in mats.items():
    pt, pv, iters = [], [], []
    for seed in SEEDS:
        m = xgb.XGBClassifier(n_estimators=1500, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                              tree_method="hist", eval_metric="aucpr", early_stopping_rounds=50, random_state=seed, n_jobs=-1)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
        pt.append(m.predict_proba(X[te])[:, 1])
        pv.append(m.predict_proba(X[va])[:, 1])
        iters.append(int(m.best_iteration))
    st, sv = np.mean(pt, axis=0), np.mean(pv, axis=0)
    scores[key] = st
    thr = op_threshold(sv, y[va], 0.90)
    flag = st >= thr if thr is not None else np.zeros(len(st), bool)
    results[key] = {
        "n_features": int(X.shape[1]), "best_iters": iters,
        "val_pr_auc": float(average_precision_score(y[va], sv)),
        "test_pr_auc": float(average_precision_score(y[te], st)), "test_roc_auc": float(roc_auc_score(y[te], st)),
        "test_recall_at_fpr_0.1pct": recall_at_fpr(y[te], st, 0.001), "test_recall_at_fpr_1pct": recall_at_fpr(y[te], st, 0.01),
        "val_chosen_threshold_precision_target_0.9": {"alert_rate": float(flag.mean()),
                                                       "precision": float(y[te][flag].mean()) if flag.any() else None,
                                                       "recall": float(flag[y[te] == 1].mean())},
    }
    log(f"{key}: test PR-AUC {results[key]['test_pr_auc']:.4f} ROC {results[key]['test_roc_auc']:.4f} "
        f"recall@1%FPR {results[key]['test_recall_at_fpr_1pct']:.3f} iters {iters}")

# ------------------------------------------------------------------ paired bootstrap on test rows
yt = y[te]
rng = np.random.default_rng(0)
keys = list(scores)
diffs = {"B_minus_A": [], "C_minus_B": [], "C_minus_A": []}
for _ in range(N_BOOT):
    idx = rng.integers(0, len(yt), len(yt))
    ap = {k: average_precision_score(yt[idx], scores[k][idx]) for k in keys}
    diffs["B_minus_A"].append(ap[keys[1]] - ap[keys[0]])
    diffs["C_minus_B"].append(ap[keys[2]] - ap[keys[1]])
    diffs["C_minus_A"].append(ap[keys[2]] - ap[keys[0]])
boot = {k: {"mean": float(np.mean(v)), "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]} for k, v in diffs.items()}
pr = {k: results[k]["test_pr_auc"] for k in keys}
c1 = bool(pr[keys[1]] - pr[keys[0]] >= 0.01 and boot["B_minus_A"]["ci95"][0] > 0)
c2 = bool(pr[keys[2]] - pr[keys[1]] >= 0.01 and boot["C_minus_B"]["ci95"][0] > 0)
summary = {"dataset": "kartik2112/fraud-detection (Sparkov simulated credit-card transactions, CC0)",
           "caveat": "simulated data: 1,000 customers, 800 merchants, rule-generated fraud; absolute scores are not real-world evidence",
           "split_rows": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())},
           "fraud_rate": {"train": float(y[tr].mean()), "val": float(y[va].mean()), "test": float(y[te].mean())},
           "feature_names": names, "sets": results, "paired_bootstrap_test_pr_auc_diff": boot,
           "criteria": {"C1_behaviour_helps": c1, "C2_fraud_history_helps": c2}}
json.dump(summary, open(f"{OUT}/results.json", "w"), indent=1)
print("\n=== SPARKOV STUDY (test period, simulated data) ===")
for k in keys:
    r = results[k]
    print(f"{k:<28} feats {r['n_features']:>2}  PR-AUC {r['test_pr_auc']:.4f}  ROC {r['test_roc_auc']:.4f}  recall@1%FPR {r['test_recall_at_fpr_1pct']:.3f}")
for k, v in boot.items():
    print(f"{k}: {v['mean']:+.4f}  95% CI [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")
print("pre-registered C1 (behaviour helps):", c1, "| C2 (fraud history helps):", c2)
log("done")
