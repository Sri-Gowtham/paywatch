"""PayWatch — adversarial stress test (Kaggle, CPU).

Black-box evasion test against the production compact model (3 seeds).
Attacker model: can probe the scorer (query budget 1 / 5 / 20) and change only fields they control; each
"move" copies a legitimate donor row's values for the feature group that field drives (including the derived
state it affects). Targets: test-period frauds the model currently flags (calibrated p >= SOFT_FLAG threshold).
Measures evasion rate per move and combined, then asks the roadmap question honestly: would an Isolation
Forest second layer catch what the model misses?
"""

import glob
import json
import os
import sys
import time

import numpy as np
import xgboost as xgb
from sklearn.ensemble import IsolationForest

T0 = time.time()
find = lambda p: sorted(glob.glob(p, recursive=True))  # noqa: E731
OUT = "/kaggle/working"
BUDGETS = (1, 5, 20)
N_TARGETS = 1500


def log(msg):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {msg}", flush=True)


src_dir = os.path.dirname(find("/kaggle/input/**/upi_fingerprint.py")[0])
sys.path.insert(0, src_dir)
import engineer  # noqa: E402

sys.path.insert(0, os.path.dirname(find("/kaggle/input/**/evasion.py")[0]))
import evasion  # noqa: E402
from entity_history import BASE_KEYS  # noqa: E402

RAW = os.path.dirname(find("/kaggle/input/**/train_transaction.csv")[0])
spec = json.load(open(find("/kaggle/input/**/prod_compact/serving_spec.json")[0]))
iso = json.load(open(find("/kaggle/input/**/isotonic.json")[0]))
cols = spec["features"]
boosters = []
for p in find("/kaggle/input/**/prod_compact/xgb_seed*.json"):
    b = xgb.Booster()
    b.load_model(p)
    boosters.append(b)
thr = {k: spec["tiers"][k]["calibrated_threshold"] for k in ("0.5", "0.7", "0.9")}
SOFT, CHALLENGE, BLOCK = thr["0.5"], thr["0.7"], thr["0.9"]
log(f"features={len(cols)} models={len(boosters)} thresholds soft/challenge/block = {SOFT:.3f}/{CHALLENGE:.3f}/{BLOCK:.3f}")

base = engineer.build_base(engineer.load_raw(RAW), lags=(7,), decay_lags=(), velocity=False, key_names=BASE_KEYS)
(X_tr, y_tr, _), (_, _, _), (X_te, y_te, _) = engineer.finalize(base, d_mode="anchor")
del base
Xtr = X_tr[cols].to_numpy(dtype=np.float32)
Xte = X_te[cols].to_numpy(dtype=np.float32)


def calibrated(X):
    raw = np.mean([bst.predict(xgb.DMatrix(X, feature_names=cols)) for bst in boosters], axis=0)
    return np.interp(raw, iso["x"], iso["y"]), raw


cal_te, raw_te = calibrated(Xte)
detected = np.where((y_te == 1) & (cal_te >= SOFT))[0]
fraud_all = int((y_te == 1).sum())
log(f"test frauds={fraud_all}; flagged at SOFT={len(detected)} ({len(detected) / fraud_all:.1%}); "
    f"at CHALLENGE={int(((y_te == 1) & (cal_te >= CHALLENGE)).sum())}; at BLOCK={int(((y_te == 1) & (cal_te >= BLOCK)).sum())}")
rng = np.random.default_rng(0)
targets = rng.choice(detected, min(N_TARGETS, len(detected)), replace=False)
legit_pool = Xtr[y_tr == 0]
donors_all = legit_pool[rng.choice(len(legit_pool), 20000, replace=False)]


GROUPS = evasion.build_groups(cols)
COMBOS = {
    "identity+amount": ["identity_rotation", "amount_mimic"],
    "all_moves": list(GROUPS),
}
log("groups: " + ", ".join(f"{k}(copy {len(v['copy'])}, set {len(v['set'])})" for k, v in GROUPS.items()))


def attack(group_names):
    """best (lowest) calibrated score reached by an adaptive attacker per fraud, for each query budget"""
    return evasion.attack(lambda X: calibrated(X)[0], Xte[targets], cols, GROUPS, group_names, donors_all, rng, BUDGETS)


def rates(c):
    return {"below_soft": float((c < SOFT).mean()), "below_challenge": float((c < CHALLENGE).mean()),
            "below_block": float((c < BLOCK).mean()), "mean_score": float(c.mean())}


results = {"targets": int(len(targets)), "thresholds": {"soft": SOFT, "challenge": CHALLENGE, "block": BLOCK},
           "baseline_flagged_share_of_all_fraud": len(detected) / fraud_all, "attacks": {}}
base_cal = cal_te[targets]
results["baseline_targets"] = {"mean_score": float(base_cal.mean()), "challenge_or_block_share": float((base_cal >= CHALLENGE).mean()),
                               "block_share": float((base_cal >= BLOCK).mean())}
best_all = None
for name, groups in {**{g: [g] for g in GROUPS}, **COMBOS}.items():
    res = attack(groups)
    results["attacks"][name] = {f"budget_{q}": rates(c) for q, c in res.items()}
    if name == "all_moves":
        best_all = res[max(BUDGETS)]
    log(f"attack {name:<18} evasion below SOFT @budget1/5/20 = "
        + " / ".join(f"{(res[q] < SOFT).mean():.1%}" for q in BUDGETS))

# ------------------------------------------------ would an Isolation Forest second layer help?
Xfit = legit_pool[rng.choice(len(legit_pool), 100000, replace=False)]
forest = IsolationForest(n_estimators=200, max_samples=2048, random_state=0, n_jobs=-1).fit(Xfit)
legit_te = Xte[y_te == 0]
if_legit = -forest.score_samples(legit_te)
if_thr = float(np.quantile(if_legit, 0.99))                      # 1% false-positive rate on legit traffic
if_fraud_orig = -forest.score_samples(Xte[targets])
# rebuild the all_moves best-attacked vectors to score them with the forest (same donor draws are not stored; re-draw)
donors = donors_all[rng.integers(0, len(donors_all), len(targets))]
Xa_all = evasion.apply_moves(Xte[targets], cols, GROUPS, list(GROUPS), donors)
if_fraud_att = -forest.score_samples(Xa_all)
cal_att, _ = calibrated(Xa_all)
evaded = cal_att < SOFT
results["isolation_forest"] = {
    "if_threshold_at_1pct_fpr": if_thr,
    "if_flags_original_flagged_frauds": float((if_fraud_orig >= if_thr).mean()),
    "if_flags_all_moves_attacked_frauds": float((if_fraud_att >= if_thr).mean()),
    "evaded_frauds_(single_draw_all_moves)": int(evaded.sum()),
    "if_flags_among_evaded": float((if_fraud_att[evaded] >= if_thr).mean()) if evaded.any() else None,
    "base_rate_of_if_flag_on_legit": 0.01,
}
# overall effect of an OR-rule on the full test (no attack): recall and FPR
flag_model = cal_te >= SOFT
flag_if = (-forest.score_samples(Xte)) >= if_thr
results["isolation_forest"]["full_test_no_attack"] = {
    "model_alone": {"recall": float(flag_model[y_te == 1].mean()), "fpr": float(flag_model[y_te == 0].mean())},
    "model_or_if": {"recall": float((flag_model | flag_if)[y_te == 1].mean()), "fpr": float((flag_model | flag_if)[y_te == 0].mean())},
}
log(f"IF: flags original {results['isolation_forest']['if_flags_original_flagged_frauds']:.1%}, "
    f"flags all-moves-attacked {results['isolation_forest']['if_flags_all_moves_attacked_frauds']:.1%}, among evaded "
    f"{results['isolation_forest']['if_flags_among_evaded']}")
json.dump(results, open(f"{OUT}/results.json", "w"), indent=1)

print("\n=== ADVERSARIAL STRESS TEST ===")
print(f"targets: {results['targets']} frauds the model flags (SOFT_FLAG or higher); model flags "
      f"{results['baseline_flagged_share_of_all_fraud']:.1%} of all fraud at SOFT")
print(f"{'attack':<20}" + "".join(f"{'budget ' + str(q):>22}" for q in BUDGETS) + "   (share pushed BELOW soft / challenge / block)")
for name, a in results["attacks"].items():
    cells = "".join(f"{a[f'budget_{q}']['below_soft']:>8.1%}/{a[f'budget_{q}']['below_challenge']:>6.1%}/{a[f'budget_{q}']['below_block']:>6.1%}" for q in BUDGETS)
    print(f"{name:<20}{cells}")
print("\nIsolation Forest second layer:")
print(json.dumps(results["isolation_forest"], indent=1))
log("done")
