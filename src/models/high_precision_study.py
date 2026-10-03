"""PayWatch — high-precision (AUTO_BLOCK) study (Kaggle, CPU).

Goal: the deepest alert threshold whose precision is >= 0.99, validated honestly.
  * production compact models (3 seeds, trained on train+val) score the TEST period
  * SELECTION on the FIRST half of the test period, REPORT on the SECOND half (time-ordered)
  * pre-registered selection rule: longest prefix of the ranked alerts, starting at 200 alerts, whose
    Clopper-Pearson 95% lower bound on precision stays >= 0.98 on the first half
  * pre-registered success criterion on the second half: precision >= 0.99 AND CP lower bound >= 0.98
Variants: A mean raw score | B min over the 3 seeds (agreement gate) | C mean logit.
"""

import glob
import json
import os
import sys
import time

import numpy as np
import xgboost as xgb
from scipy.stats import beta

T0 = time.time()
find = lambda p: sorted(glob.glob(p, recursive=True))  # noqa: E731
OUT = "/kaggle/working"
N_MIN, LOWER_TARGET, PREC_TARGET = 200, 0.98, 0.99


def log(msg):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {msg}", flush=True)


def cp_interval(tp, n, alpha=0.05):
    tp, n = np.asarray(tp, float), np.asarray(n, float)
    lo = np.where(tp > 0, beta.ppf(alpha / 2, tp, n - tp + 1), 0.0)
    hi = np.where(tp < n, beta.ppf(1 - alpha / 2, tp + 1, n - tp), 1.0)
    return lo, hi


src_dir = os.path.dirname(find("/kaggle/input/**/upi_fingerprint.py")[0])
sys.path.insert(0, src_dir)
import engineer  # noqa: E402
from entity_history import BASE_KEYS  # noqa: E402

RAW = os.path.dirname(find("/kaggle/input/**/train_transaction.csv")[0])
spec = json.load(open(find("/kaggle/input/**/prod_compact/serving_spec.json")[0]))
cols = spec["features"]
paths = find("/kaggle/input/**/prod_compact/xgb_seed*.json")
boosters = []
for p in paths:
    b = xgb.Booster()
    b.load_model(p)
    boosters.append(b)
log(f"features={len(cols)} models={len(paths)}")

base = engineer.build_base(engineer.load_raw(RAW), lags=(7,), decay_lags=(), velocity=False, key_names=BASE_KEYS)
(_, _, _), (_, _, _), (X_te, y_te, t_te) = engineer.finalize(base, d_mode="anchor")
del base
amt = X_te["TransactionAmt"].to_numpy()
uid_prior_fraud = (X_te["h7_uid_fraud_n"] > 0).to_numpy()
uid_known = (X_te["h7_uid_n"] > 0).to_numpy()
days = (t_te - t_te.min()) / 86400.0
n = len(y_te)
half = n // 2
a, b = np.arange(n) < half, np.arange(n) >= half
log(f"test rows={n} fraud={int(y_te.sum())}; first half {a.sum()} rows / {int(y_te[a].sum())} fraud, "
    f"second half {b.sum()} rows / {int(y_te[b].sum())} fraud")

P = np.vstack([bst.predict(xgb.DMatrix(X_te[cols])) for bst in boosters])       # (3, n)
eps = 1e-6
variants = {
    "A_mean_score": P.mean(axis=0),
    "B_min_over_seeds": P.min(axis=0),
    "C_mean_logit": np.log(np.clip(P, eps, 1 - eps) / (1 - np.clip(P, eps, 1 - eps))).mean(axis=0),
}
iso = json.load(open(find("/kaggle/input/**/isotonic.json")[0]))
cal = np.interp(variants["A_mean_score"], iso["x"], iso["y"])
top2 = variants["A_mean_score"] >= np.quantile(variants["A_mean_score"], 0.98)
saturated = float((cal[top2] >= 0.999).mean())
log(f"share of the top-2% rows whose isotonic probability is >= 0.999 (ties): {saturated:.2f}")


def select_threshold(score, y):
    order = np.argsort(-score)
    s_sorted, y_sorted = score[order], y[order]
    tp = np.cumsum(y_sorted)
    nn = np.arange(1, len(y_sorted) + 1)
    lo, _ = cp_interval(tp, nn)
    ok = lo[N_MIN - 1:] >= LOWER_TARGET
    if not ok[0]:
        return None
    first_fail = np.argmin(ok) if not ok.all() else len(ok)
    n_star = N_MIN - 1 + first_fail            # alerts kept = longest passing prefix (>= N_MIN)
    # do not split ties: move to the first strictly lower score
    thr = s_sorted[n_star - 1]
    return {"n_alerts": int(n_star), "threshold": float(thr), "precision": float(tp[n_star - 1] / n_star),
            "cp_lower": float(lo[n_star - 1]), "recall": float(tp[n_star - 1] / y.sum())}


def evaluate(score, y, amount, thr):
    flagged = score >= thr
    k, tp = int(flagged.sum()), int((flagged & (y == 1)).sum())
    lo, hi = cp_interval(tp, max(k, 1))
    return {"alerts": k, "tp": tp, "precision": float(tp / k) if k else None,
            "cp95": [float(lo), float(hi)], "recall": float(tp / y.sum()), "alert_rate": float(k / len(y)),
            "amount_weighted_recall": float(amount[flagged & (y == 1)].sum() / amount[y == 1].sum())}


results = {"rule": {"n_min": N_MIN, "cp_lower_target": LOWER_TARGET, "criterion": {"precision": PREC_TARGET, "cp_lower": LOWER_TARGET}},
           "isotonic_ties_share_top2pct": saturated, "variants": {}}
for name, score in variants.items():
    sel = select_threshold(score[a], y_te[a])
    entry = {"selection_first_half": sel}
    if sel is not None:
        entry["first_half_in_sample"] = evaluate(score[a], y_te[a], amt[a], sel["threshold"])
        entry["second_half_heldout"] = evaluate(score[b], y_te[b], amt[b], sel["threshold"])
        ho = entry["second_half_heldout"]
        entry["criterion_met"] = bool(ho["precision"] is not None and ho["precision"] >= PREC_TARGET and ho["cp95"][0] >= LOWER_TARGET)
    results["variants"][name] = entry
    log(f"{name}: selection={sel} heldout={entry.get('second_half_heldout')}")

valid = {k: v for k, v in results["variants"].items() if v["selection_first_half"] is not None}
chosen = max(valid, key=lambda k: valid[k]["selection_first_half"]["recall"]) if valid else None
results["chosen_variant"] = chosen
results["official"] = valid[chosen] if chosen else None

if chosen:
    score, thr = variants[chosen], valid[chosen]["selection_first_half"]["threshold"]
    # operating curve on the held-out half
    curve = []
    sb, yb, ab = score[b], y_te[b], amt[b]
    for rate in (0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03):
        t = np.quantile(sb, 1 - rate)
        r = evaluate(sb, yb, ab, t)
        curve.append({"target_alert_rate": rate, **r})
    results["heldout_curve"] = curve
    # weekly stability at the chosen threshold
    weekly = []
    for w in range(int(days.max() // 7) + 1):
        m = (days >= 7 * w) & (days < 7 * (w + 1))
        if m.sum() > 0:
            weekly.append({"week": w + 1, "rows": int(m.sum()), **evaluate(score[m], y_te[m], amt[m], thr)})
    results["weekly_at_chosen_threshold"] = weekly
    # what does the tier capture (held-out half)?
    flagged_b = score[b] >= thr
    slices = {}
    for sname, mask in (("uid_has_prior_fraud", uid_prior_fraud[b]), ("uid_known_no_prior_fraud", uid_known[b] & ~uid_prior_fraud[b]),
                        ("uid_new", ~uid_known[b]), ("amount_lt_50", amt[b] < 50), ("amount_50_200", (amt[b] >= 50) & (amt[b] < 200)),
                        ("amount_200_1000", (amt[b] >= 200) & (amt[b] < 1000)), ("amount_ge_1000", amt[b] >= 1000)):
        fraud_in = (y_te[b] == 1) & mask
        slices[sname] = {"fraud_rows": int(fraud_in.sum()), "caught": int((fraud_in & flagged_b).sum()),
                         "recall_in_slice": float((fraud_in & flagged_b).sum() / max(fraud_in.sum(), 1)),
                         "share_of_tier_alerts": float((flagged_b & mask).sum() / max(flagged_b.sum(), 1))}
    results["tier_slices_heldout"] = slices

json.dump(results, open(f"{OUT}/results.json", "w"), indent=1)

print("\n=== HIGH-PRECISION STUDY ===")
print(f"rule: longest prefix from {N_MIN} alerts with Clopper-Pearson lower bound >= {LOWER_TARGET} on the FIRST half")
print(f"criterion on the SECOND half: precision >= {PREC_TARGET} and CP lower >= {LOWER_TARGET}")
print(f"isotonic ties in the top 2%: {saturated:.0%} of rows have calibrated p >= 0.999 -> use RAW-score threshold\n")
hdr = f"{'variant':<20}{'sel alerts':>11}{'1st-half prec':>14}{'2nd: alerts':>12}{'precision':>10}{'CP95 low':>9}{'recall':>8}{'amtRecall':>10}{'alert%':>8}  met"
print(hdr)
for name, e in results["variants"].items():
    sel = e["selection_first_half"]
    if sel is None:
        print(f"{name:<20}  no threshold passes the rule on the first half")
        continue
    ho = e["second_half_heldout"]
    print(f"{name:<20}{sel['n_alerts']:>11}{sel['precision']:>14.4f}{ho['alerts']:>12}{ho['precision']:>10.4f}{ho['cp95'][0]:>9.4f}"
          f"{ho['recall']:>8.3f}{ho['amount_weighted_recall']:>10.3f}{ho['alert_rate'] * 100:>8.2f}  {e['criterion_met']}")
print(f"\nchosen on first half only: {chosen}")
if chosen:
    print("\nheld-out operating curve (second half):")
    for c in results["heldout_curve"]:
        print(f"  alert rate {c['target_alert_rate']:.4f}: alerts={c['alerts']:>4} precision={c['precision']:.4f} "
              f"CP95=[{c['cp95'][0]:.4f},{c['cp95'][1]:.4f}] recall={c['recall']:.3f}")
    print("\nweekly stability at the chosen threshold:")
    for w in results["weekly_at_chosen_threshold"]:
        print(f"  week {w['week']}: rows={w['rows']:>6} alerts={w['alerts']:>4} precision={w['precision']} recall={w['recall']:.3f}")
    print("\nwhat the tier captures (held-out half):")
    for k, v in results["tier_slices_heldout"].items():
        print(f"  {k:<26} {v}")
log("done")
