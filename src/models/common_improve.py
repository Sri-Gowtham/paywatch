"""PayWatch Phase 1 improvement study: shared data, metrics and the candidate runner (imported by the improve_* kernels).

Every candidate goes through the SAME three stages so results are comparable and nothing is tuned on test:
  1. rolling-origin validation: four expanding-window folds over train+validation (cuts 60/70/80/90%, 10% windows), 1 seed.
     Candidate SELECTION uses only these fold scores.
  2. train-only final model(s): fit on the train split, early stopping on validation, scores on validation and test.
  3. refit: fit on train+validation with a FIXED number of rounds (mean early-stopping round x 1.15), like the shipped model,
     scores on test. This is the like-for-like comparison with the shipped model (which was refit the same way).
Outputs per candidate in /kaggle/working: {name}_fold{i}.npy, {name}_val_trainonly.npy, {name}_test_trainonly.npy,
{name}_test_refit.npy, plus {name}.json. The blend kernel reads them and evaluates the pre-registered criterion once.
"""

import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

OUT = "/kaggle/working"
TAU_DAYS = 120
CUTS = (0.60, 0.70, 0.80, 0.90)
SPW = 39.3
T0 = time.time()
find = lambda p: sorted(glob.glob(p, recursive=True))  # noqa: E731


def log(msg):
    print(f"[{(time.time() - T0) / 60:6.1f} min] {msg}", flush=True)


def pr(y, p):
    return float(average_precision_score(y, p))


def recall_at_fpr(y, p, f=0.01):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.interp(f, fpr, tpr))


def recency_w(t):
    return np.exp(-(t.max() - t) / (TAU_DAYS * 86400.0))


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


# ------------------------------------------------------------------ causal linkage features (experiment E2)
def linkage_features(bs):
    """Distinct cards per device/email/address seen BEFORE each row, and the mirror per card. Counts use only earlier rows
    (the current row is added after its features are fixed). Not label based, so no label-delay logic is needed."""
    from upi_fingerprint import make_card_key
    card = make_card_key(bs)
    t = bs["TransactionDT"].to_numpy(float)
    n = len(bs)
    out = {}
    for name, col in (("device", "DeviceInfo"), ("email", "P_emaildomain"), ("addr", "addr1"), ("remail", "R_emaildomain")):
        s = bs[col]
        arr = s.astype(object).where(s.notna(), None).to_numpy()
        cards, seen_n, first_t, card_keys = {}, {}, {}, {}
        f_cards, f_seen, f_age, f_kpc, f_new = (np.full(n, -1.0) for _ in range(5))
        for i in range(n):
            k, c = arr[i], card[i]
            ck = card_keys.get(c)
            if ck is None:
                ck = card_keys[c] = set()
            f_kpc[i] = len(ck)
            if k is None:
                continue
            cs = cards.get(k)
            if cs is None:
                cs = cards[k] = set()
                first_t[k] = t[i]
            f_cards[i], f_seen[i], f_age[i] = len(cs), seen_n.get(k, 0), t[i] - first_t[k]
            f_new[i] = 0.0 if k in ck else 1.0
            cs.add(c)
            seen_n[k] = seen_n.get(k, 0) + 1
            ck.add(k)
        out[f"link_{name}_distinct_cards"], out[f"link_{name}_seen"], out[f"link_{name}_age_s"] = f_cards, f_seen, f_age
        out[f"link_card_distinct_{name}"], out[f"link_card_new_{name}"] = f_kpc, f_new
    return out


class Data:
    pass


def load(linkage=False):
    sys.path.insert(0, os.path.dirname(find("/kaggle/input/**/upi_fingerprint.py")[0]))
    import engineer
    from entity_history import BASE_KEYS
    raw_dir = os.path.dirname(find("/kaggle/input/**/train_transaction.csv")[0])
    cols = json.load(open(find("/kaggle/input/**/prod_compact/serving_spec.json")[0]))["features"]
    base = engineer.build_base(engineer.load_raw(raw_dir), lags=(7,), decay_lags=(), velocity=False, key_names=BASE_KEYS)
    (tr, va, te), _ = engineer.finalize(base, d_mode="anchor", return_artifacts=True)
    d = Data()
    d.cols = list(cols)
    parts = []
    for (X, y, t) in (tr, va, te):
        parts.append((X[cols].to_numpy(dtype=np.float32), y, t))
    if linkage:
        bs = base.sort_values("TransactionDT").reset_index(drop=True)
        lf = linkage_features(bs)
        link = np.column_stack([np.asarray(v, dtype=np.float32) for v in lf.values()])
        d.cols = list(cols) + list(lf)
        n = len(bs)
        a, b = int(n * 0.70), int(n * 0.85)
        for k, (sl, (X, y, t)) in enumerate(zip((slice(0, a), slice(a, b), slice(b, n)), parts)):
            assert np.array_equal(bs["isFraud"].to_numpy()[sl], y), "linkage rows misaligned with the pipeline rows"
            parts[k] = (np.hstack([X, link[sl]]), y, t)
        log(f"linkage features added: {len(lf)}")
    del base
    (d.Xtr, d.ytr, d.ttr), (d.Xva, d.yva, d.tva), (d.Xte, d.yte, d.tte) = parts
    d.Xall, d.yall, d.tall = np.vstack([d.Xtr, d.Xva]), np.r_[d.ytr, d.yva], np.r_[d.ttr, d.tva]
    log(f"data: train {d.Xtr.shape} val {d.Xva.shape} test {d.Xte.shape} features {len(d.cols)}")
    # shipped model scores on test (first len(cols) columns are the compact features)
    ship = []
    for p in find("/kaggle/input/**/prod_compact/xgb_seed*.json"):
        bst = xgb.Booster()
        bst.load_model(p)
        ship.append(bst)
    k = len(cols)
    d.shipped_test = np.mean([b.predict(xgb.DMatrix(pd.DataFrame(d.Xte[:, :k], columns=cols))) for b in ship], axis=0)
    np.save(f"{OUT}/y_te.npy", d.yte.astype(np.uint8))
    np.save(f"{OUT}/y_all.npy", d.yall.astype(np.uint8))
    np.save(f"{OUT}/shipped_test.npy", d.shipped_test.astype(np.float32))
    return d


# ------------------------------------------------------------------ candidate runner
def run_candidate(name, fit_predict, d, seeds=(42, 1, 2)):
    """fit_predict(Xa, ya, ta, Xv, yv, Xps, seed, n_est) -> (list of score arrays for Xps, rounds used).
    n_est None = early stopping on (Xv, yv); an int = fixed rounds, no validation set used."""
    res = {"name": name, "n_features": len(d.cols), "folds": [], "seeds": list(seeds)}
    N = len(d.Xall)

    def save():
        json.dump(res, open(f"{OUT}/{name}.json", "w"), indent=1)

    for i, cut in enumerate(CUTS):
        a, b = int(N * cut), int(N * (cut + 0.10))
        t1 = time.time()
        (pv,), best = fit_predict(d.Xall[:a], d.yall[:a], d.tall[:a], d.Xall[a:b], d.yall[a:b], [d.Xall[a:b]], 42, None)
        np.save(f"{OUT}/{name}_fold{i}.npy", pv.astype(np.float32))
        res["folds"].append({"cut": cut, "rounds": best, "val_pr_auc": pr(d.yall[a:b], pv), "seconds": round(time.time() - t1)})
        log(f"{name} fold {cut:.0%}: val PR-AUC {res['folds'][-1]['val_pr_auc']:.4f} rounds {best}")
        save()
    fm = [f["val_pr_auc"] for f in res["folds"]]
    res["fold_mean_pr_auc"], res["fold_std_pr_auc"] = float(np.mean(fm)), float(np.std(fm))

    pt, pv_list, iters = [], [], []
    for s in seeds:
        (a_te, a_va), best = fit_predict(d.Xtr, d.ytr, d.ttr, d.Xva, d.yva, [d.Xte, d.Xva], s, None)
        pt.append(a_te), pv_list.append(a_va), iters.append(best)
        log(f"{name} train-only seed {s}: test PR-AUC {pr(d.yte, a_te):.4f} rounds {best}")
    st, sv = np.mean(pt, axis=0), np.mean(pv_list, axis=0)
    np.save(f"{OUT}/{name}_test_trainonly.npy", st.astype(np.float32))
    np.save(f"{OUT}/{name}_val_trainonly.npy", sv.astype(np.float32))
    res["train_only"] = {"rounds": iters, "val_pr_auc": pr(d.yva, sv), "test_pr_auc": pr(d.yte, st), "test_roc_auc": float(roc_auc_score(d.yte, st)),
                         "test_recall_at_fpr_1pct": recall_at_fpr(d.yte, st)}
    save()

    n_fixed = int(np.mean(iters) * 1.15)
    rt = []
    for s in seeds:
        (a_te,), _ = fit_predict(d.Xall, d.yall, d.tall, d.Xva, d.yva, [d.Xte], s, n_fixed)
        rt.append(a_te)
        log(f"{name} refit seed {s} ({n_fixed} rounds): test PR-AUC {pr(d.yte, a_te):.4f}")
    sr = np.mean(rt, axis=0)
    np.save(f"{OUT}/{name}_test_refit.npy", sr.astype(np.float32))
    res["refit"] = {"rounds": n_fixed, "test_pr_auc": pr(d.yte, sr), "test_roc_auc": float(roc_auc_score(d.yte, sr)),
                    "test_recall_at_fpr_1pct": recall_at_fpr(d.yte, sr)}
    res["shipped_reference"] = {"test_pr_auc": pr(d.yte, d.shipped_test), "test_recall_at_fpr_1pct": recall_at_fpr(d.yte, d.shipped_test)}
    save()
    print(f"\n=== {name} ===\nfold mean PR-AUC {res['fold_mean_pr_auc']:.4f} (std {res['fold_std_pr_auc']:.4f}); baseline fold mean 0.6957")
    print("train-only:", res["train_only"])
    print("refit:", res["refit"], "| shipped:", res["shipped_reference"])
    return res


# ------------------------------------------------------------------ model families
def xgb_family(params, patience=300):
    def fit_predict(Xa, ya, ta, Xv, yv, Xps, seed, n_est):
        kw = dict(n_estimators=n_est or 8000, tree_method="hist", device="cpu", eval_metric="aucpr", random_state=seed, n_jobs=-1, **params)
        if n_est is None:
            kw["early_stopping_rounds"] = patience
        m = xgb.XGBClassifier(**kw)
        m.fit(Xa, ya, sample_weight=recency_w(ta), eval_set=[(Xv, yv)] if n_est is None else None, verbose=False)
        best = int(m.best_iteration) + 1 if n_est is None else n_est
        return [m.predict_proba(X)[:, 1] for X in Xps], best
    return fit_predict


XGB_BASE = dict(max_depth=10, learning_rate=0.0558, subsample=0.935, colsample_bytree=0.678, min_child_weight=1,
                reg_lambda=1.518, gamma=0.0336, scale_pos_weight=SPW)


def lgbm_family(patience=300):
    import lightgbm as lgb

    def fit_predict(Xa, ya, ta, Xv, yv, Xps, seed, n_est):
        m = lgb.LGBMClassifier(n_estimators=n_est or 8000, learning_rate=0.03, num_leaves=127, min_child_samples=40, subsample=0.8,
                               subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0, scale_pos_weight=SPW, random_state=seed,
                               n_jobs=-1, verbose=-1, metric="None")  # "None": early-stop on average_precision only, not on weighted logloss
        if n_est is None:
            m.fit(Xa, ya, sample_weight=recency_w(ta), eval_set=[(Xv, yv)], eval_metric="average_precision",
                  callbacks=[lgb.early_stopping(patience, verbose=False)])
            best = int(m.best_iteration_)
        else:
            m.fit(Xa, ya, sample_weight=recency_w(ta))
            best = n_est
        return [m.predict_proba(X)[:, 1] for X in Xps], best
    return fit_predict


def catboost_family(patience=200):
    from catboost import CatBoostClassifier

    def fit_predict(Xa, ya, ta, Xv, yv, Xps, seed, n_est):
        kw = dict(iterations=n_est or 3000, learning_rate=0.08, depth=8, l2_leaf_reg=3.0, scale_pos_weight=SPW, random_seed=seed,
                  thread_count=-1, verbose=False, allow_writing_files=False)
        if n_est is None:
            try:
                m = CatBoostClassifier(eval_metric="PRAUC", **kw)
                m.fit(Xa, ya, sample_weight=recency_w(ta), eval_set=(Xv, yv), early_stopping_rounds=patience, use_best_model=True)
            except Exception as e:  # PRAUC not available in this CatBoost build: fall back to AUC for early stopping
                log(f"CatBoost PRAUC unavailable ({str(e)[:80]}); using AUC for early stopping")
                m = CatBoostClassifier(eval_metric="AUC", **kw)
                m.fit(Xa, ya, sample_weight=recency_w(ta), eval_set=(Xv, yv), early_stopping_rounds=patience, use_best_model=True)
            best = int(m.get_best_iteration()) + 1
        else:
            m = CatBoostClassifier(**kw)
            m.fit(Xa, ya, sample_weight=recency_w(ta))
            best = n_est
        return [m.predict_proba(X)[:, 1] for X in Xps], best
    return fit_predict
