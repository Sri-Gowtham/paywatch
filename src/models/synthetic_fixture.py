"""PayWatch — build a SYNTHETIC parity fixture (Kaggle, CPU) so the repo carries no raw competition rows.

300 test-period rows of the raw data are overwritten with synthetic ones: every field is drawn independently from
the empirical marginal of that column (numeric amounts get multiplicative noise), so no synthetic row is a copy of a
real transaction. Timestamps and IDs of the replaced rows are kept, so the row count, the time split and the train-only
frequency tables are unchanged (asserted against the shipped freq_tables.json). The FULL batch pipeline is then run on
the modified data and the expected feature vector / score / calibrated score of each synthetic row come from it, so
the fixture keeps the batch-vs-API parity guarantee of the original one.
Output: /kaggle/working/parity_fixture.json
"""

import glob
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb

T0 = time.time()
find = lambda p: sorted(glob.glob(p, recursive=True))  # noqa: E731
OUT = "/kaggle/working"
N_ROWS = 300


def log(msg):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {msg}", flush=True)


sys.path.insert(0, os.path.dirname(find("/kaggle/input/**/upi_fingerprint.py")[0]))
import engineer  # noqa: E402
from entity_history import BASE_KEYS  # noqa: E402

RAW = os.path.dirname(find("/kaggle/input/**/train_transaction.csv")[0])
meta = json.load(open(find("/kaggle/input/**/feature_meta.json")[0]))
shipped_freq = json.load(open(find("/kaggle/input/**/freq_tables.json")[0]))
spec = json.load(open(find("/kaggle/input/**/prod_compact/serving_spec.json")[0]))
iso = json.load(open(find("/kaggle/input/**/isotonic.json")[0]))
cols, required_raw = spec["features"], meta["required_raw_fields"]
boosters = []
for p in find("/kaggle/input/**/prod_compact/xgb_seed*.json"):
    b = xgb.Booster()
    b.load_model(p)
    boosters.append(b)

raw = engineer.load_raw(RAW)
raw = raw.sort_values("TransactionDT").reset_index(drop=True)
n = len(raw)
rng = np.random.default_rng(2024)
test_pos = np.arange(int(n * 0.85), n)
replace_pos = np.sort(rng.choice(test_pos, N_ROWS, replace=False))
replace_ids = raw["TransactionID"].to_numpy()[replace_pos]
pool = raw.iloc[test_pos]

synth = {}
for c in required_raw:
    if c in ("TransactionDT",):
        continue
    vals = pool[c].sample(N_ROWS, replace=True, random_state=int(rng.integers(1 << 30))).to_numpy()
    if c == "TransactionAmt":
        vals = np.round(vals.astype(float) * rng.lognormal(0.0, 0.25, N_ROWS), 2)
    synth[c] = vals
for c in raw.columns:  # remaining columns: blank, they are not part of the compact feature set
    if c in ("TransactionID", "TransactionDT"):
        continue
    if c in synth:
        raw.loc[replace_pos, c] = synth[c]
    elif c == "isFraud":
        raw.loc[replace_pos, c] = 0
    else:
        raw.loc[replace_pos, c] = pool[c].sample(N_ROWS, replace=True, random_state=int(rng.integers(1 << 30))).to_numpy()
log(f"replaced {N_ROWS} test-period rows with column-independent synthetic rows")

base = engineer.build_base(raw.copy(), lags=(7,), decay_lags=(), velocity=False, key_names=BASE_KEYS)
(tr, va, te), art = engineer.finalize(base, d_mode="anchor", return_artifacts=True)
X_te = te[0]
for c, tbl in shipped_freq.items():  # the shipped frequency tables must be untouched by the replacement
    assert tbl == art["freq_tables"][c], f"freq table changed for {c}"
log("train-only frequency tables identical to the shipped ones")

sorted_base = base.sort_values("TransactionDT").reset_index(drop=True)
test_ids = sorted_base["TransactionID"].iloc[int(n * 0.85):].to_numpy()
assert len(test_ids) == len(X_te)
pos = {int(t): i for i, t in enumerate(test_ids)}
idx = np.array([pos[int(t)] for t in replace_ids])
Xs = X_te.iloc[idx][cols]
score = np.mean([b.predict(xgb.DMatrix(Xs)) for b in boosters], axis=0)
xs, ys = np.asarray(iso["x"]), np.asarray(iso["y"])
cal = np.interp(score, xs, ys)

raw_idx = raw.set_index("TransactionID")
rows = []
for k, t in enumerate(replace_ids):
    r = raw_idx.loc[t, required_raw]
    fields = {c: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)) for c, v in r.items()}
    rows.append({"transaction_id": int(t), "fields": fields, "expected_features": Xs.iloc[k].astype(float).tolist(),
                 "expected_score": float(score[k]), "expected_calibrated": float(cal[k])})
json.dump({"features": cols, "rows": rows, "synthetic": True}, open(f"{OUT}/parity_fixture.json", "w"))
log(f"wrote synthetic parity_fixture.json rows={len(rows)} score range {score.min():.4f}..{score.max():.4f}")
