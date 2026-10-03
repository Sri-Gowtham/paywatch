"""PayWatch — export serving assets for the API (Kaggle, CPU).

Produces small, pickle-free files so the API image does not need pandas/sklearn training code:
  categories.json      sorted categories per categorical raw column (== pandas category codes)
  freq_tables.json     train-only normalised frequency tables keyed by repr(float(code or value))
  isotonic.json        calibrator as a monotone (x, y) table; API applies np.interp
  feature_meta.json    compact feature order, family, serving source, required raw request fields
  parity_fixture.json  raw rows + expected feature vector + expected score (model/encoder parity test)
  stream_sample.parquet time-ordered raw test rows (+label) for the later production simulation
All asserted against the batch pipeline before being written.
"""

import glob
import json
import os
import pickle
import re
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb

T0 = time.time()


def log(msg):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {msg}", flush=True)


src_dir = os.path.dirname(glob.glob("/kaggle/input/**/upi_fingerprint.py", recursive=True)[0])
sys.path.insert(0, src_dir)
import engineer  # noqa: E402
from entity_history import BASE_KEYS  # noqa: E402

RAW = os.path.dirname(glob.glob("/kaggle/input/**/train_transaction.csv", recursive=True)[0])
OUT = "/kaggle/working"
find = lambda p: sorted(glob.glob(p, recursive=True))  # noqa: E731

spec = json.load(open(find("/kaggle/input/**/prod_compact/serving_spec.json")[0]))
model_paths = find("/kaggle/input/**/prod_compact/xgb_seed*.json")
iso = pickle.load(open(find("/kaggle/input/**/prod_compact/isotonic.pkl")[0], "rb"))
cols = spec["features"]
log(f"compact features={len(cols)} models={len(model_paths)}")

UPI_BEHAVIOR = {"amount_vs_7day_avg_ratio", "merchant_category_entropy", "hour_deviation_from_user_mean",
                "days_since_last_large_txn", "is_new_merchant", "is_p2p"}


def family(f):
    if f.startswith("h7_"):
        return "entity_fraud_history"
    if f.startswith("uid_") or f in UPI_BEHAVIOR:
        return "upi_behavior"
    if f.endswith("_freq"):
        return "frequency_encoding"
    if re.fullmatch(r"D\d+", f):
        return "time_deltas_D"
    return "raw_field"


def source(f):
    if f.startswith("h7_"):
        return "entity_history_state"
    if f.startswith("uid_") or f in UPI_BEHAVIOR - {"is_p2p"}:
        return "user_behavior_state"
    if f.endswith("_freq"):
        return "frequency_lookup"
    if re.fullmatch(r"D\d+", f):
        return "anchored_D"
    return "raw_request_field"


# ---------------------------------------------------------------- categories
raw = engineer.load_raw(RAW)
obj_cols = [c for c in raw.columns if raw[c].dtype == "object" or str(raw[c].dtype) == "str"]
categories = {c: sorted(raw[c].dropna().unique().tolist()) for c in obj_cols}
enc = engineer._encode_categoricals(raw[obj_cols].copy())
for c in obj_cols:
    mapped = raw[c].map({v: float(i) for i, v in enumerate(categories[c])}).astype("float32").to_numpy()
    assert np.array_equal(mapped, enc[c].to_numpy(), equal_nan=True), f"category mapping mismatch for {c}"
log(f"categories verified against engineer._encode_categoricals for {len(obj_cols)} columns")

# ---------------------------------------------------------------- features + artifacts
base = engineer.build_base(raw.copy(), lags=(7,), decay_lags=(), velocity=False, key_names=BASE_KEYS)
(tr, va, te), art = engineer.finalize(base, d_mode="anchor", return_artifacts=True)
X_te, y_te, t_te = te
assert list(X_te.columns) == art["feature_columns"]
sorted_base = base.sort_values("TransactionDT").reset_index(drop=True)
n = len(sorted_base)
test_ids = sorted_base["TransactionID"].iloc[int(n * 0.85):].to_numpy()
assert len(test_ids) == len(X_te)
del base, sorted_base

# ---------------------------------------------------------------- required raw fields
raw_sourced = [c for c in cols if source(c) == "raw_request_field"]
d_needed = [c for c in cols if source(c) == "anchored_D"]
freq_needed = [c[: -len("_freq")] for c in cols if c.endswith("_freq")]
always = ["TransactionDT", "TransactionAmt", "ProductCD", "card1", "card2", "card3", "card5", "addr1", "D1",
          "P_emaildomain", "R_emaildomain", "DeviceInfo"]
required_raw = sorted(set(raw_sourced) | set(d_needed) | set(freq_needed) | set(always))
missing = [c for c in required_raw if c not in raw.columns]
assert not missing, f"required raw fields not in raw data: {missing}"
log(f"required raw request fields: {len(required_raw)} (raw-sourced features {len(raw_sourced)})")

meta = {
    "features": cols,
    "feature_family": {f: family(f) for f in cols},
    "feature_source": {f: source(f) for f in cols},
    "required_raw_fields": required_raw,
    "categorical_raw_fields": [c for c in required_raw if c in categories],
    "d_columns": d_needed,
    "freq_columns": freq_needed,
    "missing_value_fill": -999.0,
}

# ---------------------------------------------------------------- isotonic as table
xs, ys = np.asarray(iso.X_thresholds_, dtype=float), np.asarray(iso.y_thresholds_, dtype=float)
probe = np.concatenate([np.random.default_rng(0).random(5000), xs[:: max(len(xs) // 500, 1)]])
diff = np.abs(np.interp(probe, xs, ys) - iso.predict(probe))
log(f"isotonic raw-table vs sklearn: n_thresholds={len(xs)} max|diff|={diff.max():.3e} at x={probe[diff.argmax()]:.6f}")
if diff.max() > 1e-3:
    grid = np.unique(np.concatenate([xs, np.linspace(0.0, 1.0, 100_001)]))
    xs, ys = grid, np.asarray(iso.predict(grid), dtype=float)
    diff = np.abs(np.interp(probe, xs, ys) - iso.predict(probe))
    log(f"isotonic dense-grid table: n={len(xs)} max|diff|={diff.max():.3e}")
    assert diff.max() < 1e-3, "isotonic table cannot reproduce sklearn predict"
isotonic_json = {"x": xs.tolist(), "y": ys.tolist(), "max_abs_diff_vs_sklearn": float(diff.max())}

# ---------------------------------------------------------------- parity fixture
boosters = []
for p in model_paths:
    b = xgb.Booster()
    b.load_model(p)
    boosters.append(b)
rng = np.random.default_rng(0)
idx = np.sort(rng.choice(len(X_te), 300, replace=False))
Xs = X_te.iloc[idx][cols]
score = np.mean([b.predict(xgb.DMatrix(Xs)) for b in boosters], axis=0)
cal = np.interp(score, xs, ys)
log(f"fixture: |interp - sklearn iso.predict| max = {np.abs(cal - iso.predict(score)).max():.3e}")
raw_idx = raw.set_index("TransactionID")
rows = []
for k, i in enumerate(idx):
    r = raw_idx.loc[test_ids[i], required_raw]
    fields = {c: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)) for c, v in r.items()}
    rows.append({"transaction_id": int(test_ids[i]), "fields": fields,
                 "expected_features": Xs.iloc[k].astype(float).tolist(),
                 "expected_score": float(score[k]), "expected_calibrated": float(cal[k])})
fixture = {"features": cols, "rows": rows}
log(f"parity fixture rows={len(rows)}")

# ---------------------------------------------------------------- stream sample
rng = np.random.default_rng(1)
sidx = np.sort(rng.choice(len(X_te), 20_000, replace=False))
stream = raw_idx.loc[test_ids[sidx], required_raw + ["isFraud"]].reset_index()
stream = stream.sort_values("TransactionDT").reset_index(drop=True)
stream.to_parquet(f"{OUT}/stream_sample.parquet", index=False)

# ---------------------------------------------------------------- write
with open(f"{OUT}/categories.json", "w") as f:
    json.dump({c: categories[c] for c in obj_cols}, f)
with open(f"{OUT}/freq_tables.json", "w") as f:
    json.dump(art["freq_tables"], f)
with open(f"{OUT}/isotonic.json", "w") as f:
    json.dump(isotonic_json, f)
with open(f"{OUT}/feature_meta.json", "w") as f:
    json.dump(meta, f, indent=1)
with open(f"{OUT}/parity_fixture.json", "w") as f:
    json.dump(fixture, f)

print("\n=== EXPORTED FILES (size bytes) ===")
for p in sorted(glob.glob(f"{OUT}/*")):
    print(f"{os.path.basename(p):<24} {os.path.getsize(p):>12,}")
print("\n=== MODEL FILES (from paywatch-b2) ===")
for p in model_paths:
    print(f"{os.path.basename(p):<24} {os.path.getsize(p):>12,}")
print("\nserving_source counts:", pd.Series(meta["feature_source"]).value_counts().to_dict())
print("categorical required fields:", meta["categorical_raw_fields"])
print("stream sample:", stream.shape, "fraud rate", float(stream["isFraud"].mean()),
      "time span days", float((stream["TransactionDT"].max() - stream["TransactionDT"].min()) / 86400))
log("done")
