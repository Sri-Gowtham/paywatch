"""IEEE-CIS feature pipeline (importable; executed inside the Kaggle pipeline kernel).

Stages: load_raw -> build_base (causal UPI features + categorical encoding, time-sorted)
-> finalize (time split, train-only frequency encoding / pruning, D-column handling).
All data-dependent choices use TRAIN rows only. No SMOTE; imbalance is handled by the models.
"""

import numpy as np
import pandas as pd

from entity_history import add_lagged_fraud_history, add_velocity_features
from upi_fingerprint import add_upi_fingerprint_features

TARGET = "isFraud"
FREQ_COLS = ["card1", "card2", "card3", "card5", "addr1", "P_emaildomain", "R_emaildomain"]
D_COLS = [f"D{i}" for i in range(1, 16) if i != 9]
MAX_NAN_FRAC = 0.90


def load_raw(raw_dir: str) -> pd.DataFrame:
    tx = pd.read_csv(f"{raw_dir}/train_transaction.csv")
    identity = pd.read_csv(f"{raw_dir}/train_identity.csv")
    return tx.merge(identity, on="TransactionID", how="left")


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = [c for c in df.columns if df[c].dtype == "object" or str(df[c].dtype) == "str"]
    for c in cat_cols:
        codes = df[c].astype("category").cat.codes.astype("float32")
        df[c] = codes.where(codes >= 0, np.nan)
    return df


def build_base(df: pd.DataFrame, lags=(1, 3, 7, 14, 30), decay_lags=(7,), velocity: bool = False,
               key_names=None) -> pd.DataFrame:
    df = add_upi_fingerprint_features(df)
    df = add_lagged_fraud_history(df, lags=lags, decay_lags=decay_lags, key_names=key_names)
    if velocity:
        df = add_velocity_features(df)
    return _encode_categoricals(df)


def finalize(base: pd.DataFrame, d_mode: str = "raw", drop_v: bool = False, return_artifacts: bool = False):
    """d_mode: 'raw' keep D columns, 'drop' remove them, 'anchor' replace with day - D.

    With return_artifacts=True returns (splits, artifacts) where artifacts holds the train-only
    frequency tables (keys are repr(float(value))) and the kept feature columns, for serving.
    """
    df = base.sort_values("TransactionDT").reset_index(drop=True).copy()
    n = len(df)
    train = df.iloc[: int(n * 0.70)].copy()
    val = df.iloc[int(n * 0.70): int(n * 0.85)].copy()
    test = df.iloc[int(n * 0.85):].copy()

    for part in (train, val, test):
        day = part["TransactionDT"] / 86400.0
        if d_mode == "anchor":
            for c in D_COLS:
                if c in part:
                    part[c] = day - part[c]
        elif d_mode == "drop":
            part.drop(columns=[c for c in D_COLS if c in part], inplace=True)
        if drop_v:
            part.drop(columns=[c for c in part.columns if c.startswith("V") and c[1:].isdigit()], inplace=True)

    freq_tables = {}
    for col in FREQ_COLS:
        freq = train[col].value_counts(normalize=True)
        freq_tables[col] = {repr(float(k)): float(v) for k, v in freq.items()}
        for part in (train, val, test):
            part[f"{col}_freq"] = part[col].map(freq).fillna(0.0).astype("float32")

    feats = [c for c in train.columns if c not in (TARGET, "TransactionID", "TransactionDT")]
    nan_frac = train[feats].isna().mean()
    nunique = train[feats].nunique(dropna=True)
    keep = [c for c in feats if nan_frac[c] <= MAX_NAN_FRAC and nunique[c] > 1]

    out = []
    for part in (train, val, test):
        x = part[keep].astype("float32").fillna(-999)
        y = part[TARGET].to_numpy()
        t = part["TransactionDT"].to_numpy(dtype=float)
        out.append((x, y, t))
    if return_artifacts:
        return out, {"freq_tables": freq_tables, "feature_columns": keep}
    return out
