"""Causal entity features: lagged fraud history (plain + recency-decayed) and short-window velocity.

Frames must be sorted by TransactionDT (build_base guarantees this).

Lagged fraud history models label delay: for a row at time t, an entity's fraud statistics only
include rows with TransactionDT <= t - lag_days, i.e. labels that would already be known. A row's
own label is never included. Missing keys get the prior (no information).
Velocity features (label-free) are kept for completeness; they were tested and added nothing.
"""

from collections import deque

import numpy as np
import pandas as pd

from upi_fingerprint import make_card_key, make_uid

PRIOR_RATE = 0.035
PRIOR_WEIGHT = 20.0
BASE_KEYS = ("card", "uid", "addr", "email", "device")


def _entity_keys(df: pd.DataFrame) -> dict:
    def col(name):
        s = df[name]
        return s.astype(object).where(s.notna(), None).to_numpy()

    card = make_card_key(df)
    return {
        "card": card,
        "uid": make_uid(df),
        "addr": col("addr1"),
        "email": col("P_emaildomain"),
        "device": col("DeviceInfo"),
        "card1": col("card1"),
        "addr2": col("addr2"),
        "remail": col("R_emaildomain"),
        "browser": col("id_31"),
        "os": col("id_30"),
        "cardaddr": (pd.Series(card) + "_" + df["addr1"].astype(str)).to_numpy(),
        "prodemail": (df["ProductCD"].astype(str) + "_" + df["P_emaildomain"].astype(str)).to_numpy(),
    }


def add_lagged_fraud_history(df, lags=(7,), decay_lags=(), halflife_days=30.0, key_names=None) -> pd.DataFrame:
    ts = df["TransactionDT"].to_numpy(dtype=float)
    labels = df["isFraud"].to_numpy()
    n = len(df)
    half = halflife_days * 86400.0
    keys = {k: v for k, v in _entity_keys(df).items() if key_names is None or k in key_names}
    new_cols = {}

    for lag in lags:
        lag_s = lag * 86400.0
        use_decay = lag in decay_lags
        for name, arr in keys.items():
            seen, frauds = {}, {}
            dec = {}
            n_out = np.zeros(n, dtype="float32")
            f_out = np.zeros(n, dtype="float32")
            r_out = np.full(n, PRIOR_RATE, dtype="float32")
            fd_out = np.zeros(n, dtype="float32") if use_decay else None
            rd_out = np.full(n, PRIOR_RATE, dtype="float32") if use_decay else None
            j = 0
            for i in range(n):
                ti = ts[i]
                while j < n and ts[j] <= ti - lag_s:
                    kj = arr[j]
                    if kj is not None:
                        seen[kj] = seen.get(kj, 0) + 1
                        frauds[kj] = frauds.get(kj, 0) + labels[j]
                        if use_decay:
                            s = dec.get(kj)
                            if s is None:
                                dec[kj] = [float(labels[j]), 1.0, ts[j]]
                            else:
                                f = 0.5 ** ((ts[j] - s[2]) / half)
                                s[0] = s[0] * f + labels[j]
                                s[1] = s[1] * f + 1.0
                                s[2] = ts[j]
                    j += 1
                ki = arr[i]
                if ki is not None and ki in seen:
                    c, fr = seen[ki], frauds[ki]
                    n_out[i], f_out[i] = c, fr
                    r_out[i] = (fr + PRIOR_WEIGHT * PRIOR_RATE) / (c + PRIOR_WEIGHT)
                    if use_decay:
                        s = dec[ki]
                        f = 0.5 ** ((ti - s[2]) / half)
                        fd, td = s[0] * f, s[1] * f
                        fd_out[i] = fd
                        rd_out[i] = (fd + PRIOR_WEIGHT * PRIOR_RATE) / (td + PRIOR_WEIGHT)
            tag = f"h{lag}_{name}"
            new_cols[f"{tag}_n"] = n_out
            new_cols[f"{tag}_fraud_n"] = f_out
            new_cols[f"{tag}_fraud_rate"] = r_out
            if use_decay:
                new_cols[f"{tag}_fraud_decay"] = fd_out
                new_cols[f"{tag}_rate_decay"] = rd_out

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["TransactionDT"].to_numpy(dtype=float)
    n = len(df)
    keys = _entity_keys(df)
    new_cols = {}

    for name in ("uid", "card", "addr", "email"):
        arr = keys[name]
        win1, win24 = {}, {}
        c1, c24 = np.zeros(n), np.zeros(n)
        for i in range(n):
            k = arr[i]
            if k is None:
                continue
            q24 = win24.setdefault(k, deque())
            q1 = win1.setdefault(k, deque())
            t = ts[i]
            while q24 and q24[0] < t - 86400:
                q24.popleft()
            while q1 and q1[0] < t - 3600:
                q1.popleft()
            c24[i], c1[i] = len(q24), len(q1)
            q24.append(t)
            q1.append(t)
        new_cols[f"vel_{name}_cnt_1h"] = c1
        new_cols[f"vel_{name}_cnt_24h"] = c24

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
