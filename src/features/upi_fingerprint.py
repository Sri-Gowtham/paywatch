"""UPI behavioral fingerprint features — causal (no look-ahead).

IEEE-CIS has no real user/merchant IDs. A user proxy ("uid") is reconstructed from the card
columns, addr1, and the account-age anchor floor(TransactionDT/86400 - D1) (D1 = days since
the card's first transaction, so this start-day is stable per card holder). The payee proxy is
P_emaildomain. This is a proxy, documented as a limitation. Every feature for a transaction is
computed only from that uid's EARLIER transactions, so nothing from the future (or from rows
that later land in val/test) can leak into a row's features.
"""

import math
from collections import deque

import numpy as np
import pandas as pd


def make_card_key(df: pd.DataFrame) -> np.ndarray:
    return (
        df["card1"].astype(str) + "_" + df["card2"].astype(str) + "_"
        + df["card3"].astype(str) + "_" + df["card5"].astype(str)
    ).to_numpy()


def make_uid(df: pd.DataFrame) -> np.ndarray:
    start_day = np.floor(df["TransactionDT"] / 86400.0 - df["D1"])
    start_day = start_day.fillna(-1).astype(int).astype(str)
    return (
        df["card1"].astype(str) + "_" + df["card2"].astype(str) + "_"
        + df["card3"].astype(str) + "_" + df["card5"].astype(str) + "_"
        + df["addr1"].astype(str) + "_" + start_day
    ).to_numpy()


def add_upi_fingerprint_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    users = make_uid(df)
    merchants = df["P_emaildomain"].fillna("none").to_numpy()
    amounts = df["TransactionAmt"].to_numpy(dtype=float)
    ts = df["TransactionDT"].to_numpy(dtype=float)
    hours = (ts // 3600 % 24).astype(float)
    days = ts / 86400.0

    n = len(df)
    ratio = np.ones(n)
    entropy = np.zeros(n)
    hour_dev = np.zeros(n)
    since_large = np.full(n, -1.0)
    new_merchant = np.zeros(n, dtype=int)
    tx_count = np.zeros(n)
    amt_mean = np.full(n, -1.0)
    amt_std = np.full(n, -1.0)
    amt_z = np.zeros(n)
    secs_prev = np.full(n, -1.0)

    # per-uid running state, updated AFTER the row's features are emitted
    recent = {}        # last 7 amounts
    m_counts = {}      # merchant -> count
    n_tx = {}          # number of prior transactions
    hour_sum = {}
    amt_sum = {}
    amt_sq = {}
    last_large_day = {}
    last_ts = {}

    for i in range(n):
        u, m, a, h, d, t = users[i], merchants[i], amounts[i], hours[i], days[i], ts[i]

        k = n_tx.get(u, 0)
        tx_count[i] = k
        if k > 0:
            window = recent[u]
            mean7 = sum(window) / len(window)
            ratio[i] = a / mean7 if mean7 > 0 else 1.0

            counts = m_counts[u]
            total = float(k)
            entropy[i] = -sum((c / total) * math.log2(c / total) for c in counts.values())
            new_merchant[i] = 0 if m in counts else 1

            hour_dev[i] = abs(h - hour_sum[u] / k)

            if u in last_large_day:
                since_large[i] = d - last_large_day[u]

            mean = amt_sum[u] / k
            var = max(amt_sq[u] / k - mean * mean, 0.0)
            std = math.sqrt(var)
            amt_mean[i] = mean
            amt_std[i] = std
            amt_z[i] = min(max((a - mean) / std, -20.0), 20.0) if std > 1e-3 else 0.0
            secs_prev[i] = t - last_ts[u]
        else:
            new_merchant[i] = 1

        # large = more than 2x the uid's prior mean amount (causal stand-in for a percentile)
        if k > 0 and a > 2 * (amt_sum[u] / k):
            last_large_day[u] = d

        recent.setdefault(u, deque(maxlen=7)).append(a)
        m_counts.setdefault(u, {})
        m_counts[u][m] = m_counts[u].get(m, 0) + 1
        n_tx[u] = k + 1
        hour_sum[u] = hour_sum.get(u, 0.0) + h
        amt_sum[u] = amt_sum.get(u, 0.0) + a
        amt_sq[u] = amt_sq.get(u, 0.0) + a * a
        last_ts[u] = t

    df["amount_vs_7day_avg_ratio"] = ratio
    df["merchant_category_entropy"] = entropy
    df["hour_deviation_from_user_mean"] = hour_dev
    df["days_since_last_large_txn"] = since_large
    df["is_new_merchant"] = new_merchant
    df["uid_prior_tx_count"] = tx_count
    df["uid_amt_mean"] = amt_mean
    df["uid_amt_std"] = amt_std
    df["uid_amt_zscore"] = amt_z
    df["uid_secs_since_prev_tx"] = secs_prev
    df["uid_is_strong"] = (df["D1"].notna() & df["addr1"].notna()).astype(int)

    # proxy: ProductCD == 'C' as a person-to-person-like product
    df["is_p2p"] = (df["ProductCD"] == "C").astype(int) if "ProductCD" in df.columns else 0

    return df
