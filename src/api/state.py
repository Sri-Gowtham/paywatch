"""Online (streaming) ports of the batch features in src/features.

Parity with `upi_fingerprint.add_upi_fingerprint_features` and `entity_history.add_lagged_fraud_history`
is enforced by tests/test_state_parity.py. Keep the arithmetic identical to the batch code.

Semantics note: batch training used a 7-day TIME lag for entity fraud history. Online, labels are
recorded when they ARRIVE (`record_label`), which is the production equivalent.
"""

import math
from collections import deque
from typing import Dict, Optional

PRIOR_RATE = 0.035
PRIOR_WEIGHT = 20.0
BASE_KEYS = ("card", "uid", "addr", "email", "device")
BEHAVIOR_FEATURES = (
    "amount_vs_7day_avg_ratio", "merchant_category_entropy", "hour_deviation_from_user_mean",
    "days_since_last_large_txn", "is_new_merchant", "uid_prior_tx_count", "uid_amt_mean",
    "uid_amt_std", "uid_amt_zscore", "uid_secs_since_prev_tx", "uid_is_strong", "is_p2p",
)


def is_missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _num(v) -> str:
    return "nan" if is_missing(v) else repr(float(v))


def card_key(f: dict) -> str:
    return "_".join(_num(f.get(c)) for c in ("card1", "card2", "card3", "card5"))


def uid_key(f: dict) -> str:
    d1 = f.get("D1")
    start_day = -1 if is_missing(d1) else int(math.floor(float(f["TransactionDT"]) / 86400.0 - float(d1)))
    return f"{card_key(f)}_{_num(f.get('addr1'))}_{start_day}"


def entity_keys(f: dict) -> Dict[str, Optional[str]]:
    def raw(name):
        v = f.get(name)
        return None if is_missing(v) else v

    addr = f.get("addr1")
    return {
        "card": card_key(f),
        "uid": uid_key(f),
        "addr": None if is_missing(addr) else _num(addr),
        "email": raw("P_emaildomain"),
        "device": raw("DeviceInfo"),
    }


class UserBehaviorState:
    """Per-uid running state; mirrors the batch loop in upi_fingerprint.py."""

    def __init__(self):
        self._s: Dict[str, dict] = {}

    def observe(self, f: dict, uid: str, commit: bool = True) -> Dict[str, float]:
        a = float(f["TransactionAmt"])
        t = float(f["TransactionDT"])
        h = float(t // 3600 % 24)
        d = t / 86400.0
        m = f.get("P_emaildomain")
        m = "none" if is_missing(m) else m

        s = self._s.get(uid)
        k = s["n"] if s else 0
        out = {
            "amount_vs_7day_avg_ratio": 1.0, "merchant_category_entropy": 0.0,
            "hour_deviation_from_user_mean": 0.0, "days_since_last_large_txn": -1.0, "is_new_merchant": 1.0,
            "uid_prior_tx_count": float(k), "uid_amt_mean": -1.0, "uid_amt_std": -1.0, "uid_amt_zscore": 0.0,
            "uid_secs_since_prev_tx": -1.0,
            "uid_is_strong": 1.0 if (not is_missing(f.get("D1")) and not is_missing(f.get("addr1"))) else 0.0,
            "is_p2p": 1.0 if f.get("ProductCD") == "C" else 0.0,
        }
        if k > 0:
            window = s["recent"]
            mean7 = sum(window) / len(window)
            out["amount_vs_7day_avg_ratio"] = a / mean7 if mean7 > 0 else 1.0
            total = float(k)
            out["merchant_category_entropy"] = -sum((c / total) * math.log2(c / total) for c in s["merchants"].values())
            out["is_new_merchant"] = 0.0 if m in s["merchants"] else 1.0
            out["hour_deviation_from_user_mean"] = abs(h - s["hour_sum"] / k)
            if s["last_large_day"] is not None:
                out["days_since_last_large_txn"] = d - s["last_large_day"]
            mean = s["amt_sum"] / k
            std = math.sqrt(max(s["amt_sq"] / k - mean * mean, 0.0))
            out["uid_amt_mean"], out["uid_amt_std"] = mean, std
            out["uid_amt_zscore"] = min(max((a - mean) / std, -20.0), 20.0) if std > 1e-3 else 0.0
            out["uid_secs_since_prev_tx"] = t - s["last_ts"]

        if commit:
            if s is None:
                s = self._s[uid] = {"n": 0, "recent": deque(maxlen=7), "merchants": {}, "hour_sum": 0.0,
                                    "amt_sum": 0.0, "amt_sq": 0.0, "last_large_day": None, "last_ts": 0.0}
            if k > 0 and a > 2 * (s["amt_sum"] / k):
                s["last_large_day"] = d
            s["recent"].append(a)
            s["merchants"][m] = s["merchants"].get(m, 0) + 1
            s["n"] = k + 1
            s["hour_sum"] += h
            s["amt_sum"] += a
            s["amt_sq"] += a * a
            s["last_ts"] = t
        return out

    def __len__(self):
        return len(self._s)


class EntityHistoryStore:
    """Lagged fraud history per entity key; counts only labels that have been recorded."""

    def __init__(self):
        self.tables: Dict[str, Dict[str, list]] = {k: {} for k in BASE_KEYS}
        self.pending: Dict[str, Dict[str, Optional[str]]] = {}

    def features(self, keys: Dict[str, Optional[str]], tag: str = "h7") -> Dict[str, float]:
        out = {}
        for name in BASE_KEYS:
            key = keys.get(name)
            row = self.tables[name].get(key) if key is not None else None
            if row:
                n, fraud = row
                rate = (fraud + PRIOR_WEIGHT * PRIOR_RATE) / (n + PRIOR_WEIGHT)
            else:
                n, fraud, rate = 0, 0, PRIOR_RATE
            out[f"{tag}_{name}_n"] = float(n)
            out[f"{tag}_{name}_fraud_n"] = float(fraud)
            out[f"{tag}_{name}_fraud_rate"] = float(rate)
        return out

    def register(self, tx_id: str, keys: Dict[str, Optional[str]]) -> None:
        self.pending[tx_id] = keys

    def add_label_for_keys(self, keys: Dict[str, Optional[str]], is_fraud: bool) -> None:
        for name in BASE_KEYS:
            key = keys.get(name)
            if key is None:
                continue
            row = self.tables[name].setdefault(key, [0, 0])
            row[0] += 1
            row[1] += int(is_fraud)

    def record_label(self, tx_id: str, is_fraud: bool) -> bool:
        keys = self.pending.pop(tx_id, None)
        if keys is None:
            return False
        self.add_label_for_keys(keys, is_fraud)
        return True
