"""UPI features: ONE code path used for training (replay) and serving (live), so they cannot drift apart.

State kept per process, updated only after a transaction's features are fixed (nothing from the future ever leaks in):
  * `UserBehaviorState` (src/api/state.py): the payer's spending history; the payee plays the role of the merchant.
  * `UpiLinkState`: who pays whom: distinct payers per payee and per device, repeat-payee count, velocity.
  * `EntityHistoryStore` (src/api/state.py): lagged fraud history per payer, payee, payer-payee pair and device. Labels are applied when
    they ARRIVE (`label`), which in a replay is `label_lag_days` after the transaction, as in production.
"""

import heapq
from collections import deque
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

from src.api.state import EntityHistoryStore, UserBehaviorState

KEY_NAMES = ("payer", "payee", "pair", "device")
BEHAVIOR = ("amount_vs_7day_avg_ratio", "merchant_category_entropy", "hour_deviation_from_user_mean", "days_since_last_large_txn",
            "is_new_merchant", "uid_prior_tx_count", "uid_amt_mean", "uid_amt_std", "uid_amt_zscore", "uid_secs_since_prev_tx")
LINK = ("payee_distinct_payers", "payee_seen", "payee_age_s", "pair_count", "device_distinct_payers", "payer_distinct_devices",
        "payer_tx_1h", "payer_tx_24h")
RAW = ("amount", "hour", "day_of_week", "channel_code", "is_collect")
HISTORY = tuple(f"h7_{k}_{s}" for k in KEY_NAMES for s in ("n", "fraud_n", "fraud_rate"))
FEATURE_NAMES: Tuple[str, ...] = RAW + BEHAVIOR + LINK + HISTORY
CHANNEL_CODE = {"app": 0.0, "qr": 1.0, "collect": 2.0, "intent": 3.0}


class UpiLinkState:
    """Graph-style counters. Memory grows with the number of distinct payers, payees and devices (prune by snapshotting if needed)."""

    def __init__(self):
        self.payee_payers: Dict[str, set] = {}
        self.payee_seen: Dict[str, int] = {}
        self.payee_first: Dict[str, float] = {}
        self.pair: Dict[str, int] = {}
        self.device_payers: Dict[str, set] = {}
        self.payer_devices: Dict[str, set] = {}
        self.payer_recent: Dict[str, deque] = {}

    def observe(self, payer: str, payee: str, device: Optional[str], t: float, commit: bool) -> Dict[str, float]:
        recent = self.payer_recent.get(payer)
        n1h = n24h = 0
        if recent:
            n24h = sum(1 for x in recent if t - x <= 86400)
            n1h = sum(1 for x in recent if t - x <= 3600)
        out = {
            "payee_distinct_payers": float(len(self.payee_payers.get(payee, ()))),
            "payee_seen": float(self.payee_seen.get(payee, 0)),
            "payee_age_s": float(t - self.payee_first[payee]) if payee in self.payee_first else -1.0,
            "pair_count": float(self.pair.get(f"{payer}>{payee}", 0)),
            "device_distinct_payers": float(len(self.device_payers.get(device, ()))) if device else -1.0,
            "payer_distinct_devices": float(len(self.payer_devices.get(payer, ()))),
            "payer_tx_1h": float(n1h), "payer_tx_24h": float(n24h),
        }
        if commit:
            self.payee_payers.setdefault(payee, set()).add(payer)
            self.payee_seen[payee] = self.payee_seen.get(payee, 0) + 1
            self.payee_first.setdefault(payee, t)
            self.pair[f"{payer}>{payee}"] = self.pair.get(f"{payer}>{payee}", 0) + 1
            if device:
                self.device_payers.setdefault(device, set()).add(payer)
                self.payer_devices.setdefault(payer, set()).add(device)
            dq = self.payer_recent.setdefault(payer, deque())
            dq.append(t)
            while dq and t - dq[0] > 86400:
                dq.popleft()
        return out


def entity_keys(txn: dict) -> Dict[str, Optional[str]]:
    payer, payee = txn["payer_id"], txn["payee_id"]
    return {"payer": payer, "payee": payee, "pair": f"{payer}>{payee}", "device": txn.get("device_id") or None}


class UpiFeaturizer:
    def __init__(self):
        self.behavior = UserBehaviorState()
        self.link = UpiLinkState()
        self.history = EntityHistoryStore(KEY_NAMES)

    def featurize(self, txn: dict, commit: bool = True, tx_id: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, Optional[str]]]:
        """Feature vector (order = FEATURE_NAMES) and the entity keys. With commit=True the state is updated afterwards."""
        t = float(txn["timestamp"])
        keys = entity_keys(txn)
        beh = self.behavior.observe_values(float(txn["amount"]), t, txn.get("payee_id") or "none", txn["payer_id"], commit=commit)
        lnk = self.link.observe(txn["payer_id"], txn["payee_id"], keys["device"], t, commit)
        ent = self.history.features(keys)
        channel = txn.get("channel") or "app"
        raw = {"amount": float(txn["amount"]), "hour": float(t // 3600 % 24), "day_of_week": float((t // 86400 + 4) % 7),
               "channel_code": CHANNEL_CODE.get(channel, 0.0), "is_collect": 1.0 if channel == "collect" else 0.0}
        row = {**raw, **beh, **lnk, **ent}
        vec = np.asarray([row[n] for n in FEATURE_NAMES], dtype=np.float32)
        if commit and tx_id is not None:
            self.history.register(tx_id, keys)
        return vec, keys

    def label(self, tx_id: str, is_fraud: bool) -> bool:
        return self.history.record_label(tx_id, is_fraud)


def replay(rows: Iterable[dict], label_lag_days: float = 7.0, featurizer: Optional["UpiFeaturizer"] = None
           ) -> Iterator[Tuple[np.ndarray, int, dict]]:
    """Walk time-ordered rows exactly as production would: features first, then the transaction joins the state, and its label only
    becomes visible `label_lag_days` later. Yields (features, label, row). Rows need transaction_id, timestamp, amount, payer_id,
    payee_id, optional device_id/channel, and is_fraud. Pass `featurizer` to keep the state the replay ends in."""
    fz = featurizer if featurizer is not None else UpiFeaturizer()   # pass one in to keep the final state (warm start)
    pending: List[Tuple[float, str, int]] = []
    lag = label_lag_days * 86400.0
    last_t = -1.0
    for row in rows:
        t = float(row["timestamp"])
        if t < last_t:
            raise ValueError("rows must be sorted by timestamp")
        last_t = t
        while pending and pending[0][0] <= t:
            _, tx, lab = heapq.heappop(pending)
            fz.label(tx, bool(lab))
        vec, _ = fz.featurize(row, commit=True, tx_id=str(row["transaction_id"]))
        label = int(row["is_fraud"])
        heapq.heappush(pending, (t + lag, str(row["transaction_id"]), label))
        yield vec, label, row
