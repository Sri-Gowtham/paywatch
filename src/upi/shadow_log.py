"""Append-only decision log for shadow mode.

One JSON object per line. Each line holds a keyed hash of the transaction id (so a later confirmed-fraud label can be joined to it) and
the model's outputs (score, calibrated probability, tier). It holds NO raw identifiers or request fields. The feature vector is logged
ONLY when explicitly enabled (`log_features`), because the vector contains the transaction amount, the hour and behavioural aggregates;
turn it on only if the partner's data policy allows it. The file rotates by size.
"""

import json
import os
import threading
import time
from typing import List, Optional

from .schema import pseudonymise


class DecisionLog:
    def __init__(self, path: str, salt: str, rotate_mb: float = 50.0):
        if not salt:
            raise ValueError("a non-empty log salt is required (PAYWATCH_LOG_SALT): it keys the transaction-id hash")
        self.path, self._salt, self._max = path, salt, int(rotate_mb * 1_000_000)
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def tx_hash(self, transaction_id: str) -> str:
        return pseudonymise(transaction_id, self._salt)

    def _write(self, obj: dict) -> None:
        line = json.dumps(obj, separators=(",", ":")) + "\n"
        with self._lock:
            if os.path.exists(self.path) and os.path.getsize(self.path) + len(line) > self._max:
                os.replace(self.path, f"{self.path}.{int(time.time())}")
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line)

    def decision(self, transaction_id: str, event_time: float, model_version: str, score: float, probability: float, tier: str,
                 features: Optional[List[float]] = None) -> None:
        rec = {"event": "decision", "tx": self.tx_hash(transaction_id), "ts": event_time, "logged_at": time.time(),
               "model": model_version, "score": round(score, 6), "p": round(probability, 6), "tier": tier}
        if features is not None:
            rec["x"] = [round(float(v), 5) for v in features]
        self._write(rec)

    def label(self, transaction_id: str, is_fraud: bool) -> None:
        self._write({"event": "label", "tx": self.tx_hash(transaction_id), "logged_at": time.time(), "is_fraud": int(bool(is_fraud))})
