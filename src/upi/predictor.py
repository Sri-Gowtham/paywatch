"""Scoring engine for the UPI toolkit: transaction -> calibrated probability, advisory tier and reasons.

The feature code is `features.UpiFeaturizer`, the same code the training replay used. Requires lightgbm and numpy only.
"""

import glob
import json
import os
import re
import threading
import time
from collections import OrderedDict
from typing import Dict, List, Optional

import numpy as np

from .features import FEATURE_NAMES, UpiFeaturizer
from .shadow_log import DecisionLog
from .state_io import import_featurizer, save_featurizer
from src.api.snapshot import load_state

ACTIONS = (("HARD_BLOCK", "0.9"), ("CHALLENGE", "0.7"), ("SOFT_FLAG", "0.5"))
WHO = {"payer": "payer", "payee": "payee", "pair": "payer-payee pair", "device": "device"}


def describe(name: str, value: Optional[float]) -> str:
    v = 0.0 if value is None else float(value)
    m = re.fullmatch(r"h7_(\w+?)_(fraud_rate|fraud_n|n)", name)
    if m:
        who = WHO.get(m.group(1), m.group(1))
        if m.group(2) == "fraud_rate":
            return f"this {who} had a {v:.0%} fraud rate in earlier labelled transactions"
        if m.group(2) == "fraud_n":
            return f"{int(v)} confirmed fraud cases earlier linked to this {who}"
        return f"{int(v)} earlier labelled transactions linked to this {who}"
    if name == "uid_secs_since_prev_tx":
        return "first payment seen for this payer" if v < 0 else f"{v / 60:.0f} minutes since this payer's previous payment"
    if name == "device_distinct_payers":
        return "no device information" if v < 0 else f"{int(v)} different payers seen on this device"
    if name == "payee_age_s":
        return "payee not seen before" if v < 0 else f"payee first seen {v / 86400:.1f} days ago"
    texts = {
        "amount": f"amount = Rs {v:,.0f}", "hour": f"hour of day = {int(v)}", "channel_code": "payment channel",
        "is_collect": "collect request" if v == 1 else "payer-initiated payment",
        "amount_vs_7day_avg_ratio": f"amount is {v:.1f}x this payer's recent average",
        "is_new_merchant": "first payment to this payee" if v == 1 else "payee paid before",
        "merchant_category_entropy": f"payee variety for this payer = {v:.2f}",
        "pair_count": f"payer has paid this payee {int(v)} time(s) before",
        "payee_distinct_payers": f"{int(v)} different payers have paid this payee",
        "payee_seen": f"payee has received {int(v)} earlier payments",
        "payer_distinct_devices": f"{int(v)} devices seen for this payer",
        "payer_tx_1h": f"{int(v)} payments by this payer in the last hour",
        "payer_tx_24h": f"{int(v)} payments by this payer in the last day",
        "uid_prior_tx_count": f"{int(v)} earlier payments seen for this payer",
        "uid_amt_zscore": f"amount z-score vs this payer's history = {v:.1f}",
    }
    return texts.get(name, f"{name} = {v:g}")


class UpiPredictor:
    def __init__(self, model_dir: str, log: Optional[DecisionLog] = None, state_path: Optional[str] = None, log_features: bool = False):
        import lightgbm as lgb

        spec = json.load(open(os.path.join(model_dir, "serving_spec.json"), encoding="utf-8"))
        if spec.get("kind") != "upi":
            raise ValueError("serving_spec.json is not a UPI model (kind != 'upi'); train one with src/upi/train.py")
        if spec["features"] != list(FEATURE_NAMES):
            raise ValueError("the model was trained with a different feature list than this code version produces")
        paths = sorted(glob.glob(os.path.join(model_dir, "lgbm_seed*.txt")))
        if not paths:
            raise FileNotFoundError(f"no lgbm_seed*.txt in {model_dir}")
        self.boosters = [lgb.Booster(model_file=p) for p in paths]
        iso = json.load(open(os.path.join(model_dir, "isotonic.json"), encoding="utf-8"))
        self._ix, self._iy = np.asarray(iso["x"]), np.asarray(iso["y"])
        self.spec = spec
        self.thresholds = {k: t["calibrated_threshold"] for _, k in ACTIONS for t in [spec["tiers"].get(k)] if t}
        self.model_version = f"paywatch-upi-lgbm-{len(FEATURE_NAMES)}f-{len(self.boosters)}seeds"
        self.synthetic = bool(spec.get("trained_on", {}).get("synthetic"))
        self.log, self.log_features, self.lock = log, log_features, threading.Lock()
        self._cache: "OrderedDict[str, dict]" = OrderedDict()
        self.fz = UpiFeaturizer()
        for candidate in (state_path, os.path.join(model_dir, "warm_state.json.gz")):
            if candidate and os.path.exists(candidate):
                import_featurizer(load_state(candidate), self.fz)
                break

    def action_for(self, p: float) -> str:
        for name, key in ACTIONS:
            thr = self.thresholds.get(key)
            if thr is not None and p >= thr:
                return name
        return "ALLOW"

    def _reasons(self, vec: np.ndarray, top_k: int) -> List[dict]:
        contrib = np.mean([b.predict(vec.reshape(1, -1), pred_contrib=True)[0][:-1] for b in self.boosters], axis=0)
        out = []
        for i in np.argsort(-np.abs(contrib))[:top_k]:
            name = FEATURE_NAMES[i]
            out.append({"feature": name, "value": float(vec[i]), "shap": float(contrib[i]),
                        "direction": "increases_risk" if contrib[i] > 0 else "decreases_risk", "text": describe(name, float(vec[i]))})
        return out

    def predict(self, txn: dict, commit: bool = True, explain: bool = True, top_k: int = 5) -> dict:
        t0 = time.perf_counter()
        tx_id = str(txn["transaction_id"])
        with self.lock:
            if commit and tx_id in self._cache:                           # a retry must not count the transaction twice
                cached = dict(self._cache[tx_id])
                cached["latency_ms"] = (time.perf_counter() - t0) * 1000.0
                return cached
            vec, _ = self.fz.featurize(txn, commit=False)
            score = float(np.mean([b.predict(vec.reshape(1, -1))[0] for b in self.boosters]))
            p = float(np.interp(score, self._ix, self._iy))
            action = self.action_for(p)
            reasons = self._reasons(vec, top_k) if explain else []
            if commit:
                self.fz.featurize(txn, commit=True, tx_id=tx_id)           # only after the request is known to be valid
                if self.log is not None:
                    self.log.decision(tx_id, float(txn["timestamp"]), self.model_version, score, p, action,
                                      vec.tolist() if self.log_features else None)
            result = {"transaction_id": tx_id, "score": score, "calibrated_probability": p, "action": action, "shadow": True,
                      "advisory": "shadow mode: this tier is a recommendation only and must not block or delay a payment",
                      "reasons": reasons, "model_version": self.model_version, "synthetic_model": self.synthetic,
                      "latency_ms": (time.perf_counter() - t0) * 1000.0}
            if commit:
                self._cache[tx_id] = result
                while len(self._cache) > 20_000:
                    self._cache.popitem(last=False)
        return result

    def feedback(self, tx_id: str, is_fraud: bool) -> bool:
        with self.lock:
            ok = self.fz.label(str(tx_id), is_fraud)
            if ok and self.log is not None:
                self.log.label(str(tx_id), is_fraud)
            return ok

    def save_state(self, path: str) -> int:
        with self.lock:
            return save_featurizer(self.fz, path)
