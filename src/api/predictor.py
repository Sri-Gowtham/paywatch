"""Scoring engine: raw transaction fields -> feature vector -> calibrated score, tier and reasons.

No pandas / sklearn: only numpy + xgboost. Encoders (categories, frequency tables, calibrator)
are loaded from the JSON assets exported by the assets kernel and must match the batch pipeline
(checked by tests/test_predictor_parity.py).
"""

import glob
import json
import math
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np
import xgboost as xgb

from .state import EntityHistoryStore, UserBehaviorState, entity_keys, is_missing

MISSING_FILL = -999.0
ACTIONS = (("HARD_BLOCK", "0.9"), ("CHALLENGE", "0.7"), ("SOFT_FLAG", "0.5"))

FAMILY_TEXT = {
    "entity_fraud_history": "entity fraud history",
    "upi_behavior": "user behaviour",
    "frequency_encoding": "how common this card/address/email is",
    "time_deltas_D": "time-delta signal",
    "raw_field": "transaction field",
}


class PredictionError(ValueError):
    pass


def _describe(feature: str, value: Optional[float]) -> str:
    v = "missing" if value is None else f"{value:g}"
    m = re.fullmatch(r"h7_(\w+?)_(fraud_rate|fraud_n|n)", feature)
    if m and value is not None:
        who = {"uid": "user", "card": "card", "addr": "address", "email": "email domain", "device": "device"}.get(m.group(1), m.group(1))
        if m.group(2) == "fraud_rate":
            return f"this {who} had a {value:.0%} fraud rate in earlier labeled transactions"
        if m.group(2) == "fraud_n":
            return f"{int(value)} confirmed fraud cases previously linked to this {who}"
        return f"{int(value)} earlier labeled transactions linked to this {who}"
    names = {
        "TransactionAmt": f"transaction amount = {v}",
        "uid_secs_since_prev_tx": f"seconds since this user's previous transaction = {v}",
        "amount_vs_7day_avg_ratio": f"amount is {v}x this user's recent average",
        "uid_amt_zscore": f"amount z-score vs this user's history = {v}",
        "merchant_category_entropy": f"payee variety for this user = {v}",
        "is_new_merchant": "first time this user pays this payee domain" if value == 1 else "known payee domain",
        "uid_prior_tx_count": f"{v} earlier transactions seen for this user",
    }
    return names.get(feature, f"{feature} = {v}")


class Predictor:
    def __init__(self, models_dir: str):
        load = lambda n: json.load(open(os.path.join(models_dir, n)))  # noqa: E731
        meta = load("feature_meta.json")
        spec = load("serving_spec.json")
        self.features: List[str] = meta["features"]
        self.family: Dict[str, str] = meta["feature_family"]
        self.source: Dict[str, str] = meta["feature_source"]
        self.required_fields: List[str] = meta["required_raw_fields"]
        cats = load("categories.json")
        self.categories = {c: {v: float(i) for i, v in enumerate(cats[c])} for c in meta["categorical_raw_fields"]}
        freq_tables = load("freq_tables.json")
        self.freq = {c: freq_tables[c] for c in meta["freq_columns"]}
        iso = load("isotonic.json")
        self._iso_x, self._iso_y = np.asarray(iso["x"]), np.asarray(iso["y"])
        self.thresholds = {k: spec["tiers"][k]["calibrated_threshold"] for _, k in ACTIONS if spec["tiers"].get(k)}
        paths = sorted(glob.glob(os.path.join(models_dir, "xgb_seed*.json")))
        if not paths:
            raise FileNotFoundError(f"no xgb_seed*.json in {models_dir}")
        self.boosters = []
        for p in paths:
            b = xgb.Booster()
            b.load_model(p)
            self.boosters.append(b)
        self.model_version = f"paywatch-compact-{len(self.features)}f-{len(self.boosters)}seeds"
        self.behavior = UserBehaviorState()
        self.history = EntityHistoryStore()
        self.lock = threading.Lock()

    # ------------------------------------------------------------------ encoding
    def _encode(self, col: str, v: Any) -> float:
        if col in self.categories:
            return float("nan") if is_missing(v) else self.categories[col].get(v, float("nan"))
        if is_missing(v):
            return float("nan")
        try:
            return float(v)
        except (TypeError, ValueError):
            raise PredictionError(f"field '{col}' must be numeric, got {v!r}")

    def build_vector(self, fields: dict, behavior: dict, entity: dict) -> np.ndarray:
        day = float(fields["TransactionDT"]) / 86400.0
        values = []
        for f in self.features:
            src = self.source[f]
            if src == "raw_request_field":
                v = (1.0 if fields.get("ProductCD") == "C" else 0.0) if f == "is_p2p" else self._encode(f, fields.get(f))
            elif src == "anchored_D":
                d = self._encode(f, fields.get(f))
                v = day - d
            elif src == "frequency_lookup":
                col = f[: -len("_freq")]
                code = self._encode(col, fields.get(col))
                v = 0.0 if math.isnan(code) else self.freq[col].get(repr(float(code)), 0.0)
            elif src == "user_behavior_state":
                v = behavior[f]
            elif src == "entity_history_state":
                v = entity[f]
            else:
                raise PredictionError(f"unknown feature source {src!r} for {f}")
            values.append(MISSING_FILL if (isinstance(v, float) and math.isnan(v)) else v)
        return np.asarray(values, dtype=np.float32)

    # ------------------------------------------------------------------ scoring
    def _score(self, vec: np.ndarray) -> float:
        dm = xgb.DMatrix(vec.reshape(1, -1), feature_names=self.features)
        return float(np.mean([b.predict(dm)[0] for b in self.boosters]))

    def calibrate(self, score: float) -> float:
        return float(np.interp(score, self._iso_x, self._iso_y))

    def action_for(self, p: float) -> str:
        for name, key in ACTIONS:
            thr = self.thresholds.get(key)
            if thr is not None and p >= thr:
                return name
        return "ALLOW"

    def explain(self, vec: np.ndarray, top_k: int) -> List[dict]:
        dm = xgb.DMatrix(vec.reshape(1, -1), feature_names=self.features)
        contrib = np.mean([b.predict(dm, pred_contribs=True)[0][:-1] for b in self.boosters], axis=0)
        reasons = []
        for i in np.argsort(-np.abs(contrib))[:top_k]:
            name = self.features[i]
            value = None if float(vec[i]) == MISSING_FILL else float(vec[i])
            reasons.append({
                "feature": name, "family": FAMILY_TEXT.get(self.family.get(name, ""), self.family.get(name, "")),
                "value": value, "shap": float(contrib[i]),
                "direction": "increases_risk" if contrib[i] > 0 else "decreases_risk",
                "text": _describe(name, value),
            })
        return reasons

    def predict(self, tx_id: str, fields: dict, explain: bool = True, commit: bool = True, top_k: int = 5) -> dict:
        for req in ("TransactionDT", "TransactionAmt"):
            if is_missing(fields.get(req)):
                raise PredictionError(f"missing required field '{req}'")
        t0 = time.perf_counter()
        with self.lock:
            keys = entity_keys(fields)
            behavior = self.behavior.observe(fields, keys["uid"], commit=commit)
            entity = self.history.features(keys)
            vec = self.build_vector(fields, behavior, entity)
            score = self._score(vec)
            p = self.calibrate(score)
            reasons = self.explain(vec, top_k) if explain else []
            if commit:
                self.history.register(tx_id, keys)
        return {
            "transaction_id": tx_id, "score": score, "calibrated_probability": p, "action": self.action_for(p),
            "reasons": reasons,
            "flags": {
                "new_user": behavior["uid_prior_tx_count"] == 0,
                "uid_has_labeled_history": entity["h7_uid_n"] > 0,
                "uid_has_prior_fraud": entity["h7_uid_fraud_n"] > 0,
                "weak_identity": behavior["uid_is_strong"] == 0,
            },
            "model_version": self.model_version,
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
        }

    def feedback(self, tx_id: str, is_fraud: bool) -> bool:
        with self.lock:
            return self.history.record_label(tx_id, is_fraud)
