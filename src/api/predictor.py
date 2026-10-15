"""Scoring engine: raw transaction fields -> feature vector -> calibrated score, tier and reasons.

No pandas / sklearn: only numpy + the model library (LightGBM for the production model; XGBoost is still supported
for older model directories). Encoders (categories, frequency tables, calibrator)
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
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import numpy as np

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


CATEGORY_NAMES = {"card4": "card network", "card6": "card type", "ProductCD": "product code", "DeviceType": "device type",
                  "P_emaildomain": "payer email domain", "R_emaildomain": "recipient email domain", "id_30": "operating system",
                  "id_31": "browser", "DeviceInfo": "device"}
NO_HISTORY = {"uid_secs_since_prev_tx": "no earlier transaction seen for this user", "uid_amt_mean": "no spending history for this user yet",
              "uid_amt_std": "no spending history for this user yet", "days_since_last_large_txn": "no earlier large transaction for this user"}


def _describe(feature: str, value: Optional[float], labels: Optional[Dict[str, Dict[int, str]]] = None) -> str:
    v = "missing" if value is None else f"{value:g}"
    if value is not None and labels and feature in labels:                   # decode a category code back to its name
        label = labels[feature].get(int(round(value)))
        if label is not None:
            return f"{CATEGORY_NAMES.get(feature, feature)}: {label}"
    if value is not None and value == -1 and feature in NO_HISTORY:           # sentinel for "nothing to compare with yet"
        return NO_HISTORY[feature]
    if value is None and feature in CATEGORY_NAMES:
        return f"{CATEGORY_NAMES[feature]} not provided"
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
    if feature in names:
        return names[feature]
    if re.fullmatch(r"C\d+", feature):
        return f"transaction-count signal {feature} = {v}"
    if re.fullmatch(r"V\d+", feature):
        return f"Vesta risk signal {feature} = {v}"
    if re.fullmatch(r"D\d+", feature):
        return f"time-delta signal {feature}"
    return f"{feature} = {v}"


REQUIRED_NUMERIC = ("TransactionDT", "TransactionAmt")
OPTIONAL_NUMERIC = ("D1", "card1", "card2", "card3", "card5", "addr1")


def coerce_numeric_fields(fields: dict) -> dict:
    """Copy of `fields` with the numerics the state code relies on converted to float; anything that cannot be
    converted is a client error (422), never an internal error."""
    out = dict(fields)
    for name in REQUIRED_NUMERIC + OPTIONAL_NUMERIC:
        v = out.get(name)
        if is_missing(v):
            if name in REQUIRED_NUMERIC:
                raise PredictionError(f"missing required field '{name}'")
            continue
        try:
            out[name] = float(v)
        except (TypeError, ValueError):
            raise PredictionError(f"field '{name}' must be numeric, got {v!r}")
    return out


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
        self.backend, self.boosters = self._load_boosters(models_dir)
        self._labels = {c: {int(code): label for label, code in m.items()} for c, m in self.categories.items()}
        self.model_version = f"paywatch-compact-{self.backend}-{len(self.features)}f-{len(self.boosters)}seeds"
        self.behavior = UserBehaviorState()
        self.history = EntityHistoryStore()
        self.lock = threading.Lock()
        self._responses: "OrderedDict[str, dict]" = OrderedDict()      # idempotent retries (committed predictions only)
        self._max_cached = 20_000

    @staticmethod
    def _load_boosters(models_dir: str):
        lgb_paths = sorted(glob.glob(os.path.join(models_dir, "lgbm_seed*.txt")))
        if lgb_paths:
            import lightgbm as lgb
            return "lgbm", [lgb.Booster(model_file=p) for p in lgb_paths]
        xgb_paths = sorted(glob.glob(os.path.join(models_dir, "xgb_seed*.json")))
        if not xgb_paths:
            raise FileNotFoundError(f"no lgbm_seed*.txt or xgb_seed*.json in {models_dir}")
        import xgboost as xgb
        boosters = []
        for p in xgb_paths:
            b = xgb.Booster()
            b.load_model(p)
            boosters.append(b)
        return "xgb", boosters

    def _predict_one(self, booster, vec: np.ndarray, contribs: bool = False) -> np.ndarray:
        """Score (or per-feature contributions, bias last) of one model for one vector."""
        if self.backend == "lgbm":
            return booster.predict(vec.reshape(1, -1), pred_contrib=contribs)[0]
        import xgboost as xgb
        dm = xgb.DMatrix(vec.reshape(1, -1), feature_names=self.features)
        return booster.predict(dm, pred_contribs=contribs)[0]

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
        return float(np.mean([self._predict_one(b, vec) for b in self.boosters]))

    def calibrate(self, score: float) -> float:
        return float(np.interp(score, self._iso_x, self._iso_y))

    def action_for(self, p: float) -> str:
        for name, key in ACTIONS:
            thr = self.thresholds.get(key)
            if thr is not None and p >= thr:
                return name
        return "ALLOW"

    def explain(self, vec: np.ndarray, top_k: int) -> List[dict]:
        contrib = np.mean([self._predict_one(b, vec, contribs=True)[:-1] for b in self.boosters], axis=0)
        reasons = []
        for i in np.argsort(-np.abs(contrib))[:top_k]:
            name = self.features[i]
            value = None if float(vec[i]) == MISSING_FILL else float(vec[i])
            reasons.append({
                "feature": name, "family": FAMILY_TEXT.get(self.family.get(name, ""), self.family.get(name, "")),
                "value": value, "shap": float(contrib[i]),
                "direction": "increases_risk" if contrib[i] > 0 else "decreases_risk",
                "text": _describe(name, value, self._labels),
            })
        return reasons

    def predict(self, tx_id: str, fields: dict, explain: bool = True, commit: bool = True, top_k: int = 5,
                include_vector: bool = False) -> dict:
        fields = coerce_numeric_fields(fields)
        t0 = time.perf_counter()
        with self.lock:
            if commit and not include_vector and tx_id in self._responses:
                cached = dict(self._responses[tx_id])               # a retry must not count the transaction twice
                cached["latency_ms"] = (time.perf_counter() - t0) * 1000.0
                return cached
            keys = entity_keys(fields)
            behavior = self.behavior.observe(fields, keys["uid"], commit=False)
            entity = self.history.features(keys)
            vec = self.build_vector(fields, behavior, entity)
            score = self._score(vec)
            p = self.calibrate(score)
            reasons = self.explain(vec, top_k) if explain else []
            if commit:                                              # only after the request is known to be valid
                self.behavior.observe(fields, keys["uid"], commit=True)
                self.history.register(tx_id, keys)
            result = {
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
            if commit and not include_vector:
                self._responses[tx_id] = result
                while len(self._responses) > self._max_cached:
                    self._responses.popitem(last=False)
        if include_vector:
            result = {**result, "_vector": vec}
        return result

    def feedback(self, tx_id: str, is_fraud: bool) -> bool:
        with self.lock:
            return self.history.record_label(tx_id, is_fraud)

    def reset_state(self) -> None:
        with self.lock:
            self.behavior = UserBehaviorState()
            self.history = EntityHistoryStore()
            self._responses.clear()

    def load_state(self, path: str) -> dict:
        """Warm start from a snapshot written by snapshot.save_state. The snapshot is validated and loaded into fresh
        objects first, so a bad file leaves the live state untouched; the swap itself is atomic."""
        from .snapshot import import_state, load_state

        obj = load_state(path)
        behavior, history = UserBehaviorState(), EntityHistoryStore()
        import_state(obj, behavior, history)
        with self.lock:
            self.behavior, self.history = behavior, history
            self._responses.clear()
        return {"users": len(obj["users"]), "entity_keys": {k: len(v) for k, v in obj["entity_tables"].items()},
                "pending_labels": len(obj["pending"])}
