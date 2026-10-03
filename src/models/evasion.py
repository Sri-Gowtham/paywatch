"""Evasion moves shared by the stress test and the adversarial-training experiment.

A move copies a legitimate donor row's values for the feature group an attacker controls (including the derived
state it drives) or resets state to what a brand-new identity looks like. Pure numpy; works on any feature order.
"""

from typing import Callable, Dict, List, Sequence

import numpy as np

PRIOR = 0.035


def build_groups(cols: Sequence[str]) -> Dict[str, Dict]:
    idx = set(cols)

    def has(*names):
        return [n for n in names if n in idx]

    def with_prefix(*prefixes):
        return [c for c in cols if c.startswith(prefixes)]

    groups = {
        "amount_mimic": {"copy": has("TransactionAmt"), "set": {"amount_vs_7day_avg_ratio": 1.0, "uid_amt_zscore": 0.0}},
        "email_rotation": {"copy": with_prefix("h7_email_", "P_emaildomain", "R_emaildomain") + has("merchant_category_entropy"),
                           "set": {"is_new_merchant": 0.0}},
        "device_rotation": {"copy": has("DeviceType", "id_30", "id_31") + with_prefix("h7_device_"), "set": {}},
        "behavior_mimic": {"copy": has("days_since_last_large_txn", "uid_secs_since_prev_tx", "merchant_category_entropy"),
                           "set": {"hour_deviation_from_user_mean": 0.0}},
        "identity_rotation": {
            "copy": [c for c in cols if c in ("card1_freq", "card2_freq", "card3_freq", "card5_freq")],
            "set": {**{c: 0.0 for c in with_prefix("h7_uid_", "h7_card_", "h7_addr_") if c.endswith(("_n", "_fraud_n"))},
                    **{c: PRIOR for c in with_prefix("h7_uid_", "h7_card_", "h7_addr_") if c.endswith("_fraud_rate")},
                    "uid_prior_tx_count": 0.0, "uid_amt_mean": -1.0, "uid_amt_std": -1.0, "uid_amt_zscore": 0.0,
                    "uid_secs_since_prev_tx": -1.0, "amount_vs_7day_avg_ratio": 1.0, "merchant_category_entropy": 0.0,
                    "hour_deviation_from_user_mean": 0.0, "days_since_last_large_txn": -1.0, "is_new_merchant": 1.0},
        },
        "product_card_swap": {"copy": has("ProductCD", "card4", "card6", "is_p2p"), "set": {}},
    }
    for g in groups.values():
        g["set"] = {k: v for k, v in g["set"].items() if k in idx}
    return groups


def apply_moves(X: np.ndarray, cols: Sequence[str], groups: Dict, names: List[str], donors: np.ndarray) -> np.ndarray:
    index = {c: i for i, c in enumerate(cols)}
    Xa = X.copy()
    for name in names:
        g = groups[name]
        for c in g["copy"]:
            Xa[:, index[c]] = donors[:, index[c]]
        for c, v in g["set"].items():
            Xa[:, index[c]] = v
    return Xa


def attack(score_fn: Callable[[np.ndarray], np.ndarray], X0: np.ndarray, cols: Sequence[str], groups: Dict,
           names: List[str], donors_all: np.ndarray, rng: np.random.Generator, budgets: Sequence[int]) -> Dict[int, np.ndarray]:
    """Adaptive black-box attacker: per query draw one random donor per target, keep the lowest score seen so far.
    Returns {budget: best score per target}."""
    best = np.full(len(X0), np.inf)
    out = {}
    for q in range(1, max(budgets) + 1):
        donors = donors_all[rng.integers(0, len(donors_all), len(X0))]
        best = np.minimum(best, score_fn(apply_moves(X0, cols, groups, names, donors)))
        if q in budgets:
            out[q] = best.copy()
    return out
