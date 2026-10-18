"""Save and load the UPI featurizer's state (behaviour, link counters, entity fraud history) as one gzip-JSON snapshot.

Reuses `src/api/snapshot.py` for the behaviour and entity-history parts and adds the link counters. A snapshot lets a restarted
service continue with its history instead of scoring every payer as new.
"""

import os
from collections import deque
from typing import Optional

from src.api.snapshot import export_state, import_state, load_state, save_state

from .features import KEY_NAMES, UpiFeaturizer

KIND = "upi-featurizer-1"


def export_featurizer(fz: UpiFeaturizer) -> dict:
    obj = export_state(fz.behavior, fz.history)
    l = fz.link
    obj["kind"] = KIND
    obj["link"] = {
        "payee_payers": {k: sorted(v) for k, v in l.payee_payers.items()}, "payee_seen": l.payee_seen, "payee_first": l.payee_first,
        "pair": l.pair, "device_payers": {k: sorted(v) for k, v in l.device_payers.items()},
        "payer_devices": {k: sorted(v) for k, v in l.payer_devices.items()}, "payer_recent": {k: list(v) for k, v in l.payer_recent.items()}}
    return obj


def import_featurizer(obj: dict, fz: Optional[UpiFeaturizer] = None) -> UpiFeaturizer:
    """Validates before swapping anything in; returns a NEW featurizer unless one is given."""
    if obj.get("kind") != KIND:
        raise ValueError(f"not a UPI featurizer snapshot (kind={obj.get('kind')!r})")
    fresh = UpiFeaturizer()
    import_state(obj, fresh.behavior, fresh.history)
    link = obj["link"]
    l = fresh.link
    l.payee_payers = {k: set(v) for k, v in link["payee_payers"].items()}
    l.payee_seen = {k: int(v) for k, v in link["payee_seen"].items()}
    l.payee_first = {k: float(v) for k, v in link["payee_first"].items()}
    l.pair = {k: int(v) for k, v in link["pair"].items()}
    l.device_payers = {k: set(v) for k, v in link["device_payers"].items()}
    l.payer_devices = {k: set(v) for k, v in link["payer_devices"].items()}
    l.payer_recent = {k: deque(v) for k, v in link["payer_recent"].items()}
    if fz is not None:
        fz.behavior, fz.link, fz.history = fresh.behavior, fresh.link, fresh.history
        return fz
    return fresh


def save_featurizer(fz: UpiFeaturizer, path: str) -> int:
    """Atomic write (temp file then rename) so a crash mid-save never leaves a corrupt snapshot."""
    tmp = path + ".tmp"
    size = save_state(export_featurizer(fz), tmp)
    os.replace(tmp, path)
    return size


def load_featurizer(path: str) -> UpiFeaturizer:
    return import_featurizer(load_state(path))
