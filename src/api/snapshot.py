"""Export / import of the API's in-memory state (user behaviour + entity fraud history).

Used for warm starts: a fresh process otherwise scores every user as new and every entity as
unseen, which costs a lot of accuracy (cold-start replay PR-AUC 0.548 vs 0.706 offline).
The snapshot is plain gzip-JSON; pruning options trade size for fidelity.
"""

import gzip
import json
from typing import Iterable, Optional

from .state import BASE_KEYS, EntityHistoryStore, UserBehaviorState

VERSION = 1


def export_state(behavior: UserBehaviorState, history: EntityHistoryStore, active_uids: Optional[Iterable[str]] = None,
                 min_entity_n: int = 1) -> dict:
    """active_uids: keep behaviour state only for these users (None = all).
    min_entity_n: drop entity keys seen fewer times unless they carry a fraud label."""
    keep = set(active_uids) if active_uids is not None else None
    users = {}
    for uid, s in behavior._s.items():
        if keep is not None and uid not in keep:
            continue
        users[uid] = [s["n"], list(s["recent"]), s["merchants"], s["hour_sum"], s["amt_sum"], s["amt_sq"],
                      s["last_large_day"], s["last_ts"]]
    tables = {name: {k: v for k, v in history.tables[name].items() if v[0] >= min_entity_n or v[1] > 0}
              for name in BASE_KEYS}
    return {"version": VERSION, "users": users, "entity_tables": tables, "pending": history.pending}


def import_state(obj: dict, behavior: UserBehaviorState, history: EntityHistoryStore) -> None:
    if obj.get("version") != VERSION:
        raise ValueError(f"unsupported snapshot version {obj.get('version')}")
    from collections import deque

    for uid, (n, recent, merchants, hour_sum, amt_sum, amt_sq, last_large_day, last_ts) in obj["users"].items():
        behavior._s[uid] = {"n": n, "recent": deque(recent, maxlen=7), "merchants": merchants, "hour_sum": hour_sum,
                            "amt_sum": amt_sum, "amt_sq": amt_sq, "last_large_day": last_large_day, "last_ts": last_ts}
    for name in BASE_KEYS:
        history.tables[name] = {k: list(v) for k, v in obj["entity_tables"].get(name, {}).items()}
    history.pending = dict(obj.get("pending", {}))


def save_state(obj: dict, path: str) -> int:
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as f:
        json.dump(obj, f, separators=(",", ":"))
    import os
    return os.path.getsize(path)


def load_state(path: str) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)
