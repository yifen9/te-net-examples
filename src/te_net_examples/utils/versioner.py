from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from te_net_examples.io.json import read_json, write_json


def _dir_ts(ts: str) -> str:
    dt = datetime.fromisoformat(ts)
    return dt.strftime("%Y%m%dT%H%M%S")


def build_version_dir(base_dir: str, meta: dict[str, Any]) -> str:
    ts = _dir_ts(meta["timestamp"])
    fp = meta["fingerprint"]
    version_dir = os.path.join(base_dir, f"{ts}_{fp}")
    os.makedirs(version_dir, exist_ok=True)
    write_json(os.path.join(version_dir, "_meta.json"), meta)
    return version_dir


def list_metas(base_dir: str) -> list[dict[str, Any]]:
    metas: list[dict[str, Any]] = []
    if not os.path.isdir(base_dir):
        return metas
    for name in sorted(os.listdir(base_dir)):
        p = os.path.join(base_dir, name, "_meta.json")
        if os.path.isfile(p):
            x = read_json(p)
            if isinstance(x, dict):
                metas.append(x)
    return metas


def index_dir(base_dir: str, meta: dict[str, Any]) -> str | None:
    if not os.path.isdir(base_dir):
        return None
    for name in sorted(os.listdir(base_dir)):
        d = os.path.join(base_dir, name)
        p = os.path.join(d, "_meta.json")
        if os.path.isfile(p):
            if read_json(p) == meta:
                return d
    return None
