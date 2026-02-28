from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from te_net_examples.io.json import read_json


@dataclass(slots=True)
class TraceNode:
    run_dir: str
    meta: dict[str, Any]
    field: str | None
    prev_run_dir: str | None


def _meta_path(run_dir: str) -> str:
    return os.path.join(run_dir, "_meta.json")


def _read_meta(run_dir: str) -> dict[str, Any]:
    p = _meta_path(run_dir)
    x = read_json(p)
    if not isinstance(x, dict):
        raise TypeError(f"_meta.json must be an object: {p}")
    return x


def trace(
    run_dir: str, *, field: str = "input_dir", limit: int = 256
) -> list[TraceNode]:
    out: list[TraceNode] = []
    seen: set[str] = set()

    cur = os.path.abspath(run_dir)

    for _ in range(limit):
        if cur in seen:
            break
        seen.add(cur)

        mp = _meta_path(cur)
        if not os.path.isfile(mp):
            break

        meta = _read_meta(cur)
        params = meta.get("params")
        prev = None
        if isinstance(params, dict):
            v = params.get(field)
            if isinstance(v, str) and v:
                prev = os.path.abspath(v)

        out.append(
            TraceNode(
                run_dir=cur,
                meta=meta,
                field=field if prev is not None else None,
                prev_run_dir=prev,
            )
        )

        if prev is None:
            break
        cur = prev

    return out


def trace_metas(
    run_dir: str, *, field: str = "input_dir", limit: int = 256
) -> list[dict[str, Any]]:
    return [n.meta for n in trace(run_dir, field=field, limit=limit)]
