from __future__ import annotations

import json
from typing import Any, Iterable


def format_jsonl(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def append_jsonl(path: str, x: Any) -> str:
    with open(path, "a", encoding="utf-8") as f:
        f.write(format_jsonl(x))
        f.write("\n")
    return path


def write_jsonl(path: str, xs: Iterable[Any]) -> str:
    with open(path, "w", encoding="utf-8") as f:
        for x in xs:
            f.write(format_jsonl(x))
            f.write("\n")
    return path


def read_jsonl(path: str) -> list[Any]:
    out: list[Any] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                out.append(json.loads(s))
    return out
