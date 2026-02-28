from __future__ import annotations

import json
from typing import Any


def format_json(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, sort_keys=True, indent=2)


def write_json(path: str, x: Any) -> str:
    s = format_json(x)
    with open(path, "w", encoding="utf-8") as f:
        f.write(s)
        f.write("\n")
    return path


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
