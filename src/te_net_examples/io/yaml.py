from __future__ import annotations

from typing import Any

import yaml


def format_yaml(x: Any) -> str:
    return yaml.safe_dump(
        x,
        sort_keys=True,
        allow_unicode=True,
        default_flow_style=False,
    )


def write_yaml(path: str, x: Any) -> str:
    s = format_yaml(x)
    with open(path, "w", encoding="utf-8") as f:
        f.write(s)
        if not s.endswith("\n"):
            f.write("\n")
    return path


def read_yaml(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
