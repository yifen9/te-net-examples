from __future__ import annotations

import json
from typing import Any, Mapping


def jdump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def jline(event: str, component: str, msg: str, **kw: Any) -> str:
    if not isinstance(event, str) or not event.strip():
        raise ValueError("event must be a non-empty string")
    if not isinstance(component, str) or not component.strip():
        raise ValueError("component must be a non-empty string")
    if not isinstance(msg, str) or not msg.strip():
        raise ValueError("msg must be a non-empty string")
    obj: dict[str, Any] = {"event": event, "component": component, "msg": msg, **kw}
    return jdump(obj)


def is_json_mapping(text: str) -> bool:
    try:
        x = json.loads(text)
    except Exception:
        return False
    return isinstance(x, Mapping)
