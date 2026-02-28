from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any


def build_meta(
    *, params: dict[str, Any], env: str, script: str, src: str, cfg: str | None = None
) -> dict[str, Any]:
    def sha256_hex(data: bytes) -> str:
        h = hashlib.sha256()
        h.update(data)
        return h.hexdigest()

    def read_bytes(path: str) -> bytes:
        with open(path, "rb") as f:
            return f.read()

    def canonical_json_bytes(obj: Any) -> bytes:
        return json.dumps(
            obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")

    def hash_tree(root: str) -> str:
        files: list[str] = []
        for r, _, fs in os.walk(root):
            fs.sort()
            for f in fs:
                files.append(os.path.join(r, f))
        files.sort()
        h = hashlib.sha256()
        for p in files:
            rel = os.path.relpath(p, root).replace(os.sep, "/").encode("utf-8")
            h.update(rel)
            h.update(b"\0")
            h.update(read_bytes(p))
            h.update(b"\0")
        return h.hexdigest()

    ts = datetime.now(timezone.utc).isoformat()

    meta_path = __file__

    h_params = sha256_hex(canonical_json_bytes(params))
    h_env = sha256_hex(read_bytes(env))
    h_script = sha256_hex(read_bytes(script))
    h_cfg = sha256_hex(read_bytes(cfg)) if cfg is not None else sha256_hex(b"")
    h_meta = sha256_hex(read_bytes(meta_path))
    h_src = hash_tree(src)

    fingerprint = sha256_hex(
        (h_params + h_env + h_script + h_cfg + h_meta + h_src).encode("utf-8")
    )[:16]

    return {
        "timestamp": ts,
        "params": params,
        "env": env,
        "script": script,
        "cfg": cfg,
        "src": src,
        "sha": {
            "params": h_params,
            "env": h_env,
            "script": h_script,
            "cfg": h_cfg,
            "meta": h_meta,
            "src": h_src,
        },
        "fingerprint": fingerprint,
    }
