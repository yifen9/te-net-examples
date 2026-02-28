from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from te_net_examples.io.json import write_json
from te_net_examples.io.jsonl import append_jsonl
from te_net_examples.utils.message import Message


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: str, obj: Any) -> str:
    tmp = path + ".tmp"
    write_json(tmp, obj)
    os.replace(tmp, path)
    return path


def _parse_json_or_none(s: str) -> Any | None:
    if not s:
        return None
    t = s.strip()
    if not t:
        return None
    if t[0] not in "{[":
        return None
    try:
        return json.loads(t)
    except Exception:
        return None


@dataclass(slots=True)
class Audit:
    run_dir: str
    meta: dict[str, Any]
    audit_path: str
    log_dir: str
    segment_idx: int
    segment_lines: int
    segments: list[str]
    status: str
    ts_start: str
    ts_end: str | None
    progress: dict[str, Any] | None
    error: dict[str, Any] | None

    @staticmethod
    def create(run_dir: str, meta: dict[str, Any]) -> "Audit":
        os.makedirs(run_dir, exist_ok=True)
        log_dir = os.path.join(run_dir, "_log")
        os.makedirs(log_dir, exist_ok=True)
        audit_path = os.path.join(run_dir, "_audit.json")
        ts = _utc_now()
        a = Audit(
            run_dir=run_dir,
            meta=meta,
            audit_path=audit_path,
            log_dir=log_dir,
            segment_idx=0,
            segment_lines=0,
            segments=["0000.jsonl"],
            status="running",
            ts_start=ts,
            ts_end=None,
            progress=None,
            error=None,
        )
        a._touch_segment()
        a._flush()
        return a

    def __call__(self, msg: Message) -> None:
        rec = self._to_record(msg)
        self._append_record(rec)
        payload = rec.get("payload")
        if isinstance(payload, dict) and payload.get("event") == "progress":
            self._update_progress_from_payload(payload)

    def finish_success(self) -> None:
        self.status = "success"
        self.ts_end = _utc_now()
        self._flush()

    def finish_error(self, exc: BaseException) -> None:
        self.status = "error"
        self.ts_end = _utc_now()
        self.error = {"type": type(exc).__name__, "message": str(exc)}
        self._flush()

    def _segment_name(self, idx: int) -> str:
        return f"{idx:04d}.jsonl"

    def _segment_path(self) -> str:
        return os.path.join(self.log_dir, self._segment_name(self.segment_idx))

    def _touch_segment(self) -> None:
        p = self._segment_path()
        if not os.path.isfile(p):
            with open(p, "w", encoding="utf-8"):
                pass

    def _rotate_if_needed(self) -> None:
        if self.segment_lines < 16384:
            return
        self.segment_idx += 1
        self.segment_lines = 0
        name = self._segment_name(self.segment_idx)
        self.segments.append(name)
        self._touch_segment()
        self._flush()

    def _append_record(self, rec: dict[str, Any]) -> None:
        self._rotate_if_needed()
        append_jsonl(self._segment_path(), rec)
        self.segment_lines += 1
        if self.segment_lines == 1 or self.segment_lines % 256 == 0:
            self._flush()

    def _to_record(self, msg: Message) -> dict[str, Any]:
        lvl = msg.level.value if hasattr(msg.level, "value") else str(msg.level)
        payload = _parse_json_or_none(msg.text)
        rec: dict[str, Any] = {
            "ts": msg.timestamp,
            "lvl": lvl,
            "text": msg.text,
        }
        if payload is not None:
            rec["payload"] = payload
        return rec

    def _update_progress_from_payload(self, p: dict[str, Any]) -> None:
        snap = {
            "name": p.get("name"),
            "current": p.get("current"),
            "total": p.get("total"),
            "elapsed_s": p.get("elapsed_s"),
            "eta_s": p.get("eta_s"),
            "rate": p.get("rate"),
            "phase": p.get("phase"),
        }
        self.progress = snap
        if p.get("phase") == "end" and self.status == "running":
            self.status = "success"
            self.ts_end = _utc_now()
        self._flush()

    def _flush(self) -> None:
        fp = self.meta.get("fingerprint")
        doc = {
            "ts_start": self.ts_start,
            "ts_end": self.ts_end,
            "status": self.status,
            "fingerprint": fp,
            "progress": self.progress,
            "log": {
                "dir": "_log",
                "segments": list(self.segments),
                "current": self._segment_name(self.segment_idx),
            },
            "error": self.error,
        }
        _atomic_write_json(self.audit_path, doc)
