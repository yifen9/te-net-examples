from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress as RichProgress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from te_net_examples.utils.message import Message


def _hhmmss(ts: str) -> str:
    if len(ts) >= 19:
        return ts[11:19]
    return ts


def _level_style(level: str) -> str:
    if level == "INFO":
        return "green"
    if level == "WARN":
        return "yellow"
    if level == "ERROR":
        return "red"
    return "white"


@dataclass(slots=True)
class ConsoleSink:
    console: Console = field(default_factory=lambda: Console(markup=False))
    transient: bool = False
    _progress: RichProgress | None = None
    _started: bool = False
    _task_id: int | None = None
    _task_name: str | None = None

    def __call__(self, msg: Message) -> None:
        payload = self._parse_payload(msg.text)
        if isinstance(payload, dict) and payload.get("event") == "progress":
            self._handle_progress(msg, payload)
            return
        self._print_log(msg, payload)

    def _parse_payload(self, text: str) -> Any:
        if not text:
            return text
        s = text.strip()
        if not s:
            return s
        if s[0] not in "{[":
            return text
        return json.loads(s)

    def _ensure_progress(self) -> None:
        if self._progress is not None:
            return
        self._progress = RichProgress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            transient=self.transient,
        )

    def _print_line(self, s: str, style: str | None = None) -> None:
        if self._progress is not None and self._started:
            self._progress.console.print(s, style=style)
        else:
            self.console.print(s, style=style)

    def _print_log(self, msg: Message, payload: Any) -> None:
        t = _hhmmss(msg.timestamp)
        lvl = msg.level.value if hasattr(msg.level, "value") else str(msg.level)
        st = _level_style(lvl)
        if isinstance(payload, dict):
            core = payload.get("event") or payload.get("component") or "app"
            body = payload.get("msg") or payload.get("message") or msg.text
            extra = []
            for k in sorted(payload.keys()):
                if k in ("event", "component", "msg", "message"):
                    continue
                extra.append(f"{k}={payload[k]}")
            tail = (" " + " ".join(extra)) if extra else ""
            line = f"[{t}] [{lvl}] [{core}] {body}{tail}"
        else:
            line = f"[{t}] [{lvl}] {msg.text}"
        self._print_line(line, style=st)

    def _handle_progress(self, msg: Message, p: dict[str, Any]) -> None:
        self._ensure_progress()
        assert self._progress is not None

        phase = p.get("phase")
        name = str(p.get("name", "task"))
        total = int(p.get("total", 0) or 0)
        current = int(p.get("current", 0) or 0)

        if phase == "start":
            self._task_name = name
            self._print_progress_start(msg, p)
            if not self._started:
                self._progress.start()
                self._started = True
            self._task_id = self._progress.add_task(name, total=total, completed=0)

        elif phase == "step":
            if self._task_id is None:
                raise RuntimeError("progress step before start")
            self._progress.update(
                self._task_id, total=total, completed=current, description=name
            )

        elif phase == "end":
            if self._task_id is None:
                raise RuntimeError("progress end before start")
            self._progress.update(
                self._task_id, total=total, completed=total, description=name
            )
            self._print_progress_end(msg, p)
            if self._started:
                self._progress.stop()
                self._started = False
            self._task_id = None
            self._task_name = None

        else:
            raise ValueError(f"unknown progress phase: {phase}")

    def _print_progress_start(self, msg: Message, p: dict[str, Any]) -> None:
        t = _hhmmss(msg.timestamp)
        st = _level_style("INFO")
        name = p.get("name")
        total = p.get("total")
        self._print_line(f"[{t}] [INFO] [progress] start", style=st)
        self._print_line(f"  name: {name}")
        self._print_line(f"  total: {total}")

    def _print_progress_end(self, msg: Message, p: dict[str, Any]) -> None:
        t = _hhmmss(msg.timestamp)
        st = _level_style("INFO")
        name = p.get("name")
        total = p.get("total")
        elapsed = p.get("elapsed_s")
        rate = p.get("rate")
        self._print_line(f"[{t}] [INFO] [progress] end", style=st)
        self._print_line(f"  name: {name}")
        self._print_line(f"  total: {total}")
        self._print_line(f"  elapsed_s: {elapsed}")
        self._print_line(f"  rate: {rate}")
