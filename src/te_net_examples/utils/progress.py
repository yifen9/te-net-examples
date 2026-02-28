from __future__ import annotations

import json
import time
from dataclasses import dataclass

from te_net_examples.utils.logger import Logger


@dataclass(slots=True)
class Progress:
    logger: Logger
    name: str
    total: int
    current: int = 0
    started: bool = False
    t0: float = 0.0

    def start(self) -> None:
        self.t0 = time.perf_counter()
        self.current = 0
        self.started = True
        self._emit("start")

    def step(self, n: int = 1) -> None:
        if not self.started:
            self.start()
        self.current += n
        if self.current > self.total:
            self.current = self.total
        self._emit("step")

    def finish(self) -> None:
        if not self.started:
            self.start()
        self.current = self.total
        self._emit("end")
        self.started = False

    def _emit(self, phase: str) -> None:
        elapsed = time.perf_counter() - self.t0
        rate = (self.current / elapsed) if elapsed > 0.0 else 0.0
        remaining = self.total - self.current
        eta = (remaining / rate) if rate > 0.0 else None
        pct = (self.current / self.total) if self.total > 0 else 1.0

        payload = {
            "event": "progress",
            "phase": phase,
            "name": self.name,
            "current": self.current,
            "total": self.total,
            "pct": pct,
            "elapsed_s": elapsed,
            "rate": rate,
            "eta_s": eta,
        }
        text = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        self.logger.info(text)
