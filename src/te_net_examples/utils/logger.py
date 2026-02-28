from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from te_net_examples.utils.message import Level, Message, make_message


Sink = Callable[[Message], None]


@dataclass(slots=True)
class Logger:
    sinks: list[Sink] = field(default_factory=list)

    def emit(self, level: Level, text: str) -> Message:
        msg = make_message(level, text)
        for s in self.sinks:
            s(msg)
        return msg

    def info(self, text: str) -> Message:
        return self.emit(Level.INFO, text)

    def warn(self, text: str) -> Message:
        return self.emit(Level.WARN, text)

    def error(self, text: str) -> Message:
        return self.emit(Level.ERROR, text)
