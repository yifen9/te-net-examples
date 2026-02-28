from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum


class Level(str, Enum):
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class Message:
    level: Level
    text: str
    timestamp: str


def make_message(level: Level, text: str) -> Message:
    ts = datetime.now(timezone.utc).isoformat()
    return Message(level=level, text=text, timestamp=ts)
