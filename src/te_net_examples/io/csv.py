from __future__ import annotations

import csv
import io
from typing import Iterable


def format_csv(
    rows: Iterable[Iterable[str]], header: Iterable[str] | None = None
) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    if header is not None:
        w.writerow(list(header))
    for r in rows:
        w.writerow(list(r))
    return buf.getvalue()


def write_csv(
    path: str, rows: Iterable[Iterable[str]], header: Iterable[str] | None = None
) -> str:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        if header is not None:
            w.writerow(list(header))
        for r in rows:
            w.writerow(list(r))
    return path


def read_csv(
    path: str, *, has_header: bool = True
) -> tuple[list[str] | None, list[list[str]]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        header: list[str] | None = None
        rows: list[list[str]] = []
        if has_header:
            try:
                header = next(r)
            except StopIteration:
                return None, []
        for row in r:
            rows.append(list(row))
        return header, rows
