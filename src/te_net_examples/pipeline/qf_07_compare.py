from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from te_net_examples.io.csv import read_csv, write_csv
from te_net_examples.io.json import write_json
from te_net_examples.utils.audit import Audit
from te_net_examples.utils.console import ConsoleSink
from te_net_examples.utils.logger import Logger
from te_net_examples.utils.meta import build_meta
from te_net_examples.utils.versioner import build_version_dir


@dataclass(frozen=True, slots=True)
class Qf07CompareOut:
    run_dir: str
    meta: dict[str, Any]
    compare_path: str
    stability_path: str


def _repo_root_from_path(p: Path) -> Path:
    cur = p.resolve()
    if cur.is_file():
        cur = cur.parent
    for _ in range(16):
        if (cur / "uv.lock").is_file() and (cur / "pyproject.toml").is_file():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    raise FileNotFoundError("repo root not found (expected uv.lock and pyproject.toml)")


def _require_file(path: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return path


def _require_dir(path: str) -> str:
    if not os.path.isdir(path):
        raise FileNotFoundError(path)
    return path


def _df_from_csv(path: str) -> pd.DataFrame:
    header, rows = read_csv(path, has_header=True)
    if header is None:
        raise ValueError(f"csv missing header: {path}")
    return pd.DataFrame(rows, columns=header)


def _rows_from_df(df: pd.DataFrame) -> list[list[str]]:
    vals = df.to_numpy(dtype=object)
    out: list[list[str]] = []
    for r in vals:
        out.append(["" if v is None else str(v) for v in r.tolist()])
    return out


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _list_runs(root: str) -> list[str]:
    base = Path(root)
    out: list[str] = []
    for p in sorted(base.iterdir()):
        if not p.is_dir():
            continue
        if (p / "report.csv").is_file():
            out.append(str(p))
    return out


def _select_latest(run_dirs: list[str], n_latest: int) -> list[str]:
    xs = sorted(run_dirs, key=lambda x: Path(x).name)
    return xs[-n_latest:] if n_latest > 0 else xs


def _load_report(run_dir: str) -> pd.DataFrame:
    p = os.path.join(run_dir, "report.csv")
    df = _df_from_csv(p)
    df = df.rename(columns={c: c.strip() for c in df.columns})
    need = ["section", "key"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"report.csv missing columns {missing}: {p}")
    return df


def _melt_metrics(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        c
        for c in df.columns
        if c not in ("section", "key", "period", "portfolio", "signal")
    ]
    keep = [
        c
        for c in cols
        if c
        in (
            "ann_ret",
            "t_stat",
            "n_periods",
            "mean_beta1",
            "t_mean_beta1",
            "reject_rate",
            "n_dates",
        )
    ]
    if not keep:
        raise ValueError("report.csv has no supported metric columns to compare")
    dd = df[["section", "key"] + keep].copy()
    m = dd.melt(
        id_vars=["section", "key"],
        value_vars=keep,
        var_name="metric",
        value_name="value",
    )
    m["value_num"] = _to_num(m["value"])
    return m


def _stability_stats(values: np.ndarray) -> dict[str, Any]:
    v = values.astype(np.float64, copy=False)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0, "mean": None, "std": None, "cv": None, "min": None, "max": None}
    mean = float(v.mean())
    std = float(v.std(ddof=1)) if v.size >= 2 else 0.0
    cv = float(std / mean) if mean != 0.0 else None
    return {
        "n": int(v.size),
        "mean": mean,
        "std": std,
        "cv": cv,
        "min": float(v.min()),
        "max": float(v.max()),
    }


def run_qf_07_compare(
    *,
    input_root: str,
    output_root: str,
    src_dir: str,
    script_path: str,
    component: str,
    n_latest: int,
) -> Qf07CompareOut:
    input_root = _require_dir(input_root)
    src_dir = _require_dir(src_dir)
    output_root_abs = os.path.abspath(output_root)

    run_dirs_all = _list_runs(input_root)
    if not run_dirs_all:
        raise FileNotFoundError(f"no report.csv runs found under: {input_root}")

    run_dirs = _select_latest(run_dirs_all, int(n_latest))
    if not run_dirs:
        raise ValueError("selected run list is empty")

    cfg_path = _require_file(os.path.join(run_dirs[0], "report.csv"))

    script_path_abs = _require_file(os.path.abspath(script_path))
    repo_root = _repo_root_from_path(Path(script_path_abs))
    env_path = _require_file(str(repo_root / "uv.lock"))

    params: dict[str, Any] = {
        "input_root": os.path.abspath(input_root),
        "n_latest": int(n_latest),
        "selected_run_ids": [Path(x).name for x in run_dirs],
        "component": component,
    }

    meta = build_meta(
        params=params,
        env=env_path,
        script=script_path_abs,
        cfg=cfg_path,
        src=src_dir,
    )

    run_dir = build_version_dir(output_root_abs, meta)
    audit = Audit.create(run_dir, meta)
    logger = Logger(sinks=[ConsoleSink(), audit])

    try:
        logger.info(
            f'{{"event":"stage","component":"{component}","msg":"start","run_dir":"{run_dir}"}}'
        )
        logger.info(
            f'{{"event":"input","component":"{component}","msg":"input_root","path":"{input_root}"}}'
        )

        long_rows: list[pd.DataFrame] = []
        for rd in run_dirs:
            df = _load_report(rd)
            m = _melt_metrics(df)
            m["run_id"] = Path(rd).name
            long_rows.append(m)

        long_df = pd.concat(long_rows, ignore_index=True)
        long_df["item"] = (
            long_df["section"].astype(str)
            + "::"
            + long_df["key"].astype(str)
            + "::"
            + long_df["metric"].astype(str)
        )

        wide = long_df.pivot_table(
            index="run_id", columns="item", values="value_num", aggfunc="first"
        )
        wide = wide.reset_index()

        compare_out = os.path.join(run_dir, "compare.csv")
        write_csv(compare_out, _rows_from_df(wide), header=list(wide.columns))

        stab: dict[str, Any] = {"n_runs": int(len(run_dirs)), "items": {}}
        for col in [c for c in wide.columns if c != "run_id"]:
            vals = wide[col].to_numpy(dtype=float)
            stab["items"][col] = _stability_stats(vals)

        stability_out = os.path.join(run_dir, "stability.json")
        write_json(stability_out, stab)

        logger.info(
            f'{{"event":"output","component":"{component}","msg":"compare","path":"{compare_out}"}}'
        )
        logger.info(
            f'{{"event":"output","component":"{component}","msg":"stability","path":"{stability_out}"}}'
        )

        audit.finish_success()
        return Qf07CompareOut(
            run_dir=run_dir,
            meta=meta,
            compare_path=compare_out,
            stability_path=stability_out,
        )
    except BaseException as e:
        audit.finish_error(e)
        raise
