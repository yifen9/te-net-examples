from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from te_net_examples.io.csv import read_csv, write_csv
from te_net_examples.io.json import write_json
from te_net_examples.utils.audit import Audit
from te_net_examples.utils.console import ConsoleSink
from te_net_examples.utils.logger import Logger
from te_net_examples.utils.meta import build_meta
from te_net_examples.utils.versioner import build_version_dir


@dataclass(frozen=True, slots=True)
class Qf03FeaturesOut:
    run_dir: str
    meta: dict[str, Any]
    features_path: str
    summary_path: str


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


def _normalize_ticker(x: Any) -> str:
    s = "" if x is None else str(x)
    s = s.strip().upper()
    s = re.sub(r"\s+", "", s)
    return s


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


def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: c.strip() for c in df.columns})

    def pick(*names: str) -> str | None:
        for n in names:
            if n in df.columns:
                return n
        return None

    col_date = pick("formation_date", "FormationDate", "date", "Date")
    col_ticker = pick("ticker", "Ticker")
    col_nio = pick("nio", "NIO")
    col_te_out = pick("te_out", "TE_OUT", "teOut")
    col_te_in = pick("te_in", "TE_IN", "teIn")
    col_ret = pick("next_week_ret", "NextWeekRet", "next_ret", "ret")

    missing = [
        ("formation_date", col_date),
        ("ticker", col_ticker),
        ("nio", col_nio),
        ("te_out", col_te_out),
        ("te_in", col_te_in),
        ("next_week_ret", col_ret),
    ]
    missing = [k for k, v in missing if v is None]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    out = pd.DataFrame(
        {
            "formation_date": df[col_date],
            "ticker": df[col_ticker].map(_normalize_ticker),
            "nio": pd.to_numeric(df[col_nio], errors="coerce"),
            "te_out": pd.to_numeric(df[col_te_out], errors="coerce"),
            "te_in": pd.to_numeric(df[col_te_in], errors="coerce"),
            "next_week_ret": pd.to_numeric(df[col_ret], errors="coerce"),
        }
    )
    return out


def _parse_formation_date(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce", utc=False)
    if dt.isna().any():
        n = int(dt.isna().sum())
        ex = s[dt.isna()].head(5).tolist()
        raise ValueError(f"formation_date parse failed for {n} rows, examples={ex}")
    return dt.dt.date.astype(str)


def _validate_unique(df: pd.DataFrame) -> None:
    dup = df.duplicated(subset=["formation_date", "ticker"], keep=False)
    if dup.any():
        n = int(dup.sum())
        ex = (
            df.loc[dup, ["formation_date", "ticker"]].head(10).to_dict(orient="records")
        )
        raise ValueError(f"duplicate (formation_date,ticker) pairs: {n}, examples={ex}")


def _validate_nonempty_ticker(df: pd.DataFrame) -> None:
    bad = df["ticker"].isna() | (df["ticker"] == "")
    if bad.any():
        n = int(bad.sum())
        ex = (
            df.loc[bad, ["formation_date", "ticker"]].head(10).to_dict(orient="records")
        )
        raise ValueError(f"ticker missing/empty for {n} rows, examples={ex}")


def _validate_numeric(df: pd.DataFrame, col: str) -> None:
    bad = df[col].isna()
    if bad.any():
        n = int(bad.sum())
        ex = (
            df.loc[bad, ["formation_date", "ticker"]].head(10).to_dict(orient="records")
        )
        raise ValueError(f"{col} contains {n} missing values, examples={ex}")


def _summary(df: pd.DataFrame) -> dict[str, Any]:
    g = df.groupby("formation_date", sort=True)
    sizes = g.size()
    date_min = df["formation_date"].min() if len(df) else None
    date_max = df["formation_date"].max() if len(df) else None

    def qstats(s: pd.Series) -> dict[str, float]:
        return {
            "min": float(s.min()),
            "p01": float(s.quantile(0.01)),
            "p05": float(s.quantile(0.05)),
            "p25": float(s.quantile(0.25)),
            "median": float(s.quantile(0.50)),
            "p75": float(s.quantile(0.75)),
            "p95": float(s.quantile(0.95)),
            "p99": float(s.quantile(0.99)),
            "max": float(s.max()),
        }

    return {
        "n_rows": int(len(df)),
        "n_dates": int(sizes.shape[0]),
        "date_range": {"start": date_min, "end": date_max},
        "n_unique_tickers": int(df["ticker"].nunique()),
        "cross_section_n": {
            "min": int(sizes.min()) if len(sizes) else 0,
            "p25": float(sizes.quantile(0.25)) if len(sizes) else 0.0,
            "median": float(sizes.quantile(0.50)) if len(sizes) else 0.0,
            "p75": float(sizes.quantile(0.75)) if len(sizes) else 0.0,
            "max": int(sizes.max()) if len(sizes) else 0,
        },
        "signals": {
            "nio": qstats(df["nio"]),
            "te_out": qstats(df["te_out"]),
            "te_in": qstats(df["te_in"]),
            "next_week_ret": qstats(df["next_week_ret"]),
        },
    }


def run_qf_03_features(
    *,
    input_dir: str,
    output_root: str,
    src_dir: str,
    raw_dir: str,
    script_path: str,
    features_name: str,
    component: str,
) -> Qf03FeaturesOut:
    input_dir = _require_dir(input_dir)
    src_dir = _require_dir(src_dir)
    raw_dir = _require_dir(raw_dir)
    output_root_abs = os.path.abspath(output_root)

    features_in = _require_file(os.path.join(raw_dir, features_name))

    script_path_abs = _require_file(os.path.abspath(script_path))
    repo_root = _repo_root_from_path(Path(script_path_abs))
    env_path = _require_file(str(repo_root / "uv.lock"))

    params: dict[str, Any] = {
        "input_dir": os.path.abspath(input_dir),
        "raw_dir": os.path.abspath(raw_dir),
        "output_root": output_root_abs,
        "features_name": features_name,
        "component": component,
    }

    meta = build_meta(
        params=params,
        env=env_path,
        script=script_path_abs,
        cfg=features_in,
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
            f'{{"event":"input","component":"{component}","msg":"features_raw","path":"{features_in}"}}'
        )

        df_in = _df_from_csv(features_in)
        df = _canonicalize_columns(df_in)

        df["formation_date"] = _parse_formation_date(df["formation_date"])
        _validate_nonempty_ticker(df)
        _validate_numeric(df, "nio")
        _validate_numeric(df, "te_out")
        _validate_numeric(df, "te_in")
        _validate_numeric(df, "next_week_ret")
        _validate_unique(df)

        df = df[
            ["formation_date", "ticker", "nio", "te_out", "te_in", "next_week_ret"]
        ].sort_values(["formation_date", "ticker"], ascending=[True, True])

        features_out = os.path.join(run_dir, "features_weekly.csv")
        write_csv(features_out, _rows_from_df(df), header=list(df.columns))

        summary = _summary(df)
        summary_out = os.path.join(run_dir, "summary.json")
        write_json(summary_out, summary)

        logger.info(
            f'{{"event":"output","component":"{component}","msg":"features","path":"{features_out}"}}'
        )
        logger.info(
            f'{{"event":"output","component":"{component}","msg":"summary","path":"{summary_out}"}}'
        )

        audit.finish_success()
        return Qf03FeaturesOut(
            run_dir=run_dir,
            meta=meta,
            features_path=features_out,
            summary_path=summary_out,
        )
    except BaseException as e:
        audit.finish_error(e)
        raise
