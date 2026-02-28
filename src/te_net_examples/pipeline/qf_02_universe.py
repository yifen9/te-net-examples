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
class Qf02UniverseOut:
    run_dir: str
    meta: dict[str, Any]
    universe_path: str
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


def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: c.strip() for c in df.columns})

    def pick(*names: str) -> str | None:
        for n in names:
            if n in df.columns:
                return n
        return None

    col_year_month = pick("YearMonth", "yearmonth", "year_month")
    col_month_start = pick("MonthStart", "monthstart", "month_start")
    col_ticker = pick("Ticker", "ticker")
    col_dv = pick("DollarVol60d", "dollarvol60d", "dollar_vol_60d", "DollarVol_60d")

    missing = [
        ("YearMonth", col_year_month),
        ("MonthStart", col_month_start),
        ("Ticker", col_ticker),
        ("DollarVol60d", col_dv),
    ]
    missing = [k for k, v in missing if v is None]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    out = pd.DataFrame(
        {
            "year_month": df[col_year_month].astype(str),
            "month_start": df[col_month_start],
            "ticker": df[col_ticker].map(_normalize_ticker),
            "dollar_vol_60d": pd.to_numeric(df[col_dv], errors="coerce"),
        }
    )
    return out


def _parse_month_start(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce", utc=False)
    if dt.isna().any():
        bad = int(dt.isna().sum())
        raise ValueError(f"MonthStart parse failed for {bad} rows")
    return dt.dt.date.astype(str)


def _validate_year_month(s: pd.Series) -> None:
    pat = re.compile(r"^\d{4}-\d{2}$")
    bad = ~s.map(lambda x: bool(pat.match(str(x))))
    if bad.any():
        n = int(bad.sum())
        ex = s[bad].head(5).tolist()
        raise ValueError(f"invalid year_month format for {n} rows, examples={ex}")


def _validate_unique(df: pd.DataFrame) -> None:
    dup = df.duplicated(subset=["year_month", "ticker"], keep=False)
    if dup.any():
        n = int(dup.sum())
        ex = df.loc[dup, ["year_month", "ticker"]].head(10).to_dict(orient="records")
        raise ValueError(f"duplicate (year_month,ticker) pairs: {n}, examples={ex}")


def _summary(universe: pd.DataFrame) -> dict[str, Any]:
    g = universe.groupby("year_month", sort=True)
    sizes = g.size()
    return {
        "n_rows": int(len(universe)),
        "n_months": int(sizes.shape[0]),
        "n_unique_tickers": int(universe["ticker"].nunique()),
        "monthly_n": {
            "min": int(sizes.min()) if len(sizes) else 0,
            "p25": float(sizes.quantile(0.25)) if len(sizes) else 0.0,
            "median": float(sizes.quantile(0.50)) if len(sizes) else 0.0,
            "p75": float(sizes.quantile(0.75)) if len(sizes) else 0.0,
            "max": int(sizes.max()) if len(sizes) else 0,
        },
        "dollar_vol_60d": {
            "min": float(universe["dollar_vol_60d"].min()),
            "p25": float(universe["dollar_vol_60d"].quantile(0.25)),
            "median": float(universe["dollar_vol_60d"].quantile(0.50)),
            "p75": float(universe["dollar_vol_60d"].quantile(0.75)),
            "max": float(universe["dollar_vol_60d"].max()),
        },
    }


def _rows_from_df(df: pd.DataFrame) -> list[list[str]]:
    cols = list(df.columns)
    vals = df.to_numpy(dtype=object)
    out: list[list[str]] = []
    for r in vals:
        out.append(["" if v is None else str(v) for v in r.tolist()])
    return out


def run_qf_02_universe(
    *,
    input_dir: str,
    output_root: str,
    src_dir: str,
    raw_dir: str,
    script_path: str,
    universe_name: str,
    component: str,
) -> Qf02UniverseOut:
    input_dir = _require_dir(input_dir)
    src_dir = _require_dir(src_dir)
    raw_dir = _require_dir(raw_dir)
    output_root_abs = os.path.abspath(output_root)

    universe_in = _require_file(os.path.join(raw_dir, universe_name))

    script_path_abs = _require_file(os.path.abspath(script_path))
    repo_root = _repo_root_from_path(Path(script_path_abs))
    env_path = _require_file(str(repo_root / "uv.lock"))

    params: dict[str, Any] = {
        "input_dir": os.path.abspath(input_dir),
        "raw_dir": os.path.abspath(raw_dir),
        "output_root": output_root_abs,
        "universe_name": universe_name,
    }

    meta = build_meta(
        params=params,
        env=env_path,
        script=script_path_abs,
        cfg=universe_in,
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
            f'{{"event":"input","component":"{component}","msg":"universe_raw","path":"{universe_in}"}}'
        )

        df_in = _df_from_csv(universe_in)
        uni = _canonicalize_columns(df_in)

        uni["month_start"] = _parse_month_start(uni["month_start"])
        _validate_year_month(uni["year_month"])

        if uni["ticker"].isna().any() or (uni["ticker"] == "").any():
            raise ValueError("ticker contains missing/empty values")

        if uni["dollar_vol_60d"].isna().any():
            n = int(uni["dollar_vol_60d"].isna().sum())
            raise ValueError(f"dollar_vol_60d contains {n} missing values")

        if (uni["dollar_vol_60d"] < 0).any():
            n = int((uni["dollar_vol_60d"] < 0).sum())
            raise ValueError(f"dollar_vol_60d contains {n} negative values")

        _validate_unique(uni)

        uni = uni[
            ["year_month", "month_start", "ticker", "dollar_vol_60d"]
        ].sort_values(
            ["year_month", "dollar_vol_60d", "ticker"], ascending=[True, False, True]
        )

        universe_out = os.path.join(run_dir, "universe.csv")
        write_csv(universe_out, _rows_from_df(uni), header=list(uni.columns))

        summ = _summary(uni)
        summary_out = os.path.join(run_dir, "summary.json")
        write_json(summary_out, summ)

        logger.info(
            f'{{"event":"output","component":"{component}","msg":"universe","path":"{universe_out}"}}'
        )
        logger.info(
            f'{{"event":"output","component":"{component}","msg":"summary","path":"{summary_out}"}}'
        )

        audit.finish_success()
        return Qf02UniverseOut(
            run_dir=run_dir,
            meta=meta,
            universe_path=universe_out,
            summary_path=summary_out,
        )
    except BaseException as e:
        audit.finish_error(e)
        raise
