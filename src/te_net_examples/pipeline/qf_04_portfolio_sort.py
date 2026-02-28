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
from te_net_lib.eval.stats import ols_1d


@dataclass(frozen=True, slots=True)
class Qf04portfolio_sortOut:
    run_dir: str
    meta: dict[str, Any]
    table_path: str
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


def _parse_date_str(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce", utc=False)
    if dt.isna().any():
        n = int(dt.isna().sum())
        ex = s[dt.isna()].head(5).tolist()
        raise ValueError(f"date parse failed for {n} rows, examples={ex}")
    return dt.dt.date.astype(str)


def _require_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")


def _mean_tstat_via_ols(x: np.ndarray) -> tuple[float, float, float, int]:
    v = x.astype(np.float64, copy=False)
    v = v[np.isfinite(v)]
    n = int(v.shape[0])
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    if n <= 2:
        m = float(v.mean())
        return m, float("nan"), float("nan"), n
    ones = np.ones((n,), dtype=np.float64)
    out = ols_1d(v, ones, False)
    return float(out.beta1), float(out.se_beta1), float(out.t_beta1), n


def _portfolio_sort_timeseries(
    df: pd.DataFrame,
    *,
    signal_col: str,
    ret_col: str,
    n_groups: int,
    min_per_group: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    rows: list[dict[str, Any]] = []
    n_dates_total = 0
    n_dates_used = 0
    n_skip_min = 0
    n_skip_qcut = 0
    n_skip_nan_q = 0

    for date, g in df.groupby("formation_date", sort=True):
        n_dates_total += 1
        gg = g.dropna(subset=[signal_col, ret_col])
        if len(gg) < int(n_groups) * int(min_per_group):
            n_skip_min += 1
            continue
        try:
            q = pd.qcut(
                gg[signal_col].astype(float),
                int(n_groups),
                labels=False,
                duplicates="drop",
            )
        except ValueError:
            n_skip_qcut += 1
            continue
        if q is None:
            n_skip_qcut += 1
            continue
        q = q.astype("float64")
        if q.isna().any():
            n_skip_nan_q += 1
            continue

        gg = gg.copy()
        gg["quintile"] = q.astype(int) + 1

        for k in range(1, int(n_groups) + 1):
            sub = gg[gg["quintile"] == k]
            if len(sub) == 0:
                continue
            rows.append(
                {
                    "formation_date": date,
                    "portfolio": f"Q{k}",
                    "ret": float(sub[ret_col].astype(float).mean()),
                    "n_stocks": int(len(sub)),
                }
            )

        n_dates_used += 1

    diag = {
        "n_dates_total": int(n_dates_total),
        "n_dates_used": int(n_dates_used),
        "n_skip_min_assets": int(n_skip_min),
        "n_skip_qcut": int(n_skip_qcut),
        "n_skip_nan_quintile": int(n_skip_nan_q),
    }
    return pd.DataFrame(rows), diag


def run_qf_04_portfolio_sort(
    *,
    input_dir: str,
    output_root: str,
    src_dir: str,
    script_path: str,
    component: str,
    features_filename: str,
    signal_col: str,
    ret_col: str,
    n_groups: int,
    min_per_group: int,
    annualization_factor: float,
    full_start: str,
    full_end: str,
    sub1_start: str,
    sub1_end: str,
    sub2_start: str,
    sub2_end: str,
) -> Qf04portfolio_sortOut:
    input_dir = _require_dir(input_dir)
    src_dir = _require_dir(src_dir)
    output_root_abs = os.path.abspath(output_root)

    features_in = _require_file(os.path.join(input_dir, features_filename))

    script_path_abs = _require_file(os.path.abspath(script_path))
    repo_root = _repo_root_from_path(Path(script_path_abs))
    env_path = _require_file(str(repo_root / "uv.lock"))

    params: dict[str, Any] = {
        "input_dir": os.path.abspath(input_dir),
        "output_root": output_root_abs,
        "features_filename": features_filename,
        "signal_col": signal_col,
        "ret_col": ret_col,
        "n_groups": int(n_groups),
        "min_per_group": int(min_per_group),
        "annualization_factor": float(annualization_factor),
        "full_start": full_start,
        "full_end": full_end,
        "sub1_start": sub1_start,
        "sub1_end": sub1_end,
        "sub2_start": sub2_start,
        "sub2_end": sub2_end,
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
            f'{{"event":"input","component":"{component}","msg":"features","path":"{features_in}"}}'
        )

        df_raw = _df_from_csv(features_in)
        df_raw = df_raw.rename(columns={c: c.strip() for c in df_raw.columns})

        _require_cols(df_raw, ["formation_date", "ticker", signal_col, ret_col])

        df = df_raw.copy()
        df["formation_date"] = _parse_date_str(df["formation_date"])
        df["ticker"] = df["ticker"].astype(str)

        df[signal_col] = pd.to_numeric(df[signal_col], errors="coerce")
        df[ret_col] = pd.to_numeric(df[ret_col], errors="coerce")

        d_full_start = str(pd.to_datetime(full_start).date())
        d_full_end = str(pd.to_datetime(full_end).date())
        d_sub1_start = str(pd.to_datetime(sub1_start).date())
        d_sub1_end = str(pd.to_datetime(sub1_end).date())
        d_sub2_start = str(pd.to_datetime(sub2_start).date())
        d_sub2_end = str(pd.to_datetime(sub2_end).date())

        def slice_period(d0: str, d1: str) -> pd.DataFrame:
            return df[(df["formation_date"] >= d0) & (df["formation_date"] <= d1)]

        periods: list[tuple[str, pd.DataFrame]] = [
            ("full", slice_period(d_full_start, d_full_end)),
            ("sub1", slice_period(d_sub1_start, d_sub1_end)),
            ("sub2", slice_period(d_sub2_start, d_sub2_end)),
        ]

        out_rows: list[dict[str, Any]] = []
        summ_periods: dict[str, Any] = {}

        for pname, dff in periods:
            ps, diag = _portfolio_sort_timeseries(
                dff,
                signal_col=signal_col,
                ret_col=ret_col,
                n_groups=int(n_groups),
                min_per_group=int(min_per_group),
            )

            stats_port = (
                ps.groupby("portfolio")["n_stocks"]
                .agg(["mean", "min", "max"])
                .reset_index()
                .to_dict(orient="records")
                if len(ps)
                else []
            )

            summ_periods[pname] = {
                **diag,
                "n_rows_input": int(len(dff)),
                "n_dates_input": int(dff["formation_date"].nunique())
                if len(dff)
                else 0,
                "n_rows_portfolio_ts": int(len(ps)),
                "portfolio_n_stocks_stats": stats_port,
            }

            for k in range(1, int(n_groups) + 1):
                wk = ps[ps["portfolio"] == f"Q{k}"]["ret"].astype(float).to_numpy()
                mean_weekly, se_mean, t_stat, n = _mean_tstat_via_ols(wk)
                out_rows.append(
                    {
                        "period": pname,
                        "portfolio": f"Q{k}",
                        "mean_weekly": mean_weekly,
                        "ann_ret": mean_weekly * float(annualization_factor),
                        "t_stat": t_stat,
                        "n_periods": n,
                    }
                )

            q5 = ps[ps["portfolio"] == f"Q{int(n_groups)}"][
                ["formation_date", "ret"]
            ].rename(columns={"ret": "q5"})
            q1 = ps[ps["portfolio"] == "Q1"][["formation_date", "ret"]].rename(
                columns={"ret": "q1"}
            )
            ls = q5.merge(q1, on="formation_date", how="inner")
            wk_ls = (
                (ls["q5"].astype(float) - ls["q1"].astype(float)).to_numpy()
                if len(ls)
                else np.zeros((0,), dtype=np.float64)
            )

            mean_weekly, se_mean, t_stat, n = _mean_tstat_via_ols(wk_ls)
            out_rows.append(
                {
                    "period": pname,
                    "portfolio": "LS",
                    "mean_weekly": mean_weekly,
                    "ann_ret": mean_weekly * float(annualization_factor),
                    "t_stat": t_stat,
                    "n_periods": n,
                }
            )

        out_df = pd.DataFrame(out_rows)
        out_df = out_df[
            ["period", "portfolio", "mean_weekly", "ann_ret", "t_stat", "n_periods"]
        ].sort_values(["period", "portfolio"], ascending=[True, True])

        table_out = os.path.join(run_dir, "portfolio_sort.csv")
        write_csv(table_out, _rows_from_df(out_df), header=list(out_df.columns))

        summary = {
            "signal_col": signal_col,
            "ret_col": ret_col,
            "n_groups": int(n_groups),
            "min_per_group": int(min_per_group),
            "annualization_factor": float(annualization_factor),
            "periods": summ_periods,
        }
        summary_out = os.path.join(run_dir, "summary.json")
        write_json(summary_out, summary)

        logger.info(
            f'{{"event":"output","component":"{component}","msg":"table","path":"{table_out}"}}'
        )
        logger.info(
            f'{{"event":"output","component":"{component}","msg":"summary","path":"{summary_out}"}}'
        )

        audit.finish_success()
        return Qf04portfolio_sortOut(
            run_dir=run_dir,
            meta=meta,
            table_path=table_out,
            summary_path=summary_out,
        )
    except BaseException as e:
        audit.finish_error(e)
        raise
