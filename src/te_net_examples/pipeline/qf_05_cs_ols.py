from __future__ import annotations

import os
import re
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
from te_net_lib.eval.stats import ols_1d, rejection_rate


@dataclass(frozen=True, slots=True)
class Qf05CsOlsOut:
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


def _parse_signal_list(s: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"[,\s]+", s) if p.strip()]
    if not parts:
        raise ValueError("signals must be non-empty")
    return parts


def _mean_tstat_beta_series(beta: np.ndarray) -> float:
    v = beta.astype(np.float64, copy=False)
    v = v[np.isfinite(v)]
    n = int(v.shape[0])
    if n <= 2:
        return float("nan")
    ones = np.ones((n,), dtype=np.float64)
    out = ols_1d(v, ones, False)
    return float(out.t_beta1)


def _qstats(x: np.ndarray) -> dict[str, float]:
    v = x.astype(np.float64, copy=False)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {
            "min": float("nan"),
            "p25": float("nan"),
            "median": float("nan"),
            "p75": float("nan"),
            "max": float("nan"),
        }
    return {
        "min": float(np.min(v)),
        "p25": float(np.quantile(v, 0.25)),
        "median": float(np.quantile(v, 0.50)),
        "p75": float(np.quantile(v, 0.75)),
        "max": float(np.max(v)),
    }


def run_qf_05_cs_ols(
    *,
    input_dir: str,
    output_root: str,
    src_dir: str,
    script_path: str,
    component: str,
    features_filename: str,
    ret_col: str,
    signals: str,
    add_intercept: bool,
    min_n: int,
    threshold: float,
    two_sided: bool,
) -> Qf05CsOlsOut:
    input_dir = _require_dir(input_dir)
    src_dir = _require_dir(src_dir)
    output_root_abs = os.path.abspath(output_root)

    features_in = _require_file(os.path.join(input_dir, features_filename))

    script_path_abs = _require_file(os.path.abspath(script_path))
    repo_root = _repo_root_from_path(Path(script_path_abs))
    env_path = _require_file(str(repo_root / "uv.lock"))

    sig_list = _parse_signal_list(signals)

    params: dict[str, Any] = {
        "input_dir": os.path.abspath(input_dir),
        "output_root": output_root_abs,
        "features_filename": features_filename,
        "ret_col": ret_col,
        "signals": sig_list,
        "add_intercept": bool(add_intercept),
        "min_n": int(min_n),
        "threshold": float(threshold),
        "two_sided": bool(two_sided),
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

        base_cols = ["formation_date", ret_col]
        _require_cols(df_raw, base_cols)
        _require_cols(df_raw, sig_list)

        df = df_raw.copy()
        df["formation_date"] = _parse_date_str(df["formation_date"])
        df[ret_col] = pd.to_numeric(df[ret_col], errors="coerce")
        for sname in sig_list:
            df[sname] = pd.to_numeric(df[sname], errors="coerce")

        per_rows: list[dict[str, Any]] = []
        agg_rows: list[dict[str, Any]] = []
        summ_signals: dict[str, Any] = {}

        for sname in sig_list:
            betas: list[float] = []
            tstats: list[float] = []
            n_dates_total = int(df["formation_date"].nunique()) if len(df) else 0
            n_dates_used = 0
            n_skip_min = 0
            n_skip_dof = 0

            for date, g in df.groupby("formation_date", sort=True):
                gg = g.dropna(subset=[ret_col, sname])
                n = int(len(gg))
                if n < int(min_n):
                    n_skip_min += 1
                    continue
                if n <= (2 if add_intercept else 1):
                    n_skip_dof += 1
                    continue

                y = gg[ret_col].astype(float).to_numpy()
                x = gg[sname].astype(float).to_numpy()

                out = ols_1d(y, x, bool(add_intercept))

                per_rows.append(
                    {
                        "level": "per_date",
                        "formation_date": date,
                        "signal": sname,
                        "n_obs": n,
                        "beta1": float(out.beta1),
                        "se_beta1": float(out.se_beta1),
                        "t_beta1": float(out.t_beta1),
                        "r2": float(out.r2),
                        "dof": int(out.dof),
                    }
                )

                betas.append(float(out.beta1))
                tstats.append(float(out.t_beta1))
                n_dates_used += 1

            b = np.array(betas, dtype=np.float64)
            t = np.array(tstats, dtype=np.float64)

            n_dates = int(b.shape[0])
            mean_beta = float(np.mean(b)) if n_dates > 0 else float("nan")
            std_beta = float(np.std(b, ddof=1)) if n_dates >= 2 else float("nan")
            t_mean = _mean_tstat_beta_series(b) if n_dates > 0 else float("nan")
            rr = (
                float(rejection_rate(t, float(threshold), bool(two_sided)))
                if n_dates > 0
                else float("nan")
            )

            agg_rows.append(
                {
                    "level": "aggregate",
                    "formation_date": "",
                    "signal": sname,
                    "n_obs": "",
                    "beta1": mean_beta,
                    "se_beta1": std_beta,
                    "t_beta1": t_mean,
                    "r2": rr,
                    "dof": n_dates,
                }
            )

            summ_signals[sname] = {
                "n_dates_total": int(n_dates_total),
                "n_dates_used": int(n_dates_used),
                "n_skip_min_n": int(n_skip_min),
                "n_skip_dof": int(n_skip_dof),
                "beta1": {
                    "mean": mean_beta,
                    "std": std_beta,
                    "q": _qstats(b),
                },
                "t_beta1": {
                    "q": _qstats(t),
                    "reject_rate": rr,
                    "threshold": float(threshold),
                    "two_sided": bool(two_sided),
                },
            }

        out_df = pd.DataFrame(per_rows + agg_rows)
        out_df = out_df[
            [
                "level",
                "formation_date",
                "signal",
                "n_obs",
                "beta1",
                "se_beta1",
                "t_beta1",
                "r2",
                "dof",
            ]
        ]

        table_out = os.path.join(run_dir, "cs_ols.csv")
        write_csv(table_out, _rows_from_df(out_df), header=list(out_df.columns))

        summary = {
            "ret_col": ret_col,
            "signals": sig_list,
            "add_intercept": bool(add_intercept),
            "min_n": int(min_n),
            "signals_summary": summ_signals,
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
        return Qf05CsOlsOut(
            run_dir=run_dir, meta=meta, table_path=table_out, summary_path=summary_out
        )
    except BaseException as e:
        audit.finish_error(e)
        raise
