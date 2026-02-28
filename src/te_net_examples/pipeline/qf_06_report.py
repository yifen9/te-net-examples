from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from te_net_examples.io.csv import read_csv, write_csv
from te_net_examples.io.json import read_json, write_json
from te_net_examples.utils.audit import Audit
from te_net_examples.utils.console import ConsoleSink
from te_net_examples.utils.logger import Logger
from te_net_examples.utils.meta import build_meta
from te_net_examples.utils.versioner import build_version_dir


@dataclass(frozen=True, slots=True)
class Qf06ReportOut:
    run_dir: str
    meta: dict[str, Any]
    report_path: str
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


def _read_json_if_exists(path: str) -> Any | None:
    if not os.path.isfile(path):
        return None
    return read_json(path)


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _extract_portfolio_sort_ls(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    df0 = df.rename(columns={c: c.strip() for c in df.columns})
    need = ["period", "portfolio", "ann_ret", "t_stat", "n_periods"]
    miss = [c for c in need if c not in df0.columns]
    if miss:
        raise ValueError(f"portfolio_sort missing columns: {miss}")

    x = df0.copy()
    x["ann_ret"] = _to_num(x["ann_ret"])
    x["t_stat"] = _to_num(x["t_stat"])
    x["n_periods"] = _to_num(x["n_periods"])

    ls = x[x["portfolio"].astype(str) == "LS"].copy()
    diag = {
        "n_rows": int(len(df0)),
        "n_ls_rows": int(len(ls)),
        "periods_found": sorted(ls["period"].astype(str).unique().tolist()),
    }

    out = pd.DataFrame(
        {
            "section": ["portfolio_sort_ls"] * int(len(ls)),
            "key": [f"{p}.LS" for p in ls["period"].astype(str).tolist()],
            "period": ls["period"].astype(str).tolist(),
            "portfolio": ["LS"] * int(len(ls)),
            "signal": [""] * int(len(ls)),
            "ann_ret": ls["ann_ret"].astype(float).tolist(),
            "t_stat": ls["t_stat"].astype(float).tolist(),
            "n_periods": ls["n_periods"].astype("Int64").astype(str).tolist(),
            "mean_beta1": ["" for _ in range(int(len(ls)))],
            "t_mean_beta1": ["" for _ in range(int(len(ls)))],
            "reject_rate": ["" for _ in range(int(len(ls)))],
            "n_dates": ["" for _ in range(int(len(ls)))],
        }
    )
    return out, diag


def _extract_cs_aggregate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    df0 = df.rename(columns={c: c.strip() for c in df.columns})
    need = ["level", "signal", "beta1", "t_beta1", "r2", "dof"]
    miss = [c for c in need if c not in df0.columns]
    if miss:
        raise ValueError(f"cs_ols missing columns: {miss}")

    x = df0.copy()
    x["beta1"] = _to_num(x["beta1"])
    x["t_beta1"] = _to_num(x["t_beta1"])
    x["r2"] = _to_num(x["r2"])
    x["dof"] = _to_num(x["dof"])

    agg = x[x["level"].astype(str) == "aggregate"].copy()
    diag = {
        "n_rows": int(len(df0)),
        "n_aggregate_rows": int(len(agg)),
        "signals_found": sorted(agg["signal"].astype(str).unique().tolist()),
    }

    out = pd.DataFrame(
        {
            "section": ["cs_ols"] * int(len(agg)),
            "key": [f"{s}.aggregate" for s in agg["signal"].astype(str).tolist()],
            "period": [""] * int(len(agg)),
            "portfolio": [""] * int(len(agg)),
            "signal": agg["signal"].astype(str).tolist(),
            "ann_ret": ["" for _ in range(int(len(agg)))],
            "t_stat": ["" for _ in range(int(len(agg)))],
            "n_periods": ["" for _ in range(int(len(agg)))],
            "mean_beta1": agg["beta1"].astype(float).tolist(),
            "t_mean_beta1": agg["t_beta1"].astype(float).tolist(),
            "reject_rate": agg["r2"].astype(float).tolist(),
            "n_dates": agg["dof"].astype("Int64").astype(str).tolist(),
        }
    )
    return out, diag


def run_qf_06_report(
    *,
    input_dir_portfolio_sort: str,
    input_dir_cs: str,
    output_root: str,
    src_dir: str,
    script_path: str,
    component: str,
    portfolio_sort_filename: str,
    cs_filename: str,
) -> Qf06ReportOut:
    input_dir_portfolio_sort = _require_dir(input_dir_portfolio_sort)
    input_dir_cs = _require_dir(input_dir_cs)
    src_dir = _require_dir(src_dir)
    output_root_abs = os.path.abspath(output_root)

    portfolio_sort_in = _require_file(
        os.path.join(input_dir_portfolio_sort, portfolio_sort_filename)
    )
    cs_in = _require_file(os.path.join(input_dir_cs, cs_filename))

    script_path_abs = _require_file(os.path.abspath(script_path))
    repo_root = _repo_root_from_path(Path(script_path_abs))
    env_path = _require_file(str(repo_root / "uv.lock"))

    params: dict[str, Any] = {
        "input_dir_portfolio_sort": os.path.abspath(input_dir_portfolio_sort),
        "input_dir_cs": os.path.abspath(input_dir_cs),
        "output_root": output_root_abs,
        "portfolio_sort_filename": portfolio_sort_filename,
        "cs_filename": cs_filename,
        "component": component,
    }

    meta = build_meta(
        params=params,
        env=env_path,
        script=script_path_abs,
        cfg=portfolio_sort_in,
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
            f'{{"event":"input","component":"{component}","msg":"portfolio_sort","path":"{portfolio_sort_in}"}}'
        )
        logger.info(
            f'{{"event":"input","component":"{component}","msg":"cs_ols","path":"{cs_in}"}}'
        )

        df_t5 = _df_from_csv(portfolio_sort_in)
        df_cs = _df_from_csv(cs_in)

        rep_t5, diag_t5 = _extract_portfolio_sort_ls(df_t5)
        rep_cs, diag_cs = _extract_cs_aggregate(df_cs)

        report = pd.concat([rep_t5, rep_cs], ignore_index=True)

        report = report[
            [
                "section",
                "key",
                "period",
                "portfolio",
                "signal",
                "ann_ret",
                "t_stat",
                "n_periods",
                "mean_beta1",
                "t_mean_beta1",
                "reject_rate",
                "n_dates",
            ]
        ].sort_values(["section", "key"], ascending=[True, True])

        report_out = os.path.join(run_dir, "report.csv")
        write_csv(report_out, _rows_from_df(report), header=list(report.columns))

        t5_summary_path = os.path.join(input_dir_portfolio_sort, "summary.json")
        cs_summary_path = os.path.join(input_dir_cs, "summary.json")

        summary = {
            "inputs": {
                "portfolio_sort_dir": os.path.abspath(input_dir_portfolio_sort),
                "cs_dir": os.path.abspath(input_dir_cs),
                "portfolio_sort_file": os.path.abspath(portfolio_sort_in),
                "cs_file": os.path.abspath(cs_in),
                "portfolio_sort_summary": os.path.abspath(t5_summary_path)
                if os.path.isfile(t5_summary_path)
                else None,
                "cs_summary": os.path.abspath(cs_summary_path)
                if os.path.isfile(cs_summary_path)
                else None,
            },
            "extract": {
                "portfolio_sort": diag_t5,
                "cs_ols": diag_cs,
                "n_report_rows": int(len(report)),
            },
            "upstream_summary": {
                "portfolio_sort": _read_json_if_exists(t5_summary_path),
                "cs_ols": _read_json_if_exists(cs_summary_path),
            },
        }

        summary_out = os.path.join(run_dir, "summary.json")
        write_json(summary_out, summary)

        logger.info(
            f'{{"event":"output","component":"{component}","msg":"report","path":"{report_out}"}}'
        )
        logger.info(
            f'{{"event":"output","component":"{component}","msg":"summary","path":"{summary_out}"}}'
        )

        audit.finish_success()
        return Qf06ReportOut(
            run_dir=run_dir, meta=meta, report_path=report_out, summary_path=summary_out
        )
    except BaseException as e:
        audit.finish_error(e)
        raise
