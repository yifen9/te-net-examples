from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from te_net_examples.io.csv import read_csv
from te_net_examples.io.json import write_json
from te_net_examples.utils.audit import Audit
from te_net_examples.utils.console import ConsoleSink
from te_net_examples.utils.logger import Logger
from te_net_examples.utils.meta import build_meta
from te_net_examples.utils.versioner import build_version_dir


@dataclass(frozen=True, slots=True)
class Qf09FiguresOut:
    run_dir: str
    meta: dict[str, Any]
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


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _fig_portfolio_sort_ls(portfolio_sort_csv: str, out_path: str) -> dict[str, Any]:
    df = _df_from_csv(portfolio_sort_csv)
    df = df.rename(columns={c: c.strip() for c in df.columns})
    need = ["period", "portfolio", "ann_ret", "t_stat"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"portfolio_sort csv missing columns: {miss}")

    x = df[df["portfolio"].astype(str) == "LS"].copy()
    x["ann_ret"] = _to_num(x["ann_ret"])
    x["t_stat"] = _to_num(x["t_stat"])
    x["period"] = x["period"].astype(str)

    order = ["full", "sub1", "sub2"]
    y = [
        float(x[x["period"] == p]["ann_ret"].iloc[0])
        if (x["period"] == p).any()
        else np.nan
        for p in order
    ]
    t = [
        float(x[x["period"] == p]["t_stat"].iloc[0])
        if (x["period"] == p).any()
        else np.nan
        for p in order
    ]

    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111)
    xs = np.arange(len(order))
    ax.bar(xs, y)
    ax.set_xticks(xs)
    ax.set_xticklabels(order)
    ax.set_ylabel("Annualized return (LS)")
    ax.set_title("Table 5: Long-Short (Q5-Q1) Annualized Return")
    for i, (yy, tt) in enumerate(zip(y, t)):
        if yy == yy:
            ax.text(
                i,
                yy,
                f"t={tt:.2f}" if tt == tt else "t=---",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    ax.axhline(0.0, linewidth=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    return {"periods": order, "ann_ret": y, "t_stat": t, "path": out_path}


def _fig_cs_beta_hist(cs_csv: str, out_dir: str) -> dict[str, Any]:
    df = _df_from_csv(cs_csv)
    df = df.rename(columns={c: c.strip() for c in df.columns})
    need = ["level", "signal", "beta1"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"cs_ols csv missing columns: {miss}")

    x = df[df["level"].astype(str) == "per_date"].copy()
    x["beta1"] = _to_num(x["beta1"])
    x["signal"] = x["signal"].astype(str)

    out: dict[str, Any] = {"signals": {}}
    for sig in sorted(x["signal"].unique().tolist()):
        v = x[x["signal"] == sig]["beta1"].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        path = os.path.join(out_dir, f"cs_beta1_hist_{sig}.png")

        fig = plt.figure(figsize=(7, 4))
        ax = fig.add_subplot(111)
        ax.hist(v, bins=30)
        ax.axvline(0.0, linewidth=1.0)
        ax.set_title(f"CS OLS per-date beta1 distribution: {sig}")
        ax.set_xlabel("beta1")
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)

        out["signals"][sig] = {"n": int(v.size), "path": path}

    return out


def run_qf_09_figures(
    *,
    input_dir_portfolio_sort: str,
    input_dir_cs: str,
    output_root: str,
    src_dir: str,
    script_path: str,
    component: str,
    portfolio_sort_filename: str,
    cs_filename: str,
) -> Qf09FiguresOut:
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
    figs_dir = os.path.join(run_dir, "figures")
    _ensure_dir(figs_dir)

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

        fig1_path = os.path.join(figs_dir, "portfolio_sort_ls.png")
        fig1 = _fig_portfolio_sort_ls(portfolio_sort_in, fig1_path)
        fig2 = _fig_cs_beta_hist(cs_in, figs_dir)

        summary = {
            "inputs": {"portfolio_sort": portfolio_sort_in, "cs_ols": cs_in},
            "figures": {"portfolio_sort_ls": fig1, "cs_beta_hist": fig2},
        }

        summary_out = os.path.join(run_dir, "summary.json")
        write_json(summary_out, summary)

        logger.info(
            f'{{"event":"output","component":"{component}","msg":"summary","path":"{summary_out}"}}'
        )

        audit.finish_success()
        return Qf09FiguresOut(run_dir=run_dir, meta=meta, summary_path=summary_out)
    except BaseException as e:
        audit.finish_error(e)
        raise
