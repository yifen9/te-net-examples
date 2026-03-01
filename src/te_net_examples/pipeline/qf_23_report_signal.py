from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from te_net_examples.io.csv import read_csv, write_csv
from te_net_examples.io.json import write_json
from te_net_examples.io.yaml import read_yaml
from te_net_examples.utils.audit import Audit
from te_net_examples.utils.console import ConsoleSink
from te_net_examples.utils.jlog import jline
from te_net_examples.utils.logger import Logger
from te_net_examples.utils.meta import build_meta
from te_net_examples.utils.progress import Progress
from te_net_examples.utils.versioner import build_version_dir


@dataclass(frozen=True, slots=True)
class Qf23ReportSignalOut:
    run_dir: str
    meta: dict[str, Any]
    report_path: str | None
    specs_path: str | None
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


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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


def _require_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _write_text(path: str, txt: str) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)


def _fig_bar_by_density(
    df: pd.DataFrame, metric: str, out_path: str, title: str
) -> dict[str, Any]:
    x = df.copy()
    x["sel_density"] = x["sel_density"].astype(str)
    g = x.groupby("sel_density", sort=True)
    dens = g.size().index.tolist()
    y = g[metric].mean().astype(float).to_numpy()
    n = g.size().astype(int).to_numpy()

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    xs = np.arange(len(dens))
    ax.bar(xs, y)
    ax.set_xticks(xs)
    ax.set_xticklabels(dens)
    ax.set_ylabel(metric)
    ax.set_title(title)
    for i, (yy, nn) in enumerate(zip(y, n)):
        if yy == yy:
            ax.text(i, yy, f"n={int(nn)}", ha="center", va="bottom", fontsize=9)
    ax.axhline(0.0, linewidth=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    return {"path": out_path, "densities": dens, "n": n.tolist()}


def run_qf_23_report_signal(
    *,
    input_dir: str,
    output_root: str,
    src_dir: str,
    config_path: str,
    script_path: str,
    component: str,
    tstat_filename: str,
    delta_filename: str,
) -> Qf23ReportSignalOut:
    input_dir = _require_dir(input_dir)
    src_dir = _require_dir(src_dir)
    output_root_abs = os.path.abspath(output_root)

    cfg_path = _require_file(os.path.abspath(config_path))
    script_path_abs = _require_file(os.path.abspath(script_path))

    tstat_in = _require_file(os.path.join(input_dir, tstat_filename))
    delta_in = _require_file(os.path.join(input_dir, delta_filename))

    repo_root = _repo_root_from_path(Path(script_path_abs))
    env_path = _require_file(str(repo_root / "uv.lock"))

    cfg = read_yaml(cfg_path)
    if not isinstance(cfg, dict):
        raise ValueError("config must be a mapping")

    groupby_cfg = cfg.get("groupby", {})
    if groupby_cfg is None:
        groupby_cfg = {}
    if not isinstance(groupby_cfg, dict):
        raise ValueError("config.groupby must be a mapping")
    keys = groupby_cfg.get("keys", [])
    if not isinstance(keys, list) or not keys:
        raise ValueError("groupby.keys must be a non-empty list")
    keys = [str(k) for k in keys]

    out_cfg = cfg.get("output", {})
    if out_cfg is None:
        out_cfg = {}
    if not isinstance(out_cfg, dict):
        raise ValueError("config.output must be a mapping")
    save_report_csv = bool(out_cfg.get("save_report_csv", True))
    save_specs_json = bool(out_cfg.get("save_specs_json", True))
    save_figures = bool(out_cfg.get("save_figures", True))

    params: dict[str, Any] = {
        "input_dir": os.path.abspath(input_dir),
        "output_root": output_root_abs,
        "config_path": cfg_path,
        "component": component,
        "tstat_path": os.path.abspath(tstat_in),
        "delta_path": os.path.abspath(delta_in),
        "groupby_keys": keys,
        "save_report_csv": bool(save_report_csv),
        "save_specs_json": bool(save_specs_json),
        "save_figures": bool(save_figures),
    }

    meta = build_meta(
        params=params, env=env_path, script=script_path_abs, cfg=cfg_path, src=src_dir
    )

    run_dir = build_version_dir(output_root_abs, meta)
    audit = Audit.create(run_dir, meta)
    logger = Logger(sinks=[ConsoleSink(), audit])

    try:
        logger.info(jline("stage", component, "start", run_dir=run_dir))
        logger.info(jline("input", component, "tstat", path=tstat_in))
        logger.info(jline("input", component, "delta", path=delta_in))
        logger.info(jline("input", component, "config", path=cfg_path))

        cfg_dir = os.path.join(run_dir, "cfg")
        os.makedirs(cfg_dir, exist_ok=True)
        cfg_copied = os.path.join(cfg_dir, os.path.basename(cfg_path))
        shutil.copy2(cfg_path, cfg_copied)
        logger.info(jline("output", component, "config_copied", path=cfg_copied))

        p = Progress(logger=logger, name="qf/23_report_signal", total=3)
        p.start()

        df_t = _df_from_csv(tstat_in)
        df_t = df_t.rename(columns={c: c.strip() for c in df_t.columns})
        _require_cols(df_t, ["design_id", "branch", "tstat"])
        df_t["tstat"] = _to_num(df_t["tstat"])
        for k in keys:
            if k not in df_t.columns:
                df_t[k] = ""
        p.step(1)

        df_d = _df_from_csv(delta_in)
        df_d = df_d.rename(columns={c: c.strip() for c in df_d.columns})
        if "delta_tstat" in df_d.columns:
            df_d["delta_tstat"] = _to_num(df_d["delta_tstat"])
        p.step(1)

        rows: list[dict[str, Any]] = []
        for branch in sorted(df_t["branch"].astype(str).unique().tolist()):
            sub = df_t[df_t["branch"].astype(str) == branch].copy()
            g = sub.groupby(keys, dropna=False, sort=True)
            for name, gg in g:
                row: dict[str, Any] = {}
                if len(keys) == 1:
                    row[keys[0]] = name
                else:
                    for i, k in enumerate(keys):
                        row[k] = name[i]
                row["subset"] = f"tstat_{branch}"
                v = gg["tstat"].to_numpy(dtype=float)
                v = v[np.isfinite(v)]
                row["n"] = int(v.size)
                row["tstat_mean"] = float(v.mean()) if v.size else float("nan")
                row["tstat_std"] = float(v.std(ddof=1)) if v.size >= 2 else float("nan")
                rows.append(row)

        report_df = pd.DataFrame(rows)
        report_df = report_df.sort_values(
            keys + ["subset"], ascending=[True] * len(keys) + [True]
        ).reset_index(drop=True)

        report_path = None
        if save_report_csv:
            report_path = os.path.join(run_dir, "report.csv")
            write_csv(
                report_path, _rows_from_df(report_df), header=list(report_df.columns)
            )
            logger.info(jline("output", component, "report", path=report_path))

        specs_path = None
        if save_specs_json:
            lines: list[str] = []
            lines.append(r"\begin{tabular}{lrr}")
            lines.append(r"\toprule")
            lines.append(r"Subset & Mean t-stat & Std \\")
            lines.append(r"\midrule")
            for _, r in report_df.iterrows():
                mean = r["tstat_mean"]
                std = r["tstat_std"]
                ss = str(r["subset"])
                lines.append(
                    f"{ss} & {float(mean):.4f} & {float(std):.4f} \\\\"
                    if mean == mean and std == std
                    else f"{ss} & --- & --- \\\\"
                )
            lines.append(r"\bottomrule")
            lines.append(r"\end{tabular}")
            tex = "\n".join(lines)
            tex_path = os.path.join(run_dir, "report.tex")
            _write_text(tex_path, tex + "\n")
            specs = {
                "groupby_keys": keys,
                "report_tex": tex,
                "report_tex_path": tex_path,
            }
            specs_path = os.path.join(run_dir, "specs.json")
            write_json(specs_path, specs)
            logger.info(jline("output", component, "specs", path=specs_path))

        figs: dict[str, Any] = {}
        if save_figures:
            fig_dir = os.path.join(run_dir, "figures")
            _ensure_dir(fig_dir)

            df_est = df_t[df_t["branch"].astype(str) == "estimated"].copy()
            if len(df_est) and df_est["tstat"].notna().any():
                figs["tstat_est_vs_density"] = _fig_bar_by_density(
                    df_est,
                    "tstat",
                    os.path.join(fig_dir, "tstat_est_vs_density.png"),
                    "Estimated signal t-stat by density",
                )
                logger.info(
                    jline(
                        "output",
                        component,
                        "figure_tstat_est_vs_density",
                        path=figs["tstat_est_vs_density"]["path"],
                    )
                )

            df_ora = df_t[df_t["branch"].astype(str) == "oracle"].copy()
            if len(df_ora) and df_ora["tstat"].notna().any():
                figs["tstat_oracle_vs_density"] = _fig_bar_by_density(
                    df_ora,
                    "tstat",
                    os.path.join(fig_dir, "tstat_oracle_vs_density.png"),
                    "Oracle signal t-stat by density",
                )
                logger.info(
                    jline(
                        "output",
                        component,
                        "figure_tstat_oracle_vs_density",
                        path=figs["tstat_oracle_vs_density"]["path"],
                    )
                )

            if "delta_tstat" in df_d.columns and df_d["delta_tstat"].notna().any():
                x = df_d.copy()
                if "sel_density" in df_t.columns:
                    dmap = df_t[df_t["branch"].astype(str) == "estimated"][
                        ["design_id", "sel_density"]
                    ].drop_duplicates()
                    x = x.merge(dmap, on="design_id", how="left")
                else:
                    x["sel_density"] = ""
                figs["delta_tstat_vs_density"] = _fig_bar_by_density(
                    x,
                    "delta_tstat",
                    os.path.join(fig_dir, "delta_tstat_vs_density.png"),
                    "Oracle - Estimated t-stat by density",
                )
                logger.info(
                    jline(
                        "output",
                        component,
                        "figure_delta_tstat_vs_density",
                        path=figs["delta_tstat_vs_density"]["path"],
                    )
                )

        p.step(1)
        p.finish()

        summary = {
            "n_tstat_rows": int(len(df_t)),
            "n_delta_rows": int(len(df_d)),
            "outputs": {
                "report_csv": report_path,
                "specs_json": specs_path,
                "figures": figs,
            },
        }

        summary_out = os.path.join(run_dir, "summary.json")
        write_json(summary_out, summary)
        logger.info(jline("output", component, "summary", path=summary_out))

        audit.finish_success()
        return Qf23ReportSignalOut(
            run_dir=run_dir,
            meta=meta,
            report_path=report_path,
            specs_path=specs_path,
            summary_path=summary_out,
        )
    except BaseException as e:
        audit.finish_error(e)
        raise
