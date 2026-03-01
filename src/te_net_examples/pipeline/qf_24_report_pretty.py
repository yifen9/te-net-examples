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
from te_net_examples.io.json import read_json, write_json
from te_net_examples.io.yaml import read_yaml
from te_net_examples.utils.audit import Audit
from te_net_examples.utils.console import ConsoleSink
from te_net_examples.utils.jlog import jline
from te_net_examples.utils.logger import Logger
from te_net_examples.utils.meta import build_meta
from te_net_examples.utils.progress import Progress
from te_net_examples.utils.versioner import build_version_dir


@dataclass(frozen=True, slots=True)
class Qf24ReportPrettyOut:
    run_dir: str
    meta: dict[str, Any]
    report_path: str | None
    figures_dir: str | None
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


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _require_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")


def _fmt_title(dgp: str, sel_density: str, fn_n_components: str) -> str:
    return f"dgp={dgp} | sel_density={sel_density} / fn_n_components={fn_n_components}"


def _unique_sorted_str(x: pd.Series) -> list[str]:
    xs = x.astype(str).tolist()
    xs = list(dict.fromkeys(xs))
    return sorted(xs)


def _unique_sorted_num(x: pd.Series) -> list[float]:
    v = pd.to_numeric(x, errors="coerce")
    v = v[np.isfinite(v)]
    xs = sorted(list(dict.fromkeys(v.tolist())))
    return [float(z) for z in xs]


def _compute_grid_layout(
    n_rows: int, n_cols: int, cell_w: float, cell_h: float
) -> tuple[float, float]:
    return float(max(1, n_cols) * cell_w), float(max(1, n_rows) * cell_h)


def _plot_metric_grid(
    *,
    df: pd.DataFrame,
    metric: str,
    out_path: str,
    row_key: str,
    col_key_a: str,
    col_key_b: str,
    x_key: str,
    hue_key: str,
    hue_order: list[str],
    x_order: list[float],
    cell_w: float,
    cell_h: float,
    dpi: int,
    band_k: float,
) -> dict[str, Any]:
    if metric not in df.columns:
        return {"path": out_path, "metric": metric, "status": "missing_metric"}

    if (
        row_key not in df.columns
        or col_key_a not in df.columns
        or col_key_b not in df.columns
    ):
        return {"path": out_path, "metric": metric, "status": "missing_facet_keys"}

    if x_key not in df.columns or hue_key not in df.columns:
        return {"path": out_path, "metric": metric, "status": "missing_x_or_hue"}

    x = df.copy()
    x[row_key] = x[row_key].astype(str)
    x[col_key_a] = x[col_key_a].astype(str)
    x[col_key_b] = x[col_key_b].astype(str)
    x[hue_key] = x[hue_key].astype(str)
    x[x_key] = _to_num(x[x_key])
    x[metric] = _to_num(x[metric])

    x = x[np.isfinite(x[x_key])]
    x = x[np.isfinite(x[metric])]

    rows = _unique_sorted_str(x[row_key])
    cols_a = _unique_sorted_str(x[col_key_a])
    cols_b = _unique_sorted_str(x[col_key_b])

    col_pairs: list[tuple[str, str]] = []
    for a in cols_a:
        for b in cols_b:
            col_pairs.append((a, b))

    n_rows = len(rows)
    n_cols = len(col_pairs)

    fig_w, fig_h = _compute_grid_layout(n_rows, n_cols, cell_w, cell_h)
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(n_rows, n_cols, wspace=0.25, hspace=0.35)

    legend_handles = None
    legend_labels = None

    for i, dgp in enumerate(rows):
        for j, (da, db) in enumerate(col_pairs):
            ax = fig.add_subplot(gs[i, j])
            sub = x[(x[row_key] == dgp) & (x[col_key_a] == da) & (x[col_key_b] == db)]
            if len(sub) == 0:
                ax.set_title(_fmt_title(dgp, da, db), fontsize=9)
                ax.set_xlabel("T/N", fontsize=9)
                ax.set_ylabel(metric, fontsize=9)
                ax.grid(True, alpha=0.2)
                continue

            gcols = [hue_key, x_key]
            agg = (
                sub.groupby(gcols, dropna=False, sort=True)[metric]
                .agg(["count", "mean", "std"])
                .reset_index()
            )
            agg["std"] = agg["std"].astype(float)
            agg["mean"] = agg["mean"].astype(float)
            agg["count"] = agg["count"].astype(int)

            for h in hue_order:
                hh = agg[agg[hue_key] == h]
                if len(hh) == 0:
                    continue
                hh = hh.sort_values(x_key, ascending=True)
                xs = hh[x_key].to_numpy(dtype=float)
                ys = hh["mean"].to_numpy(dtype=float)
                ss = hh["std"].to_numpy(dtype=float)

                line = ax.plot(xs, ys, marker="o", linewidth=1.5, markersize=3, label=h)
                if band_k > 0:
                    lo = ys - band_k * ss
                    hi = ys + band_k * ss
                    ax.fill_between(xs, lo, hi, alpha=0.18)

                if legend_handles is None:
                    legend_handles = line
                    legend_labels = [h]

            ax.set_title(_fmt_title(dgp, da, db), fontsize=9)
            ax.set_xlabel("T/N", fontsize=9)
            ax.set_ylabel(metric, fontsize=9)
            ax.grid(True, alpha=0.2)

            if x_order:
                ax.set_xticks(x_order)

    if legend_handles is not None:
        fig.legend(loc="upper center", ncol=max(1, len(hue_order)), fontsize=10)

    _ensure_dir(os.path.dirname(out_path))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)

    return {
        "path": out_path,
        "metric": metric,
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "row_levels": rows,
        "col_a_levels": cols_a,
        "col_b_levels": cols_b,
        "hue_order": hue_order,
    }


def run_qf_24_report_pretty(
    *,
    input_dir: str,
    output_root: str,
    src_dir: str,
    config_path: str,
    script_path: str,
    component: str,
    design_filename: str,
    metrics_filename: str,
) -> Qf24ReportPrettyOut:
    input_dir = _require_dir(input_dir)
    src_dir = _require_dir(src_dir)
    output_root_abs = os.path.abspath(output_root)

    cfg_path = _require_file(os.path.abspath(config_path))
    script_path_abs = _require_file(os.path.abspath(script_path))

    meta_in = _require_file(os.path.join(input_dir, "_meta.json"))
    meta_up = read_json(meta_in)
    if not isinstance(meta_up, dict):
        raise ValueError("upstream _meta.json must be a mapping")
    params_up = meta_up.get("params", None)
    if not isinstance(params_up, dict):
        raise ValueError("upstream _meta.json missing params mapping")
    design_path = params_up.get("design_path", None)
    if design_path is None:
        design_path = os.path.join(input_dir, design_filename)
    design_in = _require_file(str(design_path))

    metrics_in = _require_file(os.path.join(input_dir, metrics_filename))

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

    row_key = str(groupby_cfg.get("row_key", "dgp"))
    col_key_a = str(groupby_cfg.get("col_key_a", "sel_density"))
    col_key_b = str(groupby_cfg.get("col_key_b", "fn_n_components"))
    hue_key = str(groupby_cfg.get("hue_key", "te_name"))
    x_key = str(groupby_cfg.get("x_key", "T_N"))

    metrics_cfg = cfg.get("metrics", {})
    if metrics_cfg is None:
        metrics_cfg = {}
    if not isinstance(metrics_cfg, dict):
        raise ValueError("config.metrics must be a mapping")
    metric_cols = metrics_cfg.get("cols", [])
    if not isinstance(metric_cols, list) or not metric_cols:
        raise ValueError("metrics.cols must be a non-empty list")
    metric_cols = [str(c) for c in metric_cols]

    output_cfg = cfg.get("output", {})
    if output_cfg is None:
        output_cfg = {}
    if not isinstance(output_cfg, dict):
        raise ValueError("config.output must be a mapping")

    save_report_csv = bool(output_cfg.get("save_report_csv", True))
    save_figures = bool(output_cfg.get("save_figures", True))
    dpi = int(output_cfg.get("dpi", 200))
    band_k = float(output_cfg.get("band_k", 1.0))
    cell_w = float(output_cfg.get("cell_w", 4.5))
    cell_h = float(output_cfg.get("cell_h", 3.0))

    filters_cfg = cfg.get("filters", {})
    if filters_cfg is None:
        filters_cfg = {}
    if not isinstance(filters_cfg, dict):
        raise ValueError("config.filters must be a mapping")
    dropna_metrics = bool(filters_cfg.get("dropna_metrics", True))

    params: dict[str, Any] = {
        "input_dir": os.path.abspath(input_dir),
        "output_root": output_root_abs,
        "config_path": cfg_path,
        "component": component,
        "design_path": os.path.abspath(design_in),
        "metrics_path": os.path.abspath(metrics_in),
        "meta_in": os.path.abspath(meta_in),
        "row_key": row_key,
        "col_key_a": col_key_a,
        "col_key_b": col_key_b,
        "hue_key": hue_key,
        "x_key": x_key,
        "metric_cols": metric_cols,
        "save_report_csv": bool(save_report_csv),
        "save_figures": bool(save_figures),
        "dpi": int(dpi),
        "band_k": float(band_k),
        "cell_w": float(cell_w),
        "cell_h": float(cell_h),
        "dropna_metrics": bool(dropna_metrics),
    }

    meta = build_meta(
        params=params, env=env_path, script=script_path_abs, cfg=cfg_path, src=src_dir
    )

    run_dir = build_version_dir(output_root_abs, meta)
    audit = Audit.create(run_dir, meta)
    logger = Logger(sinks=[ConsoleSink(), audit])

    try:
        logger.info(jline("stage", component, "start", run_dir=run_dir))
        logger.info(jline("input", component, "input_dir", path=input_dir))
        logger.info(jline("input", component, "design", path=design_in))
        logger.info(jline("input", component, "metrics", path=metrics_in))
        logger.info(jline("input", component, "config", path=cfg_path))

        cfg_dir = os.path.join(run_dir, "cfg")
        os.makedirs(cfg_dir, exist_ok=True)
        cfg_copied = os.path.join(cfg_dir, os.path.basename(cfg_path))
        shutil.copy2(cfg_path, cfg_copied)
        logger.info(jline("output", component, "config_copied", path=cfg_copied))

        df_design = _df_from_csv(design_in)
        df_design = df_design.rename(columns={c: c.strip() for c in df_design.columns})
        df_metrics = _df_from_csv(metrics_in)
        df_metrics = df_metrics.rename(
            columns={c: c.strip() for c in df_metrics.columns}
        )

        _require_cols(df_design, ["design_id", "N", "T"])
        _require_cols(df_metrics, ["design_id"])

        d = df_design.copy()
        m = df_metrics.copy()

        d["design_id"] = _to_num(d["design_id"])
        m["design_id"] = _to_num(m["design_id"])
        d["N"] = _to_num(d["N"])
        d["T"] = _to_num(d["T"])

        dd = d.merge(m, on="design_id", how="left", suffixes=("", "_m"))
        dd["T_N"] = _to_num(dd["T"]) / _to_num(dd["N"])

        for c in metric_cols:
            if c in dd.columns:
                dd[c] = _to_num(dd[c])
            else:
                dd[c] = np.nan

        for k in [row_key, col_key_a, col_key_b, hue_key]:
            if k not in dd.columns:
                dd[k] = ""
            dd[k] = dd[k].astype(str)

        dd[x_key] = _to_num(dd[x_key])

        if dropna_metrics:
            dd = dd[dd[metric_cols].notna().any(axis=1)]

        hue_order = _unique_sorted_str(dd[hue_key]) if hue_key in dd.columns else []
        x_order = _unique_sorted_num(dd[x_key]) if x_key in dd.columns else []

        p = Progress(logger=logger, name="qf/24_report_pretty", total=2)
        p.start()

        report_path = None
        report_df = None
        if save_report_csv:
            keys = [row_key, hue_key, col_key_a, col_key_b, "N", "T"]
            for k in keys:
                if k not in dd.columns:
                    dd[k] = ""
            g = dd.groupby(keys, dropna=False, sort=True)
            rows: list[dict[str, Any]] = []
            for name, sub in g:
                row: dict[str, Any] = {}
                for i, k in enumerate(keys):
                    row[k] = str(name[i])
                row["n"] = int(len(sub))
                row["T_N"] = float(np.nanmean(_to_num(sub["T"]) / _to_num(sub["N"])))
                for c in metric_cols:
                    v = _to_num(sub[c]).to_numpy(dtype=float)
                    v = v[np.isfinite(v)]
                    row[f"{c}_mean"] = float(v.mean()) if v.size else float("nan")
                    row[f"{c}_std"] = (
                        float(v.std(ddof=1)) if v.size >= 2 else float("nan")
                    )
                rows.append(row)
            report_df = pd.DataFrame(rows)
            sort_keys = [row_key, hue_key, col_key_a, col_key_b, "N", "T"]
            report_df = report_df.sort_values(
                sort_keys, ascending=[True] * len(sort_keys)
            ).reset_index(drop=True)
            report_path = os.path.join(run_dir, "report_pretty.csv")
            write_csv(
                report_path, _rows_from_df(report_df), header=list(report_df.columns)
            )
            logger.info(jline("output", component, "report_pretty", path=report_path))

        p.step(1)

        figures_dir = None
        figs: dict[str, Any] = {}
        if save_figures:
            figures_dir = os.path.join(run_dir, "figures")
            _ensure_dir(figures_dir)

            for metric in metric_cols:
                out_path = os.path.join(figures_dir, f"{metric}_grid.png")
                figs[metric] = _plot_metric_grid(
                    df=dd,
                    metric=metric,
                    out_path=out_path,
                    row_key=row_key,
                    col_key_a=col_key_a,
                    col_key_b=col_key_b,
                    x_key=x_key,
                    hue_key=hue_key,
                    hue_order=hue_order,
                    x_order=x_order,
                    cell_w=cell_w,
                    cell_h=cell_h,
                    dpi=dpi,
                    band_k=band_k,
                )
                if isinstance(figs[metric], dict) and "path" in figs[metric]:
                    logger.info(
                        jline(
                            "output",
                            component,
                            f"figure_{metric}",
                            path=str(figs[metric]["path"]),
                        )
                    )

        p.step(1)
        p.finish()

        summary = {
            "inputs": {
                "input_dir": os.path.abspath(input_dir),
                "design": os.path.abspath(design_in),
                "metrics": os.path.abspath(metrics_in),
            },
            "groupby": {
                "row_key": row_key,
                "col_key_a": col_key_a,
                "col_key_b": col_key_b,
                "hue_key": hue_key,
                "x_key": x_key,
            },
            "metrics": {"cols": metric_cols},
            "outputs": {
                "report_pretty_csv": report_path,
                "figures_dir": figures_dir,
                "figures": figs,
            },
            "counts": {
                "n_design": int(len(d)),
                "n_metrics": int(len(m)),
                "n_joined": int(dd["design_id"].notna().sum()),
                "n_kept": int(len(dd)),
                "hue_levels": hue_order,
                "x_levels": x_order,
            },
        }

        summary_out = os.path.join(run_dir, "summary.json")
        write_json(summary_out, summary)
        logger.info(jline("output", component, "summary", path=summary_out))

        audit.finish_success()
        return Qf24ReportPrettyOut(
            run_dir=run_dir,
            meta=meta,
            report_path=report_path,
            figures_dir=figures_dir,
            summary_path=summary_out,
        )
    except BaseException as e:
        audit.finish_error(e)
        raise
