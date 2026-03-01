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
from te_net_examples.utils.versioner import build_version_dir


@dataclass(frozen=True, slots=True)
class Qf80PaperArtifactsOut:
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


def _fmt_cols(df: pd.DataFrame, fmt: dict[str, Any]) -> pd.DataFrame:
    x = df.copy()
    rename = fmt.get("rename", {})
    if isinstance(rename, dict) and rename:
        x = x.rename(columns={str(k): str(v) for k, v in rename.items()})
    order = fmt.get("order", None)
    if isinstance(order, list) and order:
        cols = [str(c) for c in order if str(c) in x.columns]
        rest = [c for c in x.columns if c not in cols]
        x = x[cols + rest]
    return x


def _latex_escape(s: str) -> str:
    m = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out: list[str] = []
    for ch in s:
        out.append(m.get(ch, ch))
    return "".join(out)


def _format_cell(v: Any, float_format: str | None) -> str:
    if v is None:
        return ""
    s = str(v)
    if s == "" or s.lower() in ("nan", "none", "<na>"):
        return ""
    if float_format is None:
        return s
    try:
        x = float(s)
        if x == x and np.isfinite(x):
            return float_format.format(x)
    except Exception:
        pass
    return s


def _df_to_latex_tabular(df: pd.DataFrame, fmt: dict[str, Any]) -> str:
    escape = bool(fmt.get("escape", True))
    na_rep = str(fmt.get("na_rep", ""))
    float_format = fmt.get("float_format", None)
    if float_format is not None:
        float_format = str(float_format)
    colspec = fmt.get("colspec", None)
    cols = [str(c) for c in df.columns.tolist()]
    if isinstance(colspec, str) and colspec.strip() != "":
        spec = colspec
    else:
        align = str(fmt.get("align", "l")).strip()
        if align == "":
            align = "l"
        spec = align * len(cols)
    lines: list[str] = []
    lines.append(r"\begin{tabular}{" + spec + r"}")
    lines.append(r"\toprule")
    head = [(_latex_escape(c) if escape else c) for c in cols]
    lines.append(" & ".join(head) + r" \\")
    lines.append(r"\midrule")
    vals = df.to_numpy(dtype=object)
    for r in vals:
        row: list[str] = []
        for v in r.tolist():
            cell = _format_cell(v, float_format)
            if cell == "" and na_rep != "":
                cell = na_rep
            cell = _latex_escape(cell) if escape else cell
            row.append(cell)
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def _write_table_csv_tex(
    df: pd.DataFrame, out_csv: str, out_tex: str | None, spec: dict[str, Any]
) -> dict[str, Any]:
    fmt = spec.get("format", {})
    if fmt is None:
        fmt = {}
    if not isinstance(fmt, dict):
        raise ValueError("table.format must be a mapping")
    x = _fmt_cols(df, fmt)
    _ensure_dir(os.path.dirname(out_csv))
    write_csv(out_csv, _rows_from_df(x), header=list(x.columns))
    info: dict[str, Any] = {
        "csv": out_csv,
        "n_rows": int(len(x)),
        "n_cols": int(len(x.columns)),
    }
    if out_tex is not None:
        _ensure_dir(os.path.dirname(out_tex))
        tex = _df_to_latex_tabular(x, fmt)
        with open(out_tex, "w", encoding="utf-8") as f:
            f.write(tex)
            if not tex.endswith("\n"):
                f.write("\n")
        info["tex"] = out_tex
    return info


def _apply_filters(df: pd.DataFrame, filters: Any) -> pd.DataFrame:
    if filters is None:
        return df
    if not isinstance(filters, list):
        raise ValueError("filters must be a list")
    x = df.copy()
    for f in filters:
        if not isinstance(f, dict):
            raise ValueError("filter must be a mapping")
        col = str(f.get("col", "")).strip()
        op = str(f.get("op", "eq")).strip()
        val = f.get("value", None)
        if col == "" or col not in x.columns:
            raise ValueError(f"filter column not found: {col}")
        s = x[col]
        if op == "eq":
            x = x[s.astype(str) == str(val)]
        elif op == "ne":
            x = x[s.astype(str) != str(val)]
        elif op == "in":
            if not isinstance(val, list):
                raise ValueError("filter op=in requires list value")
            xs = set([str(v) for v in val])
            x = x[s.astype(str).isin(xs)]
        elif op == "nin":
            if not isinstance(val, list):
                raise ValueError("filter op=nin requires list value")
            xs = set([str(v) for v in val])
            x = x[~s.astype(str).isin(xs)]
        else:
            raise ValueError(f"unsupported filter op: {op}")
    return x


def _read_source_df(stage_map: dict[str, str], src: dict[str, Any]) -> pd.DataFrame:
    stage = str(src.get("stage", "")).strip()
    rel = str(src.get("path", "")).strip()
    kind = str(src.get("kind", "csv")).strip()
    if stage == "" or rel == "":
        raise ValueError("source requires stage and path")
    base = stage_map.get(stage, None)
    if base is None:
        raise ValueError(f"unknown stage: {stage}")
    p = os.path.join(base, rel)
    if kind == "csv":
        return _df_from_csv(_require_file(p))
    raise ValueError(f"unsupported source kind: {kind}")


def _save_fig(path: str, spec: dict[str, Any]) -> None:
    dpi = int(spec.get("dpi", 200))
    _ensure_dir(os.path.dirname(path))
    plt.savefig(path, dpi=dpi)
    plt.close(plt.gcf())


def _fig_line(df: pd.DataFrame, out_path: str, spec: dict[str, Any]) -> dict[str, Any]:
    xcol = str(spec.get("x", "")).strip()
    ycol = str(spec.get("y", "")).strip()
    gcol = str(spec.get("group", "")).strip()
    title = str(spec.get("title", "")).strip()
    xlabel = str(spec.get("xlabel", xcol)).strip()
    ylabel = str(spec.get("ylabel", ycol)).strip()
    x = _apply_filters(df, spec.get("filters", None)).copy()
    if xcol not in x.columns or ycol not in x.columns:
        name = str(spec.get("name", "")).strip()
        raise ValueError(
            f"line figure missing columns: name={name} x={xcol} y={ycol} cols={sorted([str(c) for c in x.columns.tolist()])}"
        )
    if bool(spec.get("x_numeric", True)):
        x[xcol] = _to_num(x[xcol])
    x[ycol] = _to_num(x[ycol])
    x = x[np.isfinite(_to_num(x[ycol]))]
    plt.figure(figsize=tuple(spec.get("figsize", [7, 4])))
    ax = plt.gca()
    marker = str(spec.get("marker", "o"))
    lw = float(spec.get("linewidth", 2.0))
    if gcol != "" and gcol in x.columns:
        for g, sub in x.groupby(gcol, dropna=False, sort=True):
            ax.plot(
                sub[xcol].to_numpy(),
                sub[ycol].to_numpy(dtype=float),
                marker=marker,
                linewidth=lw,
                label=str(g),
            )
        if bool(spec.get("legend", True)):
            ax.legend(loc=str(spec.get("legend_loc", "best")))
    else:
        ax.plot(
            x[xcol].to_numpy(),
            x[ycol].to_numpy(dtype=float),
            marker=marker,
            linewidth=lw,
        )
    if title != "":
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if isinstance(spec.get("xlim", None), list) and len(spec["xlim"]) == 2:
        ax.set_xlim(float(spec["xlim"][0]), float(spec["xlim"][1]))
    if isinstance(spec.get("ylim", None), list) and len(spec["ylim"]) == 2:
        ax.set_ylim(float(spec["ylim"][0]), float(spec["ylim"][1]))
    if isinstance(spec.get("vlines", None), list):
        for v in spec["vlines"]:
            ax.axvline(
                float(v.get("x")),
                linestyle=str(v.get("linestyle", "--")),
                linewidth=float(v.get("linewidth", 1.0)),
            )
    if isinstance(spec.get("hlines", None), list):
        for h in spec["hlines"]:
            ax.axhline(
                float(h.get("y")),
                linestyle=str(h.get("linestyle", "--")),
                linewidth=float(h.get("linewidth", 1.0)),
            )
    plt.tight_layout()
    _save_fig(out_path, spec)
    return {"path": out_path, "kind": "line"}


def _fig_scatter(
    df: pd.DataFrame, out_path: str, spec: dict[str, Any]
) -> dict[str, Any]:
    xcol = str(spec.get("x", "")).strip()
    ycol = str(spec.get("y", "")).strip()
    gcol = str(spec.get("group", "")).strip()
    title = str(spec.get("title", "")).strip()
    xlabel = str(spec.get("xlabel", xcol)).strip()
    ylabel = str(spec.get("ylabel", ycol)).strip()
    x = _apply_filters(df, spec.get("filters", None)).copy()
    if xcol not in x.columns or ycol not in x.columns:
        name = str(spec.get("name", "")).strip()
        raise ValueError(
            f"scatter figure missing columns: name={name} x={xcol} y={ycol} cols={sorted([str(c) for c in x.columns.tolist()])}"
        )
    x[xcol] = _to_num(x[xcol])
    x[ycol] = _to_num(x[ycol])
    x = x[np.isfinite(_to_num(x[xcol])) & np.isfinite(_to_num(x[ycol]))]
    max_points = spec.get("max_points", None)
    if max_points is not None and int(max_points) > 0 and len(x) > int(max_points):
        x = x.sample(n=int(max_points), random_state=int(spec.get("seed", 0)))
    plt.figure(figsize=tuple(spec.get("figsize", [6, 4])))
    ax = plt.gca()
    s = float(spec.get("s", 30))
    if gcol != "" and gcol in x.columns:
        for g, sub in x.groupby(gcol, dropna=False, sort=True):
            ax.scatter(
                sub[xcol].to_numpy(dtype=float),
                sub[ycol].to_numpy(dtype=float),
                s=s,
                label=str(g),
            )
        if bool(spec.get("legend", True)):
            ax.legend(loc=str(spec.get("legend_loc", "best")))
    else:
        ax.scatter(x[xcol].to_numpy(dtype=float), x[ycol].to_numpy(dtype=float), s=s)
    if title != "":
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if isinstance(spec.get("xlim", None), list) and len(spec["xlim"]) == 2:
        ax.set_xlim(float(spec["xlim"][0]), float(spec["xlim"][1]))
    if isinstance(spec.get("ylim", None), list) and len(spec["ylim"]) == 2:
        ax.set_ylim(float(spec["ylim"][0]), float(spec["ylim"][1]))
    if isinstance(spec.get("vlines", None), list):
        for v in spec["vlines"]:
            ax.axvline(
                float(v.get("x")),
                linestyle=str(v.get("linestyle", "--")),
                linewidth=float(v.get("linewidth", 1.0)),
            )
    if isinstance(spec.get("hlines", None), list):
        for h in spec["hlines"]:
            ax.axhline(
                float(h.get("y")),
                linestyle=str(h.get("linestyle", "--")),
                linewidth=float(v.get("linewidth", 1.0))
                if isinstance(v, dict)
                else 1.0,
            )
    plt.tight_layout()
    _save_fig(out_path, spec)
    return {"path": out_path, "kind": "scatter"}


def _fig_bar(df: pd.DataFrame, out_path: str, spec: dict[str, Any]) -> dict[str, Any]:
    xcol = str(spec.get("x", "")).strip()
    ycol = str(spec.get("y", "")).strip()
    title = str(spec.get("title", "")).strip()
    xlabel = str(spec.get("xlabel", xcol)).strip()
    ylabel = str(spec.get("ylabel", ycol)).strip()
    agg = str(spec.get("agg", "mean")).strip()
    x = _apply_filters(df, spec.get("filters", None)).copy()
    if xcol not in x.columns or ycol not in x.columns:
        name = str(spec.get("name", "")).strip()
        raise ValueError(
            f"bar figure missing columns: name={name} x={xcol} y={ycol} cols={sorted([str(c) for c in x.columns.tolist()])}"
        )
    x[ycol] = _to_num(x[ycol])
    if agg == "mean":
        g = x.groupby(xcol, dropna=False, sort=True)[ycol].mean()
    elif agg == "sum":
        g = x.groupby(xcol, dropna=False, sort=True)[ycol].sum()
    else:
        raise ValueError(f"unsupported bar agg: {agg}")
    cats = [str(k) for k in g.index.tolist()]
    vals = g.to_numpy(dtype=float)
    plt.figure(figsize=tuple(spec.get("figsize", [6, 4])))
    ax = plt.gca()
    xs = np.arange(len(cats))
    ax.bar(xs, vals)
    ax.set_xticks(xs)
    ax.set_xticklabels(cats)
    if title != "":
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if isinstance(spec.get("ylim", None), list) and len(spec["ylim"]) == 2:
        ax.set_ylim(float(spec["ylim"][0]), float(spec["ylim"][1]))
    if isinstance(spec.get("xlim", None), list) and len(spec["xlim"]) == 2:
        ax.set_xlim(float(spec["xlim"][0]), float(spec["xlim"][1]))
    if isinstance(spec.get("hlines", None), list):
        for h in spec["hlines"]:
            ax.axhline(
                float(h.get("y")),
                linestyle=str(h.get("linestyle", "--")),
                linewidth=float(h.get("linewidth", 1.0)),
            )
    plt.tight_layout()
    _save_fig(out_path, spec)
    return {"path": out_path, "kind": "bar"}


def _render_tables(
    stage_map: dict[str, str], cfg: dict[str, Any], tab_dir: str
) -> list[dict[str, Any]]:
    tabs = cfg.get("tables", [])
    if tabs is None:
        tabs = []
    if not isinstance(tabs, list):
        raise ValueError("config.tables must be a list")
    out: list[dict[str, Any]] = []
    for t in tabs:
        if not isinstance(t, dict):
            raise ValueError("table spec must be a mapping")
        name = str(t.get("name", "")).strip()
        src = t.get("source", None)
        if name == "" or not isinstance(src, dict):
            raise ValueError("table requires name and source")
        df = _read_source_df(stage_map, src)
        df = df.rename(columns={c: str(c).strip() for c in df.columns})
        df = _apply_filters(df, t.get("filters", None))
        out_csv = os.path.join(tab_dir, f"{name}.csv")
        out_tex = (
            os.path.join(tab_dir, f"{name}.tex")
            if bool(t.get("save_tex", True))
            else None
        )
        info = _write_table_csv_tex(df, out_csv, out_tex, t)
        info["name"] = name
        info["source"] = src
        out.append(info)
    return out


def _render_figures(
    stage_map: dict[str, str], cfg: dict[str, Any], fig_dir: str
) -> list[dict[str, Any]]:
    figs = cfg.get("figures", [])
    if figs is None:
        figs = []
    if not isinstance(figs, list):
        raise ValueError("config.figures must be a list")
    out: list[dict[str, Any]] = []
    for f in figs:
        if not isinstance(f, dict):
            raise ValueError("figure spec must be a mapping")
        name = str(f.get("name", "")).strip()
        kind = str(f.get("kind", "")).strip()
        src = f.get("source", None)
        ext = str(f.get("ext", "png")).strip()
        if name == "" or kind == "" or not isinstance(src, dict):
            raise ValueError("figure requires name, kind, source")
        f["name"] = name
        df = _read_source_df(stage_map, src)
        df = df.rename(columns={c: str(c).strip() for c in df.columns})
        out_path = os.path.join(fig_dir, f"{name}.{ext}")
        if kind == "line":
            info = _fig_line(df, out_path, f)
        elif kind == "scatter":
            info = _fig_scatter(df, out_path, f)
        elif kind == "bar":
            info = _fig_bar(df, out_path, f)
        else:
            raise ValueError(f"unsupported figure kind: {kind}")
        info["name"] = name
        info["source"] = src
        out.append(info)
        if bool(f.get("also_pdf", False)) and ext.lower() != "pdf":
            pdf_path = os.path.join(fig_dir, f"{name}.pdf")
            shutil.copy2(out_path, pdf_path)
    return out


def run_qf_80_paper_artifacts(
    *,
    input_dir_metrics: str,
    input_dir_report_figures: str,
    input_dir_cs_tstat: str,
    input_dir_report_signal: str,
    output_root: str,
    src_dir: str,
    config_path: str,
    script_path: str,
    component: str,
) -> Qf80PaperArtifactsOut:
    input_dir_metrics = _require_dir(input_dir_metrics)
    input_dir_report_figures = _require_dir(input_dir_report_figures)
    input_dir_cs_tstat = _require_dir(input_dir_cs_tstat)
    input_dir_report_signal = _require_dir(input_dir_report_signal)
    src_dir = _require_dir(src_dir)
    output_root_abs = os.path.abspath(output_root)

    cfg_path = _require_file(os.path.abspath(config_path))
    script_path_abs = _require_file(os.path.abspath(script_path))

    repo_root = _repo_root_from_path(Path(script_path_abs))
    env_path = _require_file(str(repo_root / "uv.lock"))

    cfg = read_yaml(cfg_path)
    if not isinstance(cfg, dict):
        raise ValueError("config must be a mapping")

    stage_map = {
        "metrics": os.path.abspath(input_dir_metrics),
        "report_figures": os.path.abspath(input_dir_report_figures),
        "cs_tstat": os.path.abspath(input_dir_cs_tstat),
        "report_signal": os.path.abspath(input_dir_report_signal),
    }

    params: dict[str, Any] = {
        "input_dir_metrics": stage_map["metrics"],
        "input_dir_report_figures": stage_map["report_figures"],
        "input_dir_cs_tstat": stage_map["cs_tstat"],
        "input_dir_report_signal": stage_map["report_signal"],
        "output_root": output_root_abs,
        "config_path": cfg_path,
        "component": component,
    }

    meta = build_meta(
        params=params, env=env_path, script=script_path_abs, cfg=cfg_path, src=src_dir
    )

    run_dir = build_version_dir(output_root_abs, meta)
    audit = Audit.create(run_dir, meta)
    logger = Logger(sinks=[ConsoleSink(), audit])

    try:
        logger.info(jline("stage", component, "start", run_dir=run_dir))
        logger.info(jline("input", component, "metrics", path=stage_map["metrics"]))
        logger.info(
            jline(
                "input", component, "report_figures", path=stage_map["report_figures"]
            )
        )
        logger.info(jline("input", component, "cs_tstat", path=stage_map["cs_tstat"]))
        logger.info(
            jline("input", component, "report_signal", path=stage_map["report_signal"])
        )
        logger.info(jline("input", component, "config", path=cfg_path))

        fig_dir = os.path.join(run_dir, "fig")
        tab_dir = os.path.join(run_dir, "tab")
        _ensure_dir(fig_dir)
        _ensure_dir(tab_dir)

        copied_cfg = os.path.join(run_dir, "config.yaml")
        shutil.copy2(cfg_path, copied_cfg)
        logger.info(jline("output", component, "config_copied", path=copied_cfg))

        tables_out = _render_tables(stage_map, cfg, tab_dir)
        for t in tables_out:
            logger.info(jline("output", component, f"table_{t['name']}", path=t["csv"]))

        figures_out = _render_figures(stage_map, cfg, fig_dir)
        for f in figures_out:
            logger.info(
                jline("output", component, f"figure_{f['name']}", path=f["path"])
            )

        summary = {
            "component": component,
            "run_dir": run_dir,
            "inputs": stage_map,
            "tables": tables_out,
            "figures": figures_out,
        }
        summary_path = os.path.join(run_dir, "summary.json")
        write_json(summary_path, summary)
        logger.info(jline("output", component, "summary", path=summary_path))

        audit.finish_success()
        return Qf80PaperArtifactsOut(
            run_dir=run_dir, meta=meta, summary_path=summary_path
        )
    except BaseException as e:
        audit.finish_error(e)
        raise
