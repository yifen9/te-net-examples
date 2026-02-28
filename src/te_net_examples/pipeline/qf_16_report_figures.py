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
from te_net_examples.utils.logger import Logger
from te_net_examples.utils.meta import build_meta
from te_net_examples.utils.progress import Progress
from te_net_examples.utils.versioner import build_version_dir


@dataclass(frozen=True, slots=True)
class Qf16ReportFiguresOut:
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


def _jline(event: str, component: str, msg: str, **kw: Any) -> str:
    items = {"event": event, "component": component, "msg": msg, **kw}
    parts: list[str] = []
    for k in sorted(items.keys()):
        v = items[k]
        if v is True:
            parts.append(f'"{k}":true')
        elif v is False:
            parts.append(f'"{k}":false')
        elif v is None:
            parts.append(f'"{k}":null')
        elif isinstance(v, (int, float)):
            parts.append(f'"{k}":{v}')
        else:
            s = str(v).replace("\\", "\\\\").replace('"', '\\"')
            parts.append(f'"{k}":"{s}"')
    return "{" + ",".join(parts) + "}"


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


def _read_meta_params(run_dir: str) -> dict[str, Any]:
    meta_path = os.path.join(run_dir, "_meta.json")
    obj = read_json(_require_file(meta_path))
    if not isinstance(obj, dict):
        raise ValueError(f"_meta.json must be a mapping: {meta_path}")
    params = obj.get("params", None)
    if not isinstance(params, dict):
        raise ValueError(f"_meta.json missing params mapping: {meta_path}")
    return params


def _fmt_num(x: float) -> str:
    if x != x:
        return "---"
    return f"{x:.4f}"


def _fmt_pct(x: float) -> str:
    if x != x:
        return "---"
    return f"{x:.2%}"


def _write_text(path: str, txt: str) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)


def _latex_table_report(df: pd.DataFrame, keys: list[str], cols: list[str]) -> str:
    keep = keys + ["n"] + [f"{c}_mean" for c in cols] + [f"{c}_std" for c in cols]
    x = df[keep].copy()

    lines: list[str] = []
    colspec = "l" * len(keys) + "r" * (1 + 2 * len(cols))
    lines.append(r"\begin{tabular}{" + colspec + r"}")
    lines.append(r"\toprule")
    head = keys + ["n"]
    for c in cols:
        head.append(f"{c} mean")
        head.append(f"{c} std")
    lines.append(" & ".join(head) + r" \\")
    lines.append(r"\midrule")

    for _, r in x.iterrows():
        row: list[str] = []
        for k in keys:
            row.append(str(r[k]))
        row.append(str(int(r["n"])))
        for c in cols:
            row.append(
                _fmt_num(float(r[f"{c}_mean"]))
                if r[f"{c}_mean"] == r[f"{c}_mean"]
                else "---"
            )
            row.append(
                _fmt_num(float(r[f"{c}_std"]))
                if r[f"{c}_std"] == r[f"{c}_std"]
                else "---"
            )
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def _fig_bar_by_density(
    df: pd.DataFrame,
    metric: str,
    out_path: str,
) -> dict[str, Any]:
    x = df.copy()
    x["sel_density"] = x["sel_density"].astype(str)
    x = x.sort_values(["sel_density"], ascending=[True])

    g = x.groupby("sel_density", sort=True)
    dens = g.size().index.tolist()
    y = g[metric].mean().astype(float).to_numpy()
    n = g.size().astype(int).to_numpy()

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    xs = np.arange(len(dens))
    ax.bar(xs, y)
    ax.set_xticks(xs)
    ax.set_xticklabels(dens, rotation=0)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by selection density")
    for i, (yy, nn) in enumerate(zip(y, n)):
        if yy == yy:
            ax.text(i, yy, f"n={int(nn)}", ha="center", va="bottom", fontsize=9)
    ax.axhline(0.0, linewidth=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    return {"path": out_path, "metric": metric, "densities": dens, "n": n.tolist()}


def _fig_precision_recall_scatter(
    df: pd.DataFrame,
    out_path: str,
    max_points: int,
) -> dict[str, Any]:
    x = df.copy()
    x["precision"] = _to_num(x["precision"])
    x["recall"] = _to_num(x["recall"])
    x = x[np.isfinite(x["precision"]) & np.isfinite(x["recall"])]

    if len(x) > int(max_points):
        x = x.sample(n=int(max_points), random_state=0)

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.scatter(
        x["precision"].to_numpy(dtype=float), x["recall"].to_numpy(dtype=float), s=10
    )
    ax.set_xlabel("precision")
    ax.set_ylabel("recall")
    ax.set_title("precision vs recall (sampled)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    return {"path": out_path, "n_points": int(len(x)), "max_points": int(max_points)}


def run_qf_16_report_figures(
    *,
    input_dir: str,
    output_root: str,
    src_dir: str,
    config_path: str,
    script_path: str,
    component: str,
    design_filename: str,
    metrics_filename: str,
) -> Qf16ReportFiguresOut:
    input_dir = _require_dir(input_dir)
    src_dir = _require_dir(src_dir)
    output_root_abs = os.path.abspath(output_root)

    cfg_path = _require_file(os.path.abspath(config_path))
    script_path_abs = _require_file(os.path.abspath(script_path))

    meta_in = _require_file(os.path.join(input_dir, "_meta.json"))
    params_up = _read_meta_params(input_dir)
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
    keys = groupby_cfg.get("keys", [])
    if not isinstance(keys, list) or not keys:
        raise ValueError("groupby.keys must be a non-empty list")
    keys = [str(k) for k in keys]

    metrics_cfg = cfg.get("metrics", {})
    if metrics_cfg is None:
        metrics_cfg = {}
    if not isinstance(metrics_cfg, dict):
        raise ValueError("config.metrics must be a mapping")
    cols = metrics_cfg.get("cols", [])
    if not isinstance(cols, list) or not cols:
        raise ValueError("metrics.cols must be a non-empty list")
    cols = [str(c) for c in cols]

    output_cfg = cfg.get("output", {})
    if output_cfg is None:
        output_cfg = {}
    if not isinstance(output_cfg, dict):
        raise ValueError("config.output must be a mapping")

    save_report_csv = bool(output_cfg.get("save_report_csv", True))
    save_specs_json = bool(output_cfg.get("save_specs_json", True))
    save_figures = bool(output_cfg.get("save_figures", True))
    max_points_scatter = int(output_cfg.get("max_points_scatter", 5000))

    params: dict[str, Any] = {
        "input_dir": os.path.abspath(input_dir),
        "output_root": output_root_abs,
        "config_path": cfg_path,
        "component": component,
        "design_path": os.path.abspath(design_in),
        "metrics_path": os.path.abspath(metrics_in),
        "meta_in": os.path.abspath(meta_in),
        "groupby_keys": keys,
        "metric_cols": cols,
        "save_report_csv": bool(save_report_csv),
        "save_specs_json": bool(save_specs_json),
        "save_figures": bool(save_figures),
        "max_points_scatter": int(max_points_scatter),
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
        logger.info(_jline("stage", component, "start", run_dir=run_dir))
        logger.info(_jline("input", component, "input_dir", path=input_dir))
        logger.info(_jline("input", component, "design", path=design_in))
        logger.info(_jline("input", component, "metrics", path=metrics_in))
        logger.info(_jline("input", component, "config", path=cfg_path))

        cfg_dir = os.path.join(run_dir, "cfg")
        os.makedirs(cfg_dir, exist_ok=True)
        cfg_copied = os.path.join(cfg_dir, os.path.basename(cfg_path))
        shutil.copy2(cfg_path, cfg_copied)
        logger.info(_jline("output", component, "config_copied", path=cfg_copied))

        df_design = _df_from_csv(design_in).rename(
            columns={c: c.strip() for c in _df_from_csv(design_in).columns}
        )
        df_design = _df_from_csv(design_in)
        df_design = df_design.rename(columns={c: c.strip() for c in df_design.columns})
        df_metrics = _df_from_csv(metrics_in)
        df_metrics = df_metrics.rename(
            columns={c: c.strip() for c in df_metrics.columns}
        )

        _require_cols(df_design, ["design_id"])
        _require_cols(df_metrics, ["design_id"])

        d = df_design.copy()
        m = df_metrics.copy()
        d["design_id"] = _to_num(d["design_id"])
        m["design_id"] = _to_num(m["design_id"])
        dd = d.merge(m, on="design_id", how="inner", suffixes=("", "_m"))

        for c in cols:
            if c in dd.columns:
                dd[c] = _to_num(dd[c])

        dd["sel_density"] = (
            dd["sel_density"].astype(str) if "sel_density" in dd.columns else ""
        )

        p = Progress(logger=logger, name="qf/16_report_figures", total=3)
        p.start()

        rep = dd.copy()
        for k in keys:
            if k not in rep.columns:
                rep[k] = ""
        for c in cols:
            if c not in rep.columns:
                rep[c] = np.nan

        g = rep.groupby(keys, dropna=False, sort=True)
        rows: list[dict[str, Any]] = []
        for name, sub in g:
            row: dict[str, Any] = {}
            if len(keys) == 1:
                row[keys[0]] = name
            else:
                for i, k in enumerate(keys):
                    row[k] = name[i]
            row["n"] = int(len(sub))
            for c in cols:
                v = sub[c].to_numpy(dtype=float)
                v = v[np.isfinite(v)]
                row[f"{c}_mean"] = float(v.mean()) if v.size else float("nan")
                row[f"{c}_std"] = float(v.std(ddof=1)) if v.size >= 2 else float("nan")
            rows.append(row)

        report_df = pd.DataFrame(rows)
        report_df = report_df.sort_values(
            keys, ascending=[True] * len(keys)
        ).reset_index(drop=True)

        report_path = None
        if save_report_csv:
            report_path = os.path.join(run_dir, "report.csv")
            write_csv(
                report_path, _rows_from_df(report_df), header=list(report_df.columns)
            )
            logger.info(_jline("output", component, "report", path=report_path))

        p.step(1)

        specs_path = None
        if save_specs_json:
            tex = _latex_table_report(report_df, keys, cols)
            tex_path = os.path.join(run_dir, "report.tex")
            _write_text(tex_path, tex + "\n")
            specs = {
                "groupby_keys": keys,
                "metric_cols": cols,
                "report_tex": tex,
                "report_tex_path": tex_path,
            }
            specs_path = os.path.join(run_dir, "specs.json")
            write_json(specs_path, specs)
            logger.info(_jline("output", component, "specs", path=specs_path))

        p.step(1)

        figs: dict[str, Any] = {}
        if save_figures:
            fig_dir = os.path.join(run_dir, "figures")
            _ensure_dir(fig_dir)

            if "f1" in dd.columns:
                f1_path = os.path.join(fig_dir, "f1_vs_density.png")
                figs["f1_vs_density"] = _fig_bar_by_density(dd, "f1", f1_path)
            if "hub_rec_nio" in dd.columns:
                hr_path = os.path.join(fig_dir, "hubrec_vs_density.png")
                figs["hubrec_vs_density"] = _fig_bar_by_density(
                    dd, "hub_rec_nio", hr_path
                )
            if "precision" in dd.columns and "recall" in dd.columns:
                pr_path = os.path.join(fig_dir, "precision_recall_scatter.png")
                figs["precision_recall_scatter"] = _fig_precision_recall_scatter(
                    dd, pr_path, int(max_points_scatter)
                )

            for k, v in figs.items():
                if isinstance(v, dict) and "path" in v:
                    logger.info(
                        _jline("output", component, f"figure_{k}", path=v["path"])
                    )

        p.step(1)
        p.finish()

        summary = {
            "inputs": {
                "input_dir": os.path.abspath(input_dir),
                "design": os.path.abspath(design_in),
                "metrics": os.path.abspath(metrics_in),
            },
            "groupby_keys": keys,
            "metric_cols": cols,
            "n_joined": int(len(dd)),
            "outputs": {
                "report_csv": report_path,
                "specs_json": specs_path,
                "figures": figs,
            },
        }

        summary_out = os.path.join(run_dir, "summary.json")
        write_json(summary_out, summary)
        logger.info(_jline("output", component, "summary", path=summary_out))

        audit.finish_success()
        return Qf16ReportFiguresOut(
            run_dir=run_dir,
            meta=meta,
            report_path=report_path,
            specs_path=specs_path,
            summary_path=summary_out,
        )
    except BaseException as e:
        audit.finish_error(e)
        raise
