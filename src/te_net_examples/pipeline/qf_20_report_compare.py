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
class Qf20ReportCompareOut:
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


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _require_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")


def _write_text(path: str, txt: str) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)


def _latex_report(df: pd.DataFrame, keys: list[str], cols: list[str]) -> str:
    keep = (
        keys
        + ["subset", "n"]
        + [f"{c}_mean" for c in cols]
        + [f"{c}_std" for c in cols]
    )
    x = df[keep].copy()

    lines: list[str] = []
    colspec = "l" * (len(keys) + 1) + "r" * (1 + 2 * len(cols))
    lines.append(r"\begin{tabular}{" + colspec + r"}")
    lines.append(r"\toprule")
    head = keys + ["subset", "n"]
    for c in cols:
        head.append(f"{c} mean")
        head.append(f"{c} std")
    lines.append(" & ".join(head) + r" \\")
    lines.append(r"\midrule")
    for _, r in x.iterrows():
        row: list[str] = []
        for k in keys:
            row.append(str(r[k]))
        row.append(str(r["subset"]))
        row.append(str(int(r["n"])))
        for c in cols:
            m = r[f"{c}_mean"]
            s = r[f"{c}_std"]
            row.append(f"{float(m):.4f}" if m == m else "---")
            row.append(f"{float(s):.4f}" if s == s else "---")
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


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


def run_qf_20_report_compare(
    *,
    input_dir: str,
    output_root: str,
    src_dir: str,
    config_path: str,
    script_path: str,
    component: str,
    compare_filename: str,
) -> Qf20ReportCompareOut:
    input_dir = _require_dir(input_dir)
    src_dir = _require_dir(src_dir)
    output_root_abs = os.path.abspath(output_root)

    cfg_path = _require_file(os.path.abspath(config_path))
    script_path_abs = _require_file(os.path.abspath(script_path))

    compare_in = _require_file(os.path.join(input_dir, compare_filename))

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

    filt_cfg = cfg.get("filters", {})
    if filt_cfg is None:
        filt_cfg = {}
    if not isinstance(filt_cfg, dict):
        raise ValueError("config.filters must be a mapping")
    has_true_adj_only = bool(filt_cfg.get("has_true_adj_only", True))
    oracle_status_list = filt_cfg.get("oracle_status", ["ok", "missing"])
    if not isinstance(oracle_status_list, list) or not oracle_status_list:
        raise ValueError("filters.oracle_status must be a non-empty list")
    oracle_status_list = [str(x) for x in oracle_status_list]

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
        "compare_path": os.path.abspath(compare_in),
        "groupby_keys": keys,
        "has_true_adj_only": bool(has_true_adj_only),
        "oracle_status": oracle_status_list,
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
        logger.info(jline("input", component, "compare", path=compare_in))
        logger.info(jline("input", component, "config", path=cfg_path))

        cfg_dir = os.path.join(run_dir, "cfg")
        os.makedirs(cfg_dir, exist_ok=True)
        cfg_copied = os.path.join(cfg_dir, os.path.basename(cfg_path))
        shutil.copy2(cfg_path, cfg_copied)
        logger.info(jline("output", component, "config_copied", path=cfg_copied))

        df = _df_from_csv(compare_in)
        df = df.rename(columns={c: c.strip() for c in df.columns})
        need = ["design_id", "dgp", "te_name", "sel_density", "oracle_status"]
        _require_cols(df, need)

        for c in [
            "est_precision",
            "est_recall",
            "est_f1",
            "est_hub_rec",
            "ora_precision",
            "ora_recall",
            "ora_f1",
            "ora_hub_rec",
        ]:
            if c in df.columns:
                df[c] = _to_num(df[c])

        df["has_true_adj"] = df["est_f1"].notna()
        df["sel_density"] = df["sel_density"].astype(str)
        df["oracle_status"] = df["oracle_status"].astype(str)

        p = Progress(logger=logger, name="qf/20_report_compare", total=3)
        p.start()

        df0 = df[df["oracle_status"].isin(oracle_status_list)].copy()

        blocks: list[tuple[str, pd.DataFrame, list[str]]] = []

        df_var = (
            df0[df0["has_true_adj"]].copy() if bool(has_true_adj_only) else df0.copy()
        )
        cols_var = ["est_precision", "est_recall", "est_f1", "est_hub_rec"]
        blocks.append(("estimated_has_true_adj", df_var, cols_var))

        df_oracle = df0[df0["oracle_status"] == "ok"].copy()
        cols_oracle = ["ora_precision", "ora_recall", "ora_f1", "ora_hub_rec"]
        blocks.append(("oracle_ok", df_oracle, cols_oracle))

        rows: list[dict[str, Any]] = []
        for subset, sub, cols in blocks:
            x = sub.copy()
            for k in keys:
                if k not in x.columns:
                    x[k] = ""
            g = x.groupby(keys, dropna=False, sort=True)
            for name, gg in g:
                row: dict[str, Any] = {}
                if len(keys) == 1:
                    row[keys[0]] = name
                else:
                    for i, k in enumerate(keys):
                        row[k] = name[i]
                row["subset"] = subset
                row["n"] = int(len(gg))
                for c in cols:
                    v = gg[c].to_numpy(dtype=float)
                    v = v[np.isfinite(v)]
                    row[f"{c}_mean"] = float(v.mean()) if v.size else float("nan")
                    row[f"{c}_std"] = (
                        float(v.std(ddof=1)) if v.size >= 2 else float("nan")
                    )
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

        p.step(1)

        specs_path = None
        if save_specs_json:
            cols_all = [
                "est_precision",
                "est_recall",
                "est_f1",
                "est_hub_rec",
                "ora_precision",
                "ora_recall",
                "ora_f1",
                "ora_hub_rec",
            ]
            tex = _latex_report(report_df, keys, cols_all)
            tex_path = os.path.join(run_dir, "report.tex")
            _write_text(tex_path, tex + "\n")
            specs = {
                "report_tex": tex,
                "report_tex_path": tex_path,
                "groupby_keys": keys,
            }
            specs_path = os.path.join(run_dir, "specs.json")
            write_json(specs_path, specs)
            logger.info(jline("output", component, "specs", path=specs_path))

        p.step(1)

        figs: dict[str, Any] = {}
        if save_figures:
            fig_dir = os.path.join(run_dir, "figures")
            _ensure_dir(fig_dir)

            if (
                bool(has_true_adj_only)
                and "est_f1" in df_var.columns
                and df_var["est_f1"].notna().any()
            ):
                figs["f1_est_vs_density"] = _fig_bar_by_density(
                    df_var,
                    "est_f1",
                    os.path.join(fig_dir, "f1_est_vs_density.png"),
                    "Estimated F1 by density (has true_adj)",
                )
            if (
                bool(has_true_adj_only)
                and "est_hub_rec" in df_var.columns
                and df_var["est_hub_rec"].notna().any()
            ):
                figs["hubrec_est_vs_density"] = _fig_bar_by_density(
                    df_var,
                    "est_hub_rec",
                    os.path.join(fig_dir, "hubrec_est_vs_density.png"),
                    "Estimated hub recovery by density (has true_adj)",
                )
            if "ora_f1" in df_oracle.columns and df_oracle["ora_f1"].notna().any():
                figs["f1_oracle_vs_density"] = _fig_bar_by_density(
                    df_oracle,
                    "ora_f1",
                    os.path.join(fig_dir, "f1_oracle_vs_density.png"),
                    "Oracle F1 by density (oracle ok)",
                )

            for k, v in figs.items():
                if isinstance(v, dict) and "path" in v:
                    logger.info(
                        jline("output", component, f"figure_{k}", path=v["path"])
                    )

        p.step(1)
        p.finish()

        summary = {
            "inputs": {"compare": os.path.abspath(compare_in)},
            "groupby_keys": keys,
            "n_rows": int(len(df)),
            "n_rows_filtered": int(len(df0)),
            "n_rows_has_true_adj": int(df0["has_true_adj"].sum()),
            "n_rows_oracle_ok": int((df0["oracle_status"] == "ok").sum()),
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
        return Qf20ReportCompareOut(
            run_dir=run_dir,
            meta=meta,
            report_path=report_path,
            specs_path=specs_path,
            summary_path=summary_out,
        )
    except BaseException as e:
        audit.finish_error(e)
        raise
