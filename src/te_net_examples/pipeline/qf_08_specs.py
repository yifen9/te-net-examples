from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from te_net_examples.io.csv import read_csv
from te_net_examples.io.json import write_json
from te_net_examples.utils.audit import Audit
from te_net_examples.utils.console import ConsoleSink
from te_net_examples.utils.logger import Logger
from te_net_examples.utils.meta import build_meta
from te_net_examples.utils.versioner import build_version_dir


@dataclass(frozen=True, slots=True)
class Qf08SpecsOut:
    run_dir: str
    meta: dict[str, Any]
    specs_path: str
    portfolio_sort_tex_path: str
    cs_tex_path: str


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


def _fmt_pct(x: float) -> str:
    if x != x:
        return "---"
    return f"{x:.2%}"


def _fmt_num(x: float) -> str:
    if x != x:
        return "---"
    return f"{x:.4f}"


def _fmt_t(x: float) -> str:
    if x != x:
        return "---"
    return f"{x:.2f}"


def _write_text(path: str, txt: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)


def _portfolio_sort_ls_tex(df: pd.DataFrame) -> str:
    x = df[df["section"].astype(str) == "portfolio_sort_ls"].copy()
    x["ann_ret"] = _to_num(x["ann_ret"])
    x["t_stat"] = _to_num(x["t_stat"])
    x["period"] = x["period"].astype(str)

    order = ["full", "sub1", "sub2"]
    rows = []
    for p in order:
        r = x[x["period"] == p]
        if len(r) == 0:
            rows.append((p, float("nan"), float("nan")))
        else:
            rows.append((p, float(r.iloc[0]["ann_ret"]), float(r.iloc[0]["t_stat"])))

    lines: list[str] = []
    lines.append(r"\begin{tabular}{lrr}")
    lines.append(r"\toprule")
    lines.append(r"Period & LS Ann.\ Return & $t$ \\")
    lines.append(r"\midrule")
    for p, a, t in rows:
        lines.append(f"{p} & {_fmt_pct(a)} & {_fmt_t(t)} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def _cs_ols_tex(df: pd.DataFrame) -> str:
    x = df[df["section"].astype(str) == "cs_ols"].copy()
    x["mean_beta1"] = _to_num(x["mean_beta1"])
    x["t_mean_beta1"] = _to_num(x["t_mean_beta1"])
    x["reject_rate"] = _to_num(x["reject_rate"])
    x["signal"] = x["signal"].astype(str)

    rows = []
    for _, r in x.sort_values("signal").iterrows():
        rows.append(
            (
                r["signal"],
                float(r["mean_beta1"])
                if r["mean_beta1"] == r["mean_beta1"]
                else float("nan"),
                float(r["t_mean_beta1"])
                if r["t_mean_beta1"] == r["t_mean_beta1"]
                else float("nan"),
                float(r["reject_rate"])
                if r["reject_rate"] == r["reject_rate"]
                else float("nan"),
            )
        )

    lines: list[str] = []
    lines.append(r"\begin{tabular}{lrrr}")
    lines.append(r"\toprule")
    lines.append(r"Signal & Mean $\hat\beta$ & $t(\bar\beta)$ & RejectRate \\")
    lines.append(r"\midrule")
    for s, b, t, rr in rows:
        lines.append(f"{s} & {_fmt_num(b)} & {_fmt_t(t)} & {_fmt_pct(rr)} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def run_qf_08_specs(
    *,
    input_dir: str,
    output_root: str,
    src_dir: str,
    script_path: str,
    component: str,
    report_filename: str,
) -> Qf08SpecsOut:
    input_dir = _require_dir(input_dir)
    src_dir = _require_dir(src_dir)
    output_root_abs = os.path.abspath(output_root)

    report_in = _require_file(os.path.join(input_dir, report_filename))

    script_path_abs = _require_file(os.path.abspath(script_path))
    repo_root = _repo_root_from_path(Path(script_path_abs))
    env_path = _require_file(str(repo_root / "uv.lock"))

    params: dict[str, Any] = {
        "input_dir": os.path.abspath(input_dir),
        "output_root": output_root_abs,
        "report_filename": report_filename,
        "component": component,
    }

    meta = build_meta(
        params=params,
        env=env_path,
        script=script_path_abs,
        cfg=report_in,
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
            f'{{"event":"input","component":"{component}","msg":"report","path":"{report_in}"}}'
        )

        df = _df_from_csv(report_in).rename(
            columns={c: c.strip() for c in _df_from_csv(report_in).columns}
        )
        df = _df_from_csv(report_in)
        df = df.rename(columns={c: c.strip() for c in df.columns})

        portfolio_sort_tex = _portfolio_sort_ls_tex(df)
        cs_tex = _cs_ols_tex(df)

        portfolio_sort_path = os.path.join(run_dir, "portfolio_sort_ls.tex")
        cs_path = os.path.join(run_dir, "cs_ols.tex")
        _write_text(portfolio_sort_path, portfolio_sort_tex + "\n")
        _write_text(cs_path, cs_tex + "\n")

        specs = {
            "portfolio_sort_ls_tex": portfolio_sort_tex,
            "cs_ols_tex": cs_tex,
        }
        specs_out = os.path.join(run_dir, "specs.json")
        write_json(specs_out, specs)

        logger.info(
            f'{{"event":"output","component":"{component}","msg":"portfolio_sort_tex","path":"{portfolio_sort_path}"}}'
        )
        logger.info(
            f'{{"event":"output","component":"{component}","msg":"cs_tex","path":"{cs_path}"}}'
        )
        logger.info(
            f'{{"event":"output","component":"{component}","msg":"specs","path":"{specs_out}"}}'
        )

        audit.finish_success()
        return Qf08SpecsOut(
            run_dir=run_dir,
            meta=meta,
            specs_path=specs_out,
            portfolio_sort_tex_path=portfolio_sort_path,
            cs_tex_path=cs_path,
        )
    except BaseException as e:
        audit.finish_error(e)
        raise
