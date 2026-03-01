from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from te_net_lib.eval.stats import cross_sectional_tstat


@dataclass(frozen=True, slots=True)
class Qf22CsTstatOut:
    run_dir: str
    meta: dict[str, Any]
    tstat_path: str | None
    delta_path: str | None
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


def _read_returns(path: str) -> np.ndarray:
    x = np.load(path, allow_pickle=False)
    if not isinstance(x, np.ndarray):
        raise ValueError(f"invalid returns: {path}")
    if x.ndim != 2:
        raise ValueError(f"returns must be 2D: {path}")
    return x.astype(np.float64, copy=False)


def _read_signal(path: str) -> np.ndarray:
    x = np.load(path, allow_pickle=False)
    if not isinstance(x, np.ndarray):
        raise ValueError(f"invalid signal: {path}")
    if x.ndim != 1:
        raise ValueError(f"signal must be 1D: {path}")
    return x.astype(np.float64, copy=False)


def run_qf_22_cs_tstat(
    *,
    input_dir: str,
    output_root: str,
    src_dir: str,
    config_path: str,
    script_path: str,
    component: str,
    index_filename: str,
) -> Qf22CsTstatOut:
    input_dir = _require_dir(input_dir)
    src_dir = _require_dir(src_dir)
    output_root_abs = os.path.abspath(output_root)

    cfg_path = _require_file(os.path.abspath(config_path))
    script_path_abs = _require_file(os.path.abspath(script_path))

    index_in = _require_file(os.path.join(input_dir, index_filename))

    repo_root = _repo_root_from_path(Path(script_path_abs))
    env_path = _require_file(str(repo_root / "uv.lock"))

    cfg = read_yaml(cfg_path)
    if not isinstance(cfg, dict):
        raise ValueError("config must be a mapping")

    t_cfg = cfg.get("tstat", {})
    if t_cfg is None:
        t_cfg = {}
    if not isinstance(t_cfg, dict):
        raise ValueError("config.tstat must be a mapping")
    add_intercept = bool(t_cfg.get("add_intercept", True))

    out_cfg = cfg.get("output", {})
    if out_cfg is None:
        out_cfg = {}
    if not isinstance(out_cfg, dict):
        raise ValueError("config.output must be a mapping")
    save_tstat_csv = bool(out_cfg.get("save_tstat_csv", True))
    save_delta_csv = bool(out_cfg.get("save_delta_csv", True))

    params: dict[str, Any] = {
        "input_dir": os.path.abspath(input_dir),
        "output_root": output_root_abs,
        "config_path": cfg_path,
        "component": component,
        "index_path": os.path.abspath(index_in),
        "add_intercept": bool(add_intercept),
        "save_tstat_csv": bool(save_tstat_csv),
        "save_delta_csv": bool(save_delta_csv),
    }

    meta = build_meta(
        params=params, env=env_path, script=script_path_abs, cfg=cfg_path, src=src_dir
    )

    run_dir = build_version_dir(output_root_abs, meta)
    audit = Audit.create(run_dir, meta)
    logger = Logger(sinks=[ConsoleSink(), audit])

    try:
        logger.info(jline("stage", component, "start", run_dir=run_dir))
        logger.info(jline("input", component, "index", path=index_in))
        logger.info(jline("input", component, "config", path=cfg_path))

        cfg_dir = os.path.join(run_dir, "cfg")
        os.makedirs(cfg_dir, exist_ok=True)
        cfg_copied = os.path.join(cfg_dir, os.path.basename(cfg_path))
        shutil.copy2(cfg_path, cfg_copied)
        logger.info(jline("output", component, "config_copied", path=cfg_copied))

        df = _df_from_csv(index_in)
        df = df.rename(columns={c: c.strip() for c in df.columns})
        need = ["design_id", "branch", "returns_path", "signal_path"]
        _require_cols(df, need)

        df["design_id"] = _to_num(df["design_id"])
        df["branch"] = df["branch"].astype(str)

        records = df.to_dict(orient="records")
        p = Progress(logger=logger, name="qf/22_cs_tstat", total=int(len(records)))
        p.start()

        rows: list[dict[str, Any]] = []
        for r in records:
            did = int(float(r["design_id"]))
            branch = str(r["branch"]).strip()
            ret_path = str(r["returns_path"])
            sig_path = str(r["signal_path"])

            if sig_path.strip() == "" or (not os.path.isfile(sig_path)):
                p.step(1)
                continue
            if not os.path.isfile(ret_path):
                raise FileNotFoundError(ret_path)

            R = _read_returns(ret_path)
            s = _read_signal(sig_path)
            t = float(cross_sectional_tstat(R, s, bool(add_intercept)))

            row: dict[str, Any] = {
                "design_id": int(did),
                "branch": branch,
                "tstat": float(t),
            }
            for k in ["dgp", "te_name", "sel_density", "fn_n_components", "N", "T"]:
                if k in r:
                    row[k] = r[k]
            rows.append(row)

            p.step(1)

        p.finish()

        tstat_path = None
        delta_path = None

        out_df = pd.DataFrame(rows)
        if len(out_df):
            out_df["tstat"] = _to_num(out_df["tstat"])

        if save_tstat_csv:
            tstat_path = os.path.join(run_dir, "tstat.csv")
            if len(out_df):
                cols = ["design_id", "branch"]
                for k in ["dgp", "te_name", "sel_density", "fn_n_components", "N", "T"]:
                    if k in out_df.columns:
                        cols.append(k)
                cols.append("tstat")
                out_df2 = out_df[cols].sort_values(
                    ["design_id", "branch"], ascending=[True, True]
                )
                write_csv(
                    tstat_path, _rows_from_df(out_df2), header=list(out_df2.columns)
                )
            else:
                write_csv(tstat_path, [], header=["design_id", "branch", "tstat"])
            logger.info(jline("output", component, "tstat", path=tstat_path))

        if save_delta_csv:
            if len(out_df):
                w = out_df.pivot_table(
                    index="design_id", columns="branch", values="tstat", aggfunc="first"
                )
                w = w.reset_index()
                if "estimated" in w.columns and "oracle" in w.columns:
                    w["delta_tstat"] = _to_num(w["oracle"]) - _to_num(w["estimated"])
                else:
                    w["delta_tstat"] = np.nan
                delta_path = os.path.join(run_dir, "delta.csv")
                write_csv(delta_path, _rows_from_df(w), header=list(w.columns))
            else:
                delta_path = os.path.join(run_dir, "delta.csv")
                write_csv(
                    delta_path,
                    [],
                    header=["design_id", "estimated", "oracle", "delta_tstat"],
                )
            logger.info(jline("output", component, "delta", path=delta_path))

        summary = {
            "n_rows_index": int(len(df)),
            "n_rows_tstat": int(len(out_df)),
            "add_intercept": bool(add_intercept),
            "outputs": {"tstat_csv": tstat_path, "delta_csv": delta_path},
        }

        summary_out = os.path.join(run_dir, "summary.json")
        write_json(summary_out, summary)
        logger.info(jline("output", component, "summary", path=summary_out))

        audit.finish_success()
        return Qf22CsTstatOut(
            run_dir=run_dir,
            meta=meta,
            tstat_path=tstat_path,
            delta_path=delta_path,
            summary_path=summary_out,
        )
    except BaseException as e:
        audit.finish_error(e)
        raise
