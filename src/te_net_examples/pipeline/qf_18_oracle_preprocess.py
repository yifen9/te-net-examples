from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from te_net_examples.io.csv import read_csv
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
class Qf18OraclePreprocessOut:
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


def _require_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _trial_dir(run_dir: str, design_id: int) -> str:
    return os.path.join(run_dir, "trial", f"{int(design_id):08d}")


def _read_mat(path: str, name: str) -> np.ndarray:
    x = np.load(path, allow_pickle=False)
    if not isinstance(x, np.ndarray):
        raise ValueError(f"invalid array for {name}: {path}")
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2D: {path}")
    return x.astype(np.float64, copy=False)


def _write_npy(path: str, arr: np.ndarray) -> None:
    _ensure_dir(os.path.dirname(path))
    np.save(path, arr)


def _oracle_neutralize(
    R: np.ndarray, F: np.ndarray, add_intercept: bool
) -> tuple[np.ndarray, dict[str, Any]]:
    T, N = R.shape
    if F.shape[0] != T:
        raise ValueError("factors and returns must have same T")
    k = int(F.shape[1])
    if add_intercept:
        X = np.concatenate([np.ones((T, 1), dtype=np.float64), F], axis=1)
    else:
        X = F
    coef, _, rank, _ = np.linalg.lstsq(X, R, rcond=None)
    Rhat = X @ coef
    resid = R - Rhat

    sse = float(np.sum(resid * resid))
    tss = float(np.sum((R - R.mean(axis=0, keepdims=True)) ** 2))
    r2 = 1.0 - (sse / tss) if tss > 0.0 else 0.0

    info = {
        "T": int(T),
        "N": int(N),
        "k": int(k),
        "add_intercept": bool(add_intercept),
        "rank": int(rank),
        "coef_shape": [int(coef.shape[0]), int(coef.shape[1])],
        "r2_total": float(r2),
    }
    return resid.astype(np.float64, copy=False), info


def run_qf_18_oracle_preprocess(
    *,
    input_dir: str,
    output_root: str,
    src_dir: str,
    config_path: str,
    script_path: str,
    component: str,
    design_filename: str,
) -> Qf18OraclePreprocessOut:
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

    repo_root = _repo_root_from_path(Path(script_path_abs))
    env_path = _require_file(str(repo_root / "uv.lock"))

    cfg = read_yaml(cfg_path)
    if not isinstance(cfg, dict):
        raise ValueError("config must be a mapping")

    input_cfg = cfg.get("input", {})
    if input_cfg is None:
        input_cfg = {}
    if not isinstance(input_cfg, dict):
        raise ValueError("config.input must be a mapping")

    factors_required = bool(input_cfg.get("factors_required", True))

    oracle_cfg = cfg.get("oracle", {})
    if oracle_cfg is None:
        oracle_cfg = {}
    if not isinstance(oracle_cfg, dict):
        raise ValueError("config.oracle must be a mapping")

    add_intercept = bool(oracle_cfg.get("add_intercept", True))

    output_cfg = cfg.get("output", {})
    if output_cfg is None:
        output_cfg = {}
    if not isinstance(output_cfg, dict):
        raise ValueError("config.output must be a mapping")

    save_returns = bool(output_cfg.get("save_returns_oracle_neutral", True))
    save_oracle_json = bool(output_cfg.get("save_oracle_json", True))

    params: dict[str, Any] = {
        "input_dir": os.path.abspath(input_dir),
        "output_root": output_root_abs,
        "config_path": cfg_path,
        "component": component,
        "design_path": os.path.abspath(design_in),
        "meta_in": os.path.abspath(meta_in),
        "factors_required": bool(factors_required),
        "add_intercept": bool(add_intercept),
        "save_returns_oracle_neutral": bool(save_returns),
        "save_oracle_json": bool(save_oracle_json),
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
        logger.info(jline("input", component, "config", path=cfg_path))

        cfg_dir = os.path.join(run_dir, "cfg")
        os.makedirs(cfg_dir, exist_ok=True)
        cfg_copied = os.path.join(cfg_dir, os.path.basename(cfg_path))
        shutil.copy2(cfg_path, cfg_copied)
        logger.info(jline("output", component, "config_copied", path=cfg_copied))

        df = _df_from_csv(design_in)
        df = df.rename(columns={c: c.strip() for c in df.columns})
        _require_cols(df, ["design_id", "dgp"])
        df["design_id"] = _to_num(df["design_id"])
        df["dgp"] = df["dgp"].astype(str)

        records = df.to_dict(orient="records")
        p = Progress(
            logger=logger, name="qf/18_oracle_preprocess", total=int(len(records))
        )
        p.start()

        items: list[dict[str, Any]] = []
        n_ok = 0
        n_skip = 0

        for r in records:
            did = int(float(r["design_id"]))
            in_trial = _trial_dir(input_dir, did)
            ret_in = os.path.join(in_trial, "returns.npy")
            fac_in = os.path.join(in_trial, "factors.npy")

            dgp = str(r.get("dgp", "")).strip()
            is_factor_dgp = dgp in ("garch_factor",)
            has_f = os.path.isfile(fac_in)

            if (not has_f) and bool(factors_required) and bool(is_factor_dgp):
                raise FileNotFoundError(fac_in)

            if not has_f:
                n_skip += 1
                items.append(
                    {
                        "design_id": int(did),
                        "skipped": True,
                        "reason": "missing_factors",
                        "dgp": dgp,
                        "input_returns": ret_in if os.path.isfile(ret_in) else None,
                        "input_factors": None,
                        "output_returns": None,
                        "oracle_json": None,
                    }
                )
                p.step(1)
                continue

            R = _read_mat(_require_file(ret_in), "returns")
            F = _read_mat(_require_file(fac_in), "factors")

            resid, info = _oracle_neutralize(R, F, bool(add_intercept))

            out_trial = _trial_dir(run_dir, did)
            _ensure_dir(out_trial)

            out_ret = None
            if save_returns:
                out_ret = os.path.join(out_trial, "returns_oracle_neutral.npy")
                _write_npy(out_ret, resid)

            out_json = None
            if save_oracle_json:
                obj = {
                    "design_id": int(did),
                    "input": {"returns": ret_in, "factors": fac_in},
                    "output": {"returns_oracle_neutral": out_ret},
                    "oracle": info,
                }
                out_json = os.path.join(out_trial, "oracle.json")
                write_json(out_json, obj)

            items.append(
                {
                    "design_id": int(did),
                    "skipped": False,
                    "input_returns": ret_in,
                    "input_factors": fac_in,
                    "output_returns": out_ret,
                    "oracle_json": out_json,
                }
            )
            n_ok += 1
            p.step(1)

        p.finish()

        summary = {
            "n_rows": int(len(records)),
            "n_ok": int(n_ok),
            "n_skip": int(n_skip),
            "trial_root": os.path.join(run_dir, "trial"),
            "items": items[:2000],
        }

        summary_out = os.path.join(run_dir, "summary.json")
        write_json(summary_out, summary)
        logger.info(jline("output", component, "summary", path=summary_out))

        audit.finish_success()
        return Qf18OraclePreprocessOut(
            run_dir=run_dir, meta=meta, summary_path=summary_out
        )
    except BaseException as e:
        audit.finish_error(e)
        raise
