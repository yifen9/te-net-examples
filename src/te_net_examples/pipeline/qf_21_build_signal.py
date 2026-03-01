from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from te_net_lib.eval.nio import compute_nio
from te_net_lib.graph.edge_select import select_fixed_density
from te_net_lib.te.lasso_te import lasso_te_matrix
from te_net_lib.te.linear_ols import ols_te_matrix


@dataclass(frozen=True, slots=True)
class Qf21BuildSignalOut:
    run_dir: str
    meta: dict[str, Any]
    index_path: str | None
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


def _to_bool(x: Any, name: str) -> bool:
    if isinstance(x, bool):
        return bool(x)
    s = str(x).strip().lower()
    if s in ("1", "true", "t", "yes", "y"):
        return True
    if s in ("0", "false", "f", "no", "n"):
        return False
    raise ValueError(f"invalid bool for {name}: {x}")


def _to_int(x: Any, name: str) -> int:
    if isinstance(x, bool):
        raise ValueError(f"invalid int for {name}: {x}")
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        if float(x).is_integer():
            return int(x)
        raise ValueError(f"invalid int for {name}: {x}")
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        raise ValueError(f"invalid int for {name}: {x}")
    try:
        return int(s)
    except Exception:
        f = float(s)
        if float(f).is_integer():
            return int(f)
    raise ValueError(f"invalid int for {name}: {x}")


def _to_float(x: Any, name: str) -> float:
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return float("nan")
    return float(s)


def _trial_dir(root: str, design_id: int) -> str:
    return os.path.join(root, "trial", f"{int(design_id):08d}")


def _read_meta_params(run_dir: str) -> dict[str, Any]:
    meta_path = os.path.join(run_dir, "_meta.json")
    obj = read_json(_require_file(meta_path))
    if not isinstance(obj, dict):
        raise ValueError(f"_meta.json must be a mapping: {meta_path}")
    params = obj.get("params", None)
    if not isinstance(params, dict):
        raise ValueError(f"_meta.json missing params mapping: {meta_path}")
    return params


def _read_returns(path: str) -> np.ndarray:
    x = np.load(path, allow_pickle=False)
    if not isinstance(x, np.ndarray):
        raise ValueError(f"invalid returns: {path}")
    if x.ndim != 2:
        raise ValueError(f"returns must be 2D: {path}")
    return x.astype(np.float64, copy=False)


def _write_npy(path: str, arr: np.ndarray) -> None:
    _ensure_dir(os.path.dirname(path))
    np.save(path, arr)


def _compute_nio_signal(
    returns: np.ndarray,
    *,
    lag: int,
    exclude_self: bool,
    te_name: str,
    te_params: dict[str, Any],
    sel_density: float,
    sel_mode: str,
    sel_exclude_self: bool,
    nio_exclude_self: bool,
    nio_normalize: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    if te_name == "ols":
        add_intercept = bool(te_params.get("add_intercept", True))
        out = ols_te_matrix(returns, int(lag), bool(add_intercept), bool(exclude_self))
        beta = out.beta
    elif te_name == "lasso":
        alpha = float(te_params["alpha"])
        max_iter = int(te_params.get("max_iter", 500))
        tol = float(te_params.get("tol", 1e-8))
        add_intercept = bool(te_params.get("add_intercept", True))
        standardize = bool(te_params.get("standardize", True))
        out = lasso_te_matrix(
            returns,
            int(lag),
            float(alpha),
            int(max_iter),
            float(tol),
            bool(add_intercept),
            bool(standardize),
            bool(exclude_self),
        )
        beta = out.beta
    else:
        raise ValueError(f"unsupported te_name: {te_name}")

    adj, info = select_fixed_density(
        scores=beta,
        density=float(sel_density),
        exclude_self=bool(sel_exclude_self),
        mode=str(sel_mode),
        return_info=True,
    )
    sig = compute_nio(adj, bool(nio_exclude_self), bool(nio_normalize))
    meta = {
        "density_target": float(info.get("density_target", float(sel_density))),
        "density_real": float(info.get("density_real", float("nan"))),
        "k_selected": int(info.get("k_selected", int(adj.sum()))),
    }
    return sig.nio.astype(np.float64, copy=False), meta


def run_qf_21_build_signal(
    *,
    estimated_dir: str,
    oracle_dir: str,
    output_root: str,
    src_dir: str,
    config_path: str,
    script_path: str,
    component: str,
    design_filename: str,
) -> Qf21BuildSignalOut:
    estimated_dir = _require_dir(estimated_dir)
    oracle_dir = _require_dir(oracle_dir)
    src_dir = _require_dir(src_dir)
    output_root_abs = os.path.abspath(output_root)

    cfg_path = _require_file(os.path.abspath(config_path))
    script_path_abs = _require_file(os.path.abspath(script_path))

    repo_root = _repo_root_from_path(Path(script_path_abs))
    env_path = _require_file(str(repo_root / "uv.lock"))

    cfg = read_yaml(cfg_path)
    if not isinstance(cfg, dict):
        raise ValueError("config must be a mapping")

    filt_cfg = cfg.get("filters", {})
    if filt_cfg is None:
        filt_cfg = {}
    if not isinstance(filt_cfg, dict):
        raise ValueError("config.filters must be a mapping")
    dgp_allow = filt_cfg.get("dgp_allow", ["garch_factor"])
    if not isinstance(dgp_allow, list) or not dgp_allow:
        raise ValueError("filters.dgp_allow must be a non-empty list")
    dgp_allow = [str(x) for x in dgp_allow]

    sig_cfg = cfg.get("signal", {})
    if sig_cfg is None:
        sig_cfg = {}
    if not isinstance(sig_cfg, dict):
        raise ValueError("config.signal must be a mapping")
    nio_exclude_self = bool(sig_cfg.get("exclude_self", True))
    nio_normalize = bool(sig_cfg.get("normalize_nio", True))

    out_cfg = cfg.get("output", {})
    if out_cfg is None:
        out_cfg = {}
    if not isinstance(out_cfg, dict):
        raise ValueError("config.output must be a mapping")
    save_trial_npy = bool(out_cfg.get("save_trial_npy", True))
    save_index_csv = bool(out_cfg.get("save_index_csv", True))

    params_est = _read_meta_params(estimated_dir)
    design_path = params_est.get("design_path", None)
    if design_path is None:
        design_path = os.path.join(estimated_dir, design_filename)
    design_in = _require_file(str(design_path))

    df = _df_from_csv(design_in)
    df = df.rename(columns={c: c.strip() for c in df.columns})
    need = [
        "design_id",
        "dgp",
        "lag",
        "exclude_self",
        "te_name",
        "sel_density",
        "sel_mode",
        "sel_exclude_self",
    ]
    _require_cols(df, need)

    df["design_id"] = _to_num(df["design_id"])
    df["dgp"] = df["dgp"].astype(str)
    df = df[df["dgp"].isin(dgp_allow)].copy()

    params: dict[str, Any] = {
        "estimated_dir": os.path.abspath(estimated_dir),
        "oracle_dir": os.path.abspath(oracle_dir),
        "output_root": output_root_abs,
        "config_path": cfg_path,
        "component": component,
        "design_path": os.path.abspath(design_in),
        "dgp_allow": dgp_allow,
        "nio_exclude_self": bool(nio_exclude_self),
        "nio_normalize": bool(nio_normalize),
        "save_trial_npy": bool(save_trial_npy),
        "save_index_csv": bool(save_index_csv),
    }

    meta = build_meta(
        params=params, env=env_path, script=script_path_abs, cfg=cfg_path, src=src_dir
    )

    run_dir = build_version_dir(output_root_abs, meta)
    audit = Audit.create(run_dir, meta)
    logger = Logger(sinks=[ConsoleSink(), audit])

    try:
        logger.info(jline("stage", component, "start", run_dir=run_dir))
        logger.info(jline("input", component, "estimated_dir", path=estimated_dir))
        logger.info(jline("input", component, "oracle_dir", path=oracle_dir))
        logger.info(jline("input", component, "design", path=design_in))
        logger.info(jline("input", component, "config", path=cfg_path))

        cfg_dir = os.path.join(run_dir, "cfg")
        os.makedirs(cfg_dir, exist_ok=True)
        cfg_copied = os.path.join(cfg_dir, os.path.basename(cfg_path))
        shutil.copy2(cfg_path, cfg_copied)
        logger.info(jline("output", component, "config_copied", path=cfg_copied))

        records = df.to_dict(orient="records")
        p = Progress(
            logger=logger, name="qf/21_build_signal", total=int(len(records)) * 2
        )
        p.start()

        rows: list[dict[str, Any]] = []
        n_est = 0
        n_ora = 0

        for r in records:
            did = _to_int(r["design_id"], "design_id")
            lag = _to_int(r["lag"], "lag")
            exclude_self = _to_bool(r["exclude_self"], "exclude_self")
            te_name = str(r["te_name"]).strip()

            te_params: dict[str, Any] = {}
            if te_name == "ols":
                te_params["add_intercept"] = _to_bool(
                    r.get("te_add_intercept", True), "te_add_intercept"
                )
            elif te_name == "lasso":
                te_params["alpha"] = _to_float(r.get("te_alpha"), "te_alpha")
                te_params["max_iter"] = _to_int(
                    r.get("te_max_iter", 500), "te_max_iter"
                )
                te_params["tol"] = float(_to_float(r.get("te_tol", 1e-8), "te_tol"))
                te_params["add_intercept"] = _to_bool(
                    r.get("te_add_intercept", True), "te_add_intercept"
                )
                te_params["standardize"] = _to_bool(
                    r.get("te_standardize", True), "te_standardize"
                )
            else:
                raise ValueError(f"unsupported te_name: {te_name}")

            sel_density = float(_to_float(r["sel_density"], "sel_density"))
            sel_mode = str(r.get("sel_mode", "abs")).strip()
            sel_exclude_self = _to_bool(
                r.get("sel_exclude_self", True), "sel_exclude_self"
            )

            est_ret = _require_file(
                os.path.join(_trial_dir(estimated_dir, int(did)), "returns_neutral.npy")
            )
            R_est = _read_returns(est_ret)
            nio_est, meta_est = _compute_nio_signal(
                R_est,
                lag=int(lag),
                exclude_self=bool(exclude_self),
                te_name=te_name,
                te_params=te_params,
                sel_density=float(sel_density),
                sel_mode=sel_mode,
                sel_exclude_self=bool(sel_exclude_self),
                nio_exclude_self=bool(nio_exclude_self),
                nio_normalize=bool(nio_normalize),
            )
            est_out = None
            if save_trial_npy:
                est_out = os.path.join(
                    _trial_dir(run_dir, int(did)), "estimated", "nio.npy"
                )
                _write_npy(est_out, nio_est)
            rows.append(
                {
                    "design_id": int(did),
                    "branch": "estimated",
                    "dgp": str(r.get("dgp", "")),
                    "te_name": te_name,
                    "sel_density": str(r.get("sel_density")),
                    "fn_n_components": str(r.get("fn_n_components", "")),
                    "N": str(r.get("N", "")),
                    "T": str(r.get("T", "")),
                    "returns_path": est_ret,
                    "signal_path": est_out if est_out is not None else "",
                    "density_real": meta_est["density_real"],
                    "k_selected": meta_est["k_selected"],
                }
            )
            n_est += 1
            p.step(1)

            ora_ret = os.path.join(
                _trial_dir(oracle_dir, int(did)), "returns_oracle_neutral.npy"
            )
            if os.path.isfile(ora_ret):
                R_ora = _read_returns(ora_ret)
                nio_ora, meta_ora = _compute_nio_signal(
                    R_ora,
                    lag=int(lag),
                    exclude_self=bool(exclude_self),
                    te_name=te_name,
                    te_params=te_params,
                    sel_density=float(sel_density),
                    sel_mode=sel_mode,
                    sel_exclude_self=bool(sel_exclude_self),
                    nio_exclude_self=bool(nio_exclude_self),
                    nio_normalize=bool(nio_normalize),
                )
                ora_out = None
                if save_trial_npy:
                    ora_out = os.path.join(
                        _trial_dir(run_dir, int(did)), "oracle", "nio.npy"
                    )
                    _write_npy(ora_out, nio_ora)
                rows.append(
                    {
                        "design_id": int(did),
                        "branch": "oracle",
                        "dgp": str(r.get("dgp", "")),
                        "te_name": te_name,
                        "sel_density": str(r.get("sel_density")),
                        "fn_n_components": str(r.get("fn_n_components", "")),
                        "N": str(r.get("N", "")),
                        "T": str(r.get("T", "")),
                        "returns_path": ora_ret,
                        "signal_path": ora_out if ora_out is not None else "",
                        "density_real": meta_ora["density_real"],
                        "k_selected": meta_ora["k_selected"],
                    }
                )
                n_ora += 1
            p.step(1)

        p.finish()

        index_path = None
        if save_index_csv:
            out_df = pd.DataFrame(rows)
            cols = [
                "design_id",
                "branch",
                "dgp",
                "te_name",
                "sel_density",
                "fn_n_components",
                "N",
                "T",
                "returns_path",
                "signal_path",
                "density_real",
                "k_selected",
            ]
            out_df = out_df[cols].sort_values(
                ["design_id", "branch"], ascending=[True, True]
            )
            index_path = os.path.join(run_dir, "signals_index.csv")
            write_csv(index_path, _rows_from_df(out_df), header=list(out_df.columns))
            logger.info(jline("output", component, "signals_index", path=index_path))

        summary = {
            "n_design_rows": int(len(records)),
            "n_estimated": int(n_est),
            "n_oracle": int(n_ora),
            "trial_root": os.path.join(run_dir, "trial"),
            "signals_index": index_path,
        }

        summary_out = os.path.join(run_dir, "summary.json")
        write_json(summary_out, summary)
        logger.info(jline("output", component, "summary", path=summary_out))

        audit.finish_success()
        return Qf21BuildSignalOut(
            run_dir=run_dir, meta=meta, index_path=index_path, summary_path=summary_out
        )
    except BaseException as e:
        audit.finish_error(e)
        raise
