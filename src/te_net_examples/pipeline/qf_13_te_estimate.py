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
from te_net_examples.utils.logger import Logger
from te_net_examples.utils.meta import build_meta
from te_net_examples.utils.progress import Progress
from te_net_examples.utils.versioner import build_version_dir
from te_net_lib.te.lasso_te import lasso_te_matrix
from te_net_lib.te.linear_ols import ols_te_matrix


@dataclass(frozen=True, slots=True)
class Qf13TeEstimateOut:
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


def _require_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")


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
        try:
            f = float(s)
            if f.is_integer():
                return int(f)
        except Exception:
            pass
    raise ValueError(f"invalid int for {name}: {x}")


def _to_float(x: Any, name: str) -> float:
    try:
        return float(x)
    except Exception as e:
        raise ValueError(f"invalid float for {name}: {x}") from e


def _to_bool(x: Any, name: str) -> bool:
    if isinstance(x, bool):
        return bool(x)
    s = str(x).strip().lower()
    if s in ("1", "true", "t", "yes", "y"):
        return True
    if s in ("0", "false", "f", "no", "n"):
        return False
    raise ValueError(f"invalid bool for {name}: {x}")


def _trial_dir(run_dir: str, design_id: int) -> str:
    return os.path.join(run_dir, "trial", f"{int(design_id):08d}")


def _read_returns(path: str) -> np.ndarray:
    x = np.load(path, allow_pickle=False)
    if not isinstance(x, np.ndarray):
        raise ValueError(f"invalid returns array: {path}")
    if x.ndim != 2:
        raise ValueError(f"returns must be 2D: {path}")
    return x.astype(np.float64, copy=False)


def _write_npy(path: str, arr: np.ndarray) -> None:
    _ensure_dir(os.path.dirname(path))
    np.save(path, arr)


def _vc(df: pd.DataFrame, col: str) -> dict[str, int]:
    if col not in df.columns:
        return {}
    s = df[col].astype(str)
    m = s.value_counts(dropna=False).to_dict()
    return {str(k): int(v) for k, v in m.items()}


def _qstats(x: np.ndarray) -> dict[str, float]:
    v = x.astype(np.float64, copy=False)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {
            "min": float("nan"),
            "p25": float("nan"),
            "median": float("nan"),
            "p75": float("nan"),
            "max": float("nan"),
        }
    return {
        "min": float(np.min(v)),
        "p25": float(np.quantile(v, 0.25)),
        "median": float(np.quantile(v, 0.50)),
        "p75": float(np.quantile(v, 0.75)),
        "max": float(np.max(v)),
    }


def run_qf_13_te_estimate(
    *,
    input_dir: str,
    output_root: str,
    src_dir: str,
    config_path: str,
    script_path: str,
    component: str,
    design_filename: str,
) -> Qf13TeEstimateOut:
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

    use_neutral = bool(input_cfg.get("use_neutral", True))

    output_cfg = cfg.get("output", {})
    if output_cfg is None:
        output_cfg = {}
    if not isinstance(output_cfg, dict):
        raise ValueError("config.output must be a mapping")

    save_beta = bool(output_cfg.get("save_beta", True))
    save_intercept = bool(output_cfg.get("save_intercept", True))
    save_resid_var = bool(output_cfg.get("save_resid_var", True))
    save_n_iter = bool(output_cfg.get("save_n_iter", True))
    save_te_json = bool(output_cfg.get("save_te_json", True))

    params: dict[str, Any] = {
        "input_dir": os.path.abspath(input_dir),
        "output_root": output_root_abs,
        "config_path": cfg_path,
        "component": component,
        "design_path": os.path.abspath(design_in),
        "meta_in": os.path.abspath(meta_in),
        "use_neutral": bool(use_neutral),
        "save_beta": bool(save_beta),
        "save_intercept": bool(save_intercept),
        "save_resid_var": bool(save_resid_var),
        "save_n_iter": bool(save_n_iter),
        "save_te_json": bool(save_te_json),
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
        logger.info(_jline("input", component, "config", path=cfg_path))

        cfg_dir = os.path.join(run_dir, "cfg")
        os.makedirs(cfg_dir, exist_ok=True)
        cfg_copied = os.path.join(cfg_dir, os.path.basename(cfg_path))
        shutil.copy2(cfg_path, cfg_copied)
        logger.info(_jline("output", component, "config_copied", path=cfg_copied))

        df = _df_from_csv(design_in)
        df = df.rename(columns={c: c.strip() for c in df.columns})

        need = ["design_id", "lag", "exclude_self", "te_name"]
        _require_cols(df, need)

        records = df.to_dict(orient="records")
        p = Progress(logger=logger, name="qf/13_te_estimate", total=int(len(records)))
        p.start()

        items: list[dict[str, Any]] = []
        n_ols = 0
        n_lasso = 0

        for r in records:
            design_id = _to_int(r["design_id"], "design_id")
            lag = _to_int(r["lag"], "lag")
            exclude_self = _to_bool(r["exclude_self"], "exclude_self")
            te_name = str(r["te_name"]).strip()

            in_dir = _trial_dir(input_dir, design_id)
            ret_name = "returns_neutral.npy" if use_neutral else "returns.npy"
            ret_in = _require_file(os.path.join(in_dir, ret_name))
            R = _read_returns(ret_in)

            out_dir = _trial_dir(run_dir, design_id)
            _ensure_dir(out_dir)

            beta_path = None
            intercept_path = None
            resid_var_path = None
            n_iter_path = None
            te_json_path = None

            if te_name == "ols":
                add_intercept = _to_bool(
                    r.get("te_add_intercept", True), "te_add_intercept"
                )
                out = ols_te_matrix(
                    R, int(lag), bool(add_intercept), bool(exclude_self)
                )
                if save_beta:
                    beta_path = os.path.join(out_dir, "beta.npy")
                    _write_npy(beta_path, out.beta.astype(np.float64, copy=False))
                if save_intercept and out.intercept is not None:
                    intercept_path = os.path.join(out_dir, "intercept.npy")
                    _write_npy(
                        intercept_path, out.intercept.astype(np.float64, copy=False)
                    )
                if save_resid_var:
                    resid_var_path = os.path.join(out_dir, "resid_var.npy")
                    _write_npy(
                        resid_var_path, out.resid_var.astype(np.float64, copy=False)
                    )
                if save_te_json:
                    te = {
                        "design_id": int(design_id),
                        "te_name": "ols",
                        "lag": int(lag),
                        "exclude_self": bool(exclude_self),
                        "add_intercept": bool(add_intercept),
                        "input_returns": ret_in,
                        "beta_shape": [int(out.beta.shape[0]), int(out.beta.shape[1])],
                        "has_intercept": bool(out.intercept is not None),
                        "resid_var_q": _qstats(out.resid_var),
                        "dof": int(out.dof),
                        "paths": {
                            "beta": beta_path,
                            "intercept": intercept_path,
                            "resid_var": resid_var_path,
                            "n_iter": None,
                        },
                    }
                    te_json_path = os.path.join(out_dir, "te.json")
                    write_json(te_json_path, te)
                n_ols += 1
            elif te_name == "lasso":
                alpha = _to_float(r.get("te_alpha"), "te_alpha")
                max_iter = _to_int(r.get("te_max_iter", 500), "te_max_iter")
                tol = _to_float(r.get("te_tol", 1e-8), "te_tol")
                add_intercept = _to_bool(
                    r.get("te_add_intercept", True), "te_add_intercept"
                )
                standardize = _to_bool(r.get("te_standardize", True), "te_standardize")

                out = lasso_te_matrix(
                    R,
                    int(lag),
                    float(alpha),
                    int(max_iter),
                    float(tol),
                    bool(add_intercept),
                    bool(standardize),
                    bool(exclude_self),
                )

                if save_beta:
                    beta_path = os.path.join(out_dir, "beta.npy")
                    _write_npy(beta_path, out.beta.astype(np.float64, copy=False))
                if save_intercept and out.intercept is not None:
                    intercept_path = os.path.join(out_dir, "intercept.npy")
                    _write_npy(
                        intercept_path, out.intercept.astype(np.float64, copy=False)
                    )
                if save_n_iter:
                    n_iter_path = os.path.join(out_dir, "n_iter.npy")
                    _write_npy(n_iter_path, out.n_iter.astype(np.int64, copy=False))
                if save_te_json:
                    te = {
                        "design_id": int(design_id),
                        "te_name": "lasso",
                        "lag": int(lag),
                        "exclude_self": bool(exclude_self),
                        "alpha": float(alpha),
                        "max_iter": int(max_iter),
                        "tol": float(tol),
                        "add_intercept": bool(add_intercept),
                        "standardize": bool(standardize),
                        "input_returns": ret_in,
                        "beta_shape": [int(out.beta.shape[0]), int(out.beta.shape[1])],
                        "has_intercept": bool(out.intercept is not None),
                        "n_iter_q": _qstats(out.n_iter.astype(np.float64, copy=False)),
                        "paths": {
                            "beta": beta_path,
                            "intercept": intercept_path,
                            "resid_var": None,
                            "n_iter": n_iter_path,
                        },
                    }
                    te_json_path = os.path.join(out_dir, "te.json")
                    write_json(te_json_path, te)
                n_lasso += 1
            else:
                raise ValueError(f"unsupported te_name: {te_name}")

            items.append(
                {
                    "design_id": int(design_id),
                    "te_name": te_name,
                    "input_returns": ret_in,
                    "beta": beta_path,
                    "intercept": intercept_path,
                    "resid_var": resid_var_path,
                    "n_iter": n_iter_path,
                    "te_json": te_json_path,
                }
            )

            p.step(1)

        p.finish()

        summ = {
            "n_rows": int(len(records)),
            "n_ols": int(n_ols),
            "n_lasso": int(n_lasso),
            "by_te": _vc(df, "te_name"),
            "use_neutral": bool(use_neutral),
            "trial_root": os.path.join(run_dir, "trial"),
            "items": items[:2000],
        }

        summary_out = os.path.join(run_dir, "summary.json")
        write_json(summary_out, summ)
        logger.info(_jline("output", component, "summary", path=summary_out))

        audit.finish_success()
        return Qf13TeEstimateOut(run_dir=run_dir, meta=meta, summary_path=summary_out)
    except BaseException as e:
        audit.finish_error(e)
        raise
