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
from te_net_lib.eval.nio import compute_nio, hub_recovery_from_signal
from te_net_lib.graph.edge_select import select_fixed_density
from te_net_lib.graph.metrics import (
    confusion_counts,
    graph_density,
    hub_indices,
    precision_recall_f1,
)
from te_net_lib.te.lasso_te import lasso_te_matrix
from te_net_lib.te.linear_ols import ols_te_matrix


@dataclass(frozen=True, slots=True)
class Qf19OracleVsEstimatedOut:
    run_dir: str
    meta: dict[str, Any]
    compare_csv_path: str | None
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
    try:
        return float(s)
    except Exception as e:
        raise ValueError(f"invalid float for {name}: {x}") from e


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


def _find_true_root_from_oracle(
    oracle_dir: str, design_id: int, depth: int
) -> str | None:
    cur = oracle_dir
    for _ in range(depth):
        p = os.path.join(cur, "trial", f"{design_id:08d}", "true_adj.npy")
        if os.path.isfile(p):
            return cur
        params = _read_meta_params(cur)
        prev = params.get("input_dir", None)
        if prev is None:
            return None
        cur = str(prev)
        if not os.path.isdir(cur):
            return None
    return None


def _read_returns(path: str) -> np.ndarray:
    x = np.load(path, allow_pickle=False)
    if not isinstance(x, np.ndarray):
        raise ValueError(f"invalid returns: {path}")
    if x.ndim != 2:
        raise ValueError(f"returns must be 2D: {path}")
    return x.astype(np.float64, copy=False)


def _read_adj(path: str) -> np.ndarray:
    x = np.load(path, allow_pickle=False)
    if not isinstance(x, np.ndarray):
        raise ValueError(f"invalid adj: {path}")
    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise ValueError(f"adj must be square: {path}")
    return x.astype(np.int8, copy=False)


def _eval_one(
    returns: np.ndarray,
    true_adj: np.ndarray | None,
    *,
    lag: int,
    exclude_self: bool,
    te_name: str,
    te_params: dict[str, Any],
    sel_density: float,
    sel_mode: str,
    sel_exclude_self: bool,
    hub_k: int,
    signal_exclude_self: bool,
    signal_normalize: bool,
) -> dict[str, Any]:
    if te_name == "ols":
        add_intercept = bool(te_params.get("add_intercept", True))
        te = ols_te_matrix(returns, int(lag), bool(add_intercept), bool(exclude_self))
        beta = te.beta
    elif te_name == "lasso":
        alpha = float(te_params["alpha"])
        max_iter = int(te_params.get("max_iter", 500))
        tol = float(te_params.get("tol", 1e-8))
        add_intercept = bool(te_params.get("add_intercept", True))
        standardize = bool(te_params.get("standardize", True))
        te = lasso_te_matrix(
            returns,
            int(lag),
            float(alpha),
            int(max_iter),
            float(tol),
            bool(add_intercept),
            bool(standardize),
            bool(exclude_self),
        )
        beta = te.beta
    else:
        raise ValueError(f"unsupported te_name: {te_name}")

    adj, info = select_fixed_density(
        scores=beta,
        density=float(sel_density),
        exclude_self=bool(sel_exclude_self),
        mode=str(sel_mode),
        return_info=True,
    )

    sig = compute_nio(adj, bool(signal_exclude_self), bool(signal_normalize))
    dens_pred = float(graph_density(adj, bool(exclude_self)))

    out: dict[str, Any] = {
        "density_pred": dens_pred,
        "select": {
            "density_target": float(info.get("density_target", float(sel_density))),
            "density_real": float(info.get("density_real", float("nan"))),
            "k_target": int(info.get("k_target", -1)),
            "k_selected": int(info.get("k_selected", int(adj.sum()))),
        },
        "hub": {"k": int(hub_k), "hub_rec_nio": None},
        "edge": {
            "tp": None,
            "fp": None,
            "fn": None,
            "tn": None,
            "precision": None,
            "recall": None,
            "f1": None,
        },
    }

    if true_adj is not None:
        k0 = int(max(min(hub_k, true_adj.shape[0]), 0))
        idx = hub_indices(true_adj, int(k0), bool(exclude_self))
        true_hubs = np.zeros((true_adj.shape[0],), dtype=np.int8)
        if idx.size:
            true_hubs[idx] = 1
        out["hub"]["hub_rec_nio"] = float(
            hub_recovery_from_signal(sig.nio, true_hubs, int(k0))
        )

        tp, fp, fn, tn = confusion_counts(true_adj, adj, bool(exclude_self))
        prec, rec, f1 = precision_recall_f1(true_adj, adj, bool(exclude_self))
        out["edge"]["tp"] = int(tp)
        out["edge"]["fp"] = int(fp)
        out["edge"]["fn"] = int(fn)
        out["edge"]["tn"] = int(tn)
        out["edge"]["precision"] = float(prec)
        out["edge"]["recall"] = float(rec)
        out["edge"]["f1"] = float(f1)

    return out


def run_qf_19_oracle_vs_estimated(
    *,
    estimated_dir: str,
    oracle_dir: str,
    output_root: str,
    src_dir: str,
    config_path: str,
    script_path: str,
    component: str,
    design_filename: str,
) -> Qf19OracleVsEstimatedOut:
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

    input_cfg = cfg.get("input", {})
    if input_cfg is None:
        input_cfg = {}
    if not isinstance(input_cfg, dict):
        raise ValueError("config.input must be a mapping")

    oracle_optional = bool(input_cfg.get("oracle_optional", True))

    met_cfg = cfg.get("metrics", {})
    if met_cfg is None:
        met_cfg = {}
    if not isinstance(met_cfg, dict):
        raise ValueError("config.metrics must be a mapping")
    sig_cfg = met_cfg.get("signal", {})
    if sig_cfg is None:
        sig_cfg = {}
    if not isinstance(sig_cfg, dict):
        raise ValueError("metrics.signal must be a mapping")
    signal_normalize = bool(sig_cfg.get("normalize_nio", True))
    signal_exclude_self = bool(sig_cfg.get("exclude_self", True))

    out_cfg = cfg.get("output", {})
    if out_cfg is None:
        out_cfg = {}
    if not isinstance(out_cfg, dict):
        raise ValueError("config.output must be a mapping")
    save_compare_csv = bool(out_cfg.get("save_compare_csv", True))
    save_trial_json = bool(out_cfg.get("save_trial_json", True))

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
        "hub_k",
    ]
    _require_cols(df, need)

    params: dict[str, Any] = {
        "estimated_dir": os.path.abspath(estimated_dir),
        "oracle_dir": os.path.abspath(oracle_dir),
        "output_root": output_root_abs,
        "config_path": cfg_path,
        "component": component,
        "design_path": os.path.abspath(design_in),
        "oracle_optional": bool(oracle_optional),
        "signal_normalize_nio": bool(signal_normalize),
        "signal_exclude_self": bool(signal_exclude_self),
        "save_compare_csv": bool(save_compare_csv),
        "save_trial_json": bool(save_trial_json),
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
            logger=logger, name="qf/19_oracle_vs_estimated", total=int(len(records))
        )
        p.start()

        rows: list[dict[str, Any]] = []
        items: list[dict[str, Any]] = []

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
            hub_k = _to_int(r["hub_k"], "hub_k")

            true_root = _find_true_root_from_oracle(oracle_dir, int(did), 12)
            true_adj = None
            true_adj_path = None
            if true_root is not None:
                true_adj_path = os.path.join(
                    true_root, "trial", f"{did:08d}", "true_adj.npy"
                )
                if os.path.isfile(true_adj_path):
                    true_adj = _read_adj(true_adj_path)

            est_path = _require_file(
                os.path.join(_trial_dir(estimated_dir, int(did)), "returns_neutral.npy")
            )
            R_est = _read_returns(est_path)

            ora_path = os.path.join(
                _trial_dir(oracle_dir, int(did)), "returns_oracle_neutral.npy"
            )
            R_ora = _read_returns(ora_path) if os.path.isfile(ora_path) else None

            out_est = _eval_one(
                R_est,
                true_adj,
                lag=int(lag),
                exclude_self=bool(exclude_self),
                te_name=te_name,
                te_params=te_params,
                sel_density=float(sel_density),
                sel_mode=sel_mode,
                sel_exclude_self=bool(sel_exclude_self),
                hub_k=int(hub_k),
                signal_exclude_self=bool(signal_exclude_self),
                signal_normalize=bool(signal_normalize),
            )

            out_ora = None
            oracle_status = "missing"
            if R_ora is not None:
                out_ora = _eval_one(
                    R_ora,
                    true_adj,
                    lag=int(lag),
                    exclude_self=bool(exclude_self),
                    te_name=te_name,
                    te_params=te_params,
                    sel_density=float(sel_density),
                    sel_mode=sel_mode,
                    sel_exclude_self=bool(sel_exclude_self),
                    hub_k=int(hub_k),
                    signal_exclude_self=bool(signal_exclude_self),
                    signal_normalize=bool(signal_normalize),
                )
                oracle_status = "ok"
            else:
                if not bool(oracle_optional):
                    raise FileNotFoundError(ora_path)

            out_trial = _trial_dir(run_dir, int(did))
            _ensure_dir(out_trial)

            trial_json = None
            if save_trial_json:
                obj = {
                    "design_id": int(did),
                    "inputs": {
                        "returns_estimated": est_path,
                        "returns_oracle": ora_path
                        if os.path.isfile(ora_path)
                        else None,
                        "true_adj": true_adj_path,
                    },
                    "estimated": out_est,
                    "oracle": out_ora,
                    "oracle_status": oracle_status,
                }
                trial_json = os.path.join(out_trial, "trial.json")
                write_json(trial_json, obj)

            row_out = {
                "design_id": int(did),
                "dgp": str(r.get("dgp", "")),
                "te_name": te_name,
                "sel_density": str(r.get("sel_density")),
                "fn_n_components": str(r.get("fn_n_components", "")),
                "N": str(r.get("N", "")),
                "T": str(r.get("T", "")),
                "oracle_status": oracle_status,
                "est_precision": out_est["edge"]["precision"]
                if out_est["edge"]["precision"] is not None
                else "",
                "est_recall": out_est["edge"]["recall"]
                if out_est["edge"]["recall"] is not None
                else "",
                "est_f1": out_est["edge"]["f1"]
                if out_est["edge"]["f1"] is not None
                else "",
                "est_hub_rec": out_est["hub"]["hub_rec_nio"]
                if out_est["hub"]["hub_rec_nio"] is not None
                else "",
                "ora_precision": out_ora["edge"]["precision"]
                if (out_ora is not None and out_ora["edge"]["precision"] is not None)
                else "",
                "ora_recall": out_ora["edge"]["recall"]
                if (out_ora is not None and out_ora["edge"]["recall"] is not None)
                else "",
                "ora_f1": out_ora["edge"]["f1"]
                if (out_ora is not None and out_ora["edge"]["f1"] is not None)
                else "",
                "ora_hub_rec": out_ora["hub"]["hub_rec_nio"]
                if (out_ora is not None and out_ora["hub"]["hub_rec_nio"] is not None)
                else "",
                "trial_json": trial_json if trial_json is not None else "",
            }
            rows.append(row_out)
            items.append(
                {
                    "design_id": int(did),
                    "oracle_status": oracle_status,
                    "trial_json": trial_json,
                }
            )

            p.step(1)

        p.finish()

        compare_csv_path = None
        if save_compare_csv:
            out_df = pd.DataFrame(rows)
            cols_out = [
                "design_id",
                "dgp",
                "te_name",
                "sel_density",
                "fn_n_components",
                "N",
                "T",
                "oracle_status",
                "est_precision",
                "est_recall",
                "est_f1",
                "est_hub_rec",
                "ora_precision",
                "ora_recall",
                "ora_f1",
                "ora_hub_rec",
                "trial_json",
            ]
            out_df = out_df[cols_out].sort_values(["design_id"], ascending=[True])
            compare_csv_path = os.path.join(run_dir, "compare.csv")
            write_csv(
                compare_csv_path, _rows_from_df(out_df), header=list(out_df.columns)
            )
            logger.info(
                jline("output", component, "compare_csv", path=compare_csv_path)
            )

        summary = {
            "n_rows": int(len(records)),
            "trial_root": os.path.join(run_dir, "trial"),
            "compare_csv": compare_csv_path,
            "items": items[:2000],
        }

        summary_out = os.path.join(run_dir, "summary.json")
        write_json(summary_out, summary)
        logger.info(jline("output", component, "summary", path=summary_out))

        audit.finish_success()
        return Qf19OracleVsEstimatedOut(
            run_dir=run_dir,
            meta=meta,
            compare_csv_path=compare_csv_path,
            summary_path=summary_out,
        )
    except BaseException as e:
        audit.finish_error(e)
        raise
