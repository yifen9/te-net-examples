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
from te_net_examples.utils.logger import Logger
from te_net_examples.utils.meta import build_meta
from te_net_examples.utils.progress import Progress
from te_net_examples.utils.versioner import build_version_dir
from te_net_lib.eval.nio import compute_nio, hub_recovery_from_signal
from te_net_lib.graph.metrics import (
    confusion_counts,
    graph_density,
    hub_indices,
    precision_recall_f1,
)


@dataclass(frozen=True, slots=True)
class Qf15MetricsOut:
    run_dir: str
    meta: dict[str, Any]
    metrics_csv_path: str | None
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
        try:
            f = float(s)
            if f.is_integer():
                return int(f)
        except Exception:
            pass
    raise ValueError(f"invalid int for {name}: {x}")


def _trial_dir(run_dir: str, design_id: int) -> str:
    return os.path.join(run_dir, "trial", f"{int(design_id):08d}")


def _read_adj(path: str) -> np.ndarray:
    x = np.load(path, allow_pickle=False)
    if not isinstance(x, np.ndarray):
        raise ValueError(f"invalid adjacency array: {path}")
    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise ValueError(f"adj must be square 2D: {path}")
    return x.astype(np.int8, copy=False)


def _read_meta_params(run_dir: str) -> dict[str, Any]:
    meta_path = os.path.join(run_dir, "_meta.json")
    obj = read_json(_require_file(meta_path))
    if not isinstance(obj, dict):
        raise ValueError(f"_meta.json must be a mapping: {meta_path}")
    params = obj.get("params", None)
    if not isinstance(params, dict):
        raise ValueError(f"_meta.json missing params mapping: {meta_path}")
    return params


def _find_true_adj_root(
    start_dir: str, design_ids: list[int], depth: int
) -> str | None:
    cur = start_dir
    for _ in range(depth):
        ok = True
        for did in design_ids[:5]:
            p = os.path.join(cur, "trial", f"{did:08d}", "true_adj.npy")
            if not os.path.isfile(p):
                ok = False
                break
        if ok:
            return cur
        params = _read_meta_params(cur)
        prev = params.get("input_dir", None)
        if prev is None:
            return None
        cur = str(prev)
        if not os.path.isdir(cur):
            return None
    return None


def _vc(df: pd.DataFrame, col: str) -> dict[str, int]:
    if col not in df.columns:
        return {}
    s = df[col].astype(str)
    m = s.value_counts(dropna=False).to_dict()
    return {str(k): int(v) for k, v in m.items()}


def run_qf_15_metrics(
    *,
    input_dir: str,
    output_root: str,
    src_dir: str,
    config_path: str,
    script_path: str,
    component: str,
    design_filename: str,
) -> Qf15MetricsOut:
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

    true_adj_required = bool(input_cfg.get("true_adj_required", True))

    output_cfg = cfg.get("output", {})
    if output_cfg is None:
        output_cfg = {}
    if not isinstance(output_cfg, dict):
        raise ValueError("config.output must be a mapping")

    save_metrics_json = bool(output_cfg.get("save_metrics_json", True))
    save_metrics_csv = bool(output_cfg.get("save_metrics_csv", True))
    save_signal_json = bool(output_cfg.get("save_signal_json", True))

    sig_cfg = cfg.get("signal", {})
    if sig_cfg is None:
        sig_cfg = {}
    if not isinstance(sig_cfg, dict):
        raise ValueError("config.signal must be a mapping")

    sig_normalize = bool(sig_cfg.get("normalize_nio", True))
    sig_exclude_self = bool(sig_cfg.get("exclude_self", True))

    params: dict[str, Any] = {
        "input_dir": os.path.abspath(input_dir),
        "output_root": output_root_abs,
        "config_path": cfg_path,
        "component": component,
        "design_path": os.path.abspath(design_in),
        "meta_in": os.path.abspath(meta_in),
        "true_adj_required": bool(true_adj_required),
        "save_metrics_json": bool(save_metrics_json),
        "save_metrics_csv": bool(save_metrics_csv),
        "save_signal_json": bool(save_signal_json),
        "signal_normalize_nio": bool(sig_normalize),
        "signal_exclude_self": bool(sig_exclude_self),
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

        need = ["design_id", "exclude_self", "hub_k"]
        _require_cols(df, need)

        design_ids = [_to_int(x, "design_id") for x in df["design_id"].tolist()]
        true_root = _find_true_adj_root(input_dir, design_ids, 12)
        if true_adj_required and true_root is None:
            raise FileNotFoundError("true_adj root not found via upstream chain")
        if true_root is not None:
            logger.info(_jline("input", component, "true_adj_root", path=true_root))

        records = df.to_dict(orient="records")
        p = Progress(logger=logger, name="qf/15_metrics", total=int(len(records)))
        p.start()

        rows: list[dict[str, Any]] = []
        items: list[dict[str, Any]] = []

        for r in records:
            design_id = _to_int(r["design_id"], "design_id")
            exclude_self = _to_bool(r["exclude_self"], "exclude_self")
            hub_k = _to_int(r["hub_k"], "hub_k")

            in_dir = _trial_dir(input_dir, design_id)
            adj_hat_in = _require_file(os.path.join(in_dir, "adj_hat.npy"))
            adj_hat = _read_adj(adj_hat_in)

            true_adj_in = None
            true_adj = None
            if true_root is not None:
                true_adj_in = _require_file(
                    os.path.join(true_root, "trial", f"{design_id:08d}", "true_adj.npy")
                )
                true_adj = _read_adj(true_adj_in)

            dens_hat = float(graph_density(adj_hat, bool(exclude_self)))
            sig_hat = compute_nio(adj_hat, bool(sig_exclude_self), bool(sig_normalize))

            hub_rec = None
            if true_adj is not None:
                k0 = int(max(min(hub_k, true_adj.shape[0]), 0))
                idx = hub_indices(true_adj, int(k0), bool(exclude_self))
                true_hubs = np.zeros((true_adj.shape[0],), dtype=np.int8)
                if idx.size:
                    true_hubs[idx] = 1
                hub_rec = float(
                    hub_recovery_from_signal(sig_hat.nio, true_hubs, int(k0))
                )

            prec = rec = f1 = None
            tp = fp = fn = tn = None
            dens_true = None
            if true_adj is not None:
                tp, fp, fn, tn = confusion_counts(true_adj, adj_hat, bool(exclude_self))
                prec, rec, f1 = precision_recall_f1(
                    true_adj, adj_hat, bool(exclude_self)
                )
                dens_true = float(graph_density(true_adj, bool(exclude_self)))

            out_dir = _trial_dir(run_dir, design_id)
            _ensure_dir(out_dir)

            metrics_path = None
            signal_path = None

            if save_metrics_json:
                metrics = {
                    "design_id": int(design_id),
                    "exclude_self": bool(exclude_self),
                    "hub_k": int(hub_k),
                    "paths": {
                        "adj_hat": adj_hat_in,
                        "true_adj": true_adj_in,
                    },
                    "density": {
                        "pred": float(dens_hat),
                        "true": float(dens_true) if dens_true is not None else None,
                    },
                    "edge": {
                        "tp": int(tp) if tp is not None else None,
                        "fp": int(fp) if fp is not None else None,
                        "fn": int(fn) if fn is not None else None,
                        "tn": int(tn) if tn is not None else None,
                        "precision": float(prec) if prec is not None else None,
                        "recall": float(rec) if rec is not None else None,
                        "f1": float(f1) if f1 is not None else None,
                    },
                    "hub_recovery": {
                        "k": int(hub_k),
                        "nio_based": float(hub_rec) if hub_rec is not None else None,
                    },
                }
                metrics_path = os.path.join(out_dir, "metrics.json")
                write_json(metrics_path, metrics)

            if save_signal_json:
                signal = {
                    "design_id": int(design_id),
                    "signal": {
                        "normalize": bool(sig_normalize),
                        "exclude_self": bool(sig_exclude_self),
                    },
                    "nio": sig_hat.nio.astype(float).tolist(),
                    "out_strength": sig_hat.out_strength.astype(float).tolist(),
                    "in_strength": sig_hat.in_strength.astype(float).tolist(),
                }
                signal_path = os.path.join(out_dir, "signal.json")
                write_json(signal_path, signal)

            row_out = {
                "design_id": int(design_id),
                "exclude_self": bool(exclude_self),
                "hub_k": int(hub_k),
                "density_pred": float(dens_hat),
                "density_true": float(dens_true) if dens_true is not None else "",
                "tp": int(tp) if tp is not None else "",
                "fp": int(fp) if fp is not None else "",
                "fn": int(fn) if fn is not None else "",
                "tn": int(tn) if tn is not None else "",
                "precision": float(prec) if prec is not None else "",
                "recall": float(rec) if rec is not None else "",
                "f1": float(f1) if f1 is not None else "",
                "hub_rec_nio": float(hub_rec) if hub_rec is not None else "",
                "adj_hat": adj_hat_in,
                "true_adj": true_adj_in if true_adj_in is not None else "",
                "metrics_json": metrics_path if metrics_path is not None else "",
                "signal_json": signal_path if signal_path is not None else "",
            }
            rows.append(row_out)

            items.append(
                {
                    "design_id": int(design_id),
                    "metrics_json": metrics_path,
                    "signal_json": signal_path,
                }
            )

            p.step(1)

        p.finish()

        metrics_csv_path = None
        if save_metrics_csv:
            out_df = pd.DataFrame(rows)
            cols = [
                "design_id",
                "exclude_self",
                "hub_k",
                "density_pred",
                "density_true",
                "tp",
                "fp",
                "fn",
                "tn",
                "precision",
                "recall",
                "f1",
                "hub_rec_nio",
                "adj_hat",
                "true_adj",
                "metrics_json",
                "signal_json",
            ]
            out_df = out_df[cols].sort_values(["design_id"], ascending=[True])
            metrics_csv_path = os.path.join(run_dir, "metrics.csv")
            write_csv(
                metrics_csv_path, _rows_from_df(out_df), header=list(out_df.columns)
            )
            logger.info(
                _jline("output", component, "metrics_csv", path=metrics_csv_path)
            )

        summ = {
            "n_rows": int(len(records)),
            "by_dgp": _vc(df, "dgp"),
            "by_te": _vc(df, "te_name"),
            "by_density": _vc(df, "sel_density"),
            "true_adj_root": str(true_root) if true_root is not None else None,
            "trial_root": os.path.join(run_dir, "trial"),
            "metrics_csv": metrics_csv_path,
            "items": items[:2000],
        }

        summary_out = os.path.join(run_dir, "summary.json")
        write_json(summary_out, summ)
        logger.info(_jline("output", component, "summary", path=summary_out))

        audit.finish_success()
        return Qf15MetricsOut(
            run_dir=run_dir,
            meta=meta,
            metrics_csv_path=metrics_csv_path,
            summary_path=summary_out,
        )
    except BaseException as e:
        audit.finish_error(e)
        raise
