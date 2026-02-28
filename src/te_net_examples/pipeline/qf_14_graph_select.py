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
from te_net_lib.graph.edge_select import select_fixed_density


@dataclass(frozen=True, slots=True)
class Qf14GraphSelectOut:
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


def _to_float(x: Any, name: str) -> float:
    if isinstance(x, (int, float, np.floating, np.integer)):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return float("nan")
    try:
        return float(s)
    except Exception as e:
        raise ValueError(f"invalid float for {name}: {x}") from e


def _trial_dir(run_dir: str, design_id: int) -> str:
    return os.path.join(run_dir, "trial", f"{int(design_id):08d}")


def _read_beta(path: str) -> np.ndarray:
    x = np.load(path, allow_pickle=False)
    if not isinstance(x, np.ndarray):
        raise ValueError(f"invalid beta array: {path}")
    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise ValueError(f"beta must be square 2D: {path}")
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


def run_qf_14_graph_select(
    *,
    input_dir: str,
    output_root: str,
    src_dir: str,
    config_path: str,
    script_path: str,
    component: str,
    design_filename: str,
) -> Qf14GraphSelectOut:
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

    output_cfg = cfg.get("output", {})
    if output_cfg is None:
        output_cfg = {}
    if not isinstance(output_cfg, dict):
        raise ValueError("config.output must be a mapping")

    save_adj = bool(output_cfg.get("save_adj", True))
    save_select_json = bool(output_cfg.get("save_select_json", True))

    params: dict[str, Any] = {
        "input_dir": os.path.abspath(input_dir),
        "output_root": output_root_abs,
        "config_path": cfg_path,
        "component": component,
        "design_path": os.path.abspath(design_in),
        "meta_in": os.path.abspath(meta_in),
        "save_adj": bool(save_adj),
        "save_select_json": bool(save_select_json),
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

        need = ["design_id", "sel_name", "sel_density", "sel_mode", "sel_exclude_self"]
        _require_cols(df, need)

        records = df.to_dict(orient="records")
        p = Progress(logger=logger, name="qf/14_graph_select", total=int(len(records)))
        p.start()

        items: list[dict[str, Any]] = []
        n_fixed_density = 0

        for r in records:
            design_id = _to_int(r["design_id"], "design_id")
            sel_name = str(r["sel_name"]).strip()
            density = _to_float(r["sel_density"], "sel_density")
            mode = str(r.get("sel_mode", "abs")).strip()
            exclude_self = _to_bool(r.get("sel_exclude_self", True), "sel_exclude_self")

            in_dir = _trial_dir(input_dir, design_id)
            beta_in = _require_file(os.path.join(in_dir, "beta.npy"))
            beta = _read_beta(beta_in)

            out_dir = _trial_dir(run_dir, design_id)
            _ensure_dir(out_dir)

            adj_path = None
            sel_json_path = None

            if sel_name == "fixed_density":
                out = select_fixed_density(
                    scores=beta,
                    density=float(density),
                    exclude_self=bool(exclude_self),
                    mode=str(mode),
                    return_info=True,
                )
                adj = out[0]
                info = out[1]
                n_fixed_density += 1
            else:
                raise ValueError(f"unsupported sel_name: {sel_name}")

            if save_adj:
                adj_path = os.path.join(out_dir, "adj_hat.npy")
                _write_npy(adj_path, adj.astype(np.int8, copy=False))

            if save_select_json:
                sel = {
                    "design_id": int(design_id),
                    "sel_name": sel_name,
                    "density_target": float(info.get("density_target", float(density))),
                    "density_real": float(info.get("density_real", float("nan"))),
                    "k_target": int(info.get("k_target", -1)),
                    "k_selected": int(info.get("k_selected", int(adj.sum()))),
                    "mode": str(info.get("mode", mode)),
                    "exclude_self": bool(info.get("exclude_self", exclude_self)),
                    "input_beta": beta_in,
                    "paths": {"adj_hat": adj_path},
                }
                sel_json_path = os.path.join(out_dir, "select.json")
                write_json(sel_json_path, sel)

            items.append(
                {
                    "design_id": int(design_id),
                    "sel_name": sel_name,
                    "density": float(density),
                    "mode": mode,
                    "exclude_self": bool(exclude_self),
                    "beta": beta_in,
                    "adj_hat": adj_path,
                    "select_json": sel_json_path,
                }
            )

            p.step(1)

        p.finish()

        summ = {
            "n_rows": int(len(records)),
            "n_fixed_density": int(n_fixed_density),
            "by_sel": _vc(df, "sel_name"),
            "by_density": _vc(df, "sel_density"),
            "trial_root": os.path.join(run_dir, "trial"),
            "items": items[:2000],
        }

        summary_out = os.path.join(run_dir, "summary.json")
        write_json(summary_out, summ)
        logger.info(_jline("output", component, "summary", path=summary_out))

        audit.finish_success()
        return Qf14GraphSelectOut(run_dir=run_dir, meta=meta, summary_path=summary_out)
    except BaseException as e:
        audit.finish_error(e)
        raise
