from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from te_net_examples.io.csv import write_csv
from te_net_examples.io.json import write_json
from te_net_examples.io.yaml import read_yaml
from te_net_examples.utils.audit import Audit
from te_net_examples.utils.console import ConsoleSink
from te_net_examples.utils.logger import Logger
from te_net_examples.utils.meta import build_meta
from te_net_examples.utils.versioner import build_version_dir


@dataclass(frozen=True, slots=True)
class qf10DesignOut:
    run_dir: str
    meta: dict[str, Any]
    design_path: str
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


def _rows_from_df(df: pd.DataFrame) -> list[list[str]]:
    cols = list(df.columns)
    vals = df.to_numpy(dtype=object)
    out: list[list[str]] = []
    for r in vals:
        out.append(["" if v is None else str(v) for v in r.tolist()])
    return out


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


def _pick_list(x: Any, name: str) -> list[Any]:
    if x is None:
        raise ValueError(f"missing required field: {name}")
    if isinstance(x, list):
        if len(x) == 0:
            raise ValueError(f"{name} must be non-empty")
        return x
    return [x]


def _validate_cfg(cfg: dict[str, Any]) -> None:
    if "name" not in cfg:
        raise ValueError("config missing: name")
    if "seed" not in cfg:
        raise ValueError("config missing: seed")
    if "grid" not in cfg:
        raise ValueError("config missing: grid")
    if "dgp" not in cfg:
        raise ValueError("config missing: dgp")
    if "te" not in cfg:
        raise ValueError("config missing: te")
    if "graph" not in cfg:
        raise ValueError("config missing: graph")


def _design_rows(cfg: dict[str, Any]) -> pd.DataFrame:
    lag = int(cfg.get("lag", 1))
    exclude_self = bool(cfg.get("exclude_self", True))

    grid = cfg["grid"]
    trials = int(grid.get("trials", 1))
    Ns = _pick_list(grid.get("N"), "grid.N")
    Ts = _pick_list(grid.get("T"), "grid.T")

    fn = cfg.get("preprocess", {}).get("factor_neutral", {})
    fn_enabled = bool(fn.get("enabled", False))
    fn_center = bool(fn.get("center", True))
    fn_k_list = _pick_list(
        fn.get("n_components", [0]), "preprocess.factor_neutral.n_components"
    )

    te_list = cfg["te"].get("estimators", [])
    if not isinstance(te_list, list) or len(te_list) == 0:
        raise ValueError("te.estimators must be a non-empty list")

    sel_list = cfg["graph"].get("select", [])
    if not isinstance(sel_list, list) or len(sel_list) == 0:
        raise ValueError("graph.select must be a non-empty list")

    hub = cfg.get("metrics", {}).get("hub", {})
    hub_k_list = _pick_list(hub.get("k", [0]), "metrics.hub.k")

    rows: list[dict[str, Any]] = []
    design_id = 0

    for trial in range(trials):
        for dgp in cfg["dgp"]:
            dgp_name = str(dgp.get("name", "")).strip()
            if not dgp_name:
                raise ValueError("dgp item missing name")
            dgp_params = dgp.get("params", {})
            if not isinstance(dgp_params, dict):
                raise ValueError("dgp.params must be a mapping")

            dgp_grid: dict[str, list[Any]] = {}
            for k, v in dgp_params.items():
                dgp_grid[str(k)] = _pick_list(v, f"dgp.{dgp_name}.params.{k}")

            dgp_keys = sorted(dgp_grid.keys())

            def iter_dgp(idx: int, cur: dict[str, Any]) -> list[dict[str, Any]]:
                if idx >= len(dgp_keys):
                    return [cur]
                key = dgp_keys[idx]
                out: list[dict[str, Any]] = []
                for val in dgp_grid[key]:
                    out.extend(iter_dgp(idx + 1, {**cur, key: val}))
                return out

            dgp_combos = iter_dgp(0, {})

            for N in Ns:
                for T in Ts:
                    for dgp_par in dgp_combos:
                        for fn_k in fn_k_list:
                            for te in te_list:
                                te_name = str(te.get("name", "")).strip()
                                if not te_name:
                                    raise ValueError("te.estimators item missing name")
                                te_params = te.get("params", {})
                                if not isinstance(te_params, dict):
                                    raise ValueError(
                                        "te.estimators params must be a mapping"
                                    )

                                for sel in sel_list:
                                    sel_name = str(sel.get("name", "")).strip()
                                    if not sel_name:
                                        raise ValueError(
                                            "graph.select item missing name"
                                        )
                                    sel_params = sel.get("params", {})
                                    if not isinstance(sel_params, dict):
                                        raise ValueError(
                                            "graph.select params must be a mapping"
                                        )

                                    density_list = _pick_list(
                                        sel_params.get("density", [None]),
                                        "graph.select.params.density",
                                    )

                                    for density in density_list:
                                        for hk in hub_k_list:
                                            row: dict[str, Any] = {
                                                "design_id": int(design_id),
                                                "trial": int(trial),
                                                "dgp": dgp_name,
                                                "N": int(N),
                                                "T": int(T),
                                                "lag": int(lag),
                                                "exclude_self": bool(exclude_self),
                                                "fn_enabled": bool(fn_enabled),
                                                "fn_center": bool(fn_center),
                                                "fn_n_components": int(fn_k),
                                                "te_name": te_name,
                                                "sel_name": sel_name,
                                                "hub_k": int(hk),
                                            }

                                            for k, v in dgp_par.items():
                                                row[f"dgp_{k}"] = v

                                            for k, v in te_params.items():
                                                row[f"te_{k}"] = v

                                            row["sel_mode"] = sel_params.get(
                                                "mode", "abs"
                                            )
                                            row["sel_exclude_self"] = bool(
                                                sel_params.get("exclude_self", True)
                                            )
                                            row["sel_density"] = density

                                            seed_key = [
                                                f"trial={trial}",
                                                f"dgp={dgp_name}",
                                                f"N={int(N)}",
                                                f"T={int(T)}",
                                                f"fnk={int(fn_k)}",
                                                f"te={te_name}",
                                                f"dens={density}",
                                                f"hubk={int(hk)}",
                                            ]
                                            row["seed_key"] = "|".join(seed_key)

                                            rows.append(row)
                                            design_id += 1

    df = pd.DataFrame(rows)
    df = df.sort_values(["design_id"], ascending=[True]).reset_index(drop=True)
    return df


def _summary_from_design(df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "name": str(cfg.get("name")),
        "n_rows": int(len(df)),
        "columns": list(df.columns),
        "seed": cfg.get("seed", {}),
    }

    def vc(col: str) -> dict[str, int]:
        if col not in df.columns:
            return {}
        s = df[col].astype(str)
        m = s.value_counts(dropna=False).to_dict()
        return {str(k): int(v) for k, v in m.items()}

    out["by_dgp"] = vc("dgp")
    out["by_te"] = vc("te_name")
    out["by_fn_n_components"] = vc("fn_n_components")
    out["by_sel_density"] = vc("sel_density")
    out["by_N"] = vc("N")
    out["by_T"] = vc("T")
    out["n_trials"] = int(df["trial"].nunique()) if "trial" in df.columns else 0
    return out


def run_qf_10_design(
    *,
    input_root: str,
    output_root: str,
    src_dir: str,
    config_path: str,
    script_path: str,
    component: str,
) -> qf10DesignOut:
    input_root = _require_dir(input_root)
    src_dir = _require_dir(src_dir)
    output_root_abs = os.path.abspath(output_root)

    cfg_path = _require_file(os.path.abspath(config_path))
    script_path_abs = _require_file(os.path.abspath(script_path))

    repo_root = _repo_root_from_path(Path(script_path_abs))
    env_path = _require_file(str(repo_root / "uv.lock"))

    cfg = read_yaml(cfg_path)
    if not isinstance(cfg, dict):
        raise ValueError("config must be a mapping")
    _validate_cfg(cfg)

    params: dict[str, Any] = {
        "input_root": os.path.abspath(input_root),
        "output_root": output_root_abs,
        "config_path": cfg_path,
        "component": component,
        "config_name": cfg.get("name", None),
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
        logger.info(_jline("input", component, "input_root", path=input_root))
        logger.info(_jline("input", component, "config", path=cfg_path))

        df = _design_rows(cfg)
        design_out = os.path.join(run_dir, "design.csv")
        write_csv(design_out, _rows_from_df(df), header=list(df.columns))

        summary = _summary_from_design(df, cfg)
        summary_out = os.path.join(run_dir, "summary.json")
        write_json(summary_out, summary)

        logger.info(_jline("output", component, "design", path=design_out))
        logger.info(_jline("output", component, "summary", path=summary_out))

        audit.finish_success()
        return qf10DesignOut(
            run_dir=run_dir,
            meta=meta,
            design_path=design_out,
            summary_path=summary_out,
        )
    except BaseException as e:
        audit.finish_error(e)
        raise
