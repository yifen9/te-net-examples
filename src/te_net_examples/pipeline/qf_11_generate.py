from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shutil

from te_net_examples.io.csv import read_csv
from te_net_examples.io.json import read_json, write_json
from te_net_examples.io.yaml import read_yaml
from te_net_examples.utils.audit import Audit
from te_net_examples.utils.console import ConsoleSink
from te_net_examples.utils.logger import Logger
from te_net_examples.utils.meta import build_meta
from te_net_examples.utils.progress import Progress
from te_net_examples.utils.versioner import build_version_dir
from te_net_lib.dgp.garch_factor import simulate_garch_factor
from te_net_lib.dgp.gaussian import simulate_gaussian_var
from te_net_lib.dgp.planted_signal import simulate_planted_signal_var
from te_net_lib.rng.core import RngScope


@dataclass(frozen=True, slots=True)
class Qf11GenerateOut:
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


def _to_int(x: Any, name: str) -> int:
    try:
        return int(x)
    except Exception as e:
        raise ValueError(f"invalid int for {name}: {x}") from e


def _to_float(x: Any, name: str) -> float:
    try:
        return float(x)
    except Exception as e:
        raise ValueError(f"invalid float for {name}: {x}") from e


def _require_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")


def _person_bytes(x: Any) -> bytes:
    b = str(x).encode("utf-8")
    if len(b) > 16:
        raise ValueError("seed.person must be at most 16 bytes when UTF-8 encoded")
    return b


def _scope_from_seed(seed: dict[str, Any]) -> RngScope:
    base = seed.get("base", None)
    if base is None:
        raise ValueError("seed.base is required in design summary")
    digest_size = int(seed.get("digest_size", 8))
    person = _person_bytes(seed.get("person", "te_net_lib"))
    spawn_key = seed.get("spawn_key", [])
    if spawn_key is None:
        spawn_key = []
    if not isinstance(spawn_key, list):
        raise ValueError("seed.spawn_key must be a list")
    sk = [int(v) for v in spawn_key]
    return RngScope.from_seed(int(base), int(digest_size), person, sk)


def _adj_er(rng: np.random.Generator, N: int, edge_prob: float) -> np.ndarray:
    if not (0.0 <= float(edge_prob) <= 1.0):
        raise ValueError("edge_prob must be in [0, 1]")
    a = (rng.random(size=(N, N)) < float(edge_prob)).astype(np.int8)
    np.fill_diagonal(a, 0)
    return a


def _trial_dir(run_dir: str, design_id: int) -> str:
    return os.path.join(run_dir, "trial", f"{int(design_id):08d}")


def _write_npy(path: str, arr: np.ndarray) -> None:
    _ensure_dir(os.path.dirname(path))
    np.save(path, arr)


def _generate_one(
    row: dict[str, Any],
    *,
    scope: RngScope,
    component: str,
    run_dir: str,
    max_var_spectral_radius: float | None,
    max_tries: int,
    save_true_adj: bool,
    save_trial_json: bool,
    extras_mode: str,
) -> dict[str, Any]:
    design_id = _to_int(row["design_id"], "design_id")
    seed_key = str(row.get("seed_key", f"design_id={design_id}"))
    rng = scope.child([component, seed_key, "dgp"])

    dgp = str(row["dgp"]).strip()
    N = _to_int(row["N"], "N")
    T = _to_int(row["T"], "T")

    out_dir = _trial_dir(run_dir, design_id)
    _ensure_dir(out_dir)

    tries = 0
    accepted = False
    spectral_radius = None

    while tries < int(max_tries) and not accepted:
        tries += 1

        if dgp == "planted_var":
            edge_prob = _to_float(row.get("dgp_edge_prob"), "dgp_edge_prob")
            coef_strength = _to_float(row.get("dgp_coef_strength"), "dgp_coef_strength")
            noise_scale = _to_float(row.get("dgp_noise_scale"), "dgp_noise_scale")
            burnin = _to_int(row.get("dgp_burnin"), "dgp_burnin")
            sample = simulate_planted_signal_var(
                rng=rng,
                N=int(N),
                T=int(T),
                edge_prob=float(edge_prob),
                coef_strength=float(coef_strength),
                noise_scale=float(noise_scale),
                burnin=int(burnin),
            )
            A = sample.extras.get("A", None)
            if isinstance(A, np.ndarray) and A.size > 0:
                spectral_radius = float(np.max(np.abs(np.linalg.eigvals(A))))
        elif dgp == "gaussian_var_from_adj":
            adj_source = str(row.get("dgp_adj_source", "er")).strip()
            noise_scale = _to_float(row.get("dgp_noise_scale"), "dgp_noise_scale")
            burnin = _to_int(row.get("dgp_burnin"), "dgp_burnin")
            coef_scale = _to_float(row.get("dgp_coef_scale"), "dgp_coef_scale")
            if adj_source != "er":
                raise ValueError(f"unsupported adj_source: {adj_source}")
            edge_prob = _to_float(row.get("dgp_edge_prob"), "dgp_edge_prob")
            adj = _adj_er(rng, int(N), float(edge_prob))
            sample = simulate_gaussian_var(
                rng=rng,
                N=int(N),
                T=int(T),
                adj=adj,
                coef_scale=float(coef_scale),
                noise_scale=float(noise_scale),
                burnin=int(burnin),
            )
            spectral_radius = (
                float(sample.extras.get("spectral_radius"))
                if "spectral_radius" in sample.extras
                else None
            )
        elif dgp == "garch_factor":
            k = _to_int(row.get("dgp_k"), "dgp_k")
            omega = _to_float(row.get("dgp_omega"), "dgp_omega")
            alpha = _to_float(row.get("dgp_alpha"), "dgp_alpha")
            beta = _to_float(row.get("dgp_beta"), "dgp_beta")
            loading_scale = _to_float(row.get("dgp_loading_scale"), "dgp_loading_scale")
            burnin = _to_int(row.get("dgp_burnin"), "dgp_burnin")
            sample = simulate_garch_factor(
                rng=rng,
                N=int(N),
                T=int(T),
                k=int(k),
                omega=float(omega),
                alpha=float(alpha),
                beta=float(beta),
                loading_scale=float(loading_scale),
                burnin=int(burnin),
            )
        else:
            raise ValueError(f"unsupported dgp: {dgp}")

        if max_var_spectral_radius is None:
            accepted = True
        else:
            if spectral_radius is None:
                accepted = True
            else:
                accepted = bool(
                    float(spectral_radius) <= float(max_var_spectral_radius)
                )

        if not accepted:
            rng = scope.child([component, seed_key, "dgp", f"try={tries}"])

    returns = sample.returns.astype(np.float64, copy=False)
    ret_path = os.path.join(out_dir, "returns.npy")
    _write_npy(ret_path, returns)

    true_adj = sample.true_adj
    true_adj_path = None
    if save_true_adj and true_adj is not None:
        true_adj_path = os.path.join(out_dir, "true_adj.npy")
        _write_npy(true_adj_path, true_adj.astype(np.int8, copy=False))

    extras: dict[str, Any] = {}
    if extras_mode == "none":
        extras = {}
    elif extras_mode == "full":
        for k, v in sample.extras.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                extras[str(k)] = v
            elif isinstance(v, np.ndarray):
                extras[str(k)] = {"shape": list(v.shape), "dtype": str(v.dtype)}
            elif isinstance(v, dict):
                extras[str(k)] = v
            else:
                extras[str(k)] = str(v)
    else:
        for k, v in sample.extras.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                extras[str(k)] = v
            elif isinstance(v, np.ndarray):
                extras[str(k)] = {"shape": list(v.shape), "dtype": str(v.dtype)}
            elif isinstance(v, dict):
                ok = True
                for _, vv in v.items():
                    if not isinstance(vv, (int, float, str, bool)) and vv is not None:
                        ok = False
                        break
                extras[str(k)] = (
                    v if ok else {"keys": sorted([str(x) for x in v.keys()])}
                )
            else:
                extras[str(k)] = str(v)

    trial_json_path = None
    if save_trial_json:
        meta = {
            "design_id": int(design_id),
            "seed_key": seed_key,
            "dgp": dgp,
            "N": int(N),
            "T": int(T),
            "accepted": bool(accepted),
            "tries": int(tries),
            "spectral_radius": float(spectral_radius)
            if spectral_radius is not None
            else None,
            "paths": {"returns": ret_path, "true_adj": true_adj_path},
            "extras": extras,
        }
        trial_json_path = os.path.join(out_dir, "trial.json")
        write_json(trial_json_path, meta)

    return {
        "design_id": int(design_id),
        "seed_key": seed_key,
        "dgp": dgp,
        "accepted": bool(accepted),
        "tries": int(tries),
        "spectral_radius": float(spectral_radius)
        if spectral_radius is not None
        else None,
        "trial_dir": out_dir,
        "returns_path": ret_path,
        "true_adj_path": true_adj_path,
        "trial_json_path": trial_json_path,
    }


def run_qf_11_generate(
    *,
    input_dir: str,
    output_root: str,
    src_dir: str,
    config_path: str,
    script_path: str,
    component: str,
    design_filename: str,
) -> Qf11GenerateOut:
    input_dir = _require_dir(input_dir)
    src_dir = _require_dir(src_dir)
    output_root_abs = os.path.abspath(output_root)

    design_in = _require_file(os.path.join(input_dir, design_filename))
    design_summary_in = _require_file(os.path.join(input_dir, "summary.json"))
    cfg_path = _require_file(os.path.abspath(config_path))
    script_path_abs = _require_file(os.path.abspath(script_path))

    repo_root = _repo_root_from_path(Path(script_path_abs))
    env_path = _require_file(str(repo_root / "uv.lock"))

    cfg = read_yaml(cfg_path)
    if not isinstance(cfg, dict):
        raise ValueError("config must be a mapping")

    design_summary = read_json(design_summary_in)
    if not isinstance(design_summary, dict):
        raise ValueError("design summary must be a mapping")
    seed = design_summary.get("seed", None)
    if not isinstance(seed, dict):
        raise ValueError("design summary missing seed mapping")

    scope = _scope_from_seed(seed)

    stability = cfg.get("stability", {})
    if stability is None:
        stability = {}
    if not isinstance(stability, dict):
        raise ValueError("config.stability must be a mapping")

    max_var_spectral_radius = stability.get("max_var_spectral_radius", None)
    max_tries = int(stability.get("max_tries", 1))

    output = cfg.get("output", {})
    if output is None:
        output = {}
    if not isinstance(output, dict):
        raise ValueError("config.output must be a mapping")

    returns_format = str(output.get("returns_format", "npy"))
    if returns_format != "npy":
        raise ValueError(f"unsupported returns_format: {returns_format}")

    save_true_adj = bool(output.get("save_true_adj", True))
    save_trial_json = bool(output.get("save_trial_json", True))
    extras_mode = str(output.get("extras_mode", "summary")).strip()
    if extras_mode not in ("summary", "full", "none"):
        raise ValueError(f"unsupported extras_mode: {extras_mode}")

    params: dict[str, Any] = {
        "input_dir": os.path.abspath(input_dir),
        "output_root": output_root_abs,
        "design_filename": design_filename,
        "design_path": os.path.abspath(design_in),
        "design_summary_path": os.path.abspath(design_summary_in),
        "config_path": cfg_path,
        "component": component,
        "max_var_spectral_radius": max_var_spectral_radius,
        "max_tries": int(max_tries),
        "returns_format": returns_format,
        "save_true_adj": bool(save_true_adj),
        "save_trial_json": bool(save_trial_json),
        "extras_mode": extras_mode,
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
        logger.info(_jline("input", component, "design", path=design_in))
        logger.info(
            _jline("input", component, "design_summary", path=design_summary_in)
        )
        logger.info(_jline("input", component, "config", path=cfg_path))

        cfg_dir = os.path.join(run_dir, "cfg")
        os.makedirs(cfg_dir, exist_ok=True)
        cfg_copied = os.path.join(cfg_dir, os.path.basename(cfg_path))
        shutil.copy2(cfg_path, cfg_copied)
        logger.info(_jline("output", component, "config_copied", path=cfg_copied))

        df = _df_from_csv(design_in)
        df = df.rename(columns={c: c.strip() for c in df.columns})

        need = ["design_id", "seed_key", "dgp", "N", "T"]
        _require_cols(df, need)

        records = df.to_dict(orient="records")

        p = Progress(logger=logger, name="qf/11_generate", total=int(len(records)))
        p.start()

        items: list[dict[str, Any]] = []
        n_ok = 0
        n_reject = 0

        for r in records:
            info = _generate_one(
                r,
                scope=scope,
                component=component,
                run_dir=run_dir,
                max_var_spectral_radius=float(max_var_spectral_radius)
                if max_var_spectral_radius is not None
                else None,
                max_tries=int(max_tries),
                save_true_adj=bool(save_true_adj),
                save_trial_json=bool(save_trial_json),
                extras_mode=extras_mode,
            )
            items.append(
                {
                    "design_id": int(info["design_id"]),
                    "dgp": str(info["dgp"]),
                    "accepted": bool(info["accepted"]),
                    "tries": int(info["tries"]),
                    "spectral_radius": info["spectral_radius"],
                    "trial_dir": str(info["trial_dir"]),
                }
            )
            if bool(info["accepted"]):
                n_ok += 1
            else:
                n_reject += 1
            p.step(1)

        p.finish()

        summ = {
            "n_rows": int(len(records)),
            "n_accepted": int(n_ok),
            "n_rejected": int(n_reject),
            "max_var_spectral_radius": float(max_var_spectral_radius)
            if max_var_spectral_radius is not None
            else None,
            "max_tries": int(max_tries),
            "output": {
                "returns_format": returns_format,
                "save_true_adj": bool(save_true_adj),
                "save_trial_json": bool(save_trial_json),
                "extras_mode": extras_mode,
            },
            "trial_root": os.path.join(run_dir, "trial"),
            "items": items[:2000],
        }

        summary_out = os.path.join(run_dir, "summary.json")
        write_json(summary_out, summ)

        logger.info(_jline("output", component, "summary", path=summary_out))

        audit.finish_success()
        return Qf11GenerateOut(run_dir=run_dir, meta=meta, summary_path=summary_out)
    except BaseException as e:
        audit.finish_error(e)
        raise
