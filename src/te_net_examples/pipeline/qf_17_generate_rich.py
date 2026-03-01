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
from te_net_lib.dgp.garch_factor import simulate_garch_factor
from te_net_lib.dgp.gaussian import simulate_gaussian_var
from te_net_lib.dgp.planted_signal import simulate_planted_signal_var
from te_net_lib.rng.core import RngScope


@dataclass(frozen=True, slots=True)
class Qf17GenerateRichOut:
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


def _person_bytes(x: Any) -> bytes:
    b = str(x).encode("utf-8")
    if len(b) > 16:
        raise ValueError("seed.person must be at most 16 bytes when UTF-8 encoded")
    return b


def _scope_from_seed(seed: dict[str, Any]) -> RngScope:
    base = seed.get("base", None)
    if base is None:
        raise ValueError("seed.base is required")
    digest_size = int(seed.get("digest_size", 8))
    person = _person_bytes(seed.get("person", "te_net_lib"))
    spawn_key = seed.get("spawn_key", [])
    if spawn_key is None:
        spawn_key = []
    if not isinstance(spawn_key, list):
        raise ValueError("seed.spawn_key must be a list")
    sk = [int(v) for v in spawn_key]
    return RngScope.from_seed(int(base), int(digest_size), person, sk)


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
        raise ValueError(f"invalid float for {name}: {x}")
    return float(s)


def _trial_dir(run_dir: str, design_id: int) -> str:
    return os.path.join(run_dir, "trial", f"{int(design_id):08d}")


def _write_npy(path: str, arr: np.ndarray) -> None:
    _ensure_dir(os.path.dirname(path))
    np.save(path, arr)


def _adj_er(rng: np.random.Generator, N: int, edge_prob: float) -> np.ndarray:
    if not (0.0 <= float(edge_prob) <= 1.0):
        raise ValueError("edge_prob must be in [0, 1]")
    a = (rng.random(size=(N, N)) < float(edge_prob)).astype(np.int8)
    np.fill_diagonal(a, 0)
    return a


def _safe_array_info(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return {"shape": list(x.shape), "dtype": str(x.dtype)}
    return None


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
    save_A: bool,
    save_factors: bool,
    save_loadings: bool,
    save_sigma2: bool,
    save_eps: bool,
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
    sample = None

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
            sr = sample.extras.get("spectral_radius", None)
            spectral_radius = float(sr) if sr is not None else None
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

    if sample is None:
        raise RuntimeError("sample generation failed")

    returns = sample.returns.astype(np.float64, copy=False)
    ret_path = os.path.join(out_dir, "returns.npy")
    _write_npy(ret_path, returns)

    true_adj_path = None
    if save_true_adj and sample.true_adj is not None:
        true_adj_path = os.path.join(out_dir, "true_adj.npy")
        _write_npy(true_adj_path, sample.true_adj.astype(np.int8, copy=False))

    A_path = None
    if save_A:
        A = sample.extras.get("A", None)
        if isinstance(A, np.ndarray):
            A_path = os.path.join(out_dir, "A.npy")
            _write_npy(A_path, A.astype(np.float64, copy=False))

    factors_path = None
    if save_factors:
        F = sample.extras.get("factors", None)
        if isinstance(F, np.ndarray):
            factors_path = os.path.join(out_dir, "factors.npy")
            _write_npy(factors_path, F.astype(np.float64, copy=False))

    loadings_path = None
    if save_loadings:
        B = sample.extras.get("loadings", None)
        if isinstance(B, np.ndarray):
            loadings_path = os.path.join(out_dir, "loadings.npy")
            _write_npy(loadings_path, B.astype(np.float64, copy=False))

    sigma2_path = None
    if save_sigma2:
        s2 = sample.extras.get("sigma2", None)
        if isinstance(s2, np.ndarray):
            sigma2_path = os.path.join(out_dir, "sigma2.npy")
            _write_npy(sigma2_path, s2.astype(np.float64, copy=False))

    eps_path = None
    if save_eps:
        e = sample.extras.get("eps", None)
        if isinstance(e, np.ndarray):
            eps_path = os.path.join(out_dir, "eps.npy")
            _write_npy(eps_path, e.astype(np.float64, copy=False))

    trial_json_path = None
    if save_trial_json:
        extras_info: dict[str, Any] = {}
        for k, v in sample.extras.items():
            info = _safe_array_info(v)
            if info is not None:
                extras_info[str(k)] = info
            elif isinstance(v, (int, float, str, bool)) or v is None:
                extras_info[str(k)] = v
            elif isinstance(v, dict):
                extras_info[str(k)] = {"keys": sorted([str(x) for x in v.keys()])}
            else:
                extras_info[str(k)] = str(v)

        trial = {
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
            "paths": {
                "returns": ret_path,
                "true_adj": true_adj_path,
                "A": A_path,
                "factors": factors_path,
                "loadings": loadings_path,
                "sigma2": sigma2_path,
                "eps": eps_path,
            },
            "extras": extras_info,
        }
        trial_json_path = os.path.join(out_dir, "trial.json")
        write_json(trial_json_path, trial)

    return {
        "design_id": int(design_id),
        "dgp": dgp,
        "accepted": bool(accepted),
        "tries": int(tries),
        "spectral_radius": float(spectral_radius)
        if spectral_radius is not None
        else None,
        "trial_dir": out_dir,
        "returns": ret_path,
        "true_adj": true_adj_path,
        "trial_json": trial_json_path,
        "A": A_path,
        "factors": factors_path,
        "loadings": loadings_path,
        "sigma2": sigma2_path,
        "eps": eps_path,
    }


def run_qf_17_generate_rich(
    *,
    input_dir: str,
    output_root: str,
    src_dir: str,
    config_path: str,
    script_path: str,
    component: str,
    design_filename: str,
    summary_filename: str,
) -> Qf17GenerateRichOut:
    input_dir = _require_dir(input_dir)
    src_dir = _require_dir(src_dir)
    output_root_abs = os.path.abspath(output_root)

    cfg_path = _require_file(os.path.abspath(config_path))
    script_path_abs = _require_file(os.path.abspath(script_path))

    design_in = _require_file(os.path.join(input_dir, design_filename))
    summary_in = _require_file(os.path.join(input_dir, summary_filename))
    summ = read_json(summary_in)
    if not isinstance(summ, dict):
        raise ValueError("design summary must be a mapping")
    seed = summ.get("seed", None)
    if not isinstance(seed, dict):
        raise ValueError("design summary missing seed mapping")

    scope = _scope_from_seed(seed)

    repo_root = _repo_root_from_path(Path(script_path_abs))
    env_path = _require_file(str(repo_root / "uv.lock"))

    cfg = read_yaml(cfg_path)
    if not isinstance(cfg, dict):
        raise ValueError("config must be a mapping")

    stability = cfg.get("stability", {})
    if stability is None:
        stability = {}
    if not isinstance(stability, dict):
        raise ValueError("config.stability must be a mapping")

    max_var_spectral_radius = stability.get("max_var_spectral_radius", None)
    max_tries = int(stability.get("max_tries", 1))

    output_cfg = cfg.get("output", {})
    if output_cfg is None:
        output_cfg = {}
    if not isinstance(output_cfg, dict):
        raise ValueError("config.output must be a mapping")

    save_true_adj = bool(output_cfg.get("save_true_adj", True))
    save_trial_json = bool(output_cfg.get("save_trial_json", True))
    save_A = bool(output_cfg.get("save_A", True))
    save_factors = bool(output_cfg.get("save_factors", True))
    save_loadings = bool(output_cfg.get("save_loadings", True))
    save_sigma2 = bool(output_cfg.get("save_sigma2", True))
    save_eps = bool(output_cfg.get("save_eps", True))

    params: dict[str, Any] = {
        "input_dir": os.path.abspath(input_dir),
        "output_root": output_root_abs,
        "config_path": cfg_path,
        "component": component,
        "design_path": os.path.abspath(design_in),
        "design_summary_path": os.path.abspath(summary_in),
        "max_var_spectral_radius": max_var_spectral_radius,
        "max_tries": int(max_tries),
        "save_true_adj": bool(save_true_adj),
        "save_trial_json": bool(save_trial_json),
        "save_A": bool(save_A),
        "save_factors": bool(save_factors),
        "save_loadings": bool(save_loadings),
        "save_sigma2": bool(save_sigma2),
        "save_eps": bool(save_eps),
    }

    meta = build_meta(
        params=params, env=env_path, script=script_path_abs, cfg=cfg_path, src=src_dir
    )

    run_dir = build_version_dir(output_root_abs, meta)
    audit = Audit.create(run_dir, meta)
    logger = Logger(sinks=[ConsoleSink(), audit])

    try:
        logger.info(jline("stage", component, "start", run_dir=run_dir))
        logger.info(jline("input", component, "design", path=design_in))
        logger.info(jline("input", component, "design_summary", path=summary_in))
        logger.info(jline("input", component, "config", path=cfg_path))

        cfg_dir = os.path.join(run_dir, "cfg")
        os.makedirs(cfg_dir, exist_ok=True)
        cfg_copied = os.path.join(cfg_dir, os.path.basename(cfg_path))
        shutil.copy2(cfg_path, cfg_copied)
        logger.info(jline("output", component, "config_copied", path=cfg_copied))

        df = _df_from_csv(design_in)
        df = df.rename(columns={c: c.strip() for c in df.columns})
        need = ["design_id", "seed_key", "dgp", "N", "T"]
        _require_cols(df, need)

        records = df.to_dict(orient="records")
        p = Progress(logger=logger, name="qf/17_generate_rich", total=int(len(records)))
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
                save_A=bool(save_A),
                save_factors=bool(save_factors),
                save_loadings=bool(save_loadings),
                save_sigma2=bool(save_sigma2),
                save_eps=bool(save_eps),
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

        summ_out = {
            "n_rows": int(len(records)),
            "n_accepted": int(n_ok),
            "n_rejected": int(n_reject),
            "trial_root": os.path.join(run_dir, "trial"),
            "items": items[:2000],
        }

        summary_out = os.path.join(run_dir, "summary.json")
        write_json(summary_out, summ_out)
        logger.info(jline("output", component, "summary", path=summary_out))

        audit.finish_success()
        return Qf17GenerateRichOut(run_dir=run_dir, meta=meta, summary_path=summary_out)
    except BaseException as e:
        audit.finish_error(e)
        raise
