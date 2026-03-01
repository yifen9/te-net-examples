from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Any

from te_net_examples.io.json import read_json
from te_net_examples.io.yaml import read_yaml
from te_net_examples.utils.audit import Audit
from te_net_examples.utils.console import ConsoleSink
from te_net_examples.utils.jlog import jline
from te_net_examples.utils.logger import Logger
from te_net_examples.utils.meta import build_meta
from te_net_examples.utils.progress import Progress
from te_net_examples.utils.versioner import build_version_dir


def _repo_root_from_script(p: Path) -> Path:
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


def _require_file(p: Path) -> Path:
    if not p.is_file():
        raise FileNotFoundError(str(p))
    return p


def _require_dir(p: Path) -> Path:
    if not p.is_dir():
        raise FileNotFoundError(str(p))
    return p


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _step_sort_key(step: str) -> tuple[int, str]:
    s = step.strip()
    head = ""
    for ch in s:
        if ch.isdigit():
            head += ch
        else:
            break
    if head != "":
        return (int(head), s)
    return (10**9, s)


def _step_dirs(data_root: Path) -> list[Path]:
    out: list[Path] = []
    for e in sorted(data_root.iterdir()):
        if e.is_dir():
            out.append(e)
    return out


def _latest_artifact_dir(step_dir: Path) -> Path:
    dirs: list[Path] = []
    for e in sorted(step_dir.iterdir()):
        if e.is_dir():
            dirs.append(e)
    if not dirs:
        raise FileNotFoundError(str(step_dir))
    return dirs[-1]


def _all_files(src_root: Path) -> list[Path]:
    out: list[Path] = []
    for r, ds, fs in os.walk(src_root):
        for f in fs:
            out.append(Path(r) / f)
    out.sort()
    return out


def _meta_files(src_root: Path) -> list[Path]:
    out: list[Path] = []
    for r, ds, fs in os.walk(src_root):
        rel = Path(r).relative_to(src_root)
        if rel.parts and not rel.parts[0].startswith("_"):
            ds[:] = []
            continue
        for f in fs:
            if not f.startswith("_"):
                continue
            out.append(Path(r) / f)
    out.sort()
    return out


def _meta_dirs(src_root: Path) -> list[Path]:
    out: list[Path] = []
    for e in sorted(src_root.iterdir()):
        if e.is_dir() and e.name.startswith("_"):
            out.append(e)
    return out


def _copy_full(src_dir: Path, dst_dir: Path, prog: Progress) -> int:
    files = _all_files(src_dir)
    copied = 0
    for p in files:
        rel = p.relative_to(src_dir)
        dst = dst_dir / rel
        _ensure_dir(dst.parent)
        shutil.copy2(p, dst)
        copied += 1
        prog.step(1)
    return copied


def _copy_meta(src_dir: Path, dst_dir: Path, prog: Progress) -> int:
    copied = 0
    for d in _meta_dirs(src_dir):
        rel = d.relative_to(src_dir)
        dst = dst_dir / rel
        if dst.exists():
            shutil.rmtree(dst)
        _ensure_dir(dst.parent)
        shutil.copytree(d, dst, copy_function=shutil.copy2)
        n = len(_all_files(d))
        copied += n
        prog.step(n)
    files = _meta_files(src_dir)
    for p in files:
        rel = p.relative_to(src_dir)
        dst = dst_dir / rel
        _ensure_dir(dst.parent)
        shutil.copy2(p, dst)
        copied += 1
        prog.step(1)
    return copied


def _read_meta(meta_path: Path) -> dict[str, Any]:
    obj = read_json(str(meta_path))
    if not isinstance(obj, dict):
        raise ValueError(f"invalid _meta.json: {meta_path}")
    return obj


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pack_out")
    ap.add_argument("data_root")
    ap.add_argument("output_root")
    ap.add_argument("src_dir")
    ap.add_argument("config_path")
    ns = ap.parse_args(argv)

    script_file = Path(__file__).resolve()
    repo_root = _repo_root_from_script(script_file)

    data_root = (
        (repo_root / ns.data_root).resolve()
        if not os.path.isabs(ns.data_root)
        else Path(ns.data_root).resolve()
    )
    output_root = (
        (repo_root / ns.output_root).resolve()
        if not os.path.isabs(ns.output_root)
        else Path(ns.output_root).resolve()
    )
    src_dir = (
        (repo_root / ns.src_dir).resolve()
        if not os.path.isabs(ns.src_dir)
        else Path(ns.src_dir).resolve()
    )
    cfg_path = (
        (repo_root / ns.config_path).resolve()
        if not os.path.isabs(ns.config_path)
        else Path(ns.config_path).resolve()
    )

    _require_dir(data_root)
    _require_dir(src_dir)
    _require_file(cfg_path)

    env_file = _require_file(repo_root / "uv.lock")

    cfg = read_yaml(str(cfg_path))
    if not isinstance(cfg, dict):
        raise ValueError("config must be a mapping")
    name = str(cfg.get("name", "")).strip()
    if not name:
        raise ValueError("config missing: name")

    modes = cfg.get("modes", {})
    if modes is None:
        modes = {}
    if not isinstance(modes, dict):
        raise ValueError("config.modes must be a mapping")

    default_mode = str(modes.get("default", "meta")).strip()
    if default_mode not in ("full", "meta"):
        raise ValueError("modes.default must be full or meta")

    full_list = modes.get("full", [])
    if full_list is None:
        full_list = []
    if not isinstance(full_list, list):
        raise ValueError("modes.full must be a list")
    full_set = set([str(x).strip() for x in full_list if str(x).strip()])

    step_dirs = _step_dirs(data_root)
    if not step_dirs:
        raise FileNotFoundError(str(data_root))

    steps: list[dict[str, Any]] = []
    for sd in step_dirs:
        step = sd.name
        art = _latest_artifact_dir(sd)
        meta_path = art / "_meta.json"
        if not meta_path.is_file():
            continue
        steps.append(
            {"step": step, "artifact_dir": str(art), "meta_path": str(meta_path)}
        )

    if not steps:
        raise FileNotFoundError("no step artifacts with _meta.json under data_root")

    steps.sort(key=lambda x: _step_sort_key(str(x["step"])))

    lineage: list[dict[str, Any]] = []
    for s in steps:
        lineage.append(_read_meta(Path(str(s["meta_path"]))))

    file_count = 0
    for s in steps:
        step = str(s["step"])
        mode = "full" if step in full_set else default_mode
        art = Path(str(s["artifact_dir"]))
        if mode == "full":
            file_count += len(_all_files(art))
        else:
            for d in _meta_dirs(art):
                file_count += len(_all_files(d))
            file_count += len(_meta_files(art))

    params = {
        "data_root": str(data_root),
        "output_root": str(output_root),
        "src_dir": str(src_dir),
        "config_path": str(cfg_path),
        "file_count": int(file_count),
        "steps": [
            {"step": str(x["step"]), "artifact_dir": str(x["artifact_dir"])}
            for x in steps
        ],
        "modes_default": default_mode,
        "modes_full": sorted(list(full_set)),
        "lineage": lineage,
    }

    meta = build_meta(
        params=params,
        env=str(env_file),
        script=str(script_file),
        src=str(src_dir),
        cfg=str(cfg_path),
    )

    run_dir = Path(build_version_dir(str(output_root), meta))
    audit = Audit.create(str(run_dir), meta)
    logger = Logger(sinks=[ConsoleSink(), audit])

    try:
        logger.info(
            jline(
                "pack",
                "pack_out",
                "start",
                run_dir=str(run_dir),
                data_root=str(data_root),
                file_count=int(file_count),
            )
        )

        cfg_dir = run_dir / "cfg"
        _ensure_dir(cfg_dir)
        shutil.copy2(cfg_path, cfg_dir / cfg_path.name)

        pipe_root = run_dir / "pipeline"
        _ensure_dir(pipe_root)

        prog = Progress(logger=logger, name="pack_out", total=int(file_count))
        prog.start()

        copied_files = 0
        for s in steps:
            step = str(s["step"])
            art = Path(str(s["artifact_dir"]))
            mode = "full" if step in full_set else default_mode
            dst = pipe_root / step / art.name
            if mode == "full":
                copied_files += _copy_full(art, dst, prog)
            else:
                copied_files += _copy_meta(art, dst, prog)

        prog.finish()

        audit.finish_success()
        logger.info(
            jline(
                "pack",
                "pack_out",
                "end",
                run_dir=str(run_dir),
                copied_files=int(copied_files),
            )
        )
        return 0

    except BaseException as e:
        audit.finish_error(e)
        logger.error(
            jline("pack", "pack_out", "error", type=type(e).__name__, message=str(e))
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
