from __future__ import annotations

import argparse
import glob
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from te_net_examples.io.yaml import read_yaml
from te_net_examples.utils.audit import Audit
from te_net_examples.utils.console import ConsoleSink
from te_net_examples.utils.jlog import jline
from te_net_examples.utils.logger import Logger
from te_net_examples.utils.meta import build_meta
from te_net_examples.utils.progress import Progress
from te_net_examples.utils.versioner import build_version_dir


@dataclass(frozen=True, slots=True)
class PublishAssetsOut:
    run_dir: str
    meta: dict[str, Any]


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


def _is_dir_dst(dst: str) -> bool:
    return dst.endswith("/") or dst.endswith(os.sep)


def _expand(input_dir: str, pattern: str) -> list[str]:
    p = os.path.join(input_dir, pattern)
    xs = glob.glob(p, recursive=True)
    ys = [x for x in xs if os.path.isfile(x)]
    ys.sort()
    return ys


def _render_name(tpl: str, *, i: int, src_path: str) -> str:
    b = os.path.basename(src_path)
    if "{basename}" in tpl or "{i" in tpl or "{i}" in tpl:
        try:
            return tpl.format(i=i, basename=b)
        except Exception as e:
            raise ValueError(f"invalid rename template: {tpl}") from e
    if i != 0:
        raise ValueError("rename without template placeholders requires single match")
    return tpl


def run_publish_assets(
    *,
    input_dir: str,
    output_root: str,
    src_dir: str,
    config_path: str,
    script_path: str,
    component: str,
    overwrite: bool,
) -> PublishAssetsOut:
    input_dir = _require_dir(input_dir)
    src_dir = _require_dir(src_dir)
    output_root_abs = os.path.abspath(output_root)

    cfg_path = _require_file(os.path.abspath(config_path))
    script_path_abs = _require_file(os.path.abspath(script_path))

    repo_root = _repo_root_from_path(Path(script_path_abs))
    env_path = _require_file(str(repo_root / "uv.lock"))

    cfg = read_yaml(cfg_path)
    if not isinstance(cfg, dict):
        raise ValueError("config must be a mapping")

    rules = cfg.get("rules", None)
    if not isinstance(rules, list) or len(rules) == 0:
        raise ValueError("config.rules must be a non-empty list")

    params: dict[str, Any] = {
        "input_dir": os.path.abspath(input_dir),
        "output_root": output_root_abs,
        "src_dir": os.path.abspath(src_dir),
        "config_path": cfg_path,
        "component": component,
        "overwrite": bool(overwrite),
        "n_rules": int(len(rules)),
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
        logger.info(jline("stage", component, "start", run_dir=run_dir))
        logger.info(jline("input", component, "input_dir", path=input_dir))
        logger.info(jline("input", component, "config", path=cfg_path))

        cfg_dir = os.path.join(run_dir, "cfg")
        _ensure_dir(cfg_dir)
        shutil.copy2(cfg_path, os.path.join(cfg_dir, os.path.basename(cfg_path)))

        planned: list[tuple[str, str]] = []

        for idx, r in enumerate(rules):
            if not isinstance(r, dict):
                raise ValueError("each rule must be a mapping")

            src = r.get("src", None)
            dst = r.get("dst", None)
            if not isinstance(src, str) or not src.strip():
                raise ValueError("rule missing src")
            if not isinstance(dst, str) or not dst.strip():
                raise ValueError("rule missing dst")

            allow_missing = bool(r.get("allow_missing", False))
            rename = r.get("rename", None)
            if rename is not None and (
                not isinstance(rename, str) or not rename.strip()
            ):
                raise ValueError("rename must be a non-empty string or null")

            matches = _expand(input_dir, src.strip())
            if len(matches) == 0:
                if allow_missing:
                    logger.warn(
                        jline(
                            "map",
                            component,
                            "missing_allowed",
                            rule=idx,
                            src=src.strip(),
                        )
                    )
                    continue
                raise FileNotFoundError(f"rule {idx} matched 0 files: {src.strip()}")

            dst_is_dir = _is_dir_dst(dst.strip())

            for i, m in enumerate(matches):
                if dst_is_dir:
                    name = (
                        os.path.basename(m)
                        if rename is None
                        else _render_name(rename.strip(), i=i, src_path=m)
                    )
                    out_rel = os.path.join(dst.strip().rstrip("/"), name)
                else:
                    if rename is None:
                        if len(matches) != 1:
                            raise ValueError(
                                "dst as file requires single match (or use dst as dir)"
                            )
                        out_rel = dst.strip()
                    else:
                        base = _render_name(rename.strip(), i=i, src_path=m)
                        out_rel = os.path.join(os.path.dirname(dst.strip()), base)

                out_abs = os.path.join(run_dir, out_rel)
                planned.append((m, out_abs))

        prog = Progress(logger=logger, name=component, total=int(len(planned)))
        prog.start()

        copied = 0
        skipped = 0

        for src_abs, dst_abs in planned:
            _ensure_dir(os.path.dirname(dst_abs))
            if os.path.exists(dst_abs) and not bool(overwrite):
                skipped += 1
                prog.step(1)
                continue
            shutil.copy2(src_abs, dst_abs)
            copied += 1
            prog.step(1)

        prog.finish()

        logger.info(jline("output", component, "copied", n=int(copied)))
        logger.info(jline("output", component, "skipped", n=int(skipped)))

        audit.finish_success()
        return PublishAssetsOut(run_dir=run_dir, meta=meta)

    except BaseException as e:
        audit.finish_error(e)
        raise


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="publish_assets")
    p.add_argument("input_dir")
    p.add_argument("output_root")
    p.add_argument("src_dir")
    p.add_argument("config_path")
    p.add_argument("--component", default="publish_assets")
    p.add_argument("--overwrite", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    ns = _build_parser().parse_args(argv)
    out = run_publish_assets(
        input_dir=ns.input_dir,
        output_root=ns.output_root,
        src_dir=ns.src_dir,
        config_path=ns.config_path,
        script_path=str(Path(__file__).resolve()),
        component=str(ns.component),
        overwrite=bool(ns.overwrite),
    )
    print(out.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
