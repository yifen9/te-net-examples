from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from te_net_examples.utils.audit import Audit
from te_net_examples.utils.console import ConsoleSink
from te_net_examples.utils.logger import Logger
from te_net_examples.utils.meta import build_meta
from te_net_examples.utils.versioner import build_version_dir


@dataclass(frozen=True, slots=True)
class SourceMetaOut:
    run_dir: str
    meta: dict[str, Any]
    copied_path: str


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


def _j(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def create_source_meta_run(
    *,
    input_dir: str,
    output_root: str,
    src_dir: str,
    script_path: str,
    source_name: str,
    component: str,
) -> SourceMetaOut:
    input_dir = _require_dir(input_dir)
    src_dir = _require_dir(src_dir)
    output_root_abs = os.path.abspath(output_root)

    source_path = _require_file(os.path.join(input_dir, source_name))
    script_path_abs = _require_file(os.path.abspath(script_path))

    repo_root = _repo_root_from_path(Path(script_path_abs))
    env_path = _require_file(str(repo_root / "uv.lock"))

    params: dict[str, Any] = {
        "stage": component,
        "input_dir": os.path.abspath(input_dir),
        "output_root": output_root_abs,
        "source_file": source_name,
    }

    meta = build_meta(
        params=params,
        env=env_path,
        script=script_path_abs,
        cfg=source_path,
        src=src_dir,
    )

    run_dir = build_version_dir(output_root_abs, meta)
    audit = Audit.create(run_dir, meta)
    logger = Logger(sinks=[ConsoleSink(), audit])

    try:
        logger.info(
            _j(
                {
                    "event": "stage",
                    "component": component,
                    "msg": "start",
                    "run_dir": run_dir,
                }
            )
        )
        logger.info(
            _j(
                {
                    "event": "input",
                    "component": component,
                    "msg": "source_yaml",
                    "path": source_path,
                }
            )
        )

        dst = os.path.join(run_dir, source_name)
        shutil.copy2(source_path, dst)

        logger.info(
            _j(
                {
                    "event": "output",
                    "component": component,
                    "msg": "copied",
                    "path": dst,
                }
            )
        )
        audit.finish_success()
        return SourceMetaOut(run_dir=run_dir, meta=meta, copied_path=dst)
    except BaseException as e:
        audit.finish_error(e)
        raise


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="source_meta")
    p.add_argument("input_dir")
    p.add_argument("output_root")
    p.add_argument("src_dir")
    p.add_argument("--source-name", default="SOURCE.yaml")
    p.add_argument("--component", default="qf/01_meta")
    return p


def main(argv: list[str] | None = None) -> int:
    ns = _build_parser().parse_args(argv)
    out = create_source_meta_run(
        input_dir=ns.input_dir,
        output_root=ns.output_root,
        src_dir=ns.src_dir,
        script_path=str(Path(__file__).resolve()),
        source_name=ns.source_name,
        component=ns.component,
    )
    print(out.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
