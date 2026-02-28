from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from te_net_examples.utils.audit import Audit
from te_net_examples.utils.logger import Logger
from te_net_examples.utils.console import ConsoleSink
from te_net_examples.utils.meta import build_meta
from te_net_examples.utils.versioner import build_version_dir


@dataclass(frozen=True, slots=True)
class SourceMetaOut:
    run_dir: str
    meta: dict[str, Any]
    copied_path: str


def _repo_root_from_script(script_path: str) -> Path:
    p = Path(script_path).resolve()
    for _ in range(8):
        if (p / "uv.lock").is_file() and (p / "pyproject.toml").is_file():
            return p
        if p.parent == p:
            break
        p = p.parent
    raise FileNotFoundError("repo root not found (expected uv.lock and pyproject.toml)")


def _require_file(path: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return path


def _require_dir(path: str) -> str:
    if not os.path.isdir(path):
        raise FileNotFoundError(path)
    return path


def create_source_meta_run(
    *,
    input_dir: str,
    output_root: str,
    src_dir: str,
    script_path: str,
    source_name: str = "SOURCE.yaml",
) -> SourceMetaOut:
    input_dir = _require_dir(input_dir)
    output_root = os.path.abspath(output_root)
    src_dir = _require_dir(src_dir)

    source_path = _require_file(os.path.join(input_dir, source_name))

    root = _repo_root_from_script(script_path)
    env_path = _require_file(str(root / "uv.lock"))
    script_path = _require_file(script_path)

    params: dict[str, Any] = {
        "input_dir": os.path.abspath(input_dir),
        "output_root": os.path.abspath(output_root),
        "source_file": source_name,
    }

    meta = build_meta(
        params=params,
        env=env_path,
        script=script_path,
        cfg=source_path,
        src=src_dir,
    )

    run_dir = build_version_dir(output_root, meta)
    logger = Logger(sinks=[ConsoleSink()])
    audit = Audit.create(run_dir, meta)

    try:
        logger.info(
            f'{{"event":"stage","component":"qf/01_meta","msg":"start","run_dir":"{run_dir}"}}'
        )
        logger.info(
            f'{{"event":"input","component":"qf/01_meta","msg":"source_yaml","path":"{source_path}"}}'
        )

        dst = os.path.join(run_dir, source_name)
        shutil.copy2(source_path, dst)

        logger.info(
            f'{{"event":"output","component":"qf/01_meta","msg":"copied","path":"{dst}"}}'
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
    return p


def main(argv: list[str] | None = None) -> int:
    ns = _build_parser().parse_args(argv)
    out = create_source_meta_run(
        input_dir=ns.input_dir,
        output_root=ns.output_root,
        src_dir=ns.src_dir,
        script_path=str(Path(__file__).resolve()),
        source_name=ns.source_name,
    )
    print(out.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
