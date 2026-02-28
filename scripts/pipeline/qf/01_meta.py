from __future__ import annotations

import argparse
from pathlib import Path

from te_net_examples.tools.source_meta import create_source_meta_run


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qf-01-meta")
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
