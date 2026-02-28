from __future__ import annotations

import argparse
from pathlib import Path

from te_net_examples.pipeline.qf_07_compare import run_qf_07_compare


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qf-07-compare")
    p.add_argument("input_root")
    p.add_argument("output_root")
    p.add_argument("src_dir")
    p.add_argument("--n-latest", type=int, default=10)
    return p


def main(argv: list[str] | None = None) -> int:
    ns = _build_parser().parse_args(argv)
    out = run_qf_07_compare(
        input_root=ns.input_root,
        output_root=ns.output_root,
        src_dir=ns.src_dir,
        script_path=str(Path(__file__).resolve()),
        component="qf/07_compare",
        n_latest=int(ns.n_latest),
    )
    print(out.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
