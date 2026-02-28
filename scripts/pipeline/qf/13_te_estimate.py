from __future__ import annotations

import argparse
from pathlib import Path

from te_net_examples.pipeline.qf_13_te_estimate import run_qf_13_te_estimate


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qf-13-te-estimate")
    p.add_argument("input_dir")
    p.add_argument("output_root")
    p.add_argument("src_dir")
    p.add_argument("config_path")
    p.add_argument("--design-filename", default="design.csv")
    return p


def main(argv: list[str] | None = None) -> int:
    ns = _build_parser().parse_args(argv)
    out = run_qf_13_te_estimate(
        input_dir=ns.input_dir,
        output_root=ns.output_root,
        src_dir=ns.src_dir,
        config_path=ns.config_path,
        script_path=str(Path(__file__).resolve()),
        component="qf/13_te_estimate",
        design_filename=ns.design_filename,
    )
    print(out.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
