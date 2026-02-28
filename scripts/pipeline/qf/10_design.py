from __future__ import annotations

import argparse
from pathlib import Path

from te_net_examples.pipeline.qf_10_design import run_qf_10_design


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qf-10-design")
    p.add_argument("input_root")
    p.add_argument("output_root")
    p.add_argument("src_dir")
    p.add_argument("config_path")
    return p


def main(argv: list[str] | None = None) -> int:
    ns = _build_parser().parse_args(argv)
    out = run_qf_10_design(
        input_root=ns.input_root,
        output_root=ns.output_root,
        src_dir=ns.src_dir,
        config_path=ns.config_path,
        script_path=str(Path(__file__).resolve()),
        component="qf/10_design",
    )
    print(out.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
