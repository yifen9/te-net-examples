from __future__ import annotations

import argparse
from pathlib import Path

from te_net_examples.pipeline.qf_19_oracle_vs_estimated import (
    run_qf_19_oracle_vs_estimated,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qf-19-oracle-vs-estimated")
    p.add_argument("estimated_dir")
    p.add_argument("oracle_dir")
    p.add_argument("output_root")
    p.add_argument("src_dir")
    p.add_argument("config_path")
    p.add_argument("--design-filename", default="design.csv")
    return p


def main(argv: list[str] | None = None) -> int:
    ns = _build_parser().parse_args(argv)
    out = run_qf_19_oracle_vs_estimated(
        estimated_dir=ns.estimated_dir,
        oracle_dir=ns.oracle_dir,
        output_root=ns.output_root,
        src_dir=ns.src_dir,
        config_path=ns.config_path,
        script_path=str(Path(__file__).resolve()),
        component="qf/19_oracle_vs_estimated",
        design_filename=ns.design_filename,
    )
    print(out.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
