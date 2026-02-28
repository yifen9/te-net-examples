from __future__ import annotations

import argparse
from pathlib import Path

from te_net_examples.pipeline.qf_08_specs import run_qf_08_specs


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qf-08-specs")
    p.add_argument("input_dir")
    p.add_argument("output_root")
    p.add_argument("src_dir")
    p.add_argument("--report-filename", default="report.csv")
    return p


def main(argv: list[str] | None = None) -> int:
    ns = _build_parser().parse_args(argv)
    out = run_qf_08_specs(
        input_dir=ns.input_dir,
        output_root=ns.output_root,
        src_dir=ns.src_dir,
        script_path=str(Path(__file__).resolve()),
        component="qf/08_specs",
        report_filename=ns.report_filename,
    )
    print(out.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
