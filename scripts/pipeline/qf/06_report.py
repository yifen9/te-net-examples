from __future__ import annotations

import argparse
from pathlib import Path

from te_net_examples.pipeline.qf_06_report import run_qf_06_report


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qf-06-report")
    p.add_argument("input_dir_portfolio_sort")
    p.add_argument("input_dir_cs")
    p.add_argument("output_root")
    p.add_argument("src_dir")
    p.add_argument("--portfolio-sort-filename", default="portfolio_sort.csv")
    p.add_argument("--cs-filename", default="cs_ols.csv")
    return p


def main(argv: list[str] | None = None) -> int:
    ns = _build_parser().parse_args(argv)
    out = run_qf_06_report(
        input_dir_portfolio_sort=ns.input_dir_portfolio_sort,
        input_dir_cs=ns.input_dir_cs,
        output_root=ns.output_root,
        src_dir=ns.src_dir,
        script_path=str(Path(__file__).resolve()),
        component="qf/06_report",
        portfolio_sort_filename=ns.portfolio_sort_filename,
        cs_filename=ns.cs_filename,
    )
    print(out.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
