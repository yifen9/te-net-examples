from __future__ import annotations

import argparse
from pathlib import Path

from te_net_examples.pipeline.qf_80_paper_artifacts import run_qf_80_paper_artifacts


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qf-80-paper-artifacts")
    p.add_argument("input_dir_metrics")
    p.add_argument("input_dir_report_figures")
    p.add_argument("input_dir_cs_tstat")
    p.add_argument("input_dir_report_signal")
    p.add_argument("output_root")
    p.add_argument("src_dir")
    p.add_argument("config_path")
    p.add_argument("--component", default="qf/80_paper_artifacts")
    return p


def main(argv: list[str] | None = None) -> int:
    ns = _build_parser().parse_args(argv)
    out = run_qf_80_paper_artifacts(
        input_dir_metrics=ns.input_dir_metrics,
        input_dir_report_figures=ns.input_dir_report_figures,
        input_dir_cs_tstat=ns.input_dir_cs_tstat,
        input_dir_report_signal=ns.input_dir_report_signal,
        output_root=ns.output_root,
        src_dir=ns.src_dir,
        config_path=ns.config_path,
        script_path=str(Path(__file__).resolve()),
        component=str(ns.component),
    )
    print(out.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
