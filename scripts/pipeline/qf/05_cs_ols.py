from __future__ import annotations

import argparse
from pathlib import Path

from te_net_examples.pipeline.qf_05_cs_ols import run_qf_05_cs_ols


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qf-05-cs-ols")
    p.add_argument("input_dir")
    p.add_argument("output_root")
    p.add_argument("src_dir")
    p.add_argument("--features-filename", default="features_weekly.csv")
    p.add_argument("--ret-col", default="next_week_ret")
    p.add_argument("--signals", default="nio,te_out,te_in")
    p.add_argument("--add-intercept", action="store_true")
    p.add_argument("--min-n", type=int, default=20)
    p.add_argument("--threshold", type=float, default=1.96)
    p.add_argument("--two-sided", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    ns = _build_parser().parse_args(argv)
    out = run_qf_05_cs_ols(
        input_dir=ns.input_dir,
        output_root=ns.output_root,
        src_dir=ns.src_dir,
        script_path=str(Path(__file__).resolve()),
        component="qf/05_cs_ols",
        features_filename=ns.features_filename,
        ret_col=ns.ret_col,
        signals=ns.signals,
        add_intercept=bool(ns.add_intercept),
        min_n=int(ns.min_n),
        threshold=float(ns.threshold),
        two_sided=bool(ns.two_sided),
    )
    print(out.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
