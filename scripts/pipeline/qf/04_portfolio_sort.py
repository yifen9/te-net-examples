from __future__ import annotations

import argparse
from pathlib import Path

from te_net_examples.pipeline.qf_04_portfolio_sort import run_qf_04_portfolio_sort


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qf-04-portfolio_sort")
    p.add_argument("input_dir")
    p.add_argument("output_root")
    p.add_argument("src_dir")
    p.add_argument("--features-filename", default="features_weekly.csv")
    p.add_argument("--signal-col", default="nio")
    p.add_argument("--ret-col", default="next_week_ret")
    p.add_argument("--n-groups", type=int, default=5)
    p.add_argument("--min-per-group", type=int, default=5)
    p.add_argument("--annualization-factor", type=float, default=52.0)
    p.add_argument("--full-start", default="2021-01-01")
    p.add_argument("--full-end", default="2025-12-31")
    p.add_argument("--sub1-start", default="2021-01-01")
    p.add_argument("--sub1-end", default="2023-12-31")
    p.add_argument("--sub2-start", default="2024-01-01")
    p.add_argument("--sub2-end", default="2025-12-31")
    return p


def main(argv: list[str] | None = None) -> int:
    ns = _build_parser().parse_args(argv)
    out = run_qf_04_portfolio_sort(
        input_dir=ns.input_dir,
        output_root=ns.output_root,
        src_dir=ns.src_dir,
        script_path=str(Path(__file__).resolve()),
        component="qf/04_portfolio_sort",
        features_filename=ns.features_filename,
        signal_col=ns.signal_col,
        ret_col=ns.ret_col,
        n_groups=ns.n_groups,
        min_per_group=ns.min_per_group,
        annualization_factor=ns.annualization_factor,
        full_start=ns.full_start,
        full_end=ns.full_end,
        sub1_start=ns.sub1_start,
        sub1_end=ns.sub1_end,
        sub2_start=ns.sub2_start,
        sub2_end=ns.sub2_end,
    )
    print(out.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
