from __future__ import annotations

import argparse
from pathlib import Path

from te_net_examples.pipeline.qf_02_universe import run_qf_02_universe


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qf-02-universe")
    p.add_argument("input_dir")
    p.add_argument("output_root")
    p.add_argument("src_dir")
    p.add_argument("raw_dir")
    p.add_argument("--universe-name", default="universe_500.csv")
    return p


def main(argv: list[str] | None = None) -> int:
    ns = _build_parser().parse_args(argv)
    out = run_qf_02_universe(
        input_dir=ns.input_dir,
        output_root=ns.output_root,
        src_dir=ns.src_dir,
        raw_dir=ns.raw_dir,
        script_path=str(Path(__file__).resolve()),
        universe_name=ns.universe_name,
        component="qf/02_universe",
    )
    print(out.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
