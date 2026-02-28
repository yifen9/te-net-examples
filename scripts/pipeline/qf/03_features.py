from __future__ import annotations

import argparse
from pathlib import Path

from te_net_examples.pipeline.qf_03_features import run_qf_03_features


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qf-03-features")
    p.add_argument("input_dir")
    p.add_argument("output_root")
    p.add_argument("src_dir")
    p.add_argument("raw_dir")
    p.add_argument("--features_name", default="te_features_weekly.csv")
    return p


def main(argv: list[str] | None = None) -> int:
    ns = _build_parser().parse_args(argv)
    out = run_qf_03_features(
        input_dir=ns.input_dir,
        output_root=ns.output_root,
        src_dir=ns.src_dir,
        raw_dir=ns.raw_dir,
        script_path=str(Path(__file__).resolve()),
        features_name=ns.features_name,
        component="qf/03_features",
    )
    print(out.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
