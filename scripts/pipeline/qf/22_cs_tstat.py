from __future__ import annotations

import argparse
from pathlib import Path

from te_net_examples.pipeline.qf_22_cs_tstat import run_qf_22_cs_tstat


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qf-22-cs-tstat")
    p.add_argument("input_dir")
    p.add_argument("output_root")
    p.add_argument("src_dir")
    p.add_argument("config_path")
    p.add_argument("--index-filename", default="signals_index.csv")
    return p


def main(argv: list[str] | None = None) -> int:
    ns = _build_parser().parse_args(argv)
    out = run_qf_22_cs_tstat(
        input_dir=ns.input_dir,
        output_root=ns.output_root,
        src_dir=ns.src_dir,
        config_path=ns.config_path,
        script_path=str(Path(__file__).resolve()),
        component="qf/22_cs_tstat",
        index_filename=ns.index_filename,
    )
    print(out.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
