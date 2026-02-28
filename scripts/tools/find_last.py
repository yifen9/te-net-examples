from __future__ import annotations

import argparse
from datetime import datetime

from te_net_examples.utils.versioner import index_dir, list_metas


def _ts_key(m: dict) -> datetime:
    ts = m.get("timestamp")
    if not isinstance(ts, str) or not ts:
        raise ValueError("meta missing timestamp")
    return datetime.fromisoformat(ts)


def main() -> int:
    ap = argparse.ArgumentParser(prog="find_last")
    ap.add_argument("base_dir")
    args = ap.parse_args()

    metas = list_metas(args.base_dir)
    if not metas:
        raise FileNotFoundError(f"no version dirs under {args.base_dir}")

    metas.sort(key=_ts_key, reverse=True)
    last_meta = metas[0]

    d = index_dir(args.base_dir, last_meta)
    if d is None:
        raise FileNotFoundError(f"cannot index last meta under {args.base_dir}")

    print(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
