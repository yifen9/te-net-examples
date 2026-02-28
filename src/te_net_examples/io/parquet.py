from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq


def format_parquet(table: pa.Table) -> str:
    return str(table.schema)


def write_parquet(
    path: str,
    table: pa.Table,
    *,
    compression: str = "zstd",
    compression_level: int | None = 22,
) -> str:
    pq.write_table(
        table,
        path,
        compression=compression,
        compression_level=compression_level,
    )
    return path


def read_parquet(path: str) -> pa.Table:
    return pq.read_table(path)
