# te_net_examples.io

Lightweight IO helpers for common on-disk formats used across the project.

The API focuses on a small set of pure helpers:
`format_*` returns a string representation, while `read_*`/`write_*` handle filesystem IO.

## csv

```python
from te_net_examples.io.csv import format_csv, write_csv, read_csv

s = format_csv([["a","1"], ["b","2"]], header=["k","v"])
path = write_csv("/tmp/out.csv", [["a","1"]], header=["k","v"])
header, rows = read_csv("/tmp/out.csv", has_header=True)
```

## json

```python
from te_net_examples.io.json import format_json, write_json, read_json

txt = format_json({"a": 1})
write_json("/tmp/out.json", {"a": 1})
obj = read_json("/tmp/out.json")
```

## jsonl

```python
from te_net_examples.io.jsonl import append_jsonl, write_jsonl, read_jsonl

append_jsonl("/tmp/out.jsonl", {"x": 1})
write_jsonl("/tmp/out.jsonl", [{"x": 1}, {"x": 2}])
rows = read_jsonl("/tmp/out.jsonl")
```

## yaml

```python
from te_net_examples.io.yaml import format_yaml, write_yaml, read_yaml

txt = format_yaml({"a": 1})
write_yaml("/tmp/out.yaml", {"a": 1})
obj = read_yaml("/tmp/out.yaml")
```

### parquet

```python
import pyarrow as pa
from te_net_examples.io.parquet import write_parquet, read_parquet

t = pa.table({"a": [1,2,3]})
write_parquet("/tmp/out.parquet", t, compression="zstd", compression_level=22)
t2 = read_parquet("/tmp/out.parquet")
```
