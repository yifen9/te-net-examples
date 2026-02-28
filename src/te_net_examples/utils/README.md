# te_net_examples.utils

Runtime utilities for logging, auditing, reproducibility metadata, and run directory versioning.

The core design is "structured message + pluggable sinks":
application code emits `Message`s through `Logger`, while sinks decide how to render/persist them.

## logger — Message fan-out

```python
from te_net_examples.utils.logger import Logger
from te_net_examples.utils.console import ConsoleSink

logger = Logger(sinks=[ConsoleSink(transient=False)])
logger.info("hello")
logger.warn("warn")
logger.error("error")
```

## message — Timestamped log message

```python
from te_net_examples.utils.message import Level, make_message

msg = make_message(Level.INFO, "text")
# msg.level, msg.text, msg.timestamp
```

## console — Rich console sink

`ConsoleSink` prints human-readable logs and consumes `event=progress` payloads to show a progress bar.

```python
import json
from te_net_examples.utils.logger import Logger
from te_net_examples.utils.console import ConsoleSink

logger = Logger(sinks=[ConsoleSink(transient=True)])
logger.info(json.dumps({"event":"app","msg":"start"}))
```

## progress — Progress event emitter

`Progress` emits `event=progress` JSON through `Logger`, which can be consumed by `ConsoleSink` and `Audit`.

```python
from te_net_examples.utils.logger import Logger
from te_net_examples.utils.console import ConsoleSink
from te_net_examples.utils.progress import Progress

logger = Logger(sinks=[ConsoleSink()])
p = Progress(logger=logger, name="task", total=100)
p.start()
p.step(10)
p.finish()
```

## audit — Run audit state + segmented log

`Audit` maintains a run-local `_audit.json` and segmented `_log/*.jsonl`.

It parses message text as JSON when possible and stores progress snapshots.

```python
from te_net_examples.utils.audit import Audit
from te_net_examples.utils.logger import Logger

meta = {"fingerprint": "0123456789abcdef"}
audit = Audit.create("/path/to/run_dir", meta)

logger = Logger(sinks=[audit])
logger.info('{"event":"app","msg":"start"}')
audit.finish_success()
```

## meta — Reproducibility metadata (fingerprint)

`build_meta(...)` hashes parameters and selected environment inputs to produce a compact fingerprint.

```python
from te_net_examples.utils.meta import build_meta

meta = build_meta(
    params={"name":"exp1"},
    env="/path/to/env.txt",
    script="/path/to/run.py",
    src="/path/to/src_dir",
)
# meta["fingerprint"], meta["sha"]
```

## versioner — Versioned run directory

`build_version_dir` creates a timestamp+fingerprint directory and writes `_meta.json`.

```python
from te_net_examples.utils.versioner import build_version_dir

run_dir = build_version_dir("/path/to/output_root", meta)
# /path/to/output_root/<ts>_<fp>/_meta.json
```

## lineage — Trace run ancestry

`trace(...)` follows `params[field]` (default: `input_dir`) in `_meta.json` to reconstruct a run chain.

```python
from te_net_examples.utils.lineage import trace

nodes = trace("/path/to/run_dir", field="input_dir")
# nodes[i].run_dir, nodes[i].prev_run_dir
```
