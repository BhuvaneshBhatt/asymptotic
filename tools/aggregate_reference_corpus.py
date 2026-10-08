from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("directory")
ap.add_argument("--expected", type=int, default=1168)
ap.add_argument("--output")
ns = ap.parse_args()
rows = []
for p in sorted(Path(ns.directory).glob("shard_*.json")):
    rows.extend(json.loads(p.read_text()))
by = {r["id"]: r for r in rows}
rows = sorted(by.values(), key=lambda r: r.get("index", 10**9))
counts = collections.Counter(r["outcome"] for r in rows)
report = {
    "completed": len(rows),
    "expected": ns.expected,
    "complete": len(rows) == ns.expected,
    "counts": dict(counts),
    "nonpass": [r for r in rows if r["outcome"] != "PASS"],
}
if ns.output:
    Path(ns.output).write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k != "nonpass"}, indent=2))
