"""Fail when the capability corpus gains unexplained UNKNOWN outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default="tests/data/unknown_cause_baseline.json")
    p.add_argument("--current", required=True)
    p.add_argument("--allow-cause", action="append", default=[])
    ns = p.parse_args()
    root = Path(__file__).parents[1]
    baseline = json.loads((root / ns.baseline).read_text())
    current = json.loads((root / ns.current).read_text())
    allowed = set(ns.allow_cause)
    regressions = {}
    for cause, count in current["counts"].items():
        delta = count - baseline["counts"].get(cause, 0)
        if delta > 0 and cause not in allowed:
            regressions[cause] = delta
    total_delta = current["unknown_count"] - baseline["unknown_count"]
    if regressions or (total_delta > 0 and not allowed):
        raise SystemExit(
            f"UNKNOWN regression: total_delta={total_delta}, causes={regressions}"
        )
    print(
        json.dumps(
            {
                "baseline": baseline["unknown_count"],
                "current": current["unknown_count"],
                "delta": total_delta,
                "cause_regressions": regressions,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
