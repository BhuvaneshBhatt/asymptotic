"""Run one release-test shard with isolation for costly symbolic modules."""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_layout = importlib.import_module("tests.suite_layout")
SHARDS = _layout.SHARDS
EXECUTION_BUDGET_SECONDS = _layout.EXECUTION_BUDGET_SECONDS


def _run(paths: list[str], *, timeout: int) -> int:
    env = os.environ.copy()
    env.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    command = [sys.executable, "-m", "pytest", "-q", *paths]
    try:
        return subprocess.run(
            command, cwd=ROOT, env=env, check=False, timeout=timeout
        ).returncode
    except subprocess.TimeoutExpired:
        print(f"[timeout after {timeout}s] {' '.join(paths)}", flush=True)
        return 124


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard", choices=tuple(SHARDS))
    args = parser.parse_args()
    modules = SHARDS[args.shard]

    for module in modules:
        budget = EXECUTION_BUDGET_SECONDS[module.cost]
        print(f"[{module.cost}; budget={budget}s] {module.path}", flush=True)
        if _run([module.path], timeout=budget):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
