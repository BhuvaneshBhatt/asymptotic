#!/usr/bin/env python3
"""Run one release-test shard with isolation for costly symbolic modules."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from tests.suite_layout import SHARDS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _run(paths: list[str]) -> int:
    env = os.environ.copy()
    env.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    command = [sys.executable, "-m", "pytest", "-q", *paths]
    return subprocess.run(command, cwd=ROOT, env=env, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard", choices=tuple(SHARDS))
    args = parser.parse_args()
    modules = SHARDS[args.shard]

    shared = [module.path for module in modules if module.cost in {"cheap", "moderate"}]
    if shared and _run(shared):
        return 1

    for module in modules:
        if module.cost in {"expensive", "stateful"}:
            print(f"[{module.cost}] {module.path}", flush=True)
            if _run([module.path]):
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
