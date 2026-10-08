"""Corpus workers preserve exclusions and interrupted computations."""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_nonexecutable_row_skips_parsing(tmp_path):
    tools = tmp_path / "tools"
    data = tmp_path / "tests/data"
    tools.mkdir()
    data.mkdir(parents=True)
    for name in ("run_reference_corpus_shard.py", "reference_expression_parser.py"):
        (tools / name).write_text((ROOT / "tools" / name).read_text())
    (data / "multivariate_limit_reference_cases.json").write_text(
        json.dumps(
            [
                {
                    "id": "excluded",
                    "executable": False,
                    "expression": "invalid(",
                    "variables": [],
                    "target": [],
                    "expected_kind": "value",
                }
            ]
        )
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [
            sys.executable,
            str(tools / "run_reference_corpus_shard.py"),
            "--case-index",
            "0",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=12,
        check=True,
    )
    record = json.loads(completed.stdout.splitlines()[-1])
    assert record == {
        "id": "excluded",
        "index": 0,
        "outcome": "KNOWN_GAP",
        "detail": "explicitly non-executable reference",
        "executed": False,
    }


def test_fractional_budget_interrupts_worker():
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/run_reference_corpus_shard.py"),
            "--case-index",
            "1",
            "--timeout",
            "0.000001",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=12,
        check=True,
    )
    assert json.loads(completed.stdout.splitlines()[-1])["outcome"] == "TIMEOUT"
