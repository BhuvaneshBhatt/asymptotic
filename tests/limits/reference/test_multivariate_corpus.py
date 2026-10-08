"""Permanent table-driven multivariate limit reference corpus."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_DATA = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "multivariate_limit_reference_cases.json"
)
_CASES = json.loads(_DATA.read_text())


def test_reference_corpus_is_unique_and_complete():
    assert len(_CASES) == 1318
    assert sum(case["multiplicity"] for case in _CASES) == 2383
    assert len({case["id"] for case in _CASES}) == len(_CASES)
    assert {case["expected_kind"] for case in _CASES} <= {
        "value",
        "dne",
        "unknown",
        "unevaluated",
    }
    assert all(
        case["expression"] and case["variables"] and case["target"] for case in _CASES
    )
    assert all("<class '" not in case.get("expected", "") for case in _CASES), (
        "reference values must be mathematical values, not serialized Python heads"
    )


def test_reference_0001_records_original_unevaluated_limit_contract():
    case = _CASES[0]
    assert case["id"] == "multivariate_reference_0001"
    assert case["expected_kind"] == "unevaluated"
    assert "expected" not in case


@pytest.mark.reference_corpus
@pytest.mark.parametrize(
    "index,case", list(enumerate(_CASES)), ids=[c["id"] for c in _CASES]
)
def test_multivariate_reference_case(index, case):
    if not case.get("executable", True):
        pytest.xfail("explicitly non-executable reference")
    root = Path(__file__).resolve().parents[3]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "tools/run_reference_corpus_shard.py"),
            "--case-index",
            str(index),
            "--timeout",
            "5",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=12,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    record = json.loads(completed.stdout.splitlines()[-1])
    if record["outcome"] in {"UNKNOWN", "KNOWN_GAP"}:
        pytest.xfail(record["detail"] or "unsupported capability")
    assert record["outcome"] == "PASS", record
