"""The reference tests share the audit runner's isolation and comparison contracts."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "tests/data/univariate_limit_reference_cases.json"
CASES = json.loads(DATA.read_text())


def test_reference_corpus_is_complete_and_neutral():
    assert len(CASES) == 3536
    assert sum(case["multiplicity"] for case in CASES) == 5128
    assert len({case["id"] for case in CASES}) == len(CASES)
    assert all(
        case["variable"] and case["expression"] and case["point"] for case in CASES
    )
    assert {case["expected_kind"] for case in CASES} <= {
        "value",
        "dne",
        "unevaluated",
        "stratified",
        "cluster_set",
    }


@pytest.mark.reference_corpus
@pytest.mark.parametrize(
    "index,case", list(enumerate(CASES)), ids=[case["id"] for case in CASES]
)
def test_univariate_reference_case(index, case):
    if not case.get("executable", True):
        pytest.skip("Explicitly excluded row remains in the machine-readable audit.")
    spec = importlib.util.spec_from_file_location(
        "corpus_runner", ROOT / "tools/audit_univariate_corpus.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    record = runner.run_one(index, 5, 12)
    if record["classification"] == "unsupported_capability":
        pytest.xfail(record.get("detail", "Unresolved capability; retained in audit."))
    assert record["classification"] == "pass", json.dumps(record, indent=2)
