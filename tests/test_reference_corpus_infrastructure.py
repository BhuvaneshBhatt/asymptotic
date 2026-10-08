import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _module():
    p = Path(__file__).parents[1] / "tools" / "reference_corpus.py"
    spec = importlib.util.spec_from_file_location("reference_corpus", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_jsonl_is_incremental_and_restartable(tmp_path):
    m = _module()
    p = tmp_path / "shard.jsonl"
    m.append_atomic(p, {"index": 1, "outcome": "PASS"})
    m.append_atomic(p, {"index": 61, "outcome": "TIMEOUT"})
    assert m.load_jsonl(p) == [
        {"index": 1, "outcome": "PASS"},
        {"index": 61, "outcome": "TIMEOUT"},
    ]


def test_shard_partition_is_exact():
    corpus = json.loads(
        (
            Path(__file__).parent / "data" / "multivariate_limit_reference_cases.json"
        ).read_text()
    )
    parts = [[i for i in range(len(corpus)) if i % 60 == s] for s in range(60)]
    flat = [i for p in parts for i in p]
    assert len(corpus) == 1318
    assert sorted(flat) == list(range(1318))
    assert len(flat) == len(set(flat))


def test_aggregate_reports_complete_unique_coverage():
    m = _module()
    corpus = json.loads(
        (
            Path(__file__).parent / "data" / "multivariate_limit_reference_cases.json"
        ).read_text()
    )
    rows = [
        {"index": i, "id": case["id"], "outcome": "PASS"}
        for i, case in enumerate(corpus)
    ]
    report = m.aggregate_rows(rows, expected=len(corpus))
    assert report["total_completed"] == report["total_expected"] == 1318
    assert report["missing_count"] == 0
    assert report["duplicate_indices"] == []


def test_aggregate_detects_conflicting_duplicate_indices():
    m = _module()
    rows = [
        {"index": 7, "id": "case", "outcome": "PASS"},
        {"index": 7, "id": "case", "outcome": "WRONG"},
    ]
    report = m.aggregate_rows(rows, expected=10)
    assert report["duplicate_indices"] == [7]
    assert report["conflicting_duplicate_indices"] == [7]
    assert report["missing_count"] == 9


@pytest.mark.parametrize("index", [-1, 1, True, "0", None])
def test_aggregate_rejects_invalid_index(index):
    with pytest.raises(ValueError, match="reference index outside the corpus"):
        _module().aggregate_rows([{"index": index, "outcome": "PASS"}], expected=1)


def test_aggregate_rejects_invalid_outcome():
    with pytest.raises(ValueError, match="invalid reference outcome"):
        _module().aggregate_rows([{"index": 0, "outcome": "NONSENSE"}], expected=1)


def test_aggregate_checks_case_identity():
    with pytest.raises(ValueError, match="reference ID does not match"):
        _module().aggregate_rows(
            [{"index": 0, "id": "stale", "outcome": "PASS"}],
            expected=1,
            expected_ids=["current"],
        )


def test_fresh_run_replaces_prior_records(tmp_path):
    module = _module()
    cases = json.loads(module.CORPUS.read_text())
    out = tmp_path / "shard.jsonl"
    module.append_atomic(out, {"index": 1, "id": "stale", "outcome": "WRONG"})
    module.run(
        SimpleNamespace(
            output=str(out),
            resume=False,
            shard=1,
            shards=len(cases),
            timeout=5,
            hard_grace=2,
            pythonpath=[],
        )
    )
    rows = module.load_jsonl(out)
    assert len(rows) == 1
    assert rows[0]["index"] == rows[0]["shard"] == 1
    assert rows[0]["id"] == cases[1]["id"]
    assert rows[0]["outcome"] == "PASS"
