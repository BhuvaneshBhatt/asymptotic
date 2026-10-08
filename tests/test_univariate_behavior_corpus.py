import json
from pathlib import Path

D = Path(__file__).with_name("data")
CASES = json.loads((D / "univariate_behavior_cases.json").read_text())
FRONT = json.loads((D / "univariate_frontier_cases.json").read_text())


def test_behavioral_parity_corpus_is_category_balanced_and_neutral():
    assert CASES
    assert len({c["id"] for c in CASES}) == len(CASES)
    assert all(c["behavior"] for c in CASES)
    assert len({c["category"] for c in CASES}) == 9


def test_frontier_corpus_records_known_breadth_boundaries():
    assert len(FRONT) == 40
    assert len({c["id"] for c in FRONT}) == 40
    assert {c["category"] for c in FRONT} == {
        "oscillatory_multifrequency",
        "special_function_infinity_frontier",
        "singularity_series_frontier",
        "implicit_map_system",
    }
