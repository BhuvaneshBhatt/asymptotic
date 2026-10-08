"""Deterministic metamorphic regression layer generated from curated seeds."""

import json
from collections import Counter
from pathlib import Path

DATA = Path(__file__).with_name("data") / "multivariate_limit_metamorphic_cases.json"
ROWS = json.loads(DATA.read_text())


def test_metamorphic_layer_has_expected_size_and_unique_ids():
    assert len(ROWS) == 960
    assert len({r["id"] for r in ROWS}) == 960
    assert len({r["seed_id"] for r in ROWS}) == 320


def test_each_seed_has_exactly_three_replayable_mutations():
    counts = Counter(r["seed_id"] for r in ROWS)
    assert set(counts.values()) == {3}


def test_metamorphic_oracles_are_explicit():
    assert all(r["oracle"] == "SAME_RESULT" for r in ROWS)
    assert all(r["transformation"] for r in ROWS)
