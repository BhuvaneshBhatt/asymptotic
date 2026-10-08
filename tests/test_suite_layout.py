"""Contracts for the release-suite shard layout."""

from pathlib import Path

from .suite_layout import COSTS, EXECUTION_BUDGET_SECONDS, SHARDS


def test_every_test_module_belongs_to_one_shard():
    tests_dir = Path(__file__).resolve().parent
    expected = {
        f"tests/{path.relative_to(tests_dir).as_posix()}"
        for path in tests_dir.rglob("test_*.py")
    }
    assigned = [module.path for modules in SHARDS.values() for module in modules]
    assert len(assigned) == len(set(assigned))
    assert set(assigned) == expected


def test_every_module_has_known_cost_class():
    unknown = [
        module
        for modules in SHARDS.values()
        for module in modules
        if module.cost not in COSTS
    ]
    assert unknown == []


def test_every_cost_class_has_an_execution_budget():
    assert set(EXECUTION_BUDGET_SECONDS) == COSTS
    assert all(seconds > 0 for seconds in EXECUTION_BUDGET_SECONDS.values())


def test_reference_corpus_is_explicitly_budgeted():
    modules = [module for group in SHARDS.values() for module in group]
    for path in (
        "tests/limits/reference/test_multivariate_corpus.py",
        "tests/limits/reference/test_univariate_corpus.py",
    ):
        corpus = next(module for module in modules if module.path == path)
        assert corpus.cost == "corpus"
        assert EXECUTION_BUDGET_SECONDS[corpus.cost] >= 3600
