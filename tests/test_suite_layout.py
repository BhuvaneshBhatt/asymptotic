"""Contracts for the release-suite shard layout."""

from pathlib import Path

from .suite_layout import COSTS, SHARDS


def test_every_test_module_belongs_to_one_shard():
    tests_dir = Path(__file__).resolve().parent
    expected = {str(path.relative_to(tests_dir.parent)) for path in tests_dir.glob("test_*.py")}
    assigned = [module.path for modules in SHARDS.values() for module in modules]
    assert len(assigned) == len(set(assigned))
    assert set(assigned) == expected


def test_every_module_has_known_cost_class():
    unknown = [
        module for modules in SHARDS.values() for module in modules if module.cost not in COSTS
    ]
    assert unknown == []
