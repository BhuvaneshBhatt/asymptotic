"""Shared pytest configuration for asymptotic tests."""

import pytest
from funcprops import clear_entailment_cache
from hypothesis import settings

from asymptotic.multivariate import clear_weight_cone_cache
from asymptotic.remainder_theorems import clear_characteristic_poly_cache

settings.register_profile("symbolic", deadline=None, max_examples=32, derandomize=True)
settings.load_profile("symbolic")


def _clear_package_caches() -> None:
    clear_entailment_cache()
    clear_weight_cone_cache()
    clear_characteristic_poly_cache()


@pytest.fixture(autouse=True, scope="module")
def _isolate_package_caches_between_test_modules():
    """Reset package-owned caches without flushing SymPy's global cache.

    Destroying a large process-global SymPy cache between symbolic test
    modules can dominate runtime.  The package caches are small, explicit, and
    sufficient to keep cache-sensitive tests independent.
    """

    _clear_package_caches()
    yield
