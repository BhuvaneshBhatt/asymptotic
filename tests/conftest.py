"""Shared pytest configuration for asymptotic tests."""

import pytest
import sympy as sp

from asymptotic.function_properties.semantics import clear_entailment_cache
from asymptotic.multivariate import clear_weight_cone_cache
from asymptotic.remainder_theorems import clear_characteristic_poly_cache


def _clear_symbolic_process_caches() -> None:
    clear_entailment_cache()
    clear_weight_cone_cache()
    clear_characteristic_poly_cache()
    sp.core.cache.clear_cache()


@pytest.fixture(autouse=True, scope="module")
def _isolate_sympy_global_cache_between_test_modules():
    """Prevent one symbolic workload from poisoning later modules' timing.

    SymPy intentionally maintains process-global expression caches.  The test
    suite mixes very different symbolic workloads, so module-level isolation
    keeps tests order-independent without paying the cost of clearing between
    every individual assertion/example.
    """

    _clear_symbolic_process_caches()
    yield
