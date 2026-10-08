"""Isolation fixtures for the limit test hierarchy."""

import pytest
from sympy.core.cache import clear_cache


@pytest.fixture(autouse=True)
def _clear_sympy_cache_between_limit_tests():
    """Prevent process-global SymPy caches from coupling expensive limit tests."""
    yield
    clear_cache()
