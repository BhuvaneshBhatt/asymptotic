"""Fast cold/warm/mixed process-shape checks for representative core workloads."""

from __future__ import annotations

import time

import sympy as sp

from asymptotic.instrumentation import symbolic_metrics
from benchmarks.workloads import (
    elementary_multiseries,
    implicit_workload,
    reversion_workload,
    transseries_conversion,
)


def _clear_caches():
    from asymptotic.function_properties.semantics import clear_entailment_cache
    from asymptotic.multivariate import clear_weight_cone_cache
    from asymptotic.remainder_theorems import clear_characteristic_poly_cache

    sp.core.cache.clear_cache()
    clear_entailment_cache()
    clear_characteristic_poly_cache()
    clear_weight_cone_cache()


def _profile_once():
    start = time.perf_counter()
    with symbolic_metrics() as metrics:
        reversion_workload(4)
        implicit_workload(3)
    return time.perf_counter() - start, metrics.snapshot()


def test_core_workloads_cold_warm_mixed_profiles_keep_bounded_routes():
    _clear_caches()
    cold_t, cold = _profile_once()
    warm_t, warm = _profile_once()

    elementary_multiseries(4)
    transseries_conversion(4)
    mixed_t, mixed = _profile_once()

    for profile in (cold, warm, mixed):
        assert profile["general_rsolve_calls"] == 0
        assert profile["general_integrate_calls"] == 0
        assert profile["general_limit_calls"] <= 4
        assert profile["general_solve_calls"] <= 4

    # Only a catastrophe guard.  Structural fallback ceilings above are the
    # portable performance contract.
    baseline = max(cold_t, warm_t, 1e-6)
    assert mixed_t <= 12 * baseline
