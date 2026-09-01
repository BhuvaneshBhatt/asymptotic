"""Scaling benchmarks over mathematical rather than textual complexity."""

import pytest

from .workloads import (
    elementary_multiseries,
    implicit_workload,
    reversion_workload,
    transseries_product_workload,
)


@pytest.mark.parametrize("terms", [5, 10, 20])
def test_multiseries_term_scaling(benchmark, terms):
    benchmark(elementary_multiseries, terms)


@pytest.mark.parametrize("terms", [4, 6, 8])
def test_reversion_term_scaling(benchmark, terms):
    benchmark(reversion_workload, terms)


@pytest.mark.parametrize("terms", [3, 4, 5])
def test_implicit_term_scaling(benchmark, terms):
    benchmark(implicit_workload, terms)


@pytest.mark.parametrize("size", [4, 8, 12])
def test_transseries_product_quadratic_candidate_scaling(benchmark, size):
    benchmark(transseries_product_workload, size)
