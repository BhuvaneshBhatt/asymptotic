"""Probability/expectation asymptotic benchmarks."""

from .workloads import probability_tail_workload, saddle_expectation_workload


def test_moving_gaussian_tail(benchmark):
    benchmark(probability_tail_workload)


def test_saddle_expectation(benchmark):
    benchmark(saddle_expectation_workload)
