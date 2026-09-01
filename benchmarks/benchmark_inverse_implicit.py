"""Inverse and implicit asymptotics benchmarks."""

from .workloads import implicit_workload, reversion_workload


def test_reversion(benchmark):
    benchmark(reversion_workload)


def test_implicit(benchmark):
    benchmark(implicit_workload)
