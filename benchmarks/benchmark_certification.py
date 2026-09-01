"""Remainder and Green/Frechet certification benchmarks."""

from .workloads import green_certificate_workload, remainder_certificate_workload


def test_green(benchmark):
    benchmark(green_certificate_workload)


def test_remainder(benchmark):
    benchmark(remainder_certificate_workload)
