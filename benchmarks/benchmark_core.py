"""Core operation benchmarks for pytest-benchmark."""

from .workloads import (
    elementary_multiseries,
    nested_logexp,
    transseries_conversion,
)


def test_multiseries(benchmark):
    benchmark(elementary_multiseries)


def test_nested(benchmark):
    benchmark(nested_logexp)


def test_transseries(benchmark):
    benchmark(transseries_conversion)
