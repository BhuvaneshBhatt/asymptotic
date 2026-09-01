"""Parameter-stratification benchmark."""

from .workloads import parameter_strata_workload


def test_parameter_strata(benchmark):
    benchmark(parameter_strata_workload)
