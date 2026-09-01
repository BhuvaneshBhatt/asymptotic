"""Multivariate and nonlinear ODE benchmarks."""

from .workloads import multivariate_weight_cones, nonlinear_ode_workload


def test_weight_cones(benchmark):
    benchmark(multivariate_weight_cones)


def test_nonlinear_ode(benchmark):
    benchmark(nonlinear_ode_workload)
