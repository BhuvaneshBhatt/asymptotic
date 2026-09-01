"""Fast CI guard for catastrophic one-process benchmark degradation."""

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


from benchmarks.stateful import run_stateful_benchmark


def test_stateful_benchmark_smoke_has_bounded_degradation_and_fallbacks():
    thresholds = tomllib.loads(Path("benchmarks/thresholds.toml").read_text())["stateful_smoke"]
    result = run_stateful_benchmark(4, measure_memory=False)
    assert result.degradation_ratio <= thresholds["max_degradation_ratio"]
    for cycle in result.cycles:
        # These workloads should stay on bounded paths.  A sudden jump in the
        # general-solver count is a stronger regression signal than wall time.
        assert cycle.metrics["general_solve_calls"] <= thresholds["max_general_solve_calls"]
        assert cycle.metrics["general_limit_calls"] <= thresholds["max_general_limit_calls"]
        assert cycle.metrics["general_integrate_calls"] <= thresholds["max_general_integrate_calls"]
        assert cycle.metrics["stat_degenerate_saddles"] >= 1
        assert cycle.metrics["stat_laplace_certs"] >= 1
        assert cycle.metrics["asymptotic_sum_saddles"] >= 1
