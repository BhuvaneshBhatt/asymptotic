"""One-process degradation benchmark."""

from .stateful import run_stateful_benchmark


def test_mixed_workload_degradation(benchmark):
    result = benchmark(run_stateful_benchmark, 4, measure_memory=False)
    # This generous ceiling catches catastrophic process-state degradation but
    # deliberately avoids turning host-to-host timing differences into noise.
    assert result.degradation_ratio <= 5.0
