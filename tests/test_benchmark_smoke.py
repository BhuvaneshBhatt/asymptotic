"""Fast CI guard for catastrophic one-process benchmark degradation."""

import subprocess
import sys
import textwrap
from pathlib import Path


def test_stateful_benchmark_smoke_has_and_fallbacks():
    script = textwrap.dedent(
        """
        from pathlib import Path
        import tomllib

        from benchmarks.stateful import run_stateful_benchmark

        thresholds = tomllib.loads(Path("benchmarks/thresholds.toml").read_text())["stateful_smoke"]
        result = run_stateful_benchmark(4, measure_memory=False)
        assert result.degradation_ratio <= thresholds["max_degradation_ratio"]
        for cycle in result.cycles:
            assert cycle.metrics["general_solve_calls"] <= thresholds["max_general_solve_calls"]
            assert cycle.metrics["general_limit_calls"] <= thresholds["max_general_limit_calls"]
            assert cycle.metrics["general_integrate_calls"] <= thresholds["max_general_integrate_calls"]
            assert cycle.metrics["stat_degenerate_saddles"] >= 1
            assert cycle.metrics["stat_laplace_certs"] >= 1
            assert cycle.metrics["sum_saddles"] >= 1
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
