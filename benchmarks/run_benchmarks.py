"""Dependency-light command line runner for the stateful benchmark."""

from __future__ import annotations

import argparse

from .stateful import run_stateful_benchmark


def main() -> None:
    """Run the mixed workload and optionally write machine-readable JSON."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=6)
    parser.add_argument("--json", dest="json_path")
    parser.add_argument("--no-memory", action="store_true")
    args = parser.parse_args()
    result = run_stateful_benchmark(args.cycles, measure_memory=not args.no_memory)
    for cycle in result.cycles:
        print(
            f"cycle={cycle.cycle} seconds={cycle.seconds:.6f} "
            f"peak_mb={cycle.peak_bytes / 1024**2:.2f} "
            f"general_solve={cycle.metrics['general_solve_calls']} "
            f"general_limit={cycle.metrics['general_limit_calls']}"
        )
    print(f"degradation_ratio={result.degradation_ratio:.3f}")
    if args.json_path:
        result.to_json(args.json_path)


if __name__ == "__main__":
    main()
