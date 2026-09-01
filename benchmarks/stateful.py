"""Stateful mixed-workload benchmark and JSON result generation."""

from __future__ import annotations

import gc
import json
import resource
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import sympy as sp

from asymptotic.instrumentation import symbolic_metrics

from .workloads import MIXED_WORKLOADS


@dataclass(frozen=True)
class CycleResult:
    """Timing, memory, and symbolic counters for one heterogeneous cycle."""

    cycle: int
    seconds: float
    peak_bytes: int
    metrics: dict[str, int]


@dataclass(frozen=True)
class StatefulBenchmarkResult:
    """Serializable result of a repeated heterogeneous benchmark."""

    cycles: tuple[CycleResult, ...]
    degradation_ratio: float

    def to_json(self, path: str | Path) -> None:
        """Write benchmark data in stable, human-readable JSON format."""

        payload = {
            "degradation_ratio": self.degradation_ratio,
            "cycles": [asdict(cycle) for cycle in self.cycles],
        }
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run_stateful_benchmark(
    cycles: int = 6,
    *,
    measure_memory: bool = True,
) -> StatefulBenchmarkResult:
    """Run heterogeneous workloads repeatedly in one interpreter.

    The ratio compares the median of the last two cycles to the median of the
    first two.  Absolute timings are intentionally left as observations rather
    than correctness assertions because benchmark hosts vary substantially.
    """

    if cycles < 4:
        raise ValueError("stateful benchmark requires at least four cycles")
    results = []
    for cycle in range(cycles):
        gc.collect()
        sp.core.cache.clear_cache()
        start = time.perf_counter()
        with symbolic_metrics() as metrics:
            for workload in MIXED_WORKLOADS:
                workload()
        seconds = time.perf_counter() - start
        if measure_memory:
            max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # Linux reports KiB; macOS reports bytes.
            peak = int(max_rss if sys.platform == "darwin" else max_rss * 1024)
        else:
            peak = 0
        results.append(CycleResult(cycle, seconds, peak, metrics.snapshot()))

    first = statistics.median(item.seconds for item in results[:2])
    last = statistics.median(item.seconds for item in results[-2:])
    ratio = last / first if first else float("inf")
    return StatefulBenchmarkResult(tuple(results), ratio)
