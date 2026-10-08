"""Opt-in timing attribution for MRV/Hardy Newton lifting."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from time import perf_counter


@dataclass
class MRVTiming:
    """Timing and call counts for Newton-lifting stages."""

    seconds: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    calls: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    max_depth: int = 0
    recursive_calls: int = 0

    def snapshot(self) -> dict[str, object]:
        return {
            "seconds": dict(self.seconds),
            "calls": dict(self.calls),
            "max_depth": self.max_depth,
            "recursive_calls": self.recursive_calls,
        }


_ACTIVE: ContextVar[MRVTiming | None] = ContextVar("mrv_hardy_timing", default=None)


@contextmanager
def mrv_hardy_timing() -> Iterator[MRVTiming]:
    """Collect MRV/Hardy stage timings without changing solver semantics."""
    timing = MRVTiming()
    token = _ACTIVE.set(timing)
    try:
        yield timing
    finally:
        _ACTIVE.reset(token)


@contextmanager
def timed_mrv_stage(stage: str) -> Iterator[None]:
    """Attribute elapsed time to one active MRV/Hardy stage."""
    timing = _ACTIVE.get()
    if timing is None:
        yield
        return
    timing.calls[stage] += 1
    start = perf_counter()
    try:
        yield
    finally:
        timing.seconds[stage] += perf_counter() - start


def record_mrv_recursion(depth: int) -> None:
    """Record one recursive Newton-lifting invocation."""
    timing = _ACTIVE.get()
    if timing is None:
        return
    timing.recursive_calls += 1
    timing.max_depth = max(timing.max_depth, depth)
