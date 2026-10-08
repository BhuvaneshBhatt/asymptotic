"""Cost-aware planning and instrumentation for limit certificates."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import lru_cache
from time import perf_counter

import sympy as sp


@dataclass(frozen=True)
class LimitPlan:
    """Ordered proof stages selected from inexpensive structural features."""

    stages: tuple[str, ...]
    rational: bool
    singular: bool
    transcendental: bool


_COUNTS: Counter[str] = Counter()
_SECONDS: Counter[str] = Counter()
_OUTCOMES: Counter[str] = Counter()


def reset_limit_metrics():
    """Clear process-local proof-route counters and accumulated wall time."""
    _COUNTS.clear()
    _SECONDS.clear()
    _OUTCOMES.clear()


def limit_metrics():
    """Return snapshots of proof-route counts and accumulated wall time."""
    return {
        "counts": dict(_COUNTS),
        "seconds": dict(_SECONDS),
        "outcomes": dict(_OUTCOMES),
    }


@contextmanager
def measure_limit_stage(stage: str):
    """Record one invocation and its wall time without affecting semantics."""
    start = perf_counter()
    _COUNTS[stage] += 1
    try:
        yield
    finally:
        _SECONDS[stage] += perf_counter() - start


@dataclass
class LimitAnalysisContext:
    """Call-scoped memoization shared by limit providers and recursive helpers."""

    cache: dict = field(default_factory=dict)
    active: set = field(default_factory=set)
    trace: list = field(default_factory=list)

    def get(self, key, factory):
        if key not in self.cache:
            self.cache[key] = factory()
        return self.cache[key]

    def record(self, stage: str, outcome: str):
        _OUTCOMES[f"{stage}:{outcome}"] += 1
        self.trace.append((stage, outcome))


_CURRENT_CONTEXT: ContextVar[LimitAnalysisContext | None] = ContextVar(
    "limit_analysis_context", default=None
)


def current_limit_context():
    return _CURRENT_CONTEXT.get()


@contextmanager
def limit_analysis_context():
    existing = _CURRENT_CONTEXT.get()
    if existing is not None:
        yield existing
        return
    context = LimitAnalysisContext()
    token = _CURRENT_CONTEXT.set(context)
    try:
        yield context
    finally:
        _CURRENT_CONTEXT.reset(token)


@lru_cache(maxsize=4096)
def _plan_cached(
    expr: sp.Expr, variables: tuple[sp.Symbol, ...], target: tuple[sp.Expr, ...]
):
    rational = isinstance(expr, sp.Expr) and bool(expr.is_rational_function(*variables))
    transcendental = bool(expr.atoms(sp.Function))
    try:
        value = expr.subs(dict(zip(variables, target, strict=True)), simultaneous=True)
        singular = bool(value.has(sp.nan, sp.zoo, sp.oo, -sp.oo))
    except (TypeError, ValueError, NotImplementedError, RecursionError):
        singular = True
    stages = ["substitution", "algebra"]
    if singular and transcendental:
        stages.append("local_expansion")
    if rational:
        stages.extend(("exceptional_paths", "rational_valuation"))
    stages.extend(
        (
            "negative_witness",
            "fast_certificate",
            "cluster_set",
            "newton_fan",
            "semialgebraic",
        )
    )
    return LimitPlan(tuple(stages), rational, singular, transcendental)


def plan_limit(expr, variables, target) -> LimitPlan:
    """Build a cheap proof plan without running any heavyweight solver."""
    return _plan_cached(
        sp.sympify(expr), tuple(variables), tuple(map(sp.sympify, target))
    )
