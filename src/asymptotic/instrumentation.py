"""Lightweight counters for symbolic-policy decisions and fallbacks.

The instrumentation is intentionally opt-in and uses :mod:`contextvars`, so
normal package execution pays only a single inactive-context check per
instrumented symbolic-policy operation.  Benchmarks and regression tests can
inspect how often expensive general SymPy fallbacks were actually entered
without patching SymPy at runtime.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass


@dataclass
class SymbolicMetrics:
    """Counters collected while :func:`symbolic_metrics` is active."""

    simplify_calls: int = 0
    general_simplify_calls: int = 0
    limit_calls: int = 0
    general_limit_calls: int = 0
    solve_one_calls: int = 0
    solve_system_calls: int = 0
    general_solve_calls: int = 0
    rsolve_calls: int = 0
    general_rsolve_calls: int = 0
    primitive_calls: int = 0
    general_integrate_calls: int = 0
    assumption_sign_calls: int = 0
    assumption_entails_calls: int = 0
    ask_calls: int = 0
    satisfiable_calls: int = 0
    declined_by_budget: int = 0
    zero_oracle_calls: int = 0
    parameter_strata: int = 0
    newton_cones_generated: int = 0
    unknown_remainders: int = 0
    term_products: int = 0
    stat_exact_reductions: int = 0
    stat_density_routes: int = 0
    stat_pmf_routes: int = 0
    stat_moving_routes: int = 0
    stat_laplace_saddles: int = 0
    stat_laplace_endpoints: int = 0
    stat_degenerate_saddles: int = 0
    stat_laplace_certs: int = 0
    stat_coalescing_saddles: int = 0
    stat_stirling_routes: int = 0
    binomial_tail_routes: int = 0
    asymptotic_sum_exact: int = 0
    asymptotic_sum_series: int = 0
    asymptotic_sum_parts: int = 0
    euler_maclaurin_routes: int = 0
    asymptotic_sum_mellin: int = 0
    asymptotic_sum_riemann: int = 0
    asymptotic_sum_saddles: int = 0
    sum_zeilberger: int = 0
    loggamma_normalizations: int = 0
    factorial_normalizations: int = 0
    pmf_normalizations: int = 0

    def snapshot(self) -> dict[str, int]:
        """Return a JSON-serializable copy of the current counters."""

        return asdict(self)


_ACTIVE_METRICS: ContextVar[SymbolicMetrics | None] = ContextVar(
    "asymptotic_symbolic_metrics",
    default=None,
)


def record_symbolic_event(name: str, amount: int = 1) -> None:
    """Increment one active symbolic metric without enabling global state."""

    metrics = _ACTIVE_METRICS.get()
    if metrics is None:
        return
    if not hasattr(metrics, name):
        raise ValueError(f"unknown symbolic metric: {name}")
    setattr(metrics, name, getattr(metrics, name) + amount)


@contextmanager
def symbolic_metrics() -> Iterator[SymbolicMetrics]:
    """Collect symbolic-policy counters for the duration of one context.

    Nested contexts are independent: operations contribute only to the
    innermost active collector.  The returned object remains usable after the
    context exits, which is convenient for benchmark result serialization.
    """

    metrics = SymbolicMetrics()
    token = _ACTIVE_METRICS.set(metrics)
    try:
        yield metrics
    finally:
        _ACTIVE_METRICS.reset(token)
