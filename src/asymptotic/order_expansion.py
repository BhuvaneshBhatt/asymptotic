"""Inclusive algebraic-order expansions, with logarithmic coefficients.

This is a finite-point/infinity *formal* Laurent--logarithmic service.
It does not certify the asymptotic identity independently of SymPy's series
algorithm.  A result is complete only if the requested algebraic orders have
been exposed, with an explicit asymptotic remainder or exact equality.
"""

from __future__ import annotations

import signal
import threading
from contextlib import contextmanager
from dataclasses import dataclass

import sympy as sp

from ._symbolic_policy import bounded_limit


class ExpansionBudgetExceeded(TimeoutError):
    """A symbolic expansion exceeded its best-effort evaluation budget."""


@contextmanager
def _symbolic_budget(seconds: float):
    """Use a POSIX main-thread timer without disturbing another active alarm.

    On other platforms/threads the timeout is advisory; the non-escalating
    requested series order and expression-size guard still apply.
    """
    supports_timer = (
        seconds > 0
        and threading.current_thread() is threading.main_thread()
        and hasattr(signal, "setitimer")
        and hasattr(signal, "getitimer")
        and hasattr(signal, "SIGALRM")
    )
    if not supports_timer:
        yield
        return
    existing_delay, existing_interval = signal.getitimer(signal.ITIMER_REAL)
    if existing_delay > 0 or existing_interval > 0:
        # Never override an existing timer belonging to a caller/test harness.
        yield
        return
    previous_handler = signal.getsignal(signal.SIGALRM)

    def expire(_signal_number, _frame):
        raise ExpansionBudgetExceeded(
            "asymptotic order expansion exceeded its time budget"
        )

    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


@dataclass(frozen=True)
class OrderedMultiseries:
    """Finite Laurent--logarithmic prefix, complete *formally* through ``order``.

    ``status='complete'`` means the SymPy series yielded an explicit
    small-o/higher-order remainder. It is NOT a theorem-level certificate:
    ``certified`` is False. Unknown results do not silently return a prefix.
    The coordinate ``t`` is positive and tends to 0 from above.
    """

    expr: sp.Expr
    variable: sp.Symbol
    point: sp.Expr
    order: int
    status: str
    expression: sp.Expr | None
    remainder: sp.Expr | None
    method: str
    reason: str | None = None

    @property
    def certified(self) -> bool:
        return False

    @property
    def complete(self) -> bool:
        return self.status == "complete"

    def is_complete_through(self, order: int) -> bool:
        if isinstance(order, bool) or not isinstance(order, int):
            return False
        return self.complete and order <= self.order

    def truncate(
        self, n: int | None = None, *, recursive_coefficients: int | None = None
    ) -> sp.Expr:
        if n is not None or recursive_coefficients is not None:
            raise ValueError(
                "order-targeted expansions use truncate() without a term count"
            )
        if not self.complete or self.expression is None:
            raise NotImplementedError(
                self.reason or "order-targeted expansion is unresolved"
            )
        return self.expression


def _normalize_laurent_log(prefix: sp.Expr, t: sp.Symbol, order: int):
    """Check all algebraic coefficients are polynomial in log(t).

    Reject non-expanded special functions rather than pretending that a finite
    term count means the algebraic-order coefficient has been determined.
    """
    prefix = sp.expand(sp.expand_log(prefix))
    L = sp.Dummy("L", real=True)
    included, omitted = [], []
    for term in sp.Add.make_args(prefix):
        if term == 0:
            continue
        # log(t) is an independent coefficient generator, *not* a power of t.
        ordinary = term.xreplace({sp.log(t): L})
        exponent = ordinary.as_powers_dict().get(t, sp.S.Zero)
        if not exponent.is_Rational:
            return None
        coeff = sp.cancel(ordinary / t**exponent)
        if coeff.has(t):
            return None
        # L must occur polynomially; e.g. 1/log(t) belongs to a different scale.
        try:
            sp.Poly(coeff, L, domain="EX")
        except sp.PolynomialError:
            return None
        restored = term
        if exponent <= order:
            included.append(restored)
        else:
            omitted.append(restored)
    return sp.Add(*included), sp.Add(*omitted)


def expand_to_order(
    expr: sp.Expr,
    variable: sp.Symbol,
    *,
    point: sp.Expr,
    order: int,
    timeout: float = 10.0,
    max_ops: int = 450,
) -> OrderedMultiseries:
    """Expand to inclusive algebraic order (SymPy's ``n=order+1``).

    The call is single-pass: no sparse term-count escalation or doubling.
    Only directed-real germs at finite real points and +/- infinity are
    currently handled.  ``status='unknown'`` on unsupported input/budget.
    """
    expr, point = sp.sympify(expr), sp.sympify(point)
    if not isinstance(variable, sp.Symbol):
        raise TypeError("variable must be a SymPy Symbol")
    if isinstance(order, bool) or not isinstance(order, int):
        raise TypeError("order must be an integer")
    if not -32 <= order <= 32:
        raise ValueError("order must be between -32 and 32 (bounded formal expansion)")
    if not (0 < timeout <= 120):
        raise ValueError("timeout must be in (0, 120] seconds")
    if max_ops < 1:
        raise ValueError("max_ops must be positive")

    def unknown(reason: str):
        return OrderedMultiseries(
            expr,
            variable,
            point,
            order,
            "unknown",
            None,
            None,
            "sympy_order_series",
            reason,
        )

    if sp.count_ops(expr) > max_ops:
        return unknown("expression exceeds the symbolic operation budget")
    t = sp.Dummy("_local_small", positive=True)
    if point is sp.oo:
        substitution = 1 / t
        back = 1 / variable
    elif point is -sp.oo:
        substitution = -1 / t
        back = -1 / variable
    elif point.is_real is True and point.is_finite is True:
        substitution = point + t
        back = variable - point
    else:
        return unknown("only real finite points and real infinities are supported")

    # xreplace avoids the original variable's assumptions contaminating
    # the positive one-sided local coordinate.
    local = expr.xreplace({variable: substitution})
    try:
        with _symbolic_budget(timeout):
            # Native exact Laurent--logarithmic polynomials need no expensive
            # general-purpose series algorithm or speculative term stream.
            # Avoid expanding huge integer powers just to identify a constant.
            huge_power = any(
                p.exp.is_Integer and p.exp > 64 for p in local.atoms(sp.Pow)
            )
            if not huge_power:
                direct = _normalize_laurent_log(local, t, order)
                if direct is not None:
                    kept, dropped = direct
                    return OrderedMultiseries(
                        expr,
                        variable,
                        point,
                        order,
                        "complete",
                        sp.expand(kept.xreplace({t: back})),
                        dropped.xreplace({t: back}),
                        "exact_laurent_log",
                    )

            # Complex analytic/special-function nodes (polylog, sqrt, ...)
            # are expanded only as far as mathematically requested. There is
            # no n->2*n loop or attempt to collect 8 arbitrary nonzero terms.
            result = sp.series(local, t, 0, order + 1)
            big_o = result.getO() if hasattr(result, "getO") else None
            prefix = result.removeO() if hasattr(result, "removeO") else result
            normalized = _normalize_laurent_log(prefix, t, order)
            if normalized is None:
                return unknown("series contains a non-Laurent-logarithmic coefficient")
            included, omitted = normalized
            if big_o is None:
                # A term-free / exact polynomial expansion can omit O().
                exact = sp.simplify(local - (included + omitted)) == 0
                if not exact:
                    return unknown("series returned no justified remainder")
            else:
                # A SymPy Order term must be asymptotically smaller than
                # the requested algebraic power (even with logs).
                quotient = big_o.expr / t**order
                bound = bounded_limit(quotient, t, 0, direction="+")
                if bound is None:
                    return unknown(
                        "remainder bound could not be established within symbolic policy"
                    )
                if bound != 0:
                    return unknown("remainder does not vanish beyond requested order")
            value = sp.expand(included.xreplace({t: back}))
            remainder = omitted.xreplace({t: back})
            if big_o is not None:
                # Rebase O() explicitly: xreplace of O(t**n) alone would
                # incorrectly leave an origin-centered Order at x != 0.
                remainder += sp.Order(big_o.expr.xreplace({t: back}), (variable, point))
            return OrderedMultiseries(
                expr,
                variable,
                point,
                order,
                "complete",
                value,
                remainder,
                "sympy_order_series",
            )
    except (ExpansionBudgetExceeded, TimeoutError):
        return unknown("symbolic expansion exceeded the time budget")
    except (
        TypeError,
        ValueError,
        NotImplementedError,
        sp.PoleError,
        sp.SympifyError,
        RecursionError,
    ) as exc:
        return unknown(f"{type(exc).__name__}: {exc}")
