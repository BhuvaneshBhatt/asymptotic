from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto

import sympy as sp

from ._symbolic_errors import SYMBOLIC_ERRORS
from ._symbolic_policy import bounded_assumption_sign, bounded_limit
from .complex_domain import ComplexBranchMetadata, ComplexSector
from .instrumentation import record_symbolic_event


class GrowthComparison(Enum):
    """Relative asymptotic growth of two expressions at a fixed germ."""

    SMALLER = auto()
    SAME_ORDER = auto()
    LARGER = auto()
    UNKNOWN = auto()


@dataclass
class AsymptoticContext:
    """Shared exact-asymptotic services.

    The implementation deliberately keeps zero tests and limit/growth queries
    centralized because they are the expensive operations in Shackell-style
    algorithms.  The optional ``exprtest`` zero oracle is preferred for nontrivial identity
    tests, with conservative SymPy fallbacks for limits, signs, and growth.
    """

    variable: sp.Symbol
    point: sp.Expr = sp.oo
    direction: str = "+"
    simplify_results: bool = True
    zero_confidence: str = "certified"
    use_sympy_zero_fallback: bool = True
    zero_oracle: Callable[..., bool | None] | None = field(default=None, repr=False)
    sector: ComplexSector | None = None
    branch: ComplexBranchMetadata | None = None
    _limit_cache: dict[sp.Expr, sp.Expr] = field(default_factory=dict, init=False)
    _zero_cache: dict[sp.Expr, bool | None] = field(default_factory=dict, init=False)
    _sign_cache: dict[sp.Expr, int | None] = field(default_factory=dict, init=False)
    _growth_cache: dict[tuple[sp.Expr, sp.Expr], tuple[GrowthComparison, sp.Expr | None]] = field(
        default_factory=dict, init=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.variable, sp.Symbol):
            raise TypeError("variable must be a SymPy Symbol")
        self.point = sp.sympify(self.point)
        if self.direction not in {"+", "-"}:
            raise ValueError("direction must be '+' or '-'")
        if self.zero_confidence not in {"certified", "probable"}:
            raise ValueError("zero_confidence must be 'certified' or 'probable'")

    def normalize(self, expr: sp.Expr) -> sp.Expr:
        expr = sp.sympify(expr)
        if not self.simplify_results:
            return expr
        # ``cancel`` is excellent for rational functions but can become very
        # expensive when applied to nested exp/log expressions.  Keep the
        # normalization structural unless the expression is rational in x.
        try:
            if expr.is_rational_function(self.variable):
                expr = sp.cancel(expr)
        except SYMBOLIC_ERRORS:
            pass
        return sp.powsimp(expr, force=False)

    def limit(self, expr: sp.Expr) -> sp.Expr:
        expr = self.normalize(expr)
        if expr in self._limit_cache:
            return self._limit_cache[expr]
        value = bounded_limit(
            expr, self.variable, self.point, direction=self.direction, allow_general=True
        )
        if value is None:
            value = sp.Limit(expr, self.variable, self.point, dir=self.direction)
        self._limit_cache[expr] = value
        return value

    def is_zero(self, expr: sp.Expr) -> bool | None:
        """Return whether *expr* is identically zero.

        ``exprtest.zerotest`` is the primary nontrivial oracle.  The default
        confidence policy is ``"certified"`` because asymptotic cancellation
        must not discard a term merely because it is probably nonzero.  A
        conservative SymPy ``equals(0)`` fallback remains available for cases
        outside exprtest's current bounded oracle.
        """

        expr = self.normalize(expr)
        if expr in self._zero_cache:
            return self._zero_cache[expr]
        if expr == 0 or expr.is_zero is True:
            result: bool | None = True
        elif expr.is_zero is False:
            result = False
        else:
            oracle = self.zero_oracle
            if oracle is None:
                try:
                    import exprtest
                except ImportError:
                    result = None
                else:
                    oracle = exprtest.zerotest
            if oracle is not None:
                record_symbolic_event("zero_oracle_calls")
                # The context already memoizes zero decisions; using the
                # backend cache would retain the same symbolic graphs twice.
                result = oracle(
                    expr,
                    use_cache=False,
                    confidence=self.zero_confidence,
                )
                if result not in (True, False, None):
                    raise TypeError("zero oracle must return True, False, or None")

            if result is None and self.use_sympy_zero_fallback:
                try:
                    eq = expr.equals(0)
                    result = eq if eq in (True, False) else None
                except SYMBOLIC_ERRORS:
                    result = None
        self._zero_cache[expr] = result
        return result

    def eventual_sign(self, expr: sp.Expr) -> int | None:
        """Determine the eventual sign at the configured germ without guessing undecidable cases."""
        expr = self.normalize(expr)
        if expr in self._sign_cache:
            return self._sign_cache[expr]

        # Rational/Laurent expressions occur constantly in local ODE and
        # transseries calculations.  Asking SymPy's generic assumptions engine
        # about their sign can recurse through polynomial real-root isolation.
        # Determine the eventual sign directly from the leading Laurent term
        # whenever possible.
        result = self._eventual_sign_rational(expr)
        if result is not None:
            self._sign_cache[expr] = result
            return result

        assumption_sign = bounded_assumption_sign(expr)
        if assumption_sign in (-1, 1):
            result = assumption_sign
        elif self.is_zero(expr) is True:
            result = 0
        else:
            lim = self.limit(expr)
            if lim.is_positive is True or lim is sp.oo:
                result = 1
            elif lim.is_negative is True or lim is -sp.oo:
                result = -1
            else:
                # Directly ask for the eventual phase/sign.  This handles
                # positive expressions tending to zero such as 1/log(x).
                try:
                    phase = self.limit(expr / sp.Abs(expr))
                except SYMBOLIC_ERRORS:
                    phase = None
                if phase == 1:
                    result = 1
                elif phase == -1:
                    result = -1
                else:
                    # A leading-term query often settles eventual sign when the
                    # ordinary limit is zero or indeterminate.
                    try:
                        lead = expr.as_leading_term(self.variable)
                        result = 1 if lead.is_positive else -1 if lead.is_negative else None
                    except SYMBOLIC_ERRORS:
                        result = None
        self._sign_cache[expr] = result
        return result

    def _eventual_sign_rational(self, expr: sp.Expr) -> int | None:
        """Fast exact sign for rational/Laurent expressions at the endpoint."""
        x = self.variable
        try:
            if self.point == 0 and self.direction == "-":
                local = sp.Dummy("_h", positive=True)
                shifted = sp.cancel(expr.subs(x, -local))
                return AsymptoticContext(local, point=0)._eventual_sign_rational(shifted)
            if self.point not in (0, sp.oo, -sp.oo):
                local = sp.Dummy("_h", positive=True)
                side = -1 if self.direction == "-" else 1
                shifted = sp.cancel(expr.subs(x, self.point + side * local))
                return AsymptoticContext(local, point=0)._eventual_sign_rational(shifted)
            if self.point in (sp.oo, -sp.oo):
                local = sp.Dummy("_h", positive=True)
                sign = 1 if self.point is sp.oo else -1
                transformed = sp.cancel(expr.subs(x, sign / local))
                return AsymptoticContext(local, point=0)._eventual_sign_rational(transformed)
            num, den = sp.fraction(sp.cancel(expr))
            pnum = sp.Poly(num, x)
            pden = sp.Poly(den, x)
            if pnum.is_zero:
                return 0

            # Near 0, the first nonzero coefficient fixes the sign because the
            # local coordinate is positive by convention.
            def first_nonzero(poly: sp.Poly):
                terms = sorted(poly.terms(), key=lambda item: item[0][0])
                return next((coeff for (_power,), coeff in terms if coeff != 0), None)

            cn = first_nonzero(pnum)
            cd = first_nonzero(pden)
            if cn is None or cd is None:
                return None
            sn = 1 if cn.is_positive is True else -1 if cn.is_negative is True else None
            sd = 1 if cd.is_positive is True else -1 if cd.is_negative is True else None
            if sn is not None and sd is not None:
                return sn * sd
        except (sp.PolynomialError, TypeError, ValueError, ZeroDivisionError):
            pass
        return None

    def compare_growth(self, f: sp.Expr, g: sp.Expr) -> tuple[GrowthComparison, sp.Expr | None]:
        """Compare |f| and |g| by the limit of f/g.

        SAME_ORDER additionally returns the finite nonzero ratio when SymPy can
        determine it.  For scale *comparability classes*, use compare_log_growth.
        """

        f = self.normalize(f)
        g = self.normalize(g)
        key = (f, g)
        if key in self._growth_cache:
            return self._growth_cache[key]
        if self.is_zero(g) is True:
            result = (GrowthComparison.UNKNOWN, None)
        else:
            raw_ratio = f / g
            if raw_ratio.has(sp.gamma, sp.factorial):
                try:
                    raw_ratio = sp.combsimp(raw_ratio)
                except SYMBOLIC_ERRORS:
                    pass
            ratio = self.limit(sp.Abs(raw_ratio))
            if ratio == 0:
                result = (GrowthComparison.SMALLER, sp.S.Zero)
            elif ratio is sp.oo:
                result = (GrowthComparison.LARGER, sp.oo)
            elif ratio.is_finite is True and ratio.is_zero is False:
                result = (GrowthComparison.SAME_ORDER, ratio)
            else:
                result = (GrowthComparison.UNKNOWN, None)
        self._growth_cache[key] = result
        return result

    def compare_log_growth(self, f: sp.Expr, g: sp.Expr) -> tuple[GrowthComparison, sp.Expr | None]:
        """Compare comparability classes of positive functions tending to 0/∞.

        For vanishing scale elements this uses |log(f)| / |log(g)|.  A finite
        nonzero limit means equal comparability class; 0 and ∞ give the order.
        """

        f = self.normalize(f)
        g = self.normalize(g)
        try:
            ratio = self.limit(sp.Abs(sp.log(sp.Abs(f))) / sp.Abs(sp.log(sp.Abs(g))))
        except SYMBOLIC_ERRORS:
            return (GrowthComparison.UNKNOWN, None)
        if ratio == 0:
            return (GrowthComparison.SMALLER, sp.S.Zero)
        if ratio is sp.oo:
            return (GrowthComparison.LARGER, sp.oo)
        if ratio.is_finite is True and ratio.is_zero is False:
            return (GrowthComparison.SAME_ORDER, ratio)
        return (GrowthComparison.UNKNOWN, None)


def context_for(
    variable: sp.Symbol,
    point: sp.Expr = sp.oo,
    context: AsymptoticContext | None = None,
) -> AsymptoticContext:
    """Return a context for one germ and reject mismatched injected contexts."""

    point = sp.sympify(point)
    if context is None:
        return AsymptoticContext(variable, point)
    if context.variable != variable or context.point != point:
        raise ValueError("asymptotic context uses different coordinates")
    return context
