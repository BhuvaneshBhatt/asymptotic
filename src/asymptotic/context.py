from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto

import sympy as sp
from funcprops import normalize_assumptions

from ._polynomial_bounds import bounded_degree
from ._symbolic_errors import SYMBOLIC_ERRORS
from ._symbolic_policy import (
    bounded_assumption_entails,
    bounded_assumption_sign,
    bounded_limit,
)
from .complex_domain import ComplexBranchMetadata, ComplexSector
from .instrumentation import record_symbolic_event


def _rational_polynomial_zero(expression, variable):
    """Decide a small rational-polynomial identity without assumption search."""
    if (
        expression.free_symbols - {variable}
        or expression.has(sp.Float)
        or sp.count_ops(expression) > 160
        or bounded_degree(expression, (variable,), 64) is None
    ):
        return None
    try:
        polynomial = sp.Poly(expression, variable, domain=sp.QQ)
    except (sp.PolynomialError, sp.CoercionFailed):
        return None
    return polynomial.is_zero


class AsymptoticGrowthComparison(Enum):
    """Relative growth of two expressions near an asymptotic point."""

    SMALLER = auto()
    SAME_ORDER = auto()
    LARGER = auto()
    UNKNOWN = auto()


@dataclass
class AsymptoticContext:
    """Shared exact-asymptotic services.

    The implementation keeps zero tests and limit/growth queries
    centralized because they are the expensive operations in Shackell-style
    algorithms.  The optional ``exprtest`` zero oracle is preferred for nontrivial identity
    tests, with bounded SymPy fallbacks for limits, signs, and growth.
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
    assumptions: sp.Expr = sp.S.true
    _normalize_cache: dict[sp.Expr, sp.Expr] = field(default_factory=dict, init=False)
    _entailment_cache: dict[sp.Expr, bool | None] = field(
        default_factory=dict, init=False
    )
    _applicability_cache: dict[object, bool | None] = field(
        default_factory=dict, init=False
    )
    _cache_hits: dict[str, int] = field(default_factory=dict, init=False)
    _cache_misses: dict[str, int] = field(default_factory=dict, init=False)
    _limit_cache: dict[sp.Expr, sp.Expr] = field(default_factory=dict, init=False)
    _zero_cache: dict[sp.Expr, bool | None] = field(default_factory=dict, init=False)
    _sign_cache: dict[sp.Expr, int | None] = field(default_factory=dict, init=False)
    _growth_cache: dict[
        tuple[sp.Expr, sp.Expr], tuple[AsymptoticGrowthComparison, sp.Expr | None]
    ] = field(default_factory=dict, init=False)
    _analysis_cache: dict[tuple[str, object], object] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.variable, sp.Symbol):
            raise TypeError("variable must be a SymPy Symbol")
        self.point = sp.sympify(self.point)
        if self.direction not in {"+", "-"}:
            raise ValueError("direction must be '+' or '-'")
        self.assumptions = normalize_assumptions(self.assumptions)
        if self.zero_confidence not in {"certified", "probable"}:
            raise ValueError("zero_confidence must be 'certified' or 'probable'")

    def _cache_event(self, name: str, hit: bool) -> None:
        target = self._cache_hits if hit else self._cache_misses
        target[name] = target.get(name, 0) + 1

    def cache_metrics(self) -> dict[str, dict[str, int]]:
        return {"hits": dict(self._cache_hits), "misses": dict(self._cache_misses)}

    def cached_analysis(
        self, namespace: str, key: object, compute: Callable[[], object]
    ):
        """Memoize a context-local structural analysis result."""
        cache_key = (namespace, key)
        if cache_key in self._analysis_cache:
            self._cache_event(namespace, True)
            return self._analysis_cache[cache_key]
        self._cache_event(namespace, False)
        value = compute()
        self._analysis_cache[cache_key] = value
        return value

    def normalize(self, expr: sp.Expr) -> sp.Expr:
        expr = sp.sympify(expr)
        original = expr
        if original in self._normalize_cache:
            self._cache_event("normalize", True)
            return self._normalize_cache[original]
        self._cache_event("normalize", False)
        if not self.simplify_results:
            self._normalize_cache[expr] = expr
            return expr
        # ``cancel`` is excellent for rational functions but can become very
        # expensive when applied to nested exp/log expressions.  Keep the
        # normalization structural unless the expression is rational in x.
        try:
            if expr.is_rational_function(self.variable):
                expr = sp.cancel(expr)
        except SYMBOLIC_ERRORS:
            pass
        result = sp.powsimp(expr, force=False)
        self._normalize_cache[original] = result
        return result

    def entails(self, condition: sp.Expr) -> bool | None:
        condition = sp.sympify(condition)
        if condition in self._entailment_cache:
            self._cache_event("entails", True)
            return self._entailment_cache[condition]
        self._cache_event("entails", False)
        value = bounded_assumption_entails(condition, self.assumptions)
        self._entailment_cache[condition] = value
        return value

    def cached_applicability(
        self, key: object, compute: Callable[[], bool | None]
    ) -> bool | None:
        if key in self._applicability_cache:
            self._cache_event("applicability", True)
            return self._applicability_cache[key]
        self._cache_event("applicability", False)
        value = compute()
        self._applicability_cache[key] = value
        return value

    def limit(self, expr: sp.Expr) -> sp.Expr:
        expr = self.normalize(expr)
        if expr in self._limit_cache:
            return self._limit_cache[expr]
        value = bounded_limit(
            expr,
            self.variable,
            self.point,
            direction=self.direction,
            allow_general=True,
        )
        if value is None:
            value = sp.Limit(expr, self.variable, self.point, dir=self.direction)
        self._limit_cache[expr] = value
        return value

    def is_zero(self, expr: sp.Expr) -> bool | None:
        """Return whether *expr* is identically zero.

        Bounded rational-polynomial identities use exact coefficient arithmetic.
        ``exprtest.zerotest`` handles the remaining identities. The default
        confidence policy is ``"certified"`` because asymptotic cancellation
        must not discard a term merely because it is probably nonzero.  A
        bounded SymPy ``equals(0)`` fallback remains available for cases
        outside exprtest's current bounded oracle.
        """

        expr = self.normalize(expr)
        if expr in self._zero_cache:
            return self._zero_cache[expr]
        if expr == 0 or expr.is_zero is True:
            result: bool | None = True
        elif expr.is_zero is False:
            result = False
        elif (
            self.zero_oracle is None
            and self.assumptions is sp.S.true
            and (polynomial_zero := _rational_polynomial_zero(expr, self.variable))
            is not None
        ):
            result = polynomial_zero
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
                # The context cache handles repeated questions within one
                # asymptotic computation.  Exprtest's bounded proof-only cache
                # additionally shares certified classifications across nested
                # contexts created by localization and scale discovery.
                try:
                    result = oracle(
                        expr,
                        assumptions=self.assumptions,
                        use_cache=True,
                        confidence=self.zero_confidence,
                    )
                except (AssertionError, *SYMBOLIC_ERRORS):
                    # Third-party exact oracles may delegate to SymPy assumption
                    # solvers that reject otherwise valid symbolic expressions.
                    # An oracle failure is UNKNOWN, never a proof of zero/nonzero.
                    result = None
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
                        result = (
                            1 if lead.is_positive else -1 if lead.is_negative else None
                        )
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
                return AsymptoticContext(local, point=0)._eventual_sign_rational(
                    shifted
                )
            if self.point not in (0, sp.oo, -sp.oo):
                local = sp.Dummy("_h", positive=True)
                side = -1 if self.direction == "-" else 1
                shifted = sp.cancel(expr.subs(x, self.point + side * local))
                return AsymptoticContext(local, point=0)._eventual_sign_rational(
                    shifted
                )
            if self.point in (sp.oo, -sp.oo):
                local = sp.Dummy("_h", positive=True)
                sign = 1 if self.point is sp.oo else -1
                transformed = sp.cancel(expr.subs(x, sign / local))
                return AsymptoticContext(local, point=0)._eventual_sign_rational(
                    transformed
                )
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

    def compare_growth(
        self, f: sp.Expr, g: sp.Expr
    ) -> tuple[AsymptoticGrowthComparison, sp.Expr | None]:
        """Compare |f| and |g| by the limit of f/g.

        SAME_ORDER additionally returns the finite nonzero ratio when SymPy can
        determine it.  For scale *comparability classes*, use compare_log_growth.
        """

        from .instrumentation import record_symbolic_event

        record_symbolic_event("growth_comparisons")
        f = self.normalize(f)
        g = self.normalize(g)
        key = (f, g)
        if key in self._growth_cache:
            return self._growth_cache[key]
        if self.is_zero(g) is True:
            result = (AsymptoticGrowthComparison.UNKNOWN, None)
        else:
            raw_ratio = f / g
            if raw_ratio.has(sp.gamma, sp.factorial):
                try:
                    raw_ratio = sp.combsimp(raw_ratio)
                except SYMBOLIC_ERRORS:
                    pass
            ratio = self.limit(sp.Abs(raw_ratio))
            if ratio == 0:
                result = (AsymptoticGrowthComparison.SMALLER, sp.S.Zero)
            elif ratio is sp.oo:
                result = (AsymptoticGrowthComparison.LARGER, sp.oo)
            elif ratio.is_finite is True and ratio.is_zero is False:
                result = (AsymptoticGrowthComparison.SAME_ORDER, ratio)
            else:
                result = (AsymptoticGrowthComparison.UNKNOWN, None)
        self._growth_cache[key] = result
        return result

    def compare_log_growth(
        self, f: sp.Expr, g: sp.Expr
    ) -> tuple[AsymptoticGrowthComparison, sp.Expr | None]:
        """Compare comparability classes of positive functions tending to 0/∞.

        For vanishing scale elements this uses |log(f)| / |log(g)|.  A finite
        nonzero limit means equal comparability class; 0 and ∞ give the order.
        """

        f = self.normalize(f)
        g = self.normalize(g)
        try:
            ratio = self.limit(sp.Abs(sp.log(sp.Abs(f))) / sp.Abs(sp.log(sp.Abs(g))))
        except SYMBOLIC_ERRORS:
            return (AsymptoticGrowthComparison.UNKNOWN, None)
        if ratio == 0:
            return (AsymptoticGrowthComparison.SMALLER, sp.S.Zero)
        if ratio is sp.oo:
            return (AsymptoticGrowthComparison.LARGER, sp.oo)
        if ratio.is_finite is True and ratio.is_zero is False:
            return (AsymptoticGrowthComparison.SAME_ORDER, ratio)
        return (AsymptoticGrowthComparison.UNKNOWN, None)


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
