"""Asymptotic expectation and probability by structural statistical reduction.

The implementation follows four increasingly structural routes:

1. exact SymPy statistics reduction when it produces a usable expression;
2. reduction to the defining density or PMF integral/sum;
3. normalization of parameter-dependent continuous domains by an asymptotic
   scale substitution;
4. endpoint or interior-saddle Laplace expansion.

Laplace results are deliberately marked formal unless an exact reduction was
used.  Local saddle calculations alone do not prove global dominance or a
uniform tail bound, so the package does not manufacture a certified remainder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import sympy as sp
from sympy.stats import E, P, density, given
from sympy.stats.crv import ContinuousDistribution, SingleContinuousPSpace
from sympy.stats.drv import DiscreteDistribution, SingleDiscretePSpace
from sympy.stats.frv import SingleFiniteDistribution, SingleFinitePSpace
from sympy.stats.joint_rv import JointDistribution, JointPSpace, JointRandomSymbol
from sympy.stats.rv import RandomSymbol

from ._power_simplify import analytic_powsimp, power_expand_exact
from ._symbolic_errors import SYMBOLIC_ERRORS
from ._symbolic_policy import bounded_assumption_sign, bounded_limit, bounded_solve_one
from .instrumentation import record_symbolic_event
from .remainder import AsymptoticRemainder
from .sums import DISCRETE_STAT_METHODS, SUM_METHODS
from .transseries import TransseriesExpansion, transseries_from_expression

StatisticalStatus = Literal["EXACT", "CERTIFIED", "FORMAL", "UNKNOWN"]

_CONTINUOUS_STAT_METHODS = frozenset({"auto", "exact", "density", "laplace"})
_STATISTICAL_METHODS = _CONTINUOUS_STAT_METHODS | DISCRETE_STAT_METHODS


def _apply_random_bindings(
    expression: sp.Expr,
    bindings: dict[object, object] | None,
) -> sp.Expr:
    """Replace ordinary symbols by random symbols or raw SymPy distributions."""

    expression = sp.sympify(expression)
    if bindings is None:
        return expression
    if not isinstance(bindings, dict):
        raise TypeError("bindings must be a dict mapping Symbols to RandomSymbols")
    replacements: dict[sp.Expr, sp.Expr] = {}
    for symbol, prob_spec in bindings.items():
        if isinstance(symbol, (tuple, list, sp.Tuple)):
            symbols = tuple(symbol)
            if not symbols or not all(isinstance(item, sp.Symbol) for item in symbols):
                raise TypeError("multivariate binding keys must contain only SymPy Symbols")
            if isinstance(prob_spec, JointRandomSymbol):
                joint = prob_spec
            elif isinstance(prob_spec, JointDistribution):
                joint = JointPSpace(sp.Dummy("_joint_binding"), prob_spec).value
            else:
                raise TypeError(
                    "a multivariate binding must map a symbol tuple to a JointRandomSymbol "
                    "or JointDistribution"
                )
            support = getattr(joint.pspace.distribution, "set", None)
            if isinstance(support, sp.ProductSet) and len(support.args) != len(symbols):
                raise ValueError(
                    "multivariate binding dimension does not match the number of symbols"
                )
            # Indexed access is the stable public representation of joint RV
            # components.  Dimension mismatches are detected when components
            # are materialized by the distribution/pspace.
            replacements.update({item: joint[index] for index, item in enumerate(symbols)})
            continue
        if not isinstance(symbol, sp.Symbol):
            raise TypeError("bindings must use SymPy Symbols or symbol tuples as keys")
        if isinstance(prob_spec, RandomSymbol):
            replacements[symbol] = prob_spec
            continue
        if isinstance(prob_spec, ContinuousDistribution):
            replacements[symbol] = SingleContinuousPSpace(symbol, prob_spec).value
            continue
        if isinstance(prob_spec, DiscreteDistribution):
            replacements[symbol] = SingleDiscretePSpace(symbol, prob_spec).value
            continue
        if isinstance(prob_spec, SingleFiniteDistribution):
            replacements[symbol] = SingleFinitePSpace(symbol, prob_spec).value
            continue
        raise TypeError("bindings must map SymPy Symbols to RandomSymbols or SymPy distributions")
    return expression.xreplace(replacements)


@dataclass(frozen=True)
class LaplaceRemainderCertificate:
    """Replayable evidence for a globally valid real Laplace expansion.

    Certification is intentionally narrow: the phase must be a real polynomial
    on a real interval, all relevant critical points must be exactly enumerable,
    every dominant interior point must be an even-order strict minimum, and
    infinite tails must be coercive.  Within that class the standard real
    Laplace theorem gives a complete local expansion with a remainder of the
    scale recorded in ``remainder``.
    """

    certified: bool
    phase: sp.Expr
    variable: sp.Symbol
    domain: sp.Interval
    stationary_points: tuple[sp.Expr, ...]
    dominant_points: tuple[sp.Expr, ...]
    local_orders: tuple[int, ...]
    minimum_value: sp.Expr | None
    coercive_tails: bool
    remainder: AsymptoticRemainder | None
    reason: str

    def replay(self) -> bool | None:
        """Replay the structural conditions recorded by the Laplace theorem."""
        if not self.certified:
            return None
        try:
            sp.Poly(self.phase, self.variable)
        except sp.PolynomialError:
            return False
        if self.remainder is None or not self.remainder.is_certified:
            return False
        for point in self.stationary_points:
            if sp.simplify(sp.diff(self.phase, self.variable).subs(self.variable, point)) != 0:
                return False
            if self.domain.contains(point) is sp.S.false:
                return False
        if len(self.local_orders) != len(self.dominant_points):
            return False
        for point, order in zip(self.dominant_points, self.local_orders):
            if order < 1:
                return False
            if order > 1:
                for derivative_order in range(1, order):
                    value = sp.simplify(
                        sp.diff(self.phase, self.variable, derivative_order).subs(
                            self.variable, point
                        )
                    )
                    if value != 0:
                        return False
                lead = sp.simplify(
                    sp.diff(self.phase, self.variable, order).subs(self.variable, point)
                )
                if order % 2 == 0 and bounded_assumption_sign(lead) != 1:
                    return False
        return self.coercive_tails


@dataclass(frozen=True)
class StatisticalAsymptoticResult:
    """Result of an asymptotic probability or expectation computation.

    ``expression`` is the finite asymptotic expression returned by the chosen
    route.  ``series`` is present when that expression belongs to the package's
    finite transseries algebra.  ``reduction`` records the exact density/PMF
    problem, while ``transformation`` records a moving-domain substitution.
    """

    expression: sp.Expr
    parameter: sp.Symbol
    point: sp.Expr
    method: str
    status: StatisticalStatus
    series: TransseriesExpansion | None = None
    reduction: sp.Expr | None = None
    integration_variable: sp.Symbol | None = None
    domain: sp.Set | None = None
    transformation: tuple[sp.Symbol, sp.Expr] | None = None
    conditions: tuple[sp.Expr, ...] = ()
    remainder: AsymptoticRemainder | None = None
    certificate: object | None = None
    normalization: object | None = None

    @property
    def certified(self) -> bool:
        return self.status in ("EXACT", "CERTIFIED")

    def truncate(self, terms: int | None = None) -> sp.Expr:
        if self.series is None:
            return self.expression
        return self.series.truncate(terms)


def _prepare_statistical_query(
    expression: sp.Expr,
    *,
    bindings: dict[object, object] | None,
    condition: sp.Expr | None,
    random_symbol: RandomSymbol | None,
    method: str,
    label: str,
) -> sp.Expr:
    """Normalize bindings/conditioning and validate common query options."""

    expression = _apply_random_bindings(expression, bindings)
    if condition is not None:
        condition = _apply_random_bindings(condition, bindings)
        expression = given(expression, condition)
    if random_symbol is not None and not isinstance(random_symbol, RandomSymbol):
        raise TypeError("random_symbol must be a SymPy RandomSymbol")
    if method not in _STATISTICAL_METHODS:
        raise ValueError(f"unknown {label} method {method!r}")
    return expression


def _probability_space_kind(rv: RandomSymbol) -> str:
    """Return the structural route kind for a one-dimensional probability space."""

    if isinstance(rv.pspace, SingleContinuousPSpace):
        return "continuous"
    if isinstance(rv.pspace, (SingleDiscretePSpace, SingleFinitePSpace)):
        return "discrete"
    raise NotImplementedError(
        "only single continuous, discrete, or finite probability spaces are supported"
    )


def _validate_route_method(method: str, kind: str) -> None:
    """Reject statistical methods that are incompatible with the probability space."""

    if kind == "continuous" and method in DISCRETE_STAT_METHODS - {"auto", "exact"}:
        raise TypeError(f"{method} method requires a discrete random variable")
    if kind == "discrete" and method in {"density", "laplace"}:
        raise TypeError(f"{method} method requires a continuous random variable")


def _random_symbols(expr: sp.Expr) -> tuple[RandomSymbol, ...]:
    return tuple(sorted(expr.atoms(RandomSymbol), key=sp.default_sort_key))


def _resolve_random_symbol(expr: sp.Expr, random_symbol: RandomSymbol | None) -> RandomSymbol:
    if random_symbol is not None:
        if not isinstance(random_symbol, RandomSymbol):
            raise TypeError("random_symbol must be a SymPy RandomSymbol")
        return random_symbol
    symbols = _random_symbols(expr)
    if len(symbols) != 1:
        raise ValueError("exactly one random symbol must be supplied or inferable")
    return symbols[0]


def _finite_asymptotic_series(
    expr: sp.Expr,
    parameter: sp.Symbol,
    point: sp.Expr,
    terms: int,
    *,
    complete: bool,
    remainder: AsymptoticRemainder | None = None,
) -> TransseriesExpansion | None:
    """Convert a usable exact/formal expression into a finite transseries."""

    expr = analytic_powsimp(sp.sympify(expr))
    candidate = expr
    # Analytic small corrections such as exp(1/n) should be expanded before
    # monomial parsing.  Failure is harmless: log-exp expressions may already
    # be native transmonomials.
    try:
        expanded = sp.series(expr, parameter, point, max(2, terms)).removeO()
        # SymPy represents beyond-all-orders exponential tails by the zero
        # Poincare series.  Keep the original expression in that case so the
        # transseries layer retains its exponential scale.
        if expanded != expr and not (expanded == 0 and expr != 0):
            candidate = expanded
    except SYMBOLIC_ERRORS:
        pass
    if remainder is None:
        remainder = (
            AsymptoticRemainder.exact_zero(parameter, point, source="exact statistical reduction")
            if complete and candidate == expr
            else AsymptoticRemainder.unknown(
                parameter,
                point,
                source=(
                    "truncated exact statistical expression"
                    if complete
                    else "formal Laplace/statistical asymptotic expansion"
                ),
            )
        )
    try:
        return transseries_from_expression(
            candidate,
            parameter,
            point=point,
            complete=remainder.is_exact,
            remainder=remainder,
        ).prefix(terms)
    except (TypeError, ValueError, NotImplementedError):
        return None


def _result_from_expression(
    expr: sp.Expr,
    parameter: sp.Symbol,
    point: sp.Expr,
    terms: int,
    *,
    method: str,
    status: StatisticalStatus,
    reduction: sp.Expr | None = None,
    integration_variable: sp.Symbol | None = None,
    domain: sp.Set | None = None,
    transformation: tuple[sp.Symbol, sp.Expr] | None = None,
    conditions: tuple[sp.Expr, ...] = (),
    remainder: AsymptoticRemainder | None = None,
    certificate: LaplaceRemainderCertificate | None = None,
) -> StatisticalAsymptoticResult:
    series = _finite_asymptotic_series(
        expr, parameter, point, terms, complete=status == "EXACT", remainder=remainder
    )
    finite = (
        sp.sympify(expr)
        if status == "EXACT"
        else series.truncate()
        if series is not None
        else sp.sympify(expr)
    )
    return StatisticalAsymptoticResult(
        finite,
        parameter,
        point,
        method,
        status,
        series,
        reduction,
        integration_variable,
        domain,
        transformation,
        conditions,
        remainder,
        certificate,
    )


def _usable_exact(expr: sp.Expr, parameter: sp.Symbol) -> bool:
    if expr.has(sp.Integral, sp.Sum):
        return False
    # Unevaluated statistics objects are function applications whose names are
    # stable enough to reject without depending on private SymPy classes.
    names = {getattr(node.func, "__name__", "") for node in sp.preorder_traversal(expr)}
    if {"Expectation", "Probability"} & names:
        return False
    return bool(parameter in expr.free_symbols or not expr.free_symbols)


def _try_exact_expectation(expr: sp.Expr, parameter: sp.Symbol) -> sp.Expr | None:
    if sp.count_ops(expr) > 100:
        return None
    try:
        value = sp.sympify(E(expr))
        if sp.count_ops(value) <= 160:
            value = sp.simplify(value.rewrite(sp.erfc))
    except SYMBOLIC_ERRORS:
        return None
    if _usable_exact(value, parameter):
        record_symbolic_event("stat_exact_reductions")
        return value
    return None


def _try_exact_probability(event: sp.Expr, parameter: sp.Symbol) -> sp.Expr | None:
    if sp.count_ops(event) > 100:
        return None
    try:
        value = sp.sympify(P(event))
        if sp.count_ops(value) <= 160:
            value = sp.simplify(value.rewrite(sp.erfc))
    except SYMBOLIC_ERRORS:
        return None
    if _usable_exact(value, parameter):
        record_symbolic_event("stat_exact_reductions")
        return value
    return None


def _event_domain(event: sp.Expr, rv: RandomSymbol) -> sp.Set | None:
    """Return the one-dimensional set described by a simple event."""

    if isinstance(event, sp.Equality):
        if event.lhs == rv:
            return sp.FiniteSet(event.rhs)
        if event.rhs == rv:
            return sp.FiniteSet(event.lhs)
        return None
    if not isinstance(
        event, (sp.StrictGreaterThan, sp.GreaterThan, sp.StrictLessThan, sp.LessThan)
    ):
        return None
    if event.lhs == rv:
        value = event.rhs
        if isinstance(event, (sp.StrictGreaterThan, sp.GreaterThan)):
            return sp.Interval(value, sp.oo, left_open=isinstance(event, sp.StrictGreaterThan))
        return sp.Interval(-sp.oo, value, right_open=isinstance(event, sp.StrictLessThan))
    if event.rhs == rv:
        value = event.lhs
        if isinstance(event, (sp.StrictGreaterThan, sp.GreaterThan)):
            return sp.Interval(-sp.oo, value, right_open=isinstance(event, sp.StrictGreaterThan))
        return sp.Interval(value, sp.oo, left_open=isinstance(event, sp.StrictLessThan))
    return None


def _support(rv: RandomSymbol) -> sp.Set:
    support = getattr(rv.pspace.domain, "set", None)
    if support is None:
        support = getattr(rv.pspace.distribution, "set", None)
    if not isinstance(support, sp.Set):
        raise NotImplementedError("distribution support is not available as a SymPy set")
    return support


def _continuous_reduction(
    integrand: sp.Expr,
    rv: RandomSymbol,
    domain: sp.Set,
) -> tuple[sp.Expr, sp.Symbol]:
    z = sp.Dummy(str(rv.symbol), real=True)
    pdf = sp.sympify(density(rv)(z))
    observed = sp.sympify(integrand).xreplace({rv: z})
    record_symbolic_event("stat_density_routes")
    return analytic_powsimp(observed * pdf), z


def _discrete_components(
    integrand: sp.Expr,
    rv: RandomSymbol,
) -> tuple[sp.Expr, sp.Expr, sp.Symbol]:
    """Return observable, PMF, and lattice variable without losing positivity."""

    k = sp.Dummy(str(rv.symbol), integer=True)
    pmf = sp.sympify(density(rv)(k))
    observed = sp.sympify(integrand).xreplace({rv: k})
    record_symbolic_event("stat_pmf_routes")
    return observed, pmf, k


def _discrete_reduction(
    integrand: sp.Expr,
    rv: RandomSymbol,
) -> tuple[sp.Expr, sp.Symbol]:
    observed, pmf, k = _discrete_components(integrand, rv)
    return analytic_powsimp(observed * pmf), k


def _support_assumptions(domain: sp.Set, variable: sp.Symbol) -> sp.Expr:
    """Translate a supported integer domain into positivity predicates."""

    if domain == sp.S.Naturals0:
        return sp.Q.nonnegative(variable)
    if domain == sp.S.Integers:
        return sp.S.true
    if isinstance(domain, sp.Range) and domain.step == 1:
        pieces: list[sp.Expr] = []
        if domain.start is not -sp.oo:
            pieces.append(sp.Q.nonnegative(variable - domain.start))
        if domain.stop is not sp.oo:
            pieces.append(sp.Q.nonnegative((domain.stop - 1) - variable))
        return sp.And(*pieces) if pieces else sp.S.true
    if isinstance(domain, sp.Interval):
        pieces = []
        if domain.start is not -sp.oo:
            predicate = sp.Q.positive if domain.left_open else sp.Q.nonnegative
            pieces.append(predicate(variable - domain.start))
        if domain.end is not sp.oo:
            predicate = sp.Q.positive if domain.right_open else sp.Q.nonnegative
            pieces.append(predicate(domain.end - variable))
        return sp.And(*pieces) if pieces else sp.S.true
    if isinstance(domain, sp.Intersection):
        pieces = tuple(_support_assumptions(part, variable) for part in domain.args)
        return sp.And(*pieces) if pieces else sp.S.true
    return sp.S.true


def _restrict_piecewise_to_support(
    expression: sp.Expr,
    domain: sp.Set,
    variable: sp.Symbol,
) -> sp.Expr:
    """Select a Piecewise branch whose condition is certified on the support."""

    expression = sp.sympify(expression)
    if not isinstance(expression, sp.Piecewise):
        return expression
    assumptions = _support_assumptions(domain, variable)
    for value, condition in expression.args:
        if condition is True or condition is sp.S.true:
            continue
        try:
            refined = sp.refine(condition, assumptions)
        except (TypeError, ValueError, NotImplementedError):
            refined = condition
        if refined is sp.S.true:
            return sp.sympify(value)
    return expression


def _set_integral(expr: sp.Expr, variable: sp.Symbol, domain: sp.Set) -> sp.Expr:
    if isinstance(domain, sp.Interval):
        return sp.Integral(expr, (variable, domain.start, domain.end))
    if isinstance(domain, sp.FiniteSet):
        return sp.Add(*(expr.xreplace({variable: value}) for value in domain))
    raise NotImplementedError(f"continuous domain {domain} is not an interval")


def _discrete_bounds(domain: sp.Set) -> tuple[sp.Expr, sp.Expr] | None:
    """Return inclusive integer bounds for supported interval intersections."""

    lower: sp.Expr = -sp.oo
    upper: sp.Expr = sp.oo
    parts = domain.args if isinstance(domain, sp.Intersection) else (domain,)
    seen = False
    for part in parts:
        if part == sp.S.Integers:
            seen = True
            continue
        if part == sp.S.Naturals0:
            lower = sp.Max(lower, sp.S.Zero)
            seen = True
            continue
        if isinstance(part, sp.Range) and part.step == 1:
            lower = sp.Max(lower, part.start)
            upper = sp.Min(upper, part.stop - 1)
            seen = True
            continue
        if isinstance(part, sp.Interval):
            lo = (sp.floor(part.start) + 1) if part.left_open else sp.ceiling(part.start)
            hi = (sp.ceiling(part.end) - 1) if part.right_open else sp.floor(part.end)
            if part.start is -sp.oo:
                lo = -sp.oo
            if part.end is sp.oo:
                hi = sp.oo
            lower = sp.Max(lower, lo)
            upper = sp.Min(upper, hi)
            seen = True
            continue
        if isinstance(part, sp.FiniteSet):
            return None
        return None
    return (analytic_powsimp(lower), analytic_powsimp(upper)) if seen else None


def _set_sum(expr: sp.Expr, variable: sp.Symbol, domain: sp.Set) -> sp.Expr:
    if isinstance(domain, sp.FiniteSet):
        return sp.Add(*(expr.xreplace({variable: value}) for value in domain))
    bounds = _discrete_bounds(domain)
    if bounds is not None:
        lo, hi = bounds
        return sp.Sum(expr, (variable, lo, hi))
    raise NotImplementedError(f"discrete domain {domain} is not a supported integer range")


def _evaluate_reduction(reduction: sp.Expr) -> sp.Expr | None:
    if not reduction.has(sp.Integral, sp.Sum):
        return sp.sympify(reduction)
    try:
        value = reduction.doit()
    except SYMBOLIC_ERRORS:
        return None
    if value == reduction or value.has(sp.Integral, sp.Sum):
        return None
    return sp.sympify(value)


def _leading_boundary_scale(bound: sp.Expr, parameter: sp.Symbol, point: sp.Expr) -> sp.Expr | None:
    """Find an exact multiplicative scale making a moving bound constant."""

    if parameter not in bound.free_symbols:
        return sp.S.One
    try:
        ts = transseries_from_expression(bound, parameter, point=point).normalized()
    except (TypeError, ValueError, NotImplementedError):
        return None
    leading = ts.leading_term
    if leading is None or parameter in leading.coefficient.free_symbols:
        return None
    scale = analytic_powsimp(leading.expression / leading.coefficient)
    quotient = analytic_powsimp(bound / scale)
    if parameter in quotient.free_symbols:
        return None
    return scale


def _moving_domain_transform(
    expr: sp.Expr,
    variable: sp.Symbol,
    domain: sp.Set,
    parameter: sp.Symbol,
    point: sp.Expr,
) -> tuple[sp.Expr, sp.Symbol, sp.Set, tuple[sp.Symbol, sp.Expr]] | None:
    """Scale a parameter-dependent interval so its moving endpoint is fixed."""

    if not isinstance(domain, sp.Interval) or point is not sp.oo:
        return None
    moving = []
    if domain.start not in (-sp.oo, sp.oo) and parameter in domain.start.free_symbols:
        moving.append(domain.start)
    if domain.end not in (-sp.oo, sp.oo) and parameter in domain.end.free_symbols:
        moving.append(domain.end)
    if not moving:
        return None
    scale = _leading_boundary_scale(moving[0], parameter, point)
    if scale is None or scale == 0:
        return None
    # All finite moving boundaries must become parameter-free under one scale.
    new_start = (
        domain.start if domain.start in (-sp.oo, sp.oo) else analytic_powsimp(domain.start / scale)
    )
    new_end = domain.end if domain.end in (-sp.oo, sp.oo) else analytic_powsimp(domain.end / scale)
    if parameter in sp.Tuple(new_start, new_end).free_symbols:
        return None
    y = sp.Dummy(f"{variable}_scaled", real=True)
    replacement = analytic_powsimp(scale * y)
    transformed = analytic_powsimp(expr.xreplace({variable: replacement}) * sp.Abs(scale))
    transformed_domain = sp.Interval(
        new_start,
        new_end,
        left_open=domain.left_open,
        right_open=domain.right_open,
    )
    record_symbolic_event("stat_moving_routes")
    return transformed, y, transformed_domain, (variable, replacement)


def _extract_laplace_form(
    integrand: sp.Expr,
    variable: sp.Symbol,
    parameter: sp.Symbol,
    *,
    terms: int = 4,
) -> tuple[sp.Expr, sp.Expr, sp.Expr, bool] | None:
    """Return prefactor, amplitude, phase, and whether amplitude was truncated."""

    prefactor, dependent = sp.sympify(integrand).as_independent(variable, as_Add=False)
    exponent = sp.S.Zero
    amplitude = sp.S.One
    found = False
    for factor in sp.Mul.make_args(dependent):
        if factor.func is sp.exp:
            exponent += factor.args[0]
            found = True
        else:
            amplitude *= factor
    if not found and dependent.func is sp.exp:
        exponent = dependent.args[0]
        amplitude = sp.S.One
        found = True
    if not found:
        return None
    try:
        poly = sp.Poly(sp.expand(exponent), parameter)
    except sp.PolynomialError:
        poly = None
    truncated_amplitude = False
    if poly is not None and poly.degree() == 1:
        slope = sp.sympify(poly.coeff_monomial(parameter))
        offset = sp.sympify(poly.coeff_monomial(1))
    else:
        # Stirling-normalized PMFs naturally produce an exponent of the form
        # p*slope(x) + O(log(p)) + O(1/p).  Expand the exponent first: this is
        # substantially cheaper and more reliable than asking a generic limit
        # engine to simplify cancellations among p*log(p), p*log(p*x), etc.
        try:
            expanded_exponent = sp.series(exponent, parameter, sp.oo, max(3, terms + 2)).removeO()
        except SYMBOLIC_ERRORS:
            expanded_exponent = exponent
        slope = bounded_limit(expanded_exponent / parameter, parameter, sp.oo, allow_general=True)
        if slope is None:
            return None
        if isinstance(slope, sp.Limit) or slope in (sp.oo, -sp.oo, sp.zoo, sp.nan):
            return None
        slope = sp.simplify(slope)
        if parameter in sp.sympify(slope).free_symbols:
            return None
        offset = analytic_powsimp(expanded_exponent - parameter * slope)
        truncated_amplitude = expanded_exponent != exponent
        if not truncated_amplitude:
            relative_offset = bounded_limit(
                offset / parameter, parameter, sp.oo, allow_general=True
            )
            if relative_offset is None:
                return None
            if relative_offset != 0:
                return None
    phase = analytic_powsimp(-slope)
    if parameter in phase.free_symbols:
        return None
    amplitude = analytic_powsimp(amplitude * sp.exp(offset))
    return analytic_powsimp(prefactor), amplitude, phase, truncated_amplitude


def _interior_domain_assumptions(domain: sp.Interval, variable: sp.Symbol) -> sp.Expr:
    """Assumptions valid on the open interior of a real integration interval."""

    pieces: list[sp.Expr] = []
    if domain.start is not -sp.oo:
        pieces.append(sp.Q.positive(variable - domain.start))
    if domain.end is not sp.oo:
        pieces.append(sp.Q.positive(domain.end - variable))
    return sp.And(*pieces) if pieces else sp.S.true


def _stationary_points(phase: sp.Expr, variable: sp.Symbol) -> tuple[sp.Expr, ...]:
    derivative = sp.simplify(analytic_powsimp(sp.diff(phase, variable)))
    try:
        poly = sp.Poly(derivative, variable)
    except sp.PolynomialError:
        poly = None
    if poly is not None and poly.degree() <= 8 and sp.count_ops(derivative) <= 80:
        try:
            roots = sp.roots(poly.as_expr(), variable)
        except (sp.PolynomialError, ValueError, NotImplementedError):
            roots = {}
        if roots and sum(roots.values()) == poly.degree():
            return tuple(sorted(roots, key=sp.default_sort_key))
    solved = bounded_solve_one(sp.Eq(derivative, 0), variable, allow_general=True)
    return tuple(solved or ())


def _contains(domain: sp.Set, value: sp.Expr) -> bool | None:
    test = domain.contains(value)
    if test is sp.S.true:
        return True
    if test is sp.S.false:
        return False
    return None


def _gaussian_polynomial_integral(
    poly: sp.Expr, u: sp.Symbol, curvature: sp.Expr
) -> sp.Expr | None:
    try:
        polynomial = sp.Poly(sp.expand(poly), u)
    except sp.PolynomialError:
        return None
    total = sp.S.Zero
    for (degree,), coefficient in polynomial.terms():
        if degree % 2:
            continue
        m = degree // 2
        moment = sp.sqrt(2 * sp.pi / curvature)
        if m:
            moment *= sp.factorial2(2 * m - 1) / curvature**m
        total += coefficient * moment
    return analytic_powsimp(total)


def _interior_saddle_expansion(
    prefactor: sp.Expr,
    amplitude: sp.Expr,
    phase: sp.Expr,
    variable: sp.Symbol,
    saddle: sp.Expr,
    parameter: sp.Symbol,
    terms: int,
) -> tuple[sp.Expr, tuple[sp.Expr, ...]] | None:
    curvature = sp.simplify(analytic_powsimp(sp.diff(phase, variable, 2).subs(variable, saddle)))
    if bounded_assumption_sign(curvature) != 1:
        return None
    eps = sp.Dummy("eps", positive=True)
    u = sp.Dummy("u", real=True)
    local_x = saddle + eps * u
    phi0 = sp.simplify(analytic_powsimp(phase.subs(variable, saddle)))
    phi_local = sp.series(phase.subs(variable, local_x), eps, 0, 2 * terms + 3).removeO()
    gaussian = curvature * u**2 * eps**2 / 2
    correction = analytic_powsimp((phi_local - phi0 - gaussian) / eps**2)
    local_amp = amplitude.subs({variable: local_x, parameter: eps**-2})
    try:
        correction_series = sp.series(sp.exp(-correction), eps, 0, 2 * terms + 2).removeO()
        product = sp.series(local_amp * correction_series, eps, 0, 2 * terms + 2).removeO()
    except SYMBOLIC_ERRORS:
        return None
    integrated = sp.S.Zero
    for item in sp.Add.make_args(sp.expand(product)):
        exponent = sp.sympify(item.as_powers_dict().get(eps, 0))
        if exponent.is_integer is not True:
            return None
        coefficient = analytic_powsimp(item / eps**exponent)
        moment = _gaussian_polynomial_integral(coefficient, u, curvature)
        if moment is None:
            return None
        integrated += moment * eps**exponent
    local = analytic_powsimp(eps * integrated)
    local = sp.series(local, eps, 0, 2 * terms + 1).removeO()
    result = analytic_powsimp(
        prefactor
        * sp.exp(-parameter * phi0)
        * local.xreplace({eps: parameter ** sp.Rational(-1, 2)})
    )
    return result, (sp.Gt(curvature, 0),)


def _endpoint_expansion(
    prefactor: sp.Expr,
    amplitude: sp.Expr,
    phase: sp.Expr,
    variable: sp.Symbol,
    endpoint: sp.Expr,
    parameter: sp.Symbol,
    terms: int,
    *,
    lower: bool,
) -> tuple[sp.Expr, tuple[sp.Expr, ...]] | None:
    derivative = analytic_powsimp(sp.diff(phase, variable).subs(variable, endpoint))
    sign = bounded_assumption_sign(derivative)
    required = 1 if lower else -1
    if sign != required:
        return None
    # For an upper endpoint reverse orientation, which amounts to replacing
    # phi' by -phi' in the Watson endpoint recurrence.
    orient = sp.S.One if lower else -sp.S.One
    phi_prime = analytic_powsimp(orient * sp.diff(phase, variable))
    coeff = analytic_powsimp(amplitude / phi_prime)
    pieces = []
    for _ in range(terms):
        pieces.append(analytic_powsimp(coeff.subs(variable, endpoint)))
        coeff = analytic_powsimp(sp.diff(coeff, variable) / phi_prime)
    body = sp.Add(*(piece / parameter ** (i + 1) for i, piece in enumerate(pieces)))
    result = analytic_powsimp(
        prefactor * sp.exp(-parameter * phase.subs(variable, endpoint)) * body
    )
    condition = sp.Gt(derivative, 0) if lower else sp.Lt(derivative, 0)
    return result, (condition,)


def _first_nonzero_derivative_order(
    phase: sp.Expr,
    variable: sp.Symbol,
    location: sp.Expr,
    *,
    max_order: int = 12,
) -> tuple[int, sp.Expr] | None:
    for order in range(1, max_order + 1):
        value = sp.simplify(
            analytic_powsimp(sp.diff(phase, variable, order).subs(variable, location))
        )
        if value == 0:
            continue
        sign = bounded_assumption_sign(value)
        if sign in (-1, 1) or value.is_zero is False:
            return order, value
        return None
    return None


def _generalized_moment(
    power: int,
    order: int,
    coefficient: sp.Expr,
    *,
    half_line: bool,
) -> sp.Expr:
    if not half_line and power % 2:
        return sp.S.Zero
    factor = sp.S.One if half_line else sp.Integer(2)
    return analytic_powsimp(
        factor
        * sp.gamma(sp.Rational(power + 1, order))
        / (order * coefficient ** sp.Rational(power + 1, order))
    )


def _degenerate_local_expansion(
    prefactor: sp.Expr,
    amplitude: sp.Expr,
    phase: sp.Expr,
    variable: sp.Symbol,
    location: sp.Expr,
    parameter: sp.Symbol,
    terms: int,
    *,
    half_line: bool = False,
    lower: bool = True,
) -> tuple[sp.Expr, tuple[sp.Expr, ...], int] | None:
    """Expand an even-order interior saddle or stationary endpoint."""

    first = _first_nonzero_derivative_order(phase, variable, location)
    if first is None:
        return None
    order, derivative = first
    if half_line:
        oriented = derivative if lower else (-1) ** order * derivative
        if bounded_assumption_sign(oriented) != 1:
            return None
        derivative = oriented
    else:
        if order % 2 or bounded_assumption_sign(derivative) != 1:
            return None
    coefficient = analytic_powsimp(derivative / sp.factorial(order))
    eps = sp.Dummy("eps", positive=True)
    u = sp.Dummy("u", positive=half_line, real=True)
    orientation = sp.S.One if lower else -sp.S.One
    local_x = location + orientation * eps * u
    phi0 = analytic_powsimp(phase.subs(variable, location))
    expansion_order = max(order * terms + order + 2, order + 3)
    try:
        phi_local = sp.series(phase.subs(variable, local_x), eps, 0, expansion_order).removeO()
        leading = coefficient * eps**order * u**order
        correction = analytic_powsimp((phi_local - phi0 - leading) / eps**order)
        local_amp = amplitude.subs({variable: local_x, parameter: eps ** (-order)})
        correction_series = sp.series(sp.exp(-correction), eps, 0, expansion_order).removeO()
        product = sp.series(local_amp * correction_series, eps, 0, expansion_order).removeO()
    except SYMBOLIC_ERRORS:
        return None
    integrated = sp.S.Zero
    for item in sp.Add.make_args(sp.expand(product)):
        exponent = sp.sympify(item.as_powers_dict().get(eps, 0))
        if exponent.is_integer is not True or exponent < 0:
            return None
        coefficient_u = analytic_powsimp(item / eps**exponent)
        try:
            polynomial = sp.Poly(sp.expand(coefficient_u), u)
        except sp.PolynomialError:
            return None
        moment_sum = sp.S.Zero
        for (power,), coeff in polynomial.terms():
            moment_sum += coeff * _generalized_moment(
                power, order, coefficient, half_line=half_line
            )
        integrated += analytic_powsimp(moment_sum) * eps**exponent
    local = analytic_powsimp(eps * integrated)
    # Retain a bounded number of local-scale powers; zero parity coefficients
    # are naturally skipped at symmetric interior saddles.
    local = sp.series(local, eps, 0, order * terms + 2).removeO()
    result = analytic_powsimp(
        prefactor
        * sp.exp(-parameter * phi0)
        * local.xreplace({eps: parameter ** (-sp.Rational(1, order))})
    )
    condition = sp.Gt(derivative, 0)
    return result, (condition,), order


def _polynomial_global_laplace_certificate(
    phase: sp.Expr,
    amplitude: sp.Expr,
    prefactor: sp.Expr,
    variable: sp.Symbol,
    domain: sp.Interval,
    parameter: sp.Symbol,
    dominant: list[tuple[str, sp.Expr, sp.Expr]],
    terms: int,
    expression: sp.Expr,
) -> LaplaceRemainderCertificate:
    """Certify a narrow but useful real-polynomial Laplace theorem class."""

    try:
        sp.Poly(phase, variable)
    except sp.PolynomialError:
        return LaplaceRemainderCertificate(
            False,
            phase,
            variable,
            domain,
            (),
            tuple(item[1] for item in dominant),
            (),
            dominant[0][2] if dominant else None,
            False,
            None,
            "phase is not a polynomial in the integration variable",
        )
    if parameter in amplitude.free_symbols:
        amp_for_test = amplitude.as_independent(parameter, as_Add=False)[1]
    else:
        amp_for_test = amplitude
    if not (amp_for_test.is_polynomial(variable) or variable not in amp_for_test.free_symbols):
        return LaplaceRemainderCertificate(
            False,
            phase,
            variable,
            domain,
            (),
            tuple(item[1] for item in dominant),
            (),
            dominant[0][2] if dominant else None,
            False,
            None,
            "amplitude is not polynomial/constant in the local variable",
        )
    roots = _stationary_points(phase, variable)
    stationary = []
    for root in roots:
        if root.is_real is False:
            continue
        contained = _contains(domain, root)
        if contained is True:
            stationary.append(sp.sympify(root))
        elif contained is None:
            return LaplaceRemainderCertificate(
                False,
                phase,
                variable,
                domain,
                tuple(stationary),
                tuple(item[1] for item in dominant),
                (),
                dominant[0][2],
                False,
                None,
                "could not prove that a stationary point lies inside/outside the domain",
            )
    # Infinite real tails of a polynomial are coercive iff the relevant end
    # tends to +infinity. Exact SymPy limit is reliable for polynomial phases.
    coercive = True
    for end in (domain.start, domain.end):
        if end in (-sp.oo, sp.oo):
            lim = bounded_limit(phase, variable, end, allow_general=True)
            if lim is not sp.oo:
                coercive = False
    orders: list[int] = []
    for kind, location, _ in dominant:
        if kind in {"saddle", "degenerate", "degenerate-lower", "degenerate-upper"}:
            first = _first_nonzero_derivative_order(phase, variable, location)
            if first is None:
                return LaplaceRemainderCertificate(
                    False,
                    phase,
                    variable,
                    domain,
                    tuple(stationary),
                    tuple(item[1] for item in dominant),
                    tuple(orders),
                    dominant[0][2],
                    coercive,
                    None,
                    "could not determine the dominant local order",
                )
            local_order, local_derivative = first
            if kind == "degenerate-upper":
                local_derivative = analytic_powsimp((-1) ** local_order * local_derivative)
            if kind in {"saddle", "degenerate"} and local_order % 2:
                return LaplaceRemainderCertificate(
                    False,
                    phase,
                    variable,
                    domain,
                    tuple(stationary),
                    tuple(item[1] for item in dominant),
                    tuple(orders),
                    dominant[0][2],
                    coercive,
                    None,
                    "an interior dominant point has odd local order",
                )
            if bounded_assumption_sign(local_derivative) != 1:
                return LaplaceRemainderCertificate(
                    False,
                    phase,
                    variable,
                    domain,
                    tuple(stationary),
                    tuple(item[1] for item in dominant),
                    tuple(orders),
                    dominant[0][2],
                    coercive,
                    None,
                    "a dominant interior point is not an even-order strict minimum",
                )
            orders.append(local_order)
        else:
            orders.append(1)
    if not coercive:
        return LaplaceRemainderCertificate(
            False,
            phase,
            variable,
            domain,
            tuple(stationary),
            tuple(item[1] for item in dominant),
            tuple(orders),
            dominant[0][2],
            False,
            None,
            "an infinite tail is not polynomially coercive",
        )
    # Standard Laplace/Watson theory for polynomial phase and analytic
    # amplitude gives a full expansion. Use a conservative next local-scale
    # bound based on the slowest dominant local order.
    max_order = max(orders, default=1)
    phi0 = dominant[0][2]
    scale = analytic_powsimp(
        sp.Abs(prefactor)
        * sp.exp(-parameter * phi0)
        * parameter ** (-sp.Rational(terms + 1, max_order))
    )
    remainder = AsymptoticRemainder.big_o(
        scale,
        parameter,
        sp.oo,
        source="global real-polynomial Laplace remainder theorem",
    )
    record_symbolic_event("stat_laplace_certs")
    return LaplaceRemainderCertificate(
        True,
        phase,
        variable,
        domain,
        tuple(stationary),
        tuple(item[1] for item in dominant),
        tuple(orders),
        phi0,
        True,
        remainder,
        "real polynomial phase with exactly enumerated dominant minima and coercive tails",
    )


def airy_uniform_saddle_asymptotic(
    integrand: sp.Expr,
    variable: sp.Symbol,
    domain: sp.Interval | tuple[sp.Expr, sp.Expr],
    *,
    parameter: sp.Symbol,
    control_parameter: sp.Symbol,
    coalescence_value: sp.Expr = 0,
    location: sp.Expr = 0,
    terms: int = 1,
) -> StatisticalAsymptoticResult:
    """Leading uniform Airy approximation at a simple cubic turning point.

    This routine covers the canonical oscillatory coalescence in which an
    integral contains ``exp(I*parameter*phase(x, mu))`` and, at ``(x0, mu0)``,
    ``phase_x = phase_xx = 0`` while ``phase_xxx`` and ``phase_xmu`` are real
    and nonzero.  With ``mu-mu0 = O(parameter**(-2/3))`` two simple stationary
    points coalesce and the local integral is uniformly represented by an Airy
    function.  The implementation intentionally returns only the leading CFU
    term; higher ``terms`` are reserved for later Airy/Airy-prime transport.

    The full real line is required because the normalization uses the standard
    oscillatory identity

        integral exp(I*(t**3/3 + z*t)) dt = 2*pi*Ai(z).

    The result is FORMAL: local cubic reduction alone does not certify contour
    deformation, remote saddle dominance, or tail cancellation.
    """

    if terms != 1:
        raise NotImplementedError(
            "the current Airy uniform saddle implementation provides the leading term only"
        )
    if not isinstance(domain, sp.Interval):
        domain = sp.Interval(*domain)
    if domain.start is not -sp.oo or domain.end is not sp.oo:
        raise NotImplementedError("Airy uniform saddles currently require the full real line")

    expression = sp.sympify(integrand)
    parameter = sp.sympify(parameter)
    control_parameter = sp.sympify(control_parameter)
    exponential = None
    phase = None
    for atom in expression.atoms(sp.exp):
        candidate = analytic_powsimp(atom.args[0] / (sp.I * parameter))
        if parameter not in candidate.free_symbols:
            exponential = atom
            phase = candidate
            break
    if exponential is None or phase is None:
        raise NotImplementedError(
            "Airy uniform saddle requires an oscillatory exp(I*parameter*phase) factor"
        )
    amplitude = analytic_powsimp(expression / exponential)
    if parameter in amplitude.free_symbols:
        raise NotImplementedError("Airy leading amplitude must be parameter-independent")

    at = {variable: sp.sympify(location), control_parameter: sp.sympify(coalescence_value)}
    first = analytic_powsimp(sp.diff(phase, variable).subs(at))
    second = analytic_powsimp(sp.diff(phase, variable, 2).subs(at))
    third = analytic_powsimp(sp.diff(phase, variable, 3).subs(at))
    unfolding = analytic_powsimp(sp.diff(phase, variable, control_parameter).subs(at))
    if first != 0 or second != 0:
        raise NotImplementedError("supported Airy coalescence requires phase_x = phase_xx = 0")
    cubic_sign = bounded_assumption_sign(third)
    if cubic_sign not in {-1, 1}:
        raise NotImplementedError("phase_xxx must have a proved nonzero real sign")
    if bounded_assumption_sign(unfolding) not in {-1, 1}:
        raise NotImplementedError("control parameter must unfold the stationary equation linearly")

    # phase = phase0 + a*y**3/3 + b*(mu-mu0)*y + ...,
    # where a = phase_xxx/2.  The canonical scaling t=(p*|a|)^(1/3)y
    # produces Ai(sign(a) * b*(mu-mu0)*p^(2/3)/|a|^(1/3)).
    cubic = analytic_powsimp(third / 2)
    cubic_abs = sp.Abs(cubic)
    sign = sp.Integer(cubic_sign)
    delta = analytic_powsimp(control_parameter - coalescence_value)
    airy_argument = analytic_powsimp(
        sign * unfolding * delta * parameter ** sp.Rational(2, 3) / cubic_abs ** sp.Rational(1, 3)
    )
    phase0 = analytic_powsimp(phase.subs(variable, location))
    amplitude0 = analytic_powsimp(amplitude.subs(variable, location))
    result_expression = analytic_powsimp(
        2
        * sp.pi
        * amplitude0
        * sp.exp(sp.I * parameter * phase0)
        * sp.airyai(airy_argument)
        / (parameter * cubic_abs) ** sp.Rational(1, 3)
    )
    record_symbolic_event("stat_coalescing_saddles")
    # The Airy argument is itself a transition variable of size O(1), so it
    # must not be re-expanded by the ordinary one-parameter transseries parser.
    return StatisticalAsymptoticResult(
        expression=result_expression,
        parameter=parameter,
        point=sp.oo,
        method="oscillatory-airy-uniform-saddle",
        status="FORMAL",
        series=None,
        reduction=sp.Integral(expression, (variable, domain.start, domain.end)),
        integration_variable=variable,
        domain=domain,
        conditions=(sp.Ne(third, 0), sp.Ne(unfolding, 0)),
    )


def coalescing_saddle_asymptotic(
    integrand: sp.Expr,
    variable: sp.Symbol,
    domain: sp.Interval | tuple[sp.Expr, sp.Expr],
    *,
    parameter: sp.Symbol,
    control_parameter: sp.Symbol,
    coalescence_value: sp.Expr = 0,
    location: sp.Expr = 0,
    transition_symbol: sp.Symbol | None = None,
    terms: int = 2,
) -> StatisticalAsymptoticResult:
    """Uniform quartic transition for a symmetric pair of coalescing minima.

    The method targets the real Laplace normal form in which, at the
    coalescence, the first three x-derivatives vanish, the fourth derivative is
    positive, and the control parameter enters the quadratic term.  It uses
    ``x=x0+p^-1/4*u`` and ``mu=mu0+tau*p^-1/2`` and retains the resulting
    canonical quartic profile integrals.  This is uniform for bounded ``tau``.
    """

    if not isinstance(domain, sp.Interval):
        domain = sp.Interval(*domain)
    if domain.start is not -sp.oo or domain.end is not sp.oo:
        raise NotImplementedError("quartic coalescing-saddle profiles require the full real line")
    if transition_symbol is None:
        transition_symbol = sp.Dummy("tau", real=True)
    extracted = _extract_laplace_form(sp.sympify(integrand), variable, parameter, terms=terms)
    if extracted is None:
        raise NotImplementedError("coalescing saddle requires Laplace exponential form")
    prefactor, amplitude, phase, _truncated_amplitude = extracted
    at = {variable: location, control_parameter: coalescence_value}
    derivs = [analytic_powsimp(sp.diff(phase, variable, j).subs(at)) for j in range(1, 5)]
    if derivs[:3] != [0, 0, 0] or bounded_assumption_sign(derivs[3]) != 1:
        raise NotImplementedError("supported coalescence requires a positive quartic normal form")
    mixed = analytic_powsimp(sp.diff(phase, variable, 2, control_parameter, 1).subs(at) / 2)
    if mixed == 0:
        raise NotImplementedError("control parameter does not unfold the quadratic saddle term")
    eps = sp.Dummy("eps", positive=True)
    u = sp.Dummy("u", real=True)
    substitutions = {
        variable: location + eps * u,
        control_parameter: coalescence_value + transition_symbol * eps**2,
        parameter: eps**-4,
    }
    phi0_mu = analytic_powsimp(
        phase.subs(variable, location).subs(
            control_parameter, coalescence_value + transition_symbol * eps**2
        )
    )
    local_phase = analytic_powsimp((phase.subs(substitutions) - phi0_mu) / eps**4)
    try:
        q = sp.series(local_phase, eps, 0, max(2, terms + 1)).removeO()
        local_amp = sp.series(amplitude.subs(substitutions), eps, 0, max(1, terms)).removeO()
        correction = sp.series(sp.exp(-q), eps, 0, max(1, terms)).removeO()
        product = sp.series(local_amp * correction, eps, 0, max(1, terms)).removeO()
    except SYMBOLIC_ERRORS as exc:
        raise NotImplementedError("could not form quartic transition expansion") from exc
    pieces = []
    for j in range(terms):
        coeff = analytic_powsimp(sp.expand(product).coeff(eps, j))
        if coeff == 0:
            continue
        pieces.append(eps ** (j + 1) * sp.Integral(coeff, (u, -sp.oo, sp.oo)))
    if not pieces:
        raise NotImplementedError("empty coalescing-saddle transition expansion")
    outside = analytic_powsimp(prefactor.subs(parameter, eps**-4) * sp.exp(-(eps**-4) * phi0_mu))
    expression_eps = analytic_powsimp(outside * sp.Add(*pieces))
    expression = expression_eps.xreplace({eps: parameter ** (-sp.Rational(1, 4))})
    expression = expression.xreplace(
        {transition_symbol: (control_parameter - coalescence_value) * sp.sqrt(parameter)}
    )
    record_symbolic_event("stat_coalescing_saddles")
    return _result_from_expression(
        expression,
        parameter,
        sp.oo,
        terms,
        method="laplace-coalescing-quartic",
        status="FORMAL",
        reduction=sp.Integral(integrand, (variable, domain.start, domain.end)),
        integration_variable=variable,
        domain=domain,
        conditions=(sp.Gt(derivs[3], 0), sp.Ne(mixed, 0)),
    )


def laplace_asymptotic_integral(
    integrand: sp.Expr,
    variable: sp.Symbol,
    domain: sp.Interval | tuple[sp.Expr, sp.Expr],
    *,
    parameter: sp.Symbol,
    point: sp.Expr = sp.oo,
    terms: int = 4,
    certify: bool = True,
    _extracted_form: tuple[sp.Expr, sp.Expr, sp.Expr, bool] | None = None,
) -> StatisticalAsymptoticResult:
    """Expand a one-dimensional Laplace integral at ``parameter -> +oo``.

    Supported geometries include nondegenerate and even-order degenerate
    interior minima plus monotone/stationary finite endpoints.  When ``certify``
    is true, real polynomial phases are checked against a global Laplace
    theorem class; otherwise the local expansion remains formal.
    """

    if point is not sp.oo:
        raise NotImplementedError("Laplace asymptotics target parameter -> +oo")
    if terms < 1:
        raise ValueError("terms must be positive")
    if not isinstance(variable, sp.Symbol) or not isinstance(parameter, sp.Symbol):
        raise TypeError("variable and parameter must be symbols")
    if not isinstance(domain, sp.Interval):
        domain = sp.Interval(*domain)
    extracted = _extracted_form
    if extracted is None:
        extracted = _extract_laplace_form(sp.sympify(integrand), variable, parameter, terms=terms)
    if extracted is None:
        raise NotImplementedError("integrand is not a supported A(x,p)*exp(-p*phi(x)) Laplace form")
    prefactor, amplitude, phase, truncated_amplitude = extracted
    # Principal-branch logarithms arising from positive lattice/Stirling
    # factors can be expanded exactly on the open integration interval.  This
    # keeps the phase branch-correct while exposing entropy derivatives such as
    # log(x/(1-x)) to the stationary-point solver.
    interior_assumptions = _interior_domain_assumptions(domain, variable)
    phase = analytic_powsimp(sp.expand(power_expand_exact(phase, interior_assumptions)))

    derivative = sp.simplify(analytic_powsimp(sp.diff(phase, variable)))
    stationary = _stationary_points(phase, variable)
    local_candidates: list[tuple[str, sp.Expr, sp.Expr]] = []
    for candidate in stationary:
        if _contains(domain, candidate) is not True:
            continue
        if candidate == domain.start or candidate == domain.end:
            continue
        first = _first_nonzero_derivative_order(phase, variable, candidate)
        if first is None:
            continue
        order, derivative_value = first
        if order % 2 == 0 and bounded_assumption_sign(derivative_value) == 1:
            kind = "saddle" if order == 2 else "degenerate"
            local_candidates.append(
                (kind, candidate, analytic_powsimp(phase.subs(variable, candidate)))
            )

    if domain.start not in (-sp.oo, sp.oo):
        slope = analytic_powsimp(derivative.subs(variable, domain.start))
        slope_sign = bounded_assumption_sign(slope)
        if slope_sign == 1:
            local_candidates.append(
                ("lower", domain.start, analytic_powsimp(phase.subs(variable, domain.start)))
            )
        elif slope == 0:
            first = _first_nonzero_derivative_order(phase, variable, domain.start)
            if first is not None and bounded_assumption_sign(first[1]) == 1:
                local_candidates.append(
                    (
                        "degenerate-lower",
                        domain.start,
                        analytic_powsimp(phase.subs(variable, domain.start)),
                    )
                )
    if domain.end not in (-sp.oo, sp.oo):
        slope = analytic_powsimp(derivative.subs(variable, domain.end))
        slope_sign = bounded_assumption_sign(slope)
        if slope_sign == -1:
            local_candidates.append(
                ("upper", domain.end, analytic_powsimp(phase.subs(variable, domain.end)))
            )
        elif slope == 0:
            first = _first_nonzero_derivative_order(phase, variable, domain.end)
            if first is not None:
                order, value = first
                oriented = analytic_powsimp((-1) ** order * value)
                if bounded_assumption_sign(oriented) == 1:
                    local_candidates.append(
                        (
                            "degenerate-upper",
                            domain.end,
                            analytic_powsimp(phase.subs(variable, domain.end)),
                        )
                    )
    if not local_candidates:
        raise NotImplementedError(
            "no supported nondegenerate saddle or monotone endpoint was found"
        )

    dominant = [local_candidates[0]]
    best_value = local_candidates[0][2]
    for item in local_candidates[1:]:
        difference = analytic_powsimp(item[2] - best_value)
        if difference == 0:
            dominant.append(item)
            continue
        sign = bounded_assumption_sign(difference)
        if sign == -1:
            dominant = [item]
            best_value = item[2]
        elif sign == 1:
            continue
        else:
            raise NotImplementedError("dominant Laplace points have unresolved phase ordering")

    contributions = []
    conditions: list[sp.Expr] = []
    methods = []
    for kind, location, _ in dominant:
        if kind == "saddle":
            expanded = _interior_saddle_expansion(
                prefactor, amplitude, phase, variable, location, parameter, terms
            )
            metric = "stat_laplace_saddles"
            method = "laplace-interior-saddle"
        elif kind == "degenerate":
            degenerate = _degenerate_local_expansion(
                prefactor, amplitude, phase, variable, location, parameter, terms
            )
            expanded = None if degenerate is None else degenerate[:2]
            metric = "stat_degenerate_saddles"
            local_order = None if degenerate is None else degenerate[2]
            method = (
                "laplace-degenerate-saddle"
                if local_order is None
                else f"laplace-degenerate-saddle-order-{local_order}"
            )
        elif kind in {"degenerate-lower", "degenerate-upper"}:
            degenerate = _degenerate_local_expansion(
                prefactor,
                amplitude,
                phase,
                variable,
                location,
                parameter,
                terms,
                half_line=True,
                lower=kind == "degenerate-lower",
            )
            expanded = None if degenerate is None else degenerate[:2]
            metric = "stat_degenerate_saddles"
            local_order = None if degenerate is None else degenerate[2]
            side = "lower" if kind == "degenerate-lower" else "upper"
            method = f"laplace-{side}-degenerate-endpoint-order-{local_order}"
        else:
            expanded = _endpoint_expansion(
                prefactor,
                amplitude,
                phase,
                variable,
                location,
                parameter,
                terms,
                lower=kind == "lower",
            )
            metric = "stat_laplace_endpoints"
            method = "laplace-lower-endpoint" if kind == "lower" else "laplace-upper-endpoint"
        if expanded is None:
            raise NotImplementedError("dominant Laplace point could not be expanded")
        expression, local_conditions = expanded
        contributions.append(expression)
        conditions.extend(local_conditions)
        methods.append(method)
        record_symbolic_event(metric)

    expression = analytic_powsimp(sp.Add(*contributions))
    method = methods[0] if len(methods) == 1 else "laplace-co-dominant-points"
    certificate = (
        _polynomial_global_laplace_certificate(
            phase, amplitude, prefactor, variable, domain, parameter, dominant, terms, expression
        )
        if certify and not truncated_amplitude
        else None
    )
    certified = certificate is not None and certificate.certified
    return _result_from_expression(
        expression,
        parameter,
        point,
        terms,
        method=method,
        status="CERTIFIED" if certified else "FORMAL",
        reduction=sp.Integral(integrand, (variable, domain.start, domain.end)),
        integration_variable=variable,
        domain=domain,
        conditions=tuple(conditions),
        remainder=certificate.remainder if certified else None,
        certificate=certificate,
    )


def _continuous_statistical_route(
    observable: sp.Expr,
    rv: RandomSymbol,
    domain: sp.Set,
    parameter: sp.Symbol,
    point: sp.Expr,
    terms: int,
    *,
    prefer_laplace: bool,
    allow_exact_special: bool,
) -> StatisticalAsymptoticResult:
    """Dispatch one continuous expectation or probability to exact, density, or Laplace machinery."""
    reduction_integrand, variable = _continuous_reduction(observable, rv, domain)
    reduction = _set_integral(reduction_integrand, variable, domain)
    if not prefer_laplace:
        exact = _evaluate_reduction(reduction)
        if exact is not None:
            usable = _finite_asymptotic_series(exact, parameter, point, terms, complete=True)
            if usable is not None or allow_exact_special:
                return _result_from_expression(
                    exact,
                    parameter,
                    point,
                    terms,
                    method="density-exact-integral",
                    status="EXACT",
                    reduction=reduction,
                    integration_variable=variable,
                    domain=domain,
                )

    transformed = _moving_domain_transform(reduction_integrand, variable, domain, parameter, point)
    if transformed is not None:
        transformed_expr, new_variable, new_domain, rule = transformed
        try:
            result = laplace_asymptotic_integral(
                transformed_expr,
                new_variable,
                new_domain,
                parameter=parameter,
                point=point,
                terms=terms,
            )
            return StatisticalAsymptoticResult(
                result.expression,
                result.parameter,
                result.point,
                "moving-domain/" + result.method,
                result.status,
                result.series,
                reduction,
                new_variable,
                new_domain,
                rule,
                result.conditions,
            )
        except NotImplementedError:
            pass

    if isinstance(domain, sp.Interval):
        try:
            result = laplace_asymptotic_integral(
                reduction_integrand,
                variable,
                domain,
                parameter=parameter,
                point=point,
                terms=terms,
            )
            return StatisticalAsymptoticResult(
                result.expression,
                result.parameter,
                result.point,
                result.method,
                result.status,
                result.series,
                reduction,
                variable,
                domain,
                None,
                result.conditions,
            )
        except NotImplementedError:
            pass

    return StatisticalAsymptoticResult(
        reduction,
        parameter,
        point,
        "density-reduction",
        "UNKNOWN",
        None,
        reduction,
        variable,
        domain,
    )


def _discrete_statistical_route(
    observable: sp.Expr,
    rv: RandomSymbol,
    domain: sp.Set,
    parameter: sp.Symbol,
    point: sp.Expr,
    terms: int,
    *,
    sum_method: str = "auto",
) -> StatisticalAsymptoticResult:
    """Dispatch one discrete expectation or probability to exact, sum, Stirling, or lattice-saddle machinery."""
    observed, pmf, variable = _discrete_components(observable, rv)
    pmf = _restrict_piecewise_to_support(pmf, domain, variable)
    summand = analytic_powsimp(observed * pmf)
    reduction = _set_sum(summand, variable, domain)
    exact = _evaluate_reduction(reduction)
    if exact is not None and sum_method in {"auto", "exact", "pmf"}:
        return _result_from_expression(
            exact,
            parameter,
            point,
            terms,
            method="pmf-exact-sum",
            status="EXACT",
            reduction=reduction,
            integration_variable=variable,
            domain=domain,
        )
    if isinstance(reduction, sp.Sum) and len(reduction.limits) == 1:
        from .sums import _asymptotic_sum_impl

        _, lower, upper = reduction.limits[0]
        requested = sum_method if sum_method in SUM_METHODS else "auto"
        saddle_summand = summand
        normalization = None
        if requested in {"auto", "saddle"} and pmf.has(sp.factorial, sp.gamma, sp.binomial):
            from .stirling import normalize_positive_pmf

            support_assumptions = _support_assumptions(domain, variable)
            try:
                normalization = normalize_positive_pmf(
                    pmf,
                    variable=variable,
                    parameter=parameter,
                    point=point,
                    terms=max(2, terms),
                    assumptions=support_assumptions,
                )
            except (ValueError, NotImplementedError):
                normalization = None
            else:
                saddle_summand = analytic_powsimp(observed * normalization.expression)
                record_symbolic_event("stat_stirling_routes")

        if normalization is not None and observed == 1:
            distribution = rv.pspace.distribution
            if (
                type(distribution).__name__ == "BinomialDistribution"
                and len(distribution.args) >= 4
            ):
                count, success_probability, success_value, failure_value = distribution.args[:4]
                bounds = _discrete_bounds(domain)
                if bounds is not None and success_value == 1 and failure_value == 0:
                    from .binomial_lattice import binomial_lattice_tail_expansion

                    try:
                        lattice_tail = binomial_lattice_tail_expansion(
                            normalization,
                            variable=variable,
                            parameter=parameter,
                            probability=success_probability,
                            lower=bounds[0],
                            upper=bounds[1],
                            count=count,
                            terms=terms,
                        )
                    except (ValueError, TypeError, NotImplementedError):
                        lattice_tail = None
                    if lattice_tail is not None:
                        series = _finite_asymptotic_series(
                            lattice_tail.expression, parameter, point, terms, complete=False
                        )
                        expression = (
                            series.truncate() if series is not None else lattice_tail.expression
                        )
                        record_symbolic_event("binomial_tail_routes")
                        return StatisticalAsymptoticResult(
                            expression,
                            parameter,
                            point,
                            f"pmf/stirling+binomial-lattice-{lattice_tail.side}-endpoint",
                            "CERTIFIED",
                            series,
                            reduction,
                            variable,
                            domain,
                            None,
                            lattice_tail.conditions,
                            lattice_tail.remainder,
                            lattice_tail.certificate,
                            lattice_tail,
                        )

        result = _asymptotic_sum_impl(
            saddle_summand,
            variable,
            lower,
            upper,
            parameter=parameter,
            point=point,
            terms=terms,
            method=requested,
            normalization=normalization,
        )
        if result.status != "UNKNOWN":
            status = result.status
            method_name = "pmf/" + result.method
            remainder = result.remainder
            if normalization is not None:
                # The Stieltjes bound is pointwise and exact, but lifting it
                # through an entire moving lattice sum requires a separate
                # uniform theorem.  Until that theorem is proved, retain the
                # useful normalized saddle expansion but do not overstate the
                # overall sum as certified.
                if status == "CERTIFIED":
                    status = "FORMAL"
                    remainder = None
                method_name = "pmf/stirling+" + result.method
            return StatisticalAsymptoticResult(
                result.expression,
                parameter,
                point,
                method_name,
                status,
                result.series,
                reduction,
                variable,
                domain,
                None,
                (),
                remainder,
                result.certificate
                if isinstance(result.certificate, LaplaceRemainderCertificate)
                else None,
                normalization,
            )
    return StatisticalAsymptoticResult(
        reduction,
        parameter,
        point,
        "pmf-reduction",
        "UNKNOWN",
        None,
        reduction,
        variable,
        domain,
    )


def asymptotic_expectation(
    expr: sp.Expr,
    random_symbol: RandomSymbol | None = None,
    *,
    parameter: sp.Symbol,
    point: sp.Expr = sp.oo,
    terms: int = 4,
    method: Literal[
        "auto",
        "exact",
        "density",
        "pmf",
        "laplace",
        "sum",
        "series",
        "summation-by-parts",
        "saddle",
        "euler-maclaurin",
        "mellin",
        "riemann",
        "zeilberger",
        "poisson",
        "oscillatory",
    ] = "auto",
    bindings: dict[object, object] | None = None,
    condition: sp.Expr | None = None,
) -> StatisticalAsymptoticResult:
    """Compute the asymptotic expectation of ``expr``.

    The first argument is always the expression being averaged and may depend
    on one or more SymPy random variables. ``bindings`` may map ordinary SymPy
    symbols in that expression to ``RandomSymbol`` objects; ``condition`` forms
    a conditional expectation through SymPy's probability-space machinery.
    ``auto`` first asks SymPy for the exact joint expectation. If that route does
    not settle the problem, the package provides density/PMF and
    Laplace/saddle fallbacks for a single random variable; ``random_symbol`` can
    disambiguate that fallback. ``laplace`` skips exact integration after density
    reduction so concentration asymptotics can be inspected directly.
    """

    if terms < 1:
        raise ValueError("terms must be positive")
    expr = _prepare_statistical_query(
        expr,
        bindings=bindings,
        condition=condition,
        random_symbol=random_symbol,
        method=method,
        label="expectation",
    )
    # Exact SymPy expectation can handle products of multiple random variables.
    # Resolve a single RV only when we actually need the one-dimensional
    # density/PMF fallback.  This is important for covariance and joint moments.
    if method in {"auto", "exact"}:
        exact = _try_exact_expectation(expr, parameter)
        if exact is not None:
            usable = _finite_asymptotic_series(exact, parameter, point, terms, complete=True)
            if usable is not None or method == "exact":
                return _result_from_expression(
                    exact, parameter, point, terms, method="exact-expectation", status="EXACT"
                )
        if method == "exact":
            return StatisticalAsymptoticResult(
                E(expr, evaluate=False), parameter, point, "exact-expectation", "UNKNOWN"
            )

    rv = _resolve_random_symbol(expr, random_symbol)
    support = _support(rv)
    kind = _probability_space_kind(rv)
    _validate_route_method(method, kind)
    if kind == "continuous":
        return _continuous_statistical_route(
            expr,
            rv,
            support,
            parameter,
            point,
            terms,
            prefer_laplace=method == "laplace",
            allow_exact_special=method == "density",
        )
    return _discrete_statistical_route(
        expr, rv, support, parameter, point, terms, sum_method=method
    )


def asymptotic_probability(
    event: sp.Expr,
    random_symbol: RandomSymbol | None = None,
    *,
    parameter: sp.Symbol,
    point: sp.Expr = sp.oo,
    terms: int = 4,
    method: Literal[
        "auto",
        "exact",
        "density",
        "pmf",
        "laplace",
        "sum",
        "series",
        "summation-by-parts",
        "saddle",
        "euler-maclaurin",
        "mellin",
        "riemann",
        "zeilberger",
        "poisson",
        "oscillatory",
    ] = "auto",
    bindings: dict[object, object] | None = None,
    condition: sp.Expr | None = None,
) -> StatisticalAsymptoticResult:
    """Compute an asymptotic probability for an event.

    ``event`` may contain SymPy ``RandomSymbol`` objects directly, or ordinary
    symbols may be mapped to random symbols with ``bindings``.  Exact SymPy
    probability is attempted before a one-random-variable fallback, so exact
    joint events are supported when SymPy can evaluate them.  ``condition``
    requests conditional probability.  Structural density/PMF and Laplace
    fallbacks remain one-dimensional.
    """

    if terms < 1:
        raise ValueError("terms must be positive")
    event = _prepare_statistical_query(
        event,
        bindings=bindings,
        condition=condition,
        random_symbol=random_symbol,
        method=method,
        label="probability",
    )
    if method in {"auto", "exact"}:
        exact = _try_exact_probability(event, parameter)
        if exact is not None:
            series = _finite_asymptotic_series(exact, parameter, point, terms, complete=True)
            # Special-function tails such as erfc(a*sqrt(n)) are exact but not
            # useful to the asymptotic algebra.  In auto mode continue to the
            # defining density so the Laplace route can expose the tail scale.
            if series is not None or method == "exact":
                return _result_from_expression(
                    exact, parameter, point, terms, method="exact-probability", status="EXACT"
                )
        if method == "exact":
            return StatisticalAsymptoticResult(
                P(event, evaluate=False), parameter, point, "exact-probability", "UNKNOWN"
            )

    # Only the structural fallback requires one distinguished random variable.
    rv = _resolve_random_symbol(event, random_symbol)

    event_set = _event_domain(event, rv)
    if event_set is None:
        return StatisticalAsymptoticResult(
            P(event, evaluate=False), parameter, point, "event-reduction", "UNKNOWN"
        )
    domain = sp.Intersection(_support(rv), event_set)
    if domain is sp.S.EmptySet:
        return _result_from_expression(
            sp.S.Zero, parameter, point, terms, method="empty-event", status="EXACT"
        )
    kind = _probability_space_kind(rv)
    _validate_route_method(method, kind)
    if kind == "continuous":
        if isinstance(domain, sp.FiniteSet):
            return _result_from_expression(
                sp.S.Zero, parameter, point, terms, method="continuous-point-event", status="EXACT"
            )
        return _continuous_statistical_route(
            sp.S.One,
            rv,
            domain,
            parameter,
            point,
            terms,
            prefer_laplace=method == "laplace",
            allow_exact_special=method == "density",
        )
    return _discrete_statistical_route(
        sp.S.One, rv, domain, parameter, point, terms, sum_method=method
    )
