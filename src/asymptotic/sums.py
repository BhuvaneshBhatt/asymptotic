"""Asymptotic summation by structural discrete reductions.

The public dispatcher combines several mathematically different
routes rather than forcing all sums through a generic symbolic ``Sum``.  Exact
summation is preferred when cheap enough for SymPy, followed by termwise
parameter expansion, summation by parts, Euler–Maclaurin, Mellin-pole
expansion for supported infinite sums, scaled Riemann sums, and lattice
saddles.  Routes that need unproved interchange or contour hypotheses are
reported as formal rather than certified.
"""

from __future__ import annotations

import builtins
from dataclasses import dataclass
from typing import Literal

import sympy as sp
from funcprops import normalize_assumptions
from sympy.concrete.gosper import gosper_sum

from ._polynomial_bounds import bounded_degree
from ._power_simplify import analytic_powsimp
from ._symbolic_errors import SYMBOLIC_ERRORS
from ._symbolic_policy import bounded_assumption_sign, bounded_limit, bounded_primitive
from .context import AsymptoticContext
from .instrumentation import record_symbolic_event
from .remainder import Remainder
from .sum_advanced import (
    MellinShiftCertificate,
    UniformSummationCertificate,
    endpoint_recurrence,
    fixed_finite_uniformity,
    gamma_vertical_decay,
    geometric_uniformity,
    linear_exponential_sum,
    poisson_gaussian_sum,
    separable_multisum,
    zeilberger_recurrence,
)
from .theorem_registry import CertificationStrength, Theorem, TheoremRegistry
from .transseries import TransseriesExpansion, transseries_from_expression

SumStatus = Literal["EXACT", "CERTIFIED", "FORMAL", "UNKNOWN"]
SumMethod = Literal[
    "auto",
    "exact",
    "series",
    "summation-by-parts",
    "euler-maclaurin",
    "mellin",
    "riemann",
    "saddle",
    "zeilberger",
    "poisson",
    "oscillatory",
]

SUM_METHODS = frozenset(
    {
        "auto",
        "exact",
        "series",
        "summation-by-parts",
        "euler-maclaurin",
        "mellin",
        "riemann",
        "saddle",
        "zeilberger",
        "poisson",
        "oscillatory",
    }
)
DISCRETE_STAT_METHODS = SUM_METHODS | frozenset({"pmf", "sum"})


@dataclass(frozen=True)
class EulerMaclaurinCertificate:
    """Replayable evidence for a SymPy Euler–Maclaurin remainder estimate."""

    reduction: sp.Sum
    m: int
    n: int
    error: sp.Expr
    remainder: Remainder
    approximation: sp.Expr
    expression: sp.Expr
    theorem: str | None

    def replay(self) -> bool | None:
        if not self.remainder.is_certified:
            return False
        try:
            approximation, error = self.reduction.euler_maclaurin(m=self.m, n=self.n)
        except (
            ValueError,
            TypeError,
            NotImplementedError,
            AttributeError,
            *SYMBOLIC_ERRORS,
        ):
            return None
        bound = sp.Abs(approximation - self.expression) + sp.Abs(error)
        return (
            _euler_maclaurin_theorem(self.reduction, self.n) == self.theorem
            and self.theorem is not None
            and sp.simplify(error - self.error) == 0
            and sp.simplify(approximation - self.approximation) == 0
            and sp.simplify(
                bound - (0 if self.remainder.is_exact else self.remainder.scale)
            )
            == 0
        )


@dataclass(frozen=True)
class SumResult:
    """Finite asymptotic description of a parameter-dependent discrete sum."""

    expression: sp.Expr
    variable: sp.Symbol | tuple[sp.Symbol, ...]
    lower: sp.Expr | tuple[sp.Expr, ...]
    upper: sp.Expr | tuple[sp.Expr, ...]
    parameter: sp.Symbol
    point: sp.Expr
    method: str
    status: SumStatus
    series: TransseriesExpansion | None = None
    remainder: Remainder | None = None
    reduction: sp.Sum | None = None
    transformation: tuple[sp.Symbol, sp.Expr] | None = None
    certificate: object | None = None

    @property
    def certified(self) -> bool:
        return self.status in ("EXACT", "CERTIFIED")

    def truncate(self, terms: int | None = None) -> sp.Expr:
        if self.series is None:
            return self.expression
        return self.series.truncate(terms)


def _series_result(
    expression: sp.Expr,
    variable: sp.Symbol | tuple[sp.Symbol, ...],
    lower: sp.Expr | tuple[sp.Expr, ...],
    upper: sp.Expr | tuple[sp.Expr, ...],
    parameter: sp.Symbol,
    point: sp.Expr,
    method: str,
    status: SumStatus,
    *,
    remainder: Remainder | None,
    reduction: sp.Sum,
    transformation: tuple[sp.Symbol, sp.Expr] | None = None,
    certificate: object | None = None,
    terms: int = 4,
) -> SumResult:
    series = None
    try:
        series = transseries_from_expression(
            analytic_powsimp(expression),
            parameter,
            point=point,
            complete=status == "EXACT" and (remainder is None or remainder.is_exact),
            remainder=remainder,
        ).prefix(terms)
    except (TypeError, ValueError, NotImplementedError):
        pass
    return SumResult(
        analytic_powsimp(expression),
        variable,
        lower,
        upper,
        parameter,
        point,
        method,
        status,
        series,
        remainder,
        reduction,
        transformation,
        certificate,
    )


def _exact_sum(reduction: sp.Sum) -> sp.Expr | None:
    try:
        value = reduction.doit()
    except SYMBOLIC_ERRORS:
        return None
    if value == reduction or value.has(sp.Sum):
        return None
    return sp.sympify(value)


def _termwise_series(
    summand: sp.Expr,
    variable: sp.Symbol,
    lower: sp.Expr,
    upper: sp.Expr,
    parameter: sp.Symbol,
    point: sp.Expr,
    terms: int,
) -> tuple[sp.Expr, UniformSummationCertificate | None] | None:
    """Expand in the external parameter and sum the finite prefix termwise.

    The bounds must be independent of the expansion parameter.  This keeps the
    transformation structural: moving-boundary effects belong to
    Euler–Maclaurin or a scaled-lattice route instead.
    """

    if parameter in sp.Tuple(lower, upper).free_symbols:
        return None
    try:
        series = sp.series(summand, parameter, point, terms).removeO()
    except (TypeError, ValueError, NotImplementedError, *SYMBOLIC_ERRORS):
        return None
    if series == summand and parameter not in summand.free_symbols:
        return None
    total = sp.S.Zero
    for term in sp.Add.make_args(sp.expand(series)):
        value = _exact_sum(sp.Sum(term, (variable, lower, upper)))
        if value is None:
            return None
        total += value
    try:
        total = sp.series(total, parameter, point, terms).removeO()
    except (TypeError, ValueError, NotImplementedError, *SYMBOLIC_ERRORS):
        pass
    record_symbolic_event("sum_series")
    cert = fixed_finite_uniformity(lower, upper, parameter)
    if cert is None:
        cert = geometric_uniformity(
            summand, variable, lower, upper, parameter, point, terms
        )
    return analytic_powsimp(total), cert


def _boundary_value(
    expr: sp.Expr, variable: sp.Symbol, bound: sp.Expr, *, limit_evaluator=bounded_limit
) -> sp.Expr | None:
    """Evaluate one summation-by-parts boundary without an unrestricted limit."""

    if bound not in (-sp.oo, sp.oo):
        return analytic_powsimp(expr.xreplace({variable: bound}))
    if bound is sp.oo and expr.has(sp.binomial):
        try:
            combinatorial = sp.combsimp(expr)
        except (TypeError, ValueError, NotImplementedError):
            combinatorial = expr
        finite_support = False
        for atom in combinatorial.atoms(sp.binomial):
            top, bottom = atom.args
            if (
                variable not in top.free_symbols
                and top.is_integer is True
                and top.is_nonnegative is True
                and variable in bottom.free_symbols
            ):
                try:
                    bottom_poly = sp.Poly(bottom, variable)
                except sp.PolynomialError:
                    continue
                if bottom_poly.degree() == 1 and bottom_poly.LC() > 0:
                    finite_support = True
                    break
        if finite_support:
            return sp.S.Zero
        expr = combinatorial
    value = limit_evaluator(expr, variable, bound, allow_general=True)
    if value is None or value.has(sp.Limit) or value in (sp.oo, -sp.oo, sp.zoo, sp.nan):
        return None
    return analytic_powsimp(value)


def _summation_by_parts(
    summand: sp.Expr,
    variable: sp.Symbol,
    lower: sp.Expr,
    upper: sp.Expr,
) -> tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr] | None:
    """Return one exact Abel/summation-by-parts transformation.

    For ``summand = f(k) g(k)`` this seeks a Gosper antidifference ``G`` with
    ``G(k+1)-G(k)=f(k)`` and applies

    ``sum ΔG(k) g(k) = [G(k+1)g(k)] - sum G(k+1) Δg(k)``.

    The transformed residual sum is returned unevaluated so the caller can
    route it through the ordinary asymptotic dispatcher.
    """

    factors = sp.Mul.make_args(summand)
    if len(factors) < 2:
        return None

    def factor_key(factor: sp.Expr) -> tuple[int, int]:
        try:
            ratio = analytic_powsimp(factor.xreplace({variable: variable + 1}) / factor)
        except (TypeError, ValueError, NotImplementedError):
            return (2, int(sp.count_ops(factor, visual=False)))
        return (
            0 if variable not in ratio.free_symbols else 1,
            int(sp.count_ops(factor, visual=False)),
        )

    for factor in sorted(factors, key=factor_key):
        other = analytic_powsimp(summand / factor)
        try:
            primitive = gosper_sum(factor, variable)
        except (TypeError, ValueError, NotImplementedError, *SYMBOLIC_ERRORS):
            primitive = None
        if primitive is None:
            continue
        delta_other = analytic_powsimp(other.xreplace({variable: variable + 1}) - other)
        if delta_other == 0:
            continue
        upper_term = _boundary_value(
            primitive.xreplace({variable: variable + 1}) * other,
            variable,
            upper,
        )
        lower_term = _boundary_value(primitive * other, variable, lower)
        if upper_term is None or lower_term is None:
            continue
        residual = analytic_powsimp(
            -primitive.xreplace({variable: variable + 1}) * delta_other
        )
        residual_upper = upper if upper is sp.oo else analytic_powsimp(upper - 1)
        record_symbolic_event("sum_parts")
        return upper_term - lower_term, residual, lower, residual_upper
    return None


def _summation_by_parts_expansion(
    summand: sp.Expr,
    variable: sp.Symbol,
    lower: sp.Expr,
    upper: sp.Expr,
    terms: int,
) -> sp.Expr | None:
    """Build a finite Abel-transform prefix by repeated summation by parts.

    This is a formal route.  Each transformation is exact, but
    the final residual sum is omitted only after at least one successful
    descent; certification of that discarded residual requires additional
    monotonicity/sign information that is not inferred here.
    """

    current = summand
    current_lower = lower
    current_upper = upper
    prefix = sp.S.Zero
    transformed = False
    for _ in range(max(1, terms)):
        step = _summation_by_parts(current, variable, current_lower, current_upper)
        if step is None:
            break
        boundary, current, current_lower, current_upper = step
        prefix += boundary
        transformed = True
    if not transformed:
        return None
    return analytic_powsimp(prefix)


def _mellin_candidates(
    transform: sp.Expr, s: sp.Symbol, terms: int
) -> tuple[sp.Expr, ...]:
    """Extract a finite deterministic pole set from Gamma and zeta factors."""

    candidates: set[sp.Expr] = set()
    for zeta in transform.atoms(sp.zeta):
        arg = zeta.args[0]
        try:
            poly = sp.Poly(arg - 1, s)
        except sp.PolynomialError:
            continue
        if poly.degree() == 1:
            a, b = poly.all_coeffs()
            if a != 0:
                candidates.add(sp.cancel(-b / a))
    for gamma in transform.atoms(sp.gamma):
        arg = gamma.args[0]
        try:
            poly = sp.Poly(arg, s)
        except sp.PolynomialError:
            continue
        if poly.degree() != 1:
            continue
        a, b = poly.all_coeffs()
        if a == 0:
            continue
        for m in range(max(2, terms)):
            candidates.add(sp.cancel((-m - b) / a))
    numeric = [value for value in candidates if value.is_real is not False]
    return tuple(sorted(numeric, key=sp.default_sort_key, reverse=True))


def _simple_zeta_residue(expr: sp.Expr, s: sp.Symbol, pole: sp.Expr) -> sp.Expr | None:
    """Handle a simple ``zeta(a*s+b)`` pole without invoking a general limit."""

    singular = []
    for zeta in expr.atoms(sp.zeta):
        arg = zeta.args[0]
        if sp.simplify(arg.subs(s, pole) - 1) == 0:
            singular.append(zeta)
    if len(singular) != 1:
        return None
    zeta = singular[0]
    derivative = sp.diff(zeta.args[0], s)
    slope = derivative.subs(s, pole)
    if slope == 0:
        return None
    regular = sp.cancel(expr / zeta)
    value = regular.subs(s, pole)
    if value.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        return None
    return analytic_powsimp(value / slope)


def _proved_lt(left: sp.Expr, right: sp.Expr) -> bool:
    if left == -sp.oo or right == sp.oo:
        return True
    return bounded_assumption_sign(sp.simplify(right - left)) == 1


def _summed_mellin_strip(
    transform: sp.Expr, strip: tuple[sp.Expr, sp.Expr], s: sp.Symbol
) -> tuple[sp.Expr, sp.Expr] | None:
    """Intersect a Mellin strip with Dirichlet convergence half-planes."""

    lo, hi = map(sp.sympify, strip)
    for zeta in transform.atoms(sp.zeta):
        arg = zeta.args[0]
        slope = sp.simplify(sp.diff(arg, s))
        intercept = sp.simplify(arg - slope * s)
        if (
            s in slope.free_symbols
            or s in intercept.free_symbols
            or slope.is_real is not True
        ):
            return None
        if slope.is_positive is True:
            lo = sp.Max(lo, sp.simplify((1 - intercept) / slope))
        elif slope.is_negative is True:
            hi = sp.Min(hi, sp.simplify((1 - intercept) / slope))
        else:
            return None
    if not _proved_lt(lo, hi):
        return None
    return sp.simplify(lo), sp.simplify(hi)


def _mellin_pole_structure_is_complete(transform: sp.Expr, s: sp.Symbol) -> bool:
    """Accept certification only when poles come from recognized Gamma/zeta factors."""

    denominator = sp.together(transform).as_numer_denom()[1]
    if s in denominator.free_symbols:
        return False
    special = {func.func for func in transform.atoms(sp.Function)}
    return special <= {sp.gamma, sp.zeta}


def _mellin_sum(
    summand: sp.Expr,
    variable: sp.Symbol,
    lower: sp.Expr,
    upper: sp.Expr,
    parameter: sp.Symbol,
    point: sp.Expr,
    terms: int,
) -> tuple[sp.Expr, MellinShiftCertificate | None] | None:
    """Expand a supported infinite sum by poles of its summed Mellin transform."""

    if point != 0 or upper is not sp.oo or not lower.is_integer:
        return None
    s = sp.Dummy("mellin_s")
    try:
        transform_data = sp.mellin_transform(summand, parameter, s)
    except (TypeError, ValueError, NotImplementedError, *SYMBOLIC_ERRORS):
        return None
    if not isinstance(transform_data, tuple) or len(transform_data) < 1:
        return None
    transform = sp.sympify(transform_data[0])
    strip = transform_data[1] if len(transform_data) > 1 else None
    if transform.has(sp.MellinTransform):
        return None
    summed = _exact_sum(sp.Sum(transform, (variable, lower, upper)))
    if summed is None and upper is sp.oo and lower.is_integer:
        independent, dependent = transform.as_independent(variable, as_Add=False)
        powers = dependent.as_powers_dict()
        exponent = powers.get(variable)
        if exponent is not None and len(powers) == 1:
            order = analytic_powsimp(-exponent)
            if variable not in order.free_symbols:
                summed = analytic_powsimp(independent * sp.zeta(order, lower))
    if summed is None or not isinstance(strip, tuple) or len(strip) != 2:
        return None
    effective_strip = _summed_mellin_strip(summed, tuple(strip), s)
    if effective_strip is None:
        return None
    lo, hi = effective_strip
    initial_line = sp.simplify(lo + 1) if hi == sp.oo else sp.simplify((lo + hi) / 2)
    candidates = _mellin_candidates(summed, s, max(terms + 4, 8))
    eligible = tuple(pole for pole in candidates if _proved_lt(pole, initial_line))
    contributions = []
    crossed_poles = []
    integrand = summed * parameter ** (-s)
    for pole in eligible:
        residue = _simple_zeta_residue(integrand, s, pole)
        if residue is None:
            try:
                residue = sp.residue(integrand, s, pole)
            except (TypeError, ValueError, NotImplementedError, *SYMBOLIC_ERRORS):
                break
        if residue == 0 or residue.has(sp.Limit):
            break
        contributions.append(residue)
        crossed_poles.append(pole)
        if len(contributions) >= max(2, terms):
            break
    if not contributions:
        return None
    expression = analytic_powsimp(sp.Add(*contributions))
    cert = None
    decay = gamma_vertical_decay(summed, s)
    if (
        parameter.is_positive is True
        and decay.is_positive is True
        and _mellin_pole_structure_is_complete(summed, s)
    ):
        crossed = tuple(crossed_poles)
        last = crossed[-1]
        remaining = eligible[len(crossed) :]
        next_lower = next((pole for pole in remaining if _proved_lt(pole, last)), None)
        shifted = (
            sp.simplify((last + next_lower) / 2)
            if next_lower is not None
            else sp.simplify(last - sp.Rational(1, 2))
        )
        remainder = Remainder.big_o(
            parameter ** (-shifted),
            parameter,
            0,
            source="certified Mellin contour shift with Gamma vertical decay",
        )
        candidate = MellinShiftCertificate(
            effective_strip, initial_line, shifted, crossed, decay, True, remainder
        )
        if candidate.replay():
            cert = candidate
    record_symbolic_event("sum_mellin")
    return expression, cert


def _riemann_sum(
    summand: sp.Expr,
    variable: sp.Symbol,
    lower: sp.Expr,
    upper: sp.Expr,
    parameter: sp.Symbol,
    point: sp.Expr,
    terms: int,
) -> tuple[sp.Expr, tuple[sp.Symbol, sp.Expr]] | None:
    """Return the leading scaled-lattice integral for a Riemann-sum regime."""

    if point is not sp.oo:
        return None
    if parameter not in upper.free_symbols and upper != parameter:
        return None
    x = sp.Dummy(f"{variable}_scaled", real=True)
    lo_expr = analytic_powsimp(lower / parameter)
    hi_expr = analytic_powsimp(upper / parameter)
    lo = bounded_limit(lo_expr, parameter, sp.oo, allow_general=True)
    hi = bounded_limit(hi_expr, parameter, sp.oo, allow_general=True)
    if lo is None or hi is None:
        return None
    scaled = analytic_powsimp(parameter * summand.xreplace({variable: parameter * x}))
    primitive = bounded_primitive(scaled, x, allow_general=True)
    if primitive is None:
        return None
    integral = analytic_powsimp(primitive.subs(x, hi) - primitive.subs(x, lo))
    try:
        expression = sp.series(integral, parameter, point, terms).removeO()
    except (TypeError, ValueError, NotImplementedError, *SYMBOLIC_ERRORS):
        expression = integral
    record_symbolic_event("sum_riemann")
    return analytic_powsimp(expression), (variable, parameter * x)


def _euler_maclaurin_theorem(reduction, order):
    """Recognize families with an actual Euler–Maclaurin remainder bound.

    Polynomial exactness follows when the next even derivative vanishes.
    Positive inverse-power tails have a Laplace integral representation whose
    alternating Bernoulli remainder is bounded by the first omitted term.
    Other summands keep an estimate without a certification claim.
    """
    if len(reduction.limits) != 1:
        return None
    variable, lower, upper = reduction.limits[0]
    summand = reduction.function
    if upper.is_Integer or upper.is_integer is True:
        if (
            lower.is_integer is True
            and bounded_degree(summand, (variable,), 2 * order + 1) is not None
        ):
            polynomial = sp.Poly(summand, variable)
            if polynomial.degree() <= 2 * order + 1:
                return "finite-polynomial"
    if upper is not sp.oo or lower.is_integer is not True:
        return None
    independent, dependent = summand.as_independent(variable, as_Add=False)
    base, exponent = dependent.as_base_exp()
    shift = base - variable
    if (
        exponent.is_Rational is True
        and exponent < -1
        and not shift.has(variable)
        and (lower + shift).is_positive is True
        and independent.is_finite is True
    ):
        return "positive-inverse-power-tail"
    return None


def _euler_maclaurin(
    reduction: sp.Sum,
    parameter: sp.Symbol,
    point: sp.Expr,
    terms: int,
) -> tuple[sp.Expr, Remainder, EulerMaclaurinCertificate] | None:
    """Combine the Euler–Maclaurin estimate with the discarded prefix tail.

    The estimate returned by SymPy belongs to the full approximation. Its
    difference from the returned finite prefix must also enter the error bound.
    """

    m = max(1, terms)
    n = max(1, terms)
    try:
        approximation, error = reduction.euler_maclaurin(m=m, n=n)
    except (
        ValueError,
        TypeError,
        NotImplementedError,
        AttributeError,
        *SYMBOLIC_ERRORS,
    ):
        return None
    if approximation.has(sp.Integral, sp.Sum) or error.has(sp.Integral, sp.Sum):
        return None
    try:
        expanded = sp.series(
            approximation, parameter, point, max(2, terms + 1)
        ).removeO()
    except SYMBOLIC_ERRORS:
        expanded = approximation
    expanded = analytic_powsimp(expanded)
    bound = sp.Abs(approximation - expanded) + sp.Abs(error)
    theorem = _euler_maclaurin_theorem(reduction, n)
    remainder = (
        Remainder.exact_zero(parameter, point, source="exact Euler–Maclaurin prefix")
        if bound == 0 and theorem is not None
        else Remainder.big_o(
            bound,
            parameter,
            point,
            source="Euler–Maclaurin estimate plus discarded approximation tail",
        )
        if theorem is not None
        else Remainder.unknown(
            parameter,
            point,
            source="Euler–Maclaurin estimate lacks a proved remainder bound",
        )
    )
    record_symbolic_event("euler_maclaurin_routes")
    certificate = EulerMaclaurinCertificate(
        reduction, m, n, error, remainder, approximation, expanded, theorem
    )
    return analytic_powsimp(expanded), remainder, certificate


def _scaled_lattice_domain(
    lower: sp.Expr, upper: sp.Expr, parameter: sp.Symbol
) -> sp.Interval | None:
    def scaled(bound: sp.Expr) -> sp.Expr:
        if bound in (-sp.oo, sp.oo):
            return bound
        return analytic_powsimp(bound / parameter)

    lo = scaled(lower)
    hi = scaled(upper)
    if parameter in sp.Tuple(lo, hi).free_symbols:
        return None
    return sp.Interval(lo, hi)


def _lattice_saddle(
    summand: sp.Expr,
    variable: sp.Symbol,
    lower: sp.Expr,
    upper: sp.Expr,
    parameter: sp.Symbol,
    point: sp.Expr,
    terms: int,
    normalization=None,
):
    """Reduce a supported lattice sum to its interior or endpoint saddle expansion."""
    if point is not sp.oo:
        return None
    domain = _scaled_lattice_domain(lower, upper, parameter)
    if domain is None:
        return None
    x = sp.Dummy(f"{variable}_scaled", real=True)
    extracted_form = None
    if normalization is not None:
        from .stirling import scaled_stirling_lattice_form

        try:
            form = scaled_stirling_lattice_form(
                normalization,
                variable=variable,
                parameter=parameter,
                domain=domain,
                terms=terms,
            )
        except (ValueError, NotImplementedError):
            form = None
        if form is not None:
            x = form.variable
            try:
                observable = analytic_powsimp(summand / normalization.expression)
                scaled_observable = analytic_powsimp(
                    observable.xreplace({variable: parameter * x})
                )
            except (ValueError, TypeError, NotImplementedError):
                scaled_observable = sp.S.One
            amplitude = analytic_powsimp(form.amplitude * scaled_observable)
            scaled_integrand = analytic_powsimp(
                amplitude * sp.exp(-parameter * form.phase)
            )
            extracted_form = (sp.S.One, amplitude, form.phase, True)
        else:
            scaled_integrand = None
    else:
        scaled_integrand = None
    if scaled_integrand is None:
        scaled = analytic_powsimp(summand.xreplace({variable: parameter * x}))
        # Local lattice spacing is 1/p, so sum ~= p * integral in the scaled
        # coordinate.  Endpoint saddles are corrected separately by discrete
        # routes; this continuous proxy is used only when its geometry applies.
        scaled_integrand = parameter * scaled

    from .probability import laplace_asymptotic_integral

    try:
        integral = laplace_asymptotic_integral(
            scaled_integrand,
            x,
            domain,
            parameter=parameter,
            point=point,
            terms=terms,
            certify=True,
            _extracted_form=extracted_form,
        )
    except NotImplementedError:
        return None
    record_symbolic_event("sum_saddles")
    return integral, x, domain


def _recurrence_sum(
    summand: sp.Expr,
    variable: sp.Symbol,
    lower: sp.Expr,
    upper: sp.Expr,
    parameter: sp.Symbol,
    point: sp.Expr,
    terms: int,
):
    """Generate an exact recurrence and hand it to ``rsolve``."""
    from .rsolve import rsolve

    seq = sp.Function("S")
    endpoint = endpoint_recurrence(summand, variable, lower, upper, parameter)
    if endpoint is not None:
        _unit, rhs = endpoint
        recurrence = seq(parameter + 1) - seq(parameter) - rhs
        initial = _exact_sum(
            sp.Sum(summand, (variable, lower, upper.subs(parameter, 0)))
        )
        conditions = {seq(0): initial} if initial is not None else None
        for initial_data in (conditions, None):
            try:
                solved = rsolve(
                    recurrence,
                    seq(parameter),
                    parameter,
                    point=point,
                    terms=terms,
                    initial_conditions=initial_data,
                )
                return solved.expression, None, solved.status
            except (ValueError, NotImplementedError, TypeError, *SYMBOLIC_ERRORS):
                continue

    if upper is not sp.oo:
        return None
    cert = zeilberger_recurrence(summand, variable, parameter)
    if cert is None:
        return None
    g = analytic_powsimp(cert.rational_certificate * summand)
    lo = _boundary_value(g, variable, lower)
    hi = _boundary_value(g, variable, sp.oo)
    if lo is None or hi is None:
        return None
    boundary = analytic_powsimp(hi - lo)
    recurrence = (
        builtins.sum(c * seq(parameter + j) for j, c in enumerate(cert.coeffs))
        - boundary
    )
    conditions = {}
    for j in range(len(cert.coeffs) - 1):
        value = _exact_sum(sp.Sum(summand.subs(parameter, j), (variable, lower, upper)))
        if value is None:
            conditions = {}
            break
        conditions[seq(j)] = value
    try:
        solved = rsolve(
            recurrence,
            seq(parameter),
            parameter,
            point=point,
            terms=terms,
            initial_conditions=conditions or None,
        )
    except (ValueError, NotImplementedError, TypeError, *SYMBOLIC_ERRORS):
        return None
    record_symbolic_event("sum_zeilberger")
    return solved.expression, cert, solved.status


def _multidimensional_sum(
    summand: sp.Expr,
    variables: tuple[sp.Symbol, ...],
    lowers: tuple[sp.Expr, ...],
    uppers: tuple[sp.Expr, ...],
    *,
    parameter: sp.Symbol,
    point: sp.Expr,
    terms: int,
    method: str,
):
    """Evaluate separable multidimensional sums by exact tensor factorization."""
    if not (len(variables) == len(lowers) == len(uppers)) or not variables:
        raise ValueError(
            "multidimensional variables and bounds must have equal nonzero length"
        )
    factors = separable_multisum(summand, variables)
    if factors is None:
        # Fixed finite boxes admit rigorous termwise expansion by direct nesting.
        if all(
            fixed_finite_uniformity(a, b, parameter) for a, b in zip(lowers, uppers)
        ):
            expr = summand
            try:
                prefix = sp.series(expr, parameter, point, terms).removeO()
                for v, a, b in zip(variables, lowers, uppers):
                    value = _exact_sum(sp.Sum(prefix, (v, a, b)))
                    if value is None:
                        return None
                    prefix = value
                return analytic_powsimp(prefix), "multidimensional-termwise", None
            except (ValueError, TypeError, NotImplementedError, *SYMBOLIC_ERRORS):
                return None
        return None
    expression = sp.S.One
    certificates = []
    for factor, v, a, b in zip(factors, variables, lowers, uppers):
        result = sum(
            factor,
            v,
            a,
            b,
            parameter=parameter,
            point=point,
            terms=terms,
            method=method,
        )
        if result.status == "UNKNOWN":
            return None
        expression *= result.expression
        certificates.append(result.certificate)
    return (
        analytic_powsimp(expression),
        "multidimensional-separable",
        tuple(certificates),
    )


@dataclass(frozen=True)
class _SumRouteProblem:
    summand: sp.Expr
    variable: sp.Symbol
    lower: sp.Expr
    upper: sp.Expr
    parameter: sp.Symbol
    point: sp.Expr
    terms: int
    method: str
    normalization: object | None
    reduction: sp.Sum


def _sum_route_applicable(route: str):
    return lambda problem: problem.method in {"auto", route}


def _sum_route_hypotheses(_problem: _SumRouteProblem) -> tuple[sp.Expr, ...]:
    return ()


def _sum_route_construct(route: str, p: _SumRouteProblem) -> SumResult | None:
    if route == "exact":
        exact = _exact_sum(p.reduction)
        if exact is None:
            return None
        record_symbolic_event("sum_exact")
        return _series_result(
            exact,
            p.variable,
            p.lower,
            p.upper,
            p.parameter,
            p.point,
            "exact-sum",
            "EXACT",
            remainder=Remainder.exact_zero(
                p.parameter, p.point, source="exact summation"
            ),
            reduction=p.reduction,
            terms=p.terms,
        )
    if route == "series":
        data = _termwise_series(
            p.summand, p.variable, p.lower, p.upper, p.parameter, p.point, p.terms
        )
        if data is None:
            return None
        expression, uniformity = data
        status = (
            "CERTIFIED" if uniformity is not None and uniformity.replay() else "FORMAL"
        )
        return _series_result(
            expression,
            p.variable,
            p.lower,
            p.upper,
            p.parameter,
            p.point,
            "termwise-series",
            status,
            remainder=None,
            reduction=p.reduction,
            certificate=uniformity,
            terms=p.terms,
        )
    if route == "summation-by-parts":
        expression = _summation_by_parts_expansion(
            p.summand, p.variable, p.lower, p.upper, p.terms
        )
        if expression is None:
            return None
        return _series_result(
            expression,
            p.variable,
            p.lower,
            p.upper,
            p.parameter,
            p.point,
            "summation-by-parts",
            "FORMAL",
            remainder=None,
            reduction=p.reduction,
            terms=p.terms,
        )
    if route == "zeilberger":
        data = _recurrence_sum(
            p.summand, p.variable, p.lower, p.upper, p.parameter, p.point, p.terms
        )
        if data is None:
            return None
        expression, cert, solve_status = data
        status = solve_status if solve_status in {"EXACT", "CERTIFIED"} else "FORMAL"
        return _series_result(
            expression,
            p.variable,
            p.lower,
            p.upper,
            p.parameter,
            p.point,
            "creative-telescoping",
            status,
            remainder=None,
            reduction=p.reduction,
            certificate=cert,
            terms=p.terms,
        )
    if route == "poisson":
        data = poisson_gaussian_sum(
            p.summand, p.variable, p.lower, p.upper, p.parameter, p.point
        )
        if data is None:
            return None
        expression, remainder = data
        return _series_result(
            expression,
            p.variable,
            p.lower,
            p.upper,
            p.parameter,
            p.point,
            "poisson-summation",
            "CERTIFIED",
            remainder=remainder,
            reduction=p.reduction,
            terms=p.terms,
        )
    if route == "oscillatory":
        expression = linear_exponential_sum(p.summand, p.variable, p.lower, p.upper)
        if expression is None:
            return None
        return _series_result(
            expression,
            p.variable,
            p.lower,
            p.upper,
            p.parameter,
            p.point,
            "oscillatory-geometric",
            "EXACT",
            remainder=Remainder.exact_zero(
                p.parameter, p.point, source="exact finite geometric sum"
            ),
            reduction=p.reduction,
            terms=p.terms,
        )
    if route == "euler-maclaurin":
        data = _euler_maclaurin(p.reduction, p.parameter, p.point, p.terms)
        if data is None:
            return None
        expression, remainder, certificate = data
        return _series_result(
            expression,
            p.variable,
            p.lower,
            p.upper,
            p.parameter,
            p.point,
            "euler-maclaurin",
            "CERTIFIED" if remainder.is_certified else "FORMAL",
            remainder=remainder,
            reduction=p.reduction,
            certificate=certificate,
            terms=p.terms,
        )
    if route == "mellin":
        data = _mellin_sum(
            p.summand, p.variable, p.lower, p.upper, p.parameter, p.point, p.terms
        )
        if data is None:
            return None
        expression, cert = data
        remainder = cert.remainder if cert is not None else None
        status = "CERTIFIED" if cert is not None and cert.replay() else "FORMAL"
        return _series_result(
            expression,
            p.variable,
            p.lower,
            p.upper,
            p.parameter,
            p.point,
            "mellin-poles",
            status,
            remainder=remainder,
            reduction=p.reduction,
            certificate=cert,
            terms=p.terms,
        )
    if route == "riemann":
        data = _riemann_sum(
            p.summand, p.variable, p.lower, p.upper, p.parameter, p.point, p.terms
        )
        if data is None:
            return None
        expression, transformation = data
        return _series_result(
            expression,
            p.variable,
            p.lower,
            p.upper,
            p.parameter,
            p.point,
            "riemann-sum",
            "FORMAL",
            remainder=None,
            reduction=p.reduction,
            transformation=transformation,
            terms=p.terms,
        )
    if route == "saddle":
        data = _lattice_saddle(
            p.summand,
            p.variable,
            p.lower,
            p.upper,
            p.parameter,
            p.point,
            p.terms,
            normalization=p.normalization,
        )
        if data is None:
            return None
        integral, x, _domain = data
        return _series_result(
            integral.expression,
            p.variable,
            p.lower,
            p.upper,
            p.parameter,
            p.point,
            "discrete-" + integral.method,
            "CERTIFIED" if integral.certified else "FORMAL",
            remainder=getattr(integral, "remainder", None),
            reduction=p.reduction,
            transformation=(p.variable, p.parameter * x),
            certificate=getattr(integral, "certificate", None),
            terms=p.terms,
        )
    raise ValueError(f"unknown sum theorem route: {route}")


_SUM_ROUTE_REGISTRY = TheoremRegistry()
for _priority, (_route, _strength) in enumerate(
    (
        ("exact", CertificationStrength.EXACT),
        ("series", CertificationStrength.CERTIFIED),
        ("summation-by-parts", CertificationStrength.FORMAL),
        ("zeilberger", CertificationStrength.CERTIFIED),
        ("poisson", CertificationStrength.CERTIFIED),
        ("oscillatory", CertificationStrength.EXACT),
        ("euler-maclaurin", CertificationStrength.CERTIFIED),
        ("mellin", CertificationStrength.CERTIFIED),
        ("riemann", CertificationStrength.FORMAL),
        ("saddle", CertificationStrength.CERTIFIED),
    )
):
    _SUM_ROUTE_REGISTRY.register(
        Theorem(
            name=f"sum:{_route}",
            family="sum",
            applicability=_sum_route_applicable(_route),
            hypotheses=_sum_route_hypotheses,
            constructor=lambda problem, route=_route: _sum_route_construct(
                route, problem
            ),
            certification=_strength,
            priority=_priority,
        )
    )


def sum(
    summand: sp.Expr,
    variable: sp.Symbol | tuple[sp.Symbol, ...],
    lower: sp.Expr | tuple[sp.Expr, ...],
    upper: sp.Expr | tuple[sp.Expr, ...],
    *,
    parameter: sp.Symbol,
    point: sp.Expr = sp.oo,
    terms: int = 4,
    method: SumMethod = "auto",
    assumptions: sp.Expr = sp.S.true,
) -> SumResult:
    """Expand a parameter-dependent discrete sum asymptotically.

    ``auto`` tries exact summation, certified/formal termwise expansion, Abel
    transforms, creative telescoping into :func:`rsolve`, Poisson or
    finite oscillatory reduction, Euler–Maclaurin, certified Mellin shifts,
    scaled Riemann sums, and lattice saddles. Tuple-valued variables and bounds
    support separable multidimensional sums and fixed finite boxes.

    A route is marked ``CERTIFIED`` only when its replayable theorem obligations
    are proved; unsupported contour, uniformity, or lattice hypotheses remain
    ``FORMAL`` or ``UNKNOWN`` rather than being guessed.
    """
    assumptions = normalize_assumptions(assumptions)
    summand = sp.refine(sp.sympify(summand), assumptions)
    indices = variable if isinstance(variable, tuple) else (variable,)
    integer_indices = {
        index: sp.Dummy(index.name, integer=True)
        for index in indices
        if isinstance(index, sp.Symbol)
    }
    # Sum binds each index to integers even when its incoming Symbol is real.
    # Simplify periodic/parity identities in that scope, then restore its name.
    summand = summand.xreplace(integer_indices).xreplace(
        {integer: index for index, integer in integer_indices.items()}
    )
    return _sum_impl(
        summand,
        variable,
        lower,
        upper,
        parameter=parameter,
        point=point,
        terms=terms,
        method=method,
        normalization=None,
    )


def _sum_impl(
    summand: sp.Expr,
    variable: sp.Symbol | tuple[sp.Symbol, ...],
    lower: sp.Expr | tuple[sp.Expr, ...],
    upper: sp.Expr | tuple[sp.Expr, ...],
    *,
    parameter: sp.Symbol,
    point: sp.Expr,
    terms: int,
    method: SumMethod,
    normalization: object | None,
) -> SumResult:
    if not isinstance(parameter, sp.Symbol):
        raise TypeError("parameter must be a symbol")
    if isinstance(variable, tuple):
        variables = tuple(variable)
        lowers = tuple(map(sp.sympify, lower)) if isinstance(lower, tuple) else ()
        uppers = tuple(map(sp.sympify, upper)) if isinstance(upper, tuple) else ()
        multi = _multidimensional_sum(
            sp.sympify(summand),
            variables,
            lowers,
            uppers,
            parameter=parameter,
            point=point,
            terms=terms,
            method=method,
        )
        reduction = sp.Sum(
            sp.sympify(summand),
            *[(v, a, b) for v, a, b in zip(variables, lowers, uppers)],
        )
        if multi is None:
            return SumResult(
                reduction,
                variables,
                lowers,
                uppers,
                parameter,
                point,
                "multidimensional",
                "UNKNOWN",
                reduction=reduction,
            )
        expression, route, cert = multi
        return _series_result(
            expression,
            variables,
            lowers,
            uppers,
            parameter,
            point,
            route,
            "CERTIFIED" if route == "multidimensional-termwise" else "FORMAL",
            remainder=None,
            reduction=reduction,
            certificate=cert,
            terms=terms,
        )
    if not isinstance(variable, sp.Symbol):
        raise TypeError("variable must be a symbol or tuple of symbols")
    if terms < 1:
        raise ValueError("terms must be positive")
    methods = {
        "auto",
        "exact",
        "series",
        "summation-by-parts",
        "euler-maclaurin",
        "mellin",
        "riemann",
        "saddle",
        "zeilberger",
        "poisson",
        "oscillatory",
    }
    if method not in methods:
        raise ValueError("unknown asymptotic sum method")
    summand = sp.sympify(summand)
    lower = sp.sympify(lower)
    upper = sp.sympify(upper)
    reduction = sp.Sum(summand, (variable, lower, upper))

    problem = _SumRouteProblem(
        summand,
        variable,
        lower,
        upper,
        parameter,
        point,
        terms,
        method,
        normalization,
        reduction,
    )
    context = AsymptoticContext(parameter, point=point)
    for decision in _SUM_ROUTE_REGISTRY.candidates(problem, context=context):
        result = decision.theorem.constructor(problem)
        if result is not None:
            return result
        if method != "auto":
            break

    return SumResult(
        reduction,
        variable,
        lower,
        upper,
        parameter,
        point,
        "sum-reduction",
        "UNKNOWN",
        reduction=reduction,
    )
