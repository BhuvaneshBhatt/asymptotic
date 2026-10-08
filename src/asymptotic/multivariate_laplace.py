"""Multivariate Laplace geometry and normalized local asymptotics.

The module provides a generic backend for parameter-dependent integrals of the
form ``a(x) exp(-n phi(x))``.  It uses mathematical terminology:
Bayesian packages may interpret the phase as a negative log posterior or the
integral as an evidence calculation, but no prior/likelihood semantics live
here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from typing import Literal

import sympy as sp
from funcprops import normalize_assumptions

from ._power_simplify import analytic_powsimp
from ._symbolic_errors import SYMBOLIC_ERRORS
from ._symbolic_policy import bounded_assumption_sign, bounded_simplify
from .remainder import Remainder

LaplaceStatus = Literal["CERTIFIED", "FORMAL", "UNKNOWN"]
LaplacePointKind = Literal["interior", "boundary"]


def _as_variables(variables: Sequence[sp.Symbol]) -> tuple[sp.Symbol, ...]:
    result = tuple(variables)
    if not result or not all(isinstance(var, sp.Symbol) for var in result):
        raise TypeError("variables must be a nonempty sequence of SymPy Symbols")
    if len(set(result)) != len(result):
        raise ValueError("variables must be distinct")
    if len(result) > 4:
        raise NotImplementedError(
            "multivariate Laplace expansion supports at most four variables"
        )
    return result


def _as_box(
    domain: sp.Set | Sequence[sp.Interval] | None, dimension: int
) -> tuple[sp.Interval, ...]:
    if domain is None:
        return (sp.Interval(-sp.oo, sp.oo),) * dimension
    if isinstance(domain, sp.ProductSet):
        factors = tuple(domain.args)
    elif isinstance(domain, (tuple, list)):
        factors = tuple(domain)
    elif dimension == 1 and isinstance(domain, sp.Interval):
        factors = (domain,)
    else:
        raise TypeError("domain must be a product of real intervals")
    if len(factors) != dimension or not all(
        isinstance(item, sp.Interval) for item in factors
    ):
        raise ValueError(
            "domain dimension must match variables and contain only intervals"
        )
    return factors


def _point_in_box(point: Sequence[sp.Expr], box: Sequence[sp.Interval]) -> bool | None:
    unresolved = False
    for value, interval in zip(point, box):
        contained = interval.contains(value)
        if contained is sp.S.false:
            return False
        if contained is not sp.S.true:
            unresolved = True
    return None if unresolved else True


def _boundary_coordinates(
    point: Sequence[sp.Expr], box: Sequence[sp.Interval]
) -> tuple[int, ...]:
    active: list[int] = []
    for index, (value, interval) in enumerate(zip(point, box)):
        if (
            interval.start is not -sp.oo and sp.simplify(value - interval.start) == 0
        ) or (interval.end is not sp.oo and sp.simplify(value - interval.end) == 0):
            active.append(index)
    return tuple(active)


def _positive_definite(matrix: sp.Matrix) -> bool | None:
    if matrix.rows != matrix.cols:
        return False
    for size in range(1, matrix.rows + 1):
        minor = bounded_simplify(matrix[:size, :size].det())
        sign = bounded_assumption_sign(minor)
        if sign == -1 or sign == 0:
            return False
        if sign != 1:
            return None
    return True


def _gradient(phase: sp.Expr, variables: Sequence[sp.Symbol]) -> sp.Matrix:
    return sp.Matrix([sp.diff(phase, var) for var in variables])


def _hessian(phase: sp.Expr, variables: Sequence[sp.Symbol]) -> sp.Matrix:
    return sp.hessian(phase, variables)


def _complete_polynomial_stationary_points(
    phase: sp.Expr,
    variables: tuple[sp.Symbol, ...],
) -> tuple[tuple[sp.Expr, ...], ...] | None:
    gradient = tuple(sp.expand(sp.diff(phase, var)) for var in variables)
    if sum(int(sp.count_ops(item)) for item in gradient) > 100:
        return None
    try:
        polys = tuple(sp.Poly(item, *variables) for item in gradient)
    except sp.PolynomialError:
        return None
    if any(poly.total_degree() > 4 for poly in polys):
        return None
    try:
        solutions = sp.solve_poly_system(gradient, *variables)
    except (NotImplementedError, ValueError, sp.PolynomialError, ZeroDivisionError):
        return None
    if solutions is None:
        return None
    return tuple(tuple(sp.sympify(value) for value in point) for point in solutions)


def _quadratic_global_certificate(
    phase: sp.Expr, variables: tuple[sp.Symbol, ...]
) -> bool:
    try:
        poly = sp.Poly(phase, *variables)
    except sp.PolynomialError:
        return False
    if poly.total_degree() != 2:
        return False
    hessian = _hessian(phase, variables)
    return _positive_definite(hessian) is True


def _separable_phase(phase: sp.Expr, variables: tuple[sp.Symbol, ...]) -> bool:
    return all(
        sp.simplify(sp.diff(phase, a, b)) == 0
        for i, a in enumerate(variables)
        for b in variables[i + 1 :]
    )


def _separable_global_certificate(
    phase: sp.Expr,
    variables: tuple[sp.Symbol, ...],
    box: tuple[sp.Interval, ...],
) -> bool:
    if not _separable_phase(phase, variables):
        return False
    for var, interval in zip(variables, box):
        try:
            poly = sp.Poly(phase, var)
        except sp.PolynomialError:
            return False
        # Dependence on other variables appears as coefficients and is safe
        # only when all mixed derivatives vanish.
        degree = poly.degree()
        lead = poly.LC()
        sign = bounded_assumption_sign(lead)
        if interval.start is -sp.oo and interval.end is sp.oo:
            if degree % 2 or sign != 1:
                return False
        elif interval.end is sp.oo:
            if sign != 1:
                return False
        elif interval.start is -sp.oo:
            required = 1 if degree % 2 == 0 else -1
            if sign != required:
                return False
    return True


@dataclass(frozen=True)
class MultivariateStationaryPoint:
    """Local geometry of one candidate Laplace point."""

    point: tuple[sp.Expr, ...]
    phase_value: sp.Expr
    gradient: tuple[sp.Expr, ...]
    hessian: sp.Matrix
    hessian_pd: bool | None
    kind: LaplacePointKind
    active_boundaries: tuple[int, ...] = ()

    @property
    def nondegenerate_minimum(self) -> bool:
        """Whether the point is an interior nondegenerate local minimum."""
        return self.kind == "interior" and self.hessian_pd is True


@dataclass(frozen=True)
class MultivariateLaplaceCertificate:
    """Replayable local/global evidence for a multivariate Laplace expansion."""

    phase: sp.Expr
    variables: tuple[sp.Symbol, ...]
    domain: tuple[sp.Interval, ...]
    points: tuple[MultivariateStationaryPoint, ...]
    dominant_points: tuple[tuple[sp.Expr, ...], ...]
    local_certified: bool
    global_certified: bool
    competing: bool
    reason: str
    remainder: Remainder | None = None

    @property
    def certified(self) -> bool:
        """Whether geometry and a theorem-level remainder are certified."""
        return (
            self.local_certified
            and self.global_certified
            and self.remainder is not None
            and self.remainder.is_certified
        )

    def replay(self) -> bool | None:
        """Replay stationary-point and Hessian conditions without numerical inference."""
        if not self.points:
            return False
        for item in self.points:
            substitutions = dict(zip(self.variables, item.point))
            gradient = tuple(
                bounded_simplify(sp.diff(self.phase, var).subs(substitutions))
                for var in self.variables
            )
            if gradient != item.gradient:
                return False
            rebuilt_hessian = (
                _hessian(self.phase, self.variables)
                .subs(substitutions)
                .applyfunc(bounded_simplify)
            )
            if rebuilt_hessian != item.hessian:
                return False
            if item.point in self.dominant_points and not _candidate_is_local_minimum(
                item, self.variables, self.domain
            ):
                return False
        return True if self.certified else None


@dataclass(frozen=True)
class MultivariateLaplaceResult:
    """Finite multivariate Laplace expansion with explicit geometric evidence."""

    expression: sp.Expr
    parameter: sp.Symbol
    variables: tuple[sp.Symbol, ...]
    points: tuple[tuple[sp.Expr, ...], ...]
    method: str
    status: LaplaceStatus
    certificate: MultivariateLaplaceCertificate
    coefficients: tuple[sp.Expr, ...] = ()

    @property
    def certified(self) -> bool:
        """Whether the result has a globally certified Laplace geometry."""
        return self.status == "CERTIFIED" and self.certificate.certified


@dataclass(frozen=True)
class LaplaceRatioResult:
    """Asymptotic ratio of two Laplace integrals sharing one phase and domain."""

    expression: sp.Expr
    numerator: MultivariateLaplaceResult
    denominator: MultivariateLaplaceResult
    parameter: sp.Symbol
    status: LaplaceStatus

    @property
    def certified(self) -> bool:
        """Whether both component integrals are globally certified."""
        return self.status == "CERTIFIED"


@dataclass(frozen=True)
class LocalGaussianForm:
    """MAP-centered local normal form of a normalized Laplace measure."""

    point: tuple[sp.Expr, ...]
    hessian: sp.Matrix
    covariance_scale: sp.Matrix
    variables: tuple[sp.Symbol, ...]
    local_variables: tuple[sp.Symbol, ...]
    parameter: sp.Symbol
    density_expansion: sp.Expr
    normalization: MultivariateLaplaceResult
    status: LaplaceStatus


def _candidate_is_local_minimum(
    item: MultivariateStationaryPoint,
    variables: tuple[sp.Symbol, ...],
    box: tuple[sp.Interval, ...],
) -> bool:
    if item.kind == "interior":
        return item.nondegenerate_minimum
    active = set(item.active_boundaries)
    tangent = [index for index in range(len(variables)) if index not in active]
    for index in active:
        interval = box[index]
        derivative = item.gradient[index]
        if sp.simplify(item.point[index] - interval.start) == 0:
            if bounded_assumption_sign(derivative) != 1:
                return False
        elif sp.simplify(item.point[index] - interval.end) == 0:
            if bounded_assumption_sign(derivative) != -1:
                return False
        else:
            return False
    if any(sp.simplify(item.gradient[index]) != 0 for index in tangent):
        return False
    if tangent:
        tangent_hessian = item.hessian.extract(tangent, tangent)
        if _positive_definite(tangent_hessian) is not True:
            return False
    return True


def _box_separable_candidates(
    phase: sp.Expr,
    variables: tuple[sp.Symbol, ...],
    box: tuple[sp.Interval, ...],
) -> tuple[tuple[sp.Expr, ...], ...] | None:
    if not _separable_phase(phase, variables):
        return None
    choices: list[tuple[sp.Expr, ...]] = []
    for var, interval in zip(variables, box):
        derivative = sp.expand(sp.diff(phase, var))
        try:
            poly = sp.Poly(derivative, var)
            roots = sp.roots(poly.as_expr(), var)
        except (sp.PolynomialError, ValueError, NotImplementedError):
            return None
        values = [root for root in roots if interval.contains(root) is sp.S.true]
        if interval.start is not -sp.oo:
            values.append(interval.start)
        if interval.end is not sp.oo:
            values.append(interval.end)
        unique = tuple(sorted(set(values), key=sp.default_sort_key))
        if not unique:
            return None
        choices.append(unique)
    return tuple(tuple(point) for point in product(*choices))


def _mixed_moment(
    poly: sp.Expr,
    active_vars: tuple[sp.Symbol, ...],
    rates: tuple[sp.Expr, ...],
    gaussian_vars: tuple[sp.Symbol, ...],
    covariance: sp.Matrix,
) -> sp.Expr | None:
    all_vars = active_vars + gaussian_vars
    try:
        polynomial = sp.Poly(sp.expand(poly), *all_vars)
    except sp.PolynomialError:
        return None
    total = sp.S.Zero
    for powers, coefficient in polynomial.terms():
        active_powers = powers[: len(active_vars)]
        gaussian_powers = powers[len(active_vars) :]
        exp_moment = sp.S.One
        for power, rate in zip(active_powers, rates):
            exp_moment *= sp.factorial(power) / rate ** (power + 1)
        indices: list[int] = []
        for index, power in enumerate(gaussian_powers):
            indices.extend([index] * power)
        gauss = _gaussian_moment(tuple(indices), tuple(covariance), len(gaussian_vars))
        total += coefficient * exp_moment * gauss
    return bounded_simplify(total)


def _local_boundary_expansion(
    amplitude: sp.Expr,
    phase: sp.Expr,
    variables: tuple[sp.Symbol, ...],
    box: tuple[sp.Interval, ...],
    item: MultivariateStationaryPoint,
    parameter: sp.Symbol,
    terms: int,
) -> tuple[sp.Expr, tuple[sp.Expr, ...]] | None:
    active = tuple(item.active_boundaries)
    tangent = tuple(index for index in range(len(variables)) if index not in active)
    eps = sp.Dummy("_lap_eps", positive=True)
    active_vars = tuple(sp.Dummy(f"_s{i}", positive=True) for i in active)
    gaussian_vars = tuple(sp.Dummy(f"_z{i}", real=True) for i in tangent)
    local_subs: dict[sp.Symbol, sp.Expr] = {}
    rates: list[sp.Expr] = []
    for local_index, index in enumerate(active):
        interval = box[index]
        lower = sp.simplify(item.point[index] - interval.start) == 0
        orientation = sp.S.One if lower else -sp.S.One
        local_subs[variables[index]] = (
            item.point[index] + orientation * eps**2 * active_vars[local_index]
        )
        rate = bounded_simplify(orientation * item.gradient[index])
        if bounded_assumption_sign(rate) != 1:
            return None
        rates.append(rate)
    if tangent:
        tangent_hessian = item.hessian.extract(tangent, tangent)
        if _positive_definite(tangent_hessian) is not True:
            return None
        covariance = tangent_hessian.inv().applyfunc(bounded_simplify)
        for local_index, index in enumerate(tangent):
            local_subs[variables[index]] = (
                item.point[index] + eps * gaussian_vars[local_index]
            )
        quad = (
            sp.Matrix(gaussian_vars).T * tangent_hessian * sp.Matrix(gaussian_vars)
        )[0] / 2
    else:
        covariance = sp.zeros(0, 0)
        quad = sp.S.Zero
    linear = sp.Add(*(rate * var for rate, var in zip(rates, active_vars)))
    phi0 = item.phase_value
    try:
        phase_local = sp.series(phase.subs(local_subs), eps, 0, 2 * terms + 3).removeO()
        correction = sp.expand((phase_local - phi0 - eps**2 * (linear + quad)) / eps**2)
        amp_local = amplitude.subs(local_subs).subs(parameter, eps**-2)
        product_series = sp.series(
            amp_local * sp.exp(-correction), eps, 0, 2 * terms + 1
        ).removeO()
    except SYMBOLIC_ERRORS:
        return None
    expected = sp.S.Zero
    for term in sp.Add.make_args(sp.expand(product_series)):
        power = sp.sympify(term.as_powers_dict().get(eps, 0))
        if power.is_integer is not True or power < 0:
            return None
        coefficient = bounded_simplify(term / eps**power)
        moment = _mixed_moment(
            coefficient, active_vars, tuple(rates), gaussian_vars, covariance
        )
        if moment is None:
            return None
        expected += moment * eps**power
    tangent_norm = (2 * sp.pi) ** (sp.Rational(len(tangent), 2))
    if tangent:
        tangent_norm /= sp.sqrt(item.hessian.extract(tangent, tangent).det())
    scaling_power = 2 * len(active) + len(tangent)
    local = bounded_simplify(
        sp.exp(-parameter * phi0)
        * tangent_norm
        * parameter ** (-sp.Rational(scaling_power, 2))
        * expected.xreplace({eps: parameter ** -sp.Rational(1, 2)})
    )
    coeffs = tuple(
        bounded_simplify(expected.expand().coeff(eps, 2 * index) * tangent_norm)
        for index in range(terms)
    )
    return local, coeffs


def stationary_point_geometry(
    phase: sp.Expr,
    variables: Sequence[sp.Symbol],
    *,
    domain: sp.Set | Sequence[sp.Interval] | None = None,
    stationary_points: Sequence[Sequence[sp.Expr]] | None = None,
    assumptions: sp.Expr = sp.S.true,
) -> tuple[MultivariateStationaryPoint, ...]:
    """Analyze exact stationary-point/Hessian geometry on a product domain.

    Polynomial stationary points are discovered automatically within a bounded
    degree/complexity class.  For harder phases callers can supply exact points.
    """
    assumptions = normalize_assumptions(assumptions)
    variables = _as_variables(variables)
    phase = sp.refine(sp.sympify(phase), assumptions)
    box = _as_box(domain, len(variables))
    if stationary_points is None:
        discovered = _complete_polynomial_stationary_points(phase, variables)
        boundary_candidates = _box_separable_candidates(phase, variables, box)
        if boundary_candidates is not None:
            discovered = boundary_candidates
        if discovered is None:
            raise NotImplementedError(
                "stationary points are not exactly enumerable; supply stationary_points"
            )
    else:
        discovered = tuple(
            tuple(sp.sympify(value) for value in point) for point in stationary_points
        )
    result: list[MultivariateStationaryPoint] = []
    for point in discovered:
        if len(point) != len(variables):
            raise ValueError("stationary point dimension does not match variables")
        if _point_in_box(point, box) is False:
            continue
        substitutions = dict(zip(variables, point))
        gradient = tuple(
            bounded_simplify(sp.diff(phase, var).subs(substitutions))
            for var in variables
        )
        active = _boundary_coordinates(point, box)
        kind: LaplacePointKind = "boundary" if active else "interior"
        hessian = (
            _hessian(phase, variables).subs(substitutions).applyfunc(bounded_simplify)
        )
        pd = _positive_definite(hessian) if kind == "interior" else None
        result.append(
            MultivariateStationaryPoint(
                point,
                bounded_simplify(phase.subs(substitutions)),
                gradient,
                hessian,
                pd,
                kind,
                active,
            )
        )
    return tuple(result)


def _dominant_laplace_points(
    points: tuple[MultivariateStationaryPoint, ...],
    variables: tuple[sp.Symbol, ...],
    box: tuple[sp.Interval, ...],
) -> tuple[MultivariateStationaryPoint, ...]:
    minima = tuple(
        item for item in points if _candidate_is_local_minimum(item, variables, box)
    )
    if not minima:
        return ()
    dominant: list[MultivariateStationaryPoint] = []
    for item in minima:
        signs = [
            bounded_assumption_sign(other.phase_value - item.phase_value)
            for other in minima
        ]
        if all(sign in (0, 1) for sign in signs):
            dominant.append(item)
    return tuple(dominant)


@lru_cache(maxsize=4096)
def _gaussian_moment(
    indices: tuple[int, ...], covariance_key: tuple[sp.Expr, ...], dimension: int
) -> sp.Expr:
    if not indices:
        return sp.S.One
    if len(indices) % 2:
        return sp.S.Zero
    covariance = sp.Matrix(dimension, dimension, covariance_key)
    first = indices[0]
    total = sp.S.Zero
    for pos in range(1, len(indices)):
        second = indices[pos]
        remaining = indices[1:pos] + indices[pos + 1 :]
        total += covariance[first, second] * _gaussian_moment(
            remaining, covariance_key, dimension
        )
    return bounded_simplify(total)


def _gaussian_polynomial_expectation(
    poly: sp.Expr, z: tuple[sp.Symbol, ...], covariance: sp.Matrix
) -> sp.Expr | None:
    try:
        polynomial = sp.Poly(sp.expand(poly), *z)
    except sp.PolynomialError:
        return None
    covariance_key = tuple(covariance)
    total = sp.S.Zero
    for powers, coefficient in polynomial.terms():
        indices: list[int] = []
        for index, power in enumerate(powers):
            indices.extend([index] * power)
        total += coefficient * _gaussian_moment(tuple(indices), covariance_key, len(z))
    return bounded_simplify(total)


def _local_interior_expansion(
    amplitude: sp.Expr,
    phase: sp.Expr,
    variables: tuple[sp.Symbol, ...],
    point: tuple[sp.Expr, ...],
    parameter: sp.Symbol,
    terms: int,
) -> tuple[sp.Expr, tuple[sp.Expr, ...]] | None:
    substitutions = dict(zip(variables, point))
    hessian = _hessian(phase, variables).subs(substitutions).applyfunc(bounded_simplify)
    if _positive_definite(hessian) is not True:
        return None
    try:
        covariance = hessian.inv().applyfunc(bounded_simplify)
    except (ValueError, ZeroDivisionError):
        return None
    eps = sp.Dummy("_lap_eps", positive=True)
    z = tuple(sp.Dummy(f"_z{i}", real=True) for i in range(len(variables)))
    local_subs = {var: value + eps * zi for var, value, zi in zip(variables, point, z)}
    phi0 = bounded_simplify(phase.subs(substitutions))
    quad = (sp.Matrix(z).T * hessian * sp.Matrix(z))[0] / 2
    order = max(2 * terms + 3, 5)
    try:
        phase_local = sp.series(phase.subs(local_subs), eps, 0, order).removeO()
        correction = sp.expand((phase_local - phi0 - eps**2 * quad) / eps**2)
        amp_local = amplitude.subs(local_subs).subs(parameter, eps**-2)
        product_series = sp.series(
            amp_local * sp.exp(-correction), eps, 0, 2 * terms + 1
        ).removeO()
    except SYMBOLIC_ERRORS:
        return None
    expected = sp.S.Zero
    for item in sp.Add.make_args(sp.expand(product_series)):
        power = sp.sympify(item.as_powers_dict().get(eps, 0))
        if power.is_integer is not True or power < 0:
            return None
        coefficient = bounded_simplify(item / eps**power)
        moment = _gaussian_polynomial_expectation(coefficient, z, covariance)
        if moment is None:
            return None
        expected += moment * eps**power
    # Odd Gaussian moments disappear. Keep terms through n^{-(terms-1)}.
    try:
        expected = sp.series(expected, eps, 0, 2 * terms).removeO()
    except SYMBOLIC_ERRORS:
        return None
    gaussian_norm = (2 * sp.pi) ** (sp.Rational(len(variables), 2)) / sp.sqrt(
        hessian.det()
    )
    local = bounded_simplify(
        sp.exp(-parameter * phi0)
        * parameter ** (-sp.Rational(len(variables), 2))
        * gaussian_norm
        * expected.xreplace({eps: parameter ** -sp.Rational(1, 2)})
    )
    coeffs: list[sp.Expr] = []
    for index in range(terms):
        coeffs.append(
            bounded_simplify(expected.expand().coeff(eps, 2 * index) * gaussian_norm)
        )
    return local, tuple(coeffs)


def _make_certificate(
    phase: sp.Expr,
    variables: tuple[sp.Symbol, ...],
    box: tuple[sp.Interval, ...],
    points: tuple[MultivariateStationaryPoint, ...],
    dominant: tuple[MultivariateStationaryPoint, ...],
    parameter: sp.Symbol,
    terms: int,
) -> MultivariateLaplaceCertificate:
    real_problem = (
        parameter.is_positive is True
        and all(variable.is_real is True for variable in variables)
        and phase.is_real is True
    )
    local = (
        real_problem
        and bool(dominant)
        and all(_candidate_is_local_minimum(item, variables, box) for item in dominant)
    )
    global_ok = False
    reason = (
        "local nondegenerate minima are verified; global dominance is formal"
        if real_problem
        else "positive-real parameter and real phase/domain variables were not certified"
    )
    if local and (
        _quadratic_global_certificate(phase, variables)
        or _separable_global_certificate(phase, variables, box)
    ):
        # For the supported globally certified classes exact stationary-point
        # enumeration plus strict convexity/coercivity proves dominance.
        global_ok = True
        reason = "stationary points, positive Hessian, and global coercive geometry are certified"
    remainder = None
    if global_ok:
        minimum = dominant[0].phase_value
        remainder = Remainder.big_o(
            sp.exp(-parameter * minimum) * parameter ** (-terms),
            parameter,
            sp.oo,
            source="multivariate Laplace theorem on certified coercive geometry",
        )
    return MultivariateLaplaceCertificate(
        phase,
        variables,
        box,
        points,
        tuple(item.point for item in dominant),
        local,
        global_ok,
        len(dominant) > 1,
        reason,
        remainder,
    )


def multivariate_laplace_integral(
    amplitude: sp.Expr,
    phase: sp.Expr,
    variables: Sequence[sp.Symbol],
    parameter: sp.Symbol,
    *,
    domain: sp.Set | Sequence[sp.Interval] | None = None,
    terms: int = 2,
    stationary_points: Sequence[Sequence[sp.Expr]] | None = None,
    assumptions: sp.Expr = sp.S.true,
) -> MultivariateLaplaceResult:
    """Expand a real multivariate Laplace integral around dominant minima.

    The integral is ``Integral(amplitude*exp(-parameter*phase), variables)``.
    Multiple exactly co-dominant minima are summed.  Axis-aligned boundary minima
    use mixed normal/tangent scaling; unsupported constrained geometry returns
    ``UNKNOWN`` rather than being approximated by an interior Gaussian.
    """
    assumptions = normalize_assumptions(assumptions)
    amplitude = sp.refine(sp.sympify(amplitude), assumptions)
    phase = sp.refine(sp.sympify(phase), assumptions)
    if not isinstance(parameter, sp.Symbol):
        raise TypeError("parameter must be a SymPy Symbol")
    if terms < 1 or terms > 4:
        raise ValueError("terms must be between 1 and 4")
    variables = _as_variables(variables)
    box = _as_box(domain, len(variables))
    points = stationary_point_geometry(
        phase,
        variables,
        domain=box,
        stationary_points=stationary_points,
        assumptions=assumptions,
    )
    dominant = _dominant_laplace_points(points, variables, box)
    certificate = _make_certificate(
        phase, variables, box, points, dominant, parameter, terms
    )
    if not dominant:
        return MultivariateLaplaceResult(
            sp.S.NaN,
            parameter,
            variables,
            (),
            "multivariate-laplace-unsupported-geometry",
            "UNKNOWN",
            certificate,
        )
    pieces: list[sp.Expr] = []
    all_coeffs: list[sp.Expr] = []
    for item in dominant:
        if item.kind == "interior":
            expanded = _local_interior_expansion(
                amplitude, phase, variables, item.point, parameter, terms
            )
        else:
            expanded = _local_boundary_expansion(
                amplitude, phase, variables, box, item, parameter, terms
            )
        if expanded is None:
            return MultivariateLaplaceResult(
                sp.S.NaN,
                parameter,
                variables,
                tuple(obj.point for obj in dominant),
                "multivariate-laplace-local-expansion-failed",
                "UNKNOWN",
                certificate,
            )
        expression, coefficients = expanded
        pieces.append(expression)
        all_coeffs.extend(coefficients)
    status: LaplaceStatus = "CERTIFIED" if certificate.certified else "FORMAL"
    if len(dominant) > 1:
        method = "multivariate-laplace-co-dominant"
    elif dominant[0].kind == "boundary":
        method = "multivariate-laplace-boundary"
    else:
        method = "multivariate-laplace-interior"
    return MultivariateLaplaceResult(
        bounded_simplify(sp.Add(*pieces)),
        parameter,
        variables,
        tuple(item.point for item in dominant),
        method,
        status,
        certificate,
        tuple(all_coeffs),
    )


def log_laplace_asymptotic(
    amplitude: sp.Expr,
    phase: sp.Expr,
    variables: Sequence[sp.Symbol],
    parameter: sp.Symbol,
    *,
    domain: sp.Set | Sequence[sp.Interval] | None = None,
    terms: int = 2,
    stationary_points: Sequence[Sequence[sp.Expr]] | None = None,
    assumptions: sp.Expr = sp.S.true,
) -> MultivariateLaplaceResult:
    """Return a log-domain expansion of a multivariate Laplace integral."""
    result = multivariate_laplace_integral(
        amplitude,
        phase,
        variables,
        parameter,
        domain=domain,
        terms=terms,
        stationary_points=stationary_points,
        assumptions=assumptions,
    )
    if result.status == "UNKNOWN":
        return result
    try:
        inverse = sp.Dummy("_inv_n", positive=True)
        expression = (
            sp.series(
                sp.log(result.expression.subs(parameter, 1 / inverse)),
                inverse,
                0,
                terms,
            )
            .removeO()
            .subs(inverse, 1 / parameter)
        )
        expression = sp.expand_log(expression, force=False)
    except SYMBOLIC_ERRORS:
        expression = sp.log(result.expression)
    return MultivariateLaplaceResult(
        analytic_powsimp(expression),
        result.parameter,
        result.variables,
        result.points,
        "log-" + result.method,
        result.status,
        result.certificate,
        result.coefficients,
    )


def laplace_ratio_asymptotic(
    numerator_amplitude: sp.Expr,
    denominator_amplitude: sp.Expr,
    phase: sp.Expr,
    variables: Sequence[sp.Symbol],
    parameter: sp.Symbol,
    *,
    domain: sp.Set | Sequence[sp.Interval] | None = None,
    terms: int = 2,
    stationary_points: Sequence[Sequence[sp.Expr]] | None = None,
    assumptions: sp.Expr = sp.S.true,
) -> LaplaceRatioResult:
    """Expand a normalized ratio of two Laplace integrals around common saddles."""
    numerator = multivariate_laplace_integral(
        numerator_amplitude,
        phase,
        variables,
        parameter,
        domain=domain,
        terms=terms,
        stationary_points=stationary_points,
        assumptions=assumptions,
    )
    denominator = multivariate_laplace_integral(
        denominator_amplitude,
        phase,
        variables,
        parameter,
        domain=domain,
        terms=terms,
        stationary_points=stationary_points,
        assumptions=assumptions,
    )
    if "UNKNOWN" in (numerator.status, denominator.status):
        return LaplaceRatioResult(
            sp.S.NaN, numerator, denominator, parameter, "UNKNOWN"
        )
    quotient = bounded_simplify(numerator.expression / denominator.expression)
    try:
        quotient = sp.series(quotient, parameter, sp.oo, terms).removeO()
    except SYMBOLIC_ERRORS:
        pass
    status: LaplaceStatus = (
        "CERTIFIED" if numerator.certified and denominator.certified else "FORMAL"
    )
    return LaplaceRatioResult(
        analytic_powsimp(quotient), numerator, denominator, parameter, status
    )


def local_gaussian_form(
    amplitude: sp.Expr,
    phase: sp.Expr,
    variables: Sequence[sp.Symbol],
    parameter: sp.Symbol,
    *,
    domain: sp.Set | Sequence[sp.Interval] | None = None,
    terms: int = 2,
    stationary_point: Sequence[sp.Expr] | None = None,
    assumptions: sp.Expr = sp.S.true,
) -> LocalGaussianForm:
    """Construct a normalized local density in ``sqrt(parameter)`` coordinates.

    The returned local variables use ``x = x_hat + z/sqrt(parameter)``.  The
    leading density is the Gaussian with precision equal to the phase Hessian;
    higher polynomial corrections are normalized by the Laplace denominator.
    """
    assumptions = normalize_assumptions(assumptions)
    variables = _as_variables(variables)
    amplitude = sp.sympify(amplitude)
    phase = sp.sympify(phase)
    point_arg = None if stationary_point is None else (tuple(stationary_point),)
    normalization = multivariate_laplace_integral(
        amplitude,
        phase,
        variables,
        parameter,
        domain=domain,
        terms=terms,
        stationary_points=point_arg,
        assumptions=assumptions,
    )
    if len(normalization.points) != 1:
        raise ValueError("local Gaussian form requires one dominant stationary point")
    point = normalization.points[0]
    substitutions = dict(zip(variables, point))
    hessian = (
        _hessian(sp.sympify(phase), variables)
        .subs(substitutions)
        .applyfunc(bounded_simplify)
    )
    if _positive_definite(hessian) is not True:
        raise ValueError("stationary point must have a positive-definite Hessian")
    covariance = hessian.inv().applyfunc(
        lambda value: bounded_simplify(value / parameter)
    )
    z = tuple(sp.Symbol(f"z_{index}", real=True) for index in range(len(variables)))
    local_subs = {
        var: value + zi / sp.sqrt(parameter)
        for var, value, zi in zip(variables, point, z)
    }
    raw = amplitude.subs(local_subs) * sp.exp(
        -parameter * (phase.subs(local_subs) - phase.subs(substitutions))
    )
    jacobian = parameter ** (-sp.Rational(len(variables), 2))
    density = bounded_simplify(
        raw
        * jacobian
        / (normalization.expression * sp.exp(parameter * phase.subs(substitutions)))
    )
    try:
        eps = sp.Dummy("_eps", positive=True)
        density = (
            sp.series(density.subs(parameter, eps**-2), eps, 0, 2 * terms)
            .removeO()
            .subs(eps, parameter ** -sp.Rational(1, 2))
        )
    except SYMBOLIC_ERRORS:
        pass
    return LocalGaussianForm(
        point,
        hessian,
        covariance,
        variables,
        z,
        parameter,
        analytic_powsimp(density),
        normalization,
        normalization.status,
    )
