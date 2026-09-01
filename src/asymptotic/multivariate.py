"""Scaling-path dominant balance for multivariate asymptotic problems."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache

import sympy as sp

from ._power_simplify import analytic_powsimp
from .canonical import canonical_equal, canonical_expr
from .dominant import DominantBalanceCandidate, dominant_balance_candidates
from .instrumentation import record_symbolic_event
from .parameter_auto import (
    automatic_parameter_stratification,
    parameter_symbols,
    specialize_expression,
)
from .stratification import AsymptoticStratification, ParameterStratum


@dataclass(frozen=True)
class ScalingPath:
    """A one-parameter anisotropic path approaching a multivariate point.

    The path is ``x_i = center_i + amplitude_i * epsilon**weight_i``.  Signed
    rational weights are allowed, so paths to infinity can be represented by
    negative weights with zero centers.
    """

    variables: tuple[sp.Symbol, ...]
    weights: tuple[sp.Rational, ...]
    parameter: sp.Symbol
    centers: tuple[sp.Expr, ...]
    amplitudes: tuple[sp.Expr, ...]

    @property
    def substitution(self) -> dict[sp.Symbol, sp.Expr]:
        return {
            variable: sp.sympify(center) + sp.sympify(amplitude) * self.parameter**weight
            for variable, weight, center, amplitude in zip(
                self.variables, self.weights, self.centers, self.amplitudes
            )
        }

    def transform(self, expr: sp.Expr) -> sp.Expr:
        return analytic_powsimp(sp.expand(sp.sympify(expr).subs(self.substitution)))


@dataclass(frozen=True)
class MultivariateDominantBalanceCandidate:
    """A univariate balance induced by a certified multivariate scaling path."""

    path: ScalingPath
    transformed_equation: sp.Expr
    candidate: DominantBalanceCandidate

    @property
    def dependent_exponent(self) -> sp.Rational:
        return self.candidate.exponent

    @property
    def coefficients(self) -> tuple[sp.Expr, ...]:
        return self.candidate.coefficients


def scaling_path(
    variables: tuple[sp.Symbol, ...] | list[sp.Symbol],
    weights: tuple[sp.Expr, ...] | list[sp.Expr] | Mapping[sp.Symbol, sp.Expr],
    *,
    parameter: sp.Symbol | None = None,
    centers: tuple[sp.Expr, ...] | list[sp.Expr] | Mapping[sp.Symbol, sp.Expr] | None = None,
    amplitudes: tuple[sp.Expr, ...] | list[sp.Expr] | Mapping[sp.Symbol, sp.Expr] | None = None,
) -> ScalingPath:
    """Construct a weighted one-parameter scaling path for multivariate limits."""
    variables = tuple(variables)
    if not variables:
        raise ValueError("at least one scaling variable is required")
    if isinstance(weights, Mapping):
        weight_values = tuple(sp.Rational(weights[v]) for v in variables)
    else:
        weight_values = tuple(sp.Rational(w) for w in weights)
    if len(weight_values) != len(variables):
        raise ValueError("weights must match variables")
    epsilon = parameter or sp.Dummy("epsilon", positive=True)

    def values(spec: object, default: sp.Expr) -> tuple[sp.Expr, ...]:
        if spec is None:
            return (default,) * len(variables)
        if isinstance(spec, Mapping):
            return tuple(sp.sympify(spec.get(v, default)) for v in variables)
        result = tuple(sp.sympify(v) for v in spec)  # type: ignore[arg-type]
        if len(result) != len(variables):
            raise ValueError("path data must match variables")
        return result

    center_values = values(centers, sp.S.Zero)
    amplitude_values = values(amplitudes, sp.S.One)
    if any(a == 0 for a in amplitude_values):
        raise ValueError("scaling-path amplitudes must be nonzero")
    return ScalingPath(variables, weight_values, epsilon, center_values, amplitude_values)


def _wrap_stratified(
    stratification: AsymptoticStratification[tuple[DominantBalanceCandidate, ...]],
    path: ScalingPath,
    transformed: sp.Expr,
) -> AsymptoticStratification[tuple[MultivariateDominantBalanceCandidate, ...]]:
    strata = tuple(
        ParameterStratum(
            stratum.condition,
            tuple(
                MultivariateDominantBalanceCandidate(path, transformed, item)
                for item in stratum.result
            ),
            stratum.knowledge,
            stratum.provenance,
            stratum.decisions,
            stratum.complete,
            stratum.limitations,
        )
        for stratum in stratification.strata
    )
    return AsymptoticStratification(
        stratification.parameters,
        strata,
        stratification.assumptions,
        stratification.exhaustive,
        stratification.provenance,
    )


def multivariate_dominant_balance_candidates(
    equation: sp.Expr,
    dependent: sp.Symbol,
    variables: tuple[sp.Symbol, ...] | list[sp.Symbol],
    weights: tuple[sp.Expr, ...] | list[sp.Expr] | Mapping[sp.Symbol, sp.Expr] | None = None,
    *,
    parameter: sp.Symbol | None = None,
    centers: tuple[sp.Expr, ...] | list[sp.Expr] | Mapping[sp.Symbol, sp.Expr] | None = None,
    amplitudes: tuple[sp.Expr, ...] | list[sp.Expr] | Mapping[sp.Symbol, sp.Expr] | None = None,
    assumptions: sp.Expr | bool = sp.S.true,
    stratify_parameters: bool = True,
    max_parameter_splits: int = 6,
) -> (
    tuple[MultivariateDominantBalanceCandidate, ...]
    | AsymptoticStratification[tuple[MultivariateDominantBalanceCandidate, ...]]
):
    """Compute dominant balances on an explicit or automatically discovered path.

    When ``weights`` is omitted, all admissible automatic Newton weight cones
    are discovered and the balances from their rational representative paths
    are returned.  Supplying ``weights`` preserves the historical single-path
    behavior.
    """

    if weights is None:
        regimes = multivariate_scaling_regimes(
            equation,
            dependent,
            variables,
            assumptions=assumptions,
            stratify_parameters=stratify_parameters,
            max_parameter_splits=max_parameter_splits,
        )
        if isinstance(regimes, AsymptoticStratification):
            return AsymptoticStratification(
                regimes.parameters,
                tuple(
                    ParameterStratum(
                        stratum.condition,
                        tuple(balance for regime in stratum.result for balance in regime.balances),
                        stratum.knowledge,
                        stratum.provenance,
                        stratum.decisions,
                        stratum.complete,
                        stratum.limitations,
                    )
                    for stratum in regimes.strata
                ),
                regimes.assumptions,
                regimes.exhaustive,
                regimes.provenance,
            )
        return tuple(balance for regime in regimes for balance in regime.balances)

    path = scaling_path(
        variables,
        weights,
        parameter=parameter,
        centers=centers,
        amplitudes=amplitudes,
    )
    transformed = path.transform(equation)
    result = dominant_balance_candidates(
        transformed,
        dependent,
        path.parameter,
        assumptions=assumptions,
        stratify_parameters=stratify_parameters,
        max_parameter_splits=max_parameter_splits,
    )
    if isinstance(result, AsymptoticStratification):
        return _wrap_stratified(result, path, transformed)
    return tuple(MultivariateDominantBalanceCandidate(path, transformed, item) for item in result)


@dataclass(frozen=True)
class NewtonPolyhedronTerm:
    """One support point of a multivariate Newton diagram."""

    variable_exponents: tuple[sp.Rational, ...]
    dependent_power: sp.Rational
    coefficient: sp.Expr
    expression: sp.Expr

    @property
    def support_point(self) -> tuple[sp.Rational, ...]:
        return self.variable_exponents + (self.dependent_power,)


@dataclass(frozen=True)
class WeightCone:
    """A rational polyhedral cone of anisotropic independent-variable weights.

    ``equalities`` and ``inequalities`` are homogeneous linear expressions in
    ``weight_symbols``; equalities are zero and inequalities are nonnegative.
    ``dependent_weight`` is the Newton balance exponent as a linear function
    of the independent weights.
    """

    weight_symbols: tuple[sp.Symbol, ...]
    equalities: tuple[sp.Expr, ...]
    inequalities: tuple[sp.Expr, ...]
    dependent_weight: sp.Expr
    active_indices: tuple[int, ...]
    representative: tuple[sp.Rational, ...]

    @property
    def dimension(self) -> int:
        if not self.equalities:
            return len(self.weight_symbols)
        matrix = sp.Matrix(
            [[sp.expand(eq).coeff(w) for w in self.weight_symbols] for eq in self.equalities]
        )
        return len(self.weight_symbols) - matrix.rank()

    def contains(self, weights: tuple[sp.Expr, ...] | list[sp.Expr]) -> bool | None:
        if len(weights) != len(self.weight_symbols):
            raise ValueError("weights must match the cone dimension")
        subs = dict(zip(self.weight_symbols, map(sp.sympify, weights)))
        vals_eq = [sp.simplify(e.subs(subs)) for e in self.equalities]
        vals_ge = [sp.simplify(e.subs(subs)) for e in self.inequalities]
        if any(v.is_zero is False for v in vals_eq):
            return False
        if any(v.is_negative is True for v in vals_ge):
            return False
        if all(v.is_zero is True for v in vals_eq) and all(
            v.is_nonnegative is True for v in vals_ge
        ):
            return True
        return None


@dataclass(frozen=True)
class ScalingRegime:
    """An automatically discovered Newton face and its admissible weight cone."""

    variables: tuple[sp.Symbol, ...]
    terms: tuple[NewtonPolyhedronTerm, ...]
    active_terms: tuple[NewtonPolyhedronTerm, ...]
    cone: WeightCone
    balances: tuple[MultivariateDominantBalanceCandidate, ...]

    @property
    def representative_weights(self) -> tuple[sp.Rational, ...]:
        return self.cone.representative


def newton_polyhedron_terms(
    equation: sp.Expr,
    dependent: sp.Symbol,
    variables: tuple[sp.Symbol, ...] | list[sp.Symbol],
) -> tuple[NewtonPolyhedronTerm, ...]:
    """Extract the exact finite Newton support of an algebraic/Puiseux expression."""

    variables = tuple(variables)
    generators = set(variables) | {dependent}
    out = []
    for term in sp.Add.make_args(sp.expand(sp.sympify(equation))):
        powers = term.as_powers_dict()
        exponents = []
        valid = True
        for variable in variables:
            exponent = sp.sympify(powers.get(variable, 0))
            if not exponent.is_Rational:
                valid = False
                break
            exponents.append(sp.Rational(exponent))
        dep_power = sp.sympify(powers.get(dependent, 0))
        if not valid or not dep_power.is_Rational:
            continue
        dep_power = sp.Rational(dep_power)
        monomial = dependent**dep_power
        for variable, exponent in zip(variables, exponents):
            monomial *= variable**exponent
        coefficient = sp.simplify(term / monomial)
        if coefficient.free_symbols & generators:
            continue
        out.append(NewtonPolyhedronTerm(tuple(exponents), dep_power, coefficient, term))
    return tuple(out)


def _linear_form(
    term: NewtonPolyhedronTerm, weights: tuple[sp.Symbol, ...], rho: sp.Expr
) -> sp.Expr:
    return sp.expand(
        sum(a * w for a, w in zip(term.variable_exponents, weights)) + term.dependent_power * rho
    )


def _normalize_linear(expr: sp.Expr, weights: tuple[sp.Symbol, ...]) -> sp.Expr:
    expr = sp.expand(expr)
    coeffs = [sp.Rational(expr.coeff(w)) for w in weights]
    if not any(coeffs):
        return sp.S.Zero
    den = sp.ilcm(*[c.q for c in coeffs])
    ints = [int(c * den) for c in coeffs]
    gcd = abs(sp.igcd(*ints)) if ints else 1
    ints = [i // (gcd or 1) for i in ints]
    for i in ints:
        if i:
            if i < 0:
                ints = [-j for j in ints]
            break
    return sp.Add(*[i * w for i, w in zip(ints, weights)])


def _rational_representative(
    weights: tuple[sp.Symbol, ...],
    equalities: tuple[sp.Expr, ...],
    inequalities: tuple[sp.Expr, ...],
    *,
    max_denominator: int = 20,
    strict: bool = False,
) -> tuple[sp.Rational, ...] | None:
    """Find a small positive rational point in a projectivized cone."""
    import itertools

    n = len(weights)
    # Search the positive simplex; this selects a canonical local path to the origin.
    for den in range(n, max_denominator + 1):
        for numerators in itertools.product(range(1, den), repeat=n - 1):
            if sum(numerators) >= den:
                continue
            nums = numerators + (den - sum(numerators),)
            point = tuple(sp.Rational(v, den) for v in nums)
            subs = dict(zip(weights, point))
            if any(sp.simplify(eq.subs(subs)) != 0 for eq in equalities):
                continue
            vals = [sp.simplify(ineq.subs(subs)) for ineq in inequalities]
            if strict:
                if all(v.is_positive is True for v in vals):
                    return point
            elif all(v.is_nonnegative is True for v in vals):
                return point
    return None


def _discover_regimes_uncached(
    equation: sp.Expr,
    dependent: sp.Symbol,
    variables: tuple[sp.Symbol, ...],
    *,
    assumptions: sp.Expr | bool,
) -> tuple[ScalingRegime, ...]:
    """Discover Newton weight cones without parameter stratification.

    Candidate pairs determine the dependent weight as an affine function of
    the independent weights. Rational interior points identify active faces;
    repeated intersections then close the resulting fan along its boundaries.
    """

    terms = newton_polyhedron_terms(equation, dependent, variables)
    if len(terms) < 2:
        return ()
    ws = tuple(sp.Dummy(f"w_{v}", positive=True) for v in variables)
    regimes = {}
    # Every Newton face relevant to solving for y contains two terms of distinct
    # dependent degree.  A pair determines rho(w); the remaining face equations
    # and lower-face inequalities determine its normal cone.
    for i, left in enumerate(terms):
        for j in range(i + 1, len(terms)):
            right = terms[j]
            if left.dependent_power == right.dependent_power:
                continue
            numerator = sum(
                (b - a) * w
                for a, b, w in zip(left.variable_exponents, right.variable_exponents, ws)
            )
            rho = sp.factor(numerator / (left.dependent_power - right.dependent_power))
            base = _linear_form(left, ws, rho)
            differences = tuple(sp.factor(_linear_form(term, ws, rho) - base) for term in terms)
            # Cone inequalities must retain their sign; normalizing a linear
            # form up to sign would reverse part of the cone.
            inequalities = tuple(sp.factor(d) for d in differences if d != 0)
            representative = _rational_representative(ws, (), inequalities, strict=True)
            if representative is None:
                representative = _rational_representative(ws, (), inequalities)
            if representative is None:
                continue
            subs = dict(zip(ws, representative))
            vals = tuple(sp.simplify(d.subs(subs)) for d in differences)
            if any(v.is_negative is True for v in vals):
                continue
            active = tuple(k for k, v in enumerate(vals) if v == 0)
            if len(active) < 2:
                continue
            # Refine to the exact face seen at the representative.
            equalities = tuple(
                _normalize_linear(differences[k], ws) for k in active if differences[k] != 0
            )
            inequalities2 = tuple(
                differences[k] for k in range(len(terms)) if k not in active and differences[k] != 0
            )
            representative2 = (
                _rational_representative(ws, equalities, inequalities2) or representative
            )
            weights_value = representative2
            balances_raw = multivariate_dominant_balance_candidates(
                equation,
                dependent,
                variables,
                weights_value,
                assumptions=assumptions,
                stratify_parameters=False,
            )
            if isinstance(balances_raw, AsymptoticStratification):
                continue
            # Keep only the candidate corresponding to rho at this path.
            expected_rho = sp.simplify(rho.subs(dict(zip(ws, weights_value))))
            balances = tuple(
                b for b in balances_raw if canonical_equal(b.dependent_exponent, expected_rho)
            )
            if not balances:
                continue
            cone = WeightCone(ws, equalities, inequalities2, rho, active, weights_value)
            key = active
            regimes[key] = ScalingRegime(
                variables, terms, tuple(terms[k] for k in active), cone, balances
            )

    # Close the fan under intersections.  Intersections expose lower-dimensional
    # faces (walls, rays, and higher-codimension ties) which would otherwise be
    # hidden on boundaries of the maximal chambers.
    changed = True
    while changed:
        changed = False
        current = list(regimes.values())
        for a_idx, first in enumerate(current):
            for second in current[a_idx + 1 :]:
                difference = sp.simplify(first.cone.dependent_weight - second.cone.dependent_weight)
                extra_equalities = ()
                if difference != 0:
                    eq = _normalize_linear(difference, ws)
                    if eq != 0:
                        extra_equalities = (eq,)
                equalities = tuple(
                    dict.fromkeys(first.cone.equalities + second.cone.equalities + extra_equalities)
                )
                inequalities = first.cone.inequalities + second.cone.inequalities
                rep = _rational_representative(ws, equalities, inequalities)
                if rep is None:
                    continue
                path = scaling_path(variables, rep)
                transformed = path.transform(equation)
                raw = dominant_balance_candidates(
                    transformed,
                    dependent,
                    path.parameter,
                    assumptions=assumptions,
                    stratify_parameters=False,
                )
                if isinstance(raw, AsymptoticStratification):
                    continue
                expected = sp.simplify(first.cone.dependent_weight.subs(dict(zip(ws, rep))))
                raw = tuple(item for item in raw if canonical_equal(item.exponent, expected))
                if not raw:
                    continue
                wrapped = tuple(
                    MultivariateDominantBalanceCandidate(path, transformed, item) for item in raw
                )
                subs = dict(zip(ws, rep))
                rho_value = raw[0].exponent
                weighted = [
                    sp.simplify(_linear_form(term, ws, rho_value).subs(subs)) for term in terms
                ]
                minimum = min(weighted)
                active = tuple(k for k, value in enumerate(weighted) if value == minimum)
                if len(active) < 2 or active in regimes:
                    continue
                rho = first.cone.dependent_weight
                cone = WeightCone(ws, equalities, inequalities, rho, active, rep)
                regimes[active] = ScalingRegime(
                    variables,
                    terms,
                    tuple(terms[k] for k in active),
                    cone,
                    wrapped,
                )
                changed = True

    return tuple(
        sorted(regimes.values(), key=lambda r: (len(r.cone.active_indices), r.cone.representative))
    )


@lru_cache(maxsize=256)
def _discover_regimes_cached(
    equation: sp.Expr,
    dependent: sp.Symbol,
    variables: tuple[sp.Symbol, ...],
    assumptions: sp.Expr,
) -> tuple[ScalingRegime, ...]:
    return _discover_regimes_uncached(equation, dependent, variables, assumptions=assumptions)


def clear_weight_cone_cache() -> None:
    """Clear the bounded Newton weight-cone/regime cache."""
    _discover_regimes_cached.cache_clear()


def weight_cone_cache_info():
    """Return ``functools.lru_cache`` statistics for regime discovery."""
    return _discover_regimes_cached.cache_info()


def multivariate_scaling_regimes(
    equation: sp.Expr,
    dependent: sp.Symbol,
    variables: tuple[sp.Symbol, ...] | list[sp.Symbol],
    *,
    assumptions: sp.Expr | bool = sp.S.true,
    stratify_parameters: bool = True,
    max_parameter_splits: int = 6,
) -> tuple[ScalingRegime, ...] | AsymptoticStratification[tuple[ScalingRegime, ...]]:
    """Discover admissible positive weight cones of the multivariate Newton diagram.

    The function enumerates lower Newton faces that contain terms of distinct
    dependent degree, derives the dependent balance exponent ``rho(w)`` exactly,
    and returns a rational representative scaling path for each nonempty cone.
    Parameter values that delete support points are automatically stratified.
    """

    variables = tuple(variables)
    equation = sp.sympify(equation)
    if not variables:
        raise ValueError("at least one scaling variable is required")
    parameters = parameter_symbols(equation, variables + (dependent,))
    terms = newton_polyhedron_terms(equation, dependent, variables)
    if stratify_parameters and parameters:
        structural = tuple(term.coefficient for term in terms)

        def evaluate(condition: sp.Expr) -> tuple[ScalingRegime, ...]:
            specialized = specialize_expression(equation, condition, parameters=parameters)
            return _discover_regimes_cached(
                canonical_expr(specialized),
                dependent,
                variables,
                canonical_expr(sp.And(sp.sympify(assumptions), condition)),
            )

        strat = automatic_parameter_stratification(
            structural,
            evaluate,
            parameters=parameters,
            assumptions=assumptions,
            max_splits=max_parameter_splits,
            provenance_source="asymptotic.multivariate_scaling_regimes",
        )
        if strat is not None:
            record_symbolic_event(
                "newton_cones_generated",
                sum(len(stratum.result) for stratum in strat.strata),
            )
            return strat
    result = _discover_regimes_cached(
        canonical_expr(equation),
        dependent,
        variables,
        canonical_expr(assumptions),
    )
    record_symbolic_event("newton_cones_generated", len(result))
    return result
