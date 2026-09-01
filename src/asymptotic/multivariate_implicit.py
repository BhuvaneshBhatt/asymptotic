from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from itertools import product

import sympy as sp

from ._integer_utils import integer_lcm as _lcm
from ._symbolic_policy import bounded_solve_system
from .multivariate import WeightCone, _rational_representative
from .parameter_auto import (
    automatic_parameter_stratification,
    parameter_symbols,
    specialize_expression,
)
from .stratification import AsymptoticStratification


@dataclass(frozen=True)
class JointNewtonTerm:
    equation_index: int
    variable_exponents: tuple[sp.Rational, ...]
    dependent_powers: tuple[sp.Rational, ...]
    coefficient: sp.Expr
    expression: sp.Expr


@dataclass(frozen=True)
class MultivariateImplicitBranch:
    """One jointly lifted branch along an automatically discovered weight cone."""

    variables: tuple[sp.Symbol, ...]
    dependents: tuple[sp.Symbol, ...]
    cone: WeightCone
    dependent_weights: tuple[sp.Expr, ...]
    leading_coefficients: tuple[sp.Expr, ...]
    representative_weights: tuple[sp.Rational, ...]
    parameter: sp.Symbol
    series: tuple[sp.Expr, ...]
    residuals: tuple[sp.Expr, ...]
    jacobian_determinant: sp.Expr
    complete: bool


@dataclass(frozen=True)
class MultivariateImplicitRegime:
    cone: WeightCone
    dependent_weights: tuple[sp.Expr, ...]
    active_terms: tuple[tuple[JointNewtonTerm, ...], ...]
    branches: tuple[MultivariateImplicitBranch, ...]


def _joint_terms(
    equations: tuple[sp.Expr, ...],
    dependents: tuple[sp.Symbol, ...],
    variables: tuple[sp.Symbol, ...],
) -> tuple[tuple[JointNewtonTerm, ...], ...]:
    generators = set(variables) | set(dependents)
    all_terms = []
    for eq_index, equation in enumerate(equations):
        row = []
        for term in sp.Add.make_args(sp.expand(equation)):
            pd = term.as_powers_dict()
            vx = tuple(sp.sympify(pd.get(v, 0)) for v in variables)
            dy = tuple(sp.sympify(pd.get(y, 0)) for y in dependents)
            if not all(e.is_Rational for e in vx + dy):
                continue
            monomial = sp.S.One
            for v, e in zip(variables, vx):
                monomial *= v**e
            for y, e in zip(dependents, dy):
                monomial *= y**e
            coeff = sp.simplify(term / monomial)
            if coeff.free_symbols & generators:
                continue
            row.append(
                JointNewtonTerm(
                    eq_index,
                    tuple(sp.Rational(e) for e in vx),
                    tuple(sp.Rational(e) for e in dy),
                    coeff,
                    term,
                )
            )
        all_terms.append(tuple(row))
    return tuple(all_terms)


def _weight(
    term: JointNewtonTerm, ws: tuple[sp.Symbol, ...], rhos: tuple[sp.Symbol, ...]
) -> sp.Expr:
    return sp.expand(
        sum(a * w for a, w in zip(term.variable_exponents, ws))
        + sum(p * r for p, r in zip(term.dependent_powers, rhos))
    )


def _lowest_epsilon_coefficient(
    expr: sp.Expr, eps: sp.Symbol
) -> tuple[sp.Rational, sp.Expr] | None:
    terms = sp.Add.make_args(sp.expand(expr))
    data = []
    for term in terms:
        power = sp.sympify(term.as_powers_dict().get(eps, 0))
        if not power.is_Rational:
            return None
        coeff = sp.simplify(term / eps**power)
        data.append((sp.Rational(power), coeff))
    if not data:
        return None
    minimum = min(p for p, _ in data)
    return minimum, sp.simplify(sum(c for p, c in data if p == minimum))


def _lift_joint_branch(
    equations: tuple[sp.Expr, ...],
    variables: tuple[sp.Symbol, ...],
    dependents: tuple[sp.Symbol, ...],
    weights: tuple[sp.Rational, ...],
    rhos: tuple[sp.Rational, ...],
    leading: tuple[sp.Expr, ...],
    *,
    terms: int,
) -> tuple[sp.Symbol, tuple[sp.Expr, ...], tuple[sp.Expr, ...], bool]:
    """Lift one multivariate implicit branch by solving the first nonzero residual layer jointly."""
    eps = sp.Dummy("epsilon", positive=True)
    xsubs = {x: eps**w for x, w in zip(variables, weights)}
    series = [sp.simplify(c * eps**rho) for c, rho in zip(leading, rhos)]
    denoms = [int(v.q) for v in weights + rhos]
    ram = reduce(_lcm, denoms, 1) or 1
    step = sp.Rational(1, ram)

    # Joint Newton correction: at each admissible ramified order, solve all
    # new coefficients simultaneously from the first nonzero residual layer.
    accepted = 1
    scan = 1
    max_scan = max(8 * terms * ram, terms)
    while accepted < terms and scan <= max_scan:
        delta = step * scan
        coeffs = tuple(sp.Dummy(f"c{j}_{scan}") for j in range(len(dependents)))
        trial = [sp.expand(s + c * eps ** (rho + delta)) for s, c, rho in zip(series, coeffs, rhos)]
        eqs = []
        for equation in equations:
            transformed = sp.expand(equation.subs(xsubs).subs(dict(zip(dependents, trial))))
            low = _lowest_epsilon_coefficient(transformed, eps)
            if low is None:
                eqs = []
                break
            _, coefficient = low
            eqs.append(sp.simplify(coefficient))
        if not eqs:
            break
        if not any(set(coeffs) & e.free_symbols for e in eqs):
            scan += 1
            continue
        solutions = bounded_solve_system(eqs, coeffs, allow_general=True) or ()
        if not solutions:
            break
        sol = min(solutions, key=lambda d: sum(sp.count_ops(v) for v in d.values()))
        if not all(c in sol for c in coeffs):
            break
        for j, (c, rho) in enumerate(zip(coeffs, rhos)):
            value = sp.simplify(sol[c])
            if value != 0:
                series[j] = sp.expand(series[j] + value * eps ** (rho + delta))
        accepted += 1
        scan += 1

    residuals = tuple(
        sp.simplify(eq.subs(xsubs).subs(dict(zip(dependents, series)))) for eq in equations
    )
    complete = all(r == 0 for r in residuals)
    return eps, tuple(series), residuals, complete


def _unstratified_multivariate_implicit(
    equations: tuple[sp.Expr, ...],
    dependents: tuple[sp.Symbol, ...],
    variables: tuple[sp.Symbol, ...],
    *,
    terms: int,
) -> tuple[MultivariateImplicitRegime, ...]:
    """Construct multivariate implicit branches before parameter-stratum splitting."""
    if len(equations) != len(dependents):
        raise ValueError("a square implicit system is required")
    support = _joint_terms(equations, dependents, variables)
    if any(len(row) < 2 for row in support):
        return ()
    ws = tuple(sp.Dummy(f"w_{v}", positive=True) for v in variables)
    rhos = tuple(sp.Dummy(f"rho_{y}") for y in dependents)
    pair_choices = []
    for row in support:
        choices = []
        for i, left in enumerate(row):
            for j in range(i + 1, len(row)):
                right = row[j]
                if left.dependent_powers != right.dependent_powers:
                    choices.append((i, j))
        if not choices:
            return ()
        pair_choices.append(tuple(choices))

    regimes = {}
    for selected in product(*pair_choices):
        equal_weight_eqs = []
        bases = []
        for row, (i, j) in zip(support, selected):
            left, right = row[i], row[j]
            equal_weight_eqs.append(sp.Eq(_weight(left, ws, rhos), _weight(right, ws, rhos)))
            bases.append(left)
        solved = bounded_solve_system(equal_weight_eqs, rhos) or ()
        for rho_sol in solved:
            if not all(r in rho_sol for r in rhos):
                continue
            rho_exprs = tuple(sp.factor(rho_sol[r]) for r in rhos)
            differences = []
            for row, base in zip(support, bases):
                base_w = _weight(base, ws, rhos).subs(rho_sol)
                differences.extend(
                    sp.factor(_weight(term, ws, rhos).subs(rho_sol) - base_w) for term in row
                )
            inequalities = tuple(d for d in differences if d != 0)
            rep = _rational_representative(
                ws, (), inequalities, strict=True
            ) or _rational_representative(ws, (), inequalities)
            if rep is None:
                continue
            rho_values = tuple(sp.simplify(r.subs(dict(zip(ws, rep)))) for r in rho_exprs)
            if not all(r.is_Rational for r in rho_values):
                continue
            rho_values = tuple(sp.Rational(r) for r in rho_values)
            active_rows = []
            leading_eqs = []
            cs = tuple(sp.Dummy(f"C_{y}") for y in dependents)
            for row in support:
                values = [
                    sp.simplify(_weight(term, ws, rhos).subs(rho_sol).subs(dict(zip(ws, rep))))
                    for term in row
                ]
                minimum = min(values)
                active = tuple(term for term, value in zip(row, values) if value == minimum)
                active_rows.append(active)
                leading_eqs.append(
                    sp.simplify(
                        sum(
                            term.coefficient
                            * sp.prod(c**p for c, p in zip(cs, term.dependent_powers))
                            for term in active
                        )
                    )
                )
            leading_solutions = bounded_solve_system(leading_eqs, cs, allow_general=True) or ()
            branches = []
            jac = sp.Matrix(leading_eqs).jacobian(cs)
            for sol in leading_solutions:
                if not all(c in sol for c in cs):
                    continue
                leading = tuple(sp.simplify(sol[c]) for c in cs)
                if any(c == 0 for c in leading):
                    continue
                jac_det = sp.simplify(jac.det().subs(sol))
                eps, series, residuals, complete = _lift_joint_branch(
                    equations, variables, dependents, rep, rho_values, leading, terms=terms
                )
                cone = WeightCone(ws, (), inequalities, sp.S.Zero, (), rep)
                branches.append(
                    MultivariateImplicitBranch(
                        variables,
                        dependents,
                        cone,
                        rho_exprs,
                        leading,
                        rep,
                        eps,
                        series,
                        residuals,
                        jac_det,
                        complete,
                    )
                )
            if not branches:
                continue
            key = tuple(
                tuple(row.index(term) for term in active)
                for row, active in zip(support, active_rows)
            )
            cone = branches[0].cone
            regimes[key] = MultivariateImplicitRegime(
                cone, rho_exprs, tuple(active_rows), tuple(branches)
            )
    return tuple(
        sorted(
            regimes.values(),
            key=lambda r: (-r.cone.dimension, tuple(map(str, r.cone.representative))),
        )
    )


def multivariate_implicit_asymptotics(
    equations: tuple[sp.Expr, ...] | list[sp.Expr],
    dependents: tuple[sp.Symbol, ...] | list[sp.Symbol],
    variables: tuple[sp.Symbol, ...] | list[sp.Symbol],
    *,
    terms: int = 4,
    assumptions: sp.Expr | bool = sp.S.true,
    stratify_parameters: bool = True,
    max_parameter_splits: int = 6,
) -> (
    tuple[MultivariateImplicitRegime, ...]
    | AsymptoticStratification[tuple[MultivariateImplicitRegime, ...]]
):
    """Discover and jointly lift multivariate implicit-system asymptotics.

    Unlike one-dimensional path reduction, dependent weights are solved
    simultaneously from one active Newton face in every equation.  The leading
    coefficient system and subsequent ramified corrections are then solved as
    a coupled system.
    """

    equations = tuple(map(sp.sympify, equations))
    dependents = tuple(dependents)
    variables = tuple(variables)
    if terms < 1:
        raise ValueError("terms must be positive")

    if stratify_parameters:
        params = parameter_symbols(sp.Add(*equations), variables + dependents)
        if params:
            structural = tuple(
                term.coefficient
                for row in _joint_terms(equations, dependents, variables)
                for term in row
            )

            def evaluate(condition: sp.Expr) -> tuple[MultivariateImplicitRegime, ...]:
                specialized = tuple(
                    specialize_expression(eq, condition, parameters=params) for eq in equations
                )
                result = multivariate_implicit_asymptotics(
                    specialized,
                    dependents,
                    variables,
                    terms=terms,
                    assumptions=condition,
                    stratify_parameters=False,
                    max_parameter_splits=max_parameter_splits,
                )
                if isinstance(result, AsymptoticStratification):
                    raise TypeError("unstratified solver returned a stratification")
                return result

            stratified = automatic_parameter_stratification(
                structural,
                evaluate,
                parameters=params,
                assumptions=assumptions,
                max_splits=max_parameter_splits,
                provenance_source="asymptotic.multivariate_implicit",
            )
            if stratified is not None:
                return stratified

    return _unstratified_multivariate_implicit(equations, dependents, variables, terms=terms)
