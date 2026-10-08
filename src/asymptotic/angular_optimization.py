"""Certified parameterized optimization of homogeneous angular forms."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from ._symbolic_policy import bounded_ask, bounded_solve_one


@dataclass(frozen=True)
class AngularOptimizationResult:
    """Exact image bounds of a homogeneous form on the real unit sphere."""

    expression: sp.Expr
    variables: tuple[sp.Symbol, ...]
    minimum: sp.Expr | None
    maximum: sp.Expr | None
    critical_values: tuple[sp.Expr, ...]
    parameter_discriminant: sp.Expr | None
    certified: bool
    provider: str
    statement: str

    @property
    def cluster_set(self):
        if not self.certified or self.minimum is None or self.maximum is None:
            return None
        if sp.simplify(self.minimum - self.maximum) == 0:
            return sp.FiniteSet(self.minimum)
        return sp.Interval(self.minimum, self.maximum)


def _homogeneous_degree(expr, variables):
    try:
        poly = sp.Poly(sp.expand(expr), *variables)
    except sp.PolynomialError:
        return None
    degrees = {sum(mon) for mon, coeff in poly.terms() if coeff != 0}
    return next(iter(degrees)) if len(degrees) == 1 else None


def _quadratic_sphere_range(expr, variables):
    """Exact eigenvalue range for a real symmetric quadratic form in 2D."""
    if len(variables) != 2:
        return None
    x, y = variables
    expanded = sp.expand(expr)
    a = sp.simplify(expanded.coeff(x, 2))
    c = sp.simplify(expanded.coeff(y, 2))
    xy = sp.simplify(expanded.coeff(x, 1).coeff(y, 1))
    if sp.simplify(expanded - a * x**2 - xy * x * y - c * y**2) != 0:
        return None
    if any(v in (a.free_symbols | xy.free_symbols | c.free_symbols) for v in variables):
        return None
    disc = sp.simplify((a - c) ** 2 + xy**2)
    root = sp.sqrt(disc)
    lo = sp.simplify((a + c - root) / 2)
    hi = sp.simplify((a + c + root) / 2)
    return lo, hi, (lo, hi), disc, "symmetric_quadratic_eigenvalues"


def _even_bivariate_sphere_range(expr, variables):
    """Optimize an even homogeneous bivariate form via z=x**2 in [0,1]."""
    if len(variables) != 2:
        return None
    x, y = variables
    degree = _homogeneous_degree(expr, variables)
    if degree is None or degree % 2:
        return None
    try:
        poly = sp.Poly(sp.expand(expr), x, y)
    except sp.PolynomialError:
        return None
    if any(
        any(power % 2 for power in mon) for mon, coeff in poly.terms() if coeff != 0
    ):
        return None
    z = sp.Dummy("_angular_z", real=True)
    descended = sp.expand(expr)
    # Homogeneity and even exponents make this substitution exact on x^2+y^2=1.
    descended = descended.subs(x**2, z).subs(y**2, 1 - z)
    descended = sp.expand(descended)
    if descended.free_symbols & {x, y}:
        return None
    derivative = sp.factor(sp.diff(descended, z))
    try:
        roots = (
            tuple(bounded_solve_one(derivative, z, allow_general=True) or ())
            if derivative != 0
            else ()
        )
    except (NotImplementedError, TypeError, ValueError):
        return None
    # A symbolic root is usable only when its membership in [0,1] is provable.
    interior = []
    for root in roots:
        ge0 = bounded_ask(sp.Q.nonnegative(root))
        le1 = bounded_ask(sp.Q.nonpositive(root - 1))
        if ge0 is True and le1 is True:
            interior.append(sp.simplify(root))
        elif root.free_symbols:
            # Parameter-dependent root membership requires a later semialgebraic
            # parameter-cell refinement; do not discard it.
            return None
    candidates = (
        sp.simplify(descended.subs(z, 0)),
        sp.simplify(descended.subs(z, 1)),
        *(sp.simplify(descended.subs(z, root)) for root in interior),
    )
    lo = sp.Min(*candidates)
    hi = sp.Max(*candidates)
    discriminant = None
    if derivative != 0:
        try:
            discriminant = sp.factor(sp.discriminant(derivative, z))
        except (sp.PolynomialError, TypeError, ValueError):
            pass
    return (
        lo,
        hi,
        tuple(candidates),
        discriminant,
        "even_bivariate_simplex_critical_values",
    )


def parameterized_angular_range(expr, variables):
    """Certify exact parameter-dependent extrema on the unit sphere.

    The implementation is algebraic, not sampled.  It currently covers every
    real symmetric bivariate quadratic form (including cross terms and symbolic
    coefficients) and even homogeneous bivariate forms whose critical points
    are uniformly known to remain in the simplex.  Unsupported parameter-cell
    changes return an uncertified result for later QE/CAD refinement.
    """
    expr = sp.sympify(expr)
    variables = tuple(variables)
    degree = _homogeneous_degree(expr, variables)
    if degree is None:
        return AngularOptimizationResult(
            expr,
            variables,
            None,
            None,
            (),
            None,
            False,
            "none",
            "angular form is not homogeneous",
        )
    for solver in (_quadratic_sphere_range, _even_bivariate_sphere_range):
        data = solver(expr, variables)
        if data is None:
            continue
        lo, hi, critical, discriminant, provider = data
        return AngularOptimizationResult(
            expr,
            variables,
            lo,
            hi,
            critical,
            discriminant,
            True,
            provider,
            "exact continuous angular image obtained by algebraic sphere optimization",
        )
    parameters = expr.free_symbols - set(variables)
    if not parameters:
        from .semialgebraic_angular import semialgebraic_angular_image

        sphere = sp.Eq(sp.Add(*(v**2 for v in variables)), 1)
        general = semialgebraic_angular_image(expr, variables, sphere)
        if general.certified:
            return AngularOptimizationResult(
                expr,
                variables,
                general.minimum,
                general.maximum,
                (),
                None,
                True,
                general.provider,
                general.statement,
            )
    return AngularOptimizationResult(
        expr,
        variables,
        None,
        None,
        (),
        None,
        False,
        "none",
        "parameter-dependent angular critical topology requires parameter-cell refinement",
    )


@dataclass(frozen=True)
class AngularParameterStratum:
    """An exact angular image valid on one parameter condition."""

    condition: sp.Expr
    minimum: sp.Expr
    maximum: sp.Expr
    provider: str

    @property
    def cluster_set(self):
        if sp.simplify(self.minimum - self.maximum) == 0:
            return sp.FiniteSet(self.minimum)
        return sp.Interval(self.minimum, self.maximum)


def _even_bivariate_parameter_cells(expr, variables, base_condition):
    """Exact cells induced by symbolic stationary-point simplex membership."""
    from .parameter_cells import parameter_truth_cells, root_interval_membership

    if len(variables) != 2:
        return ()
    x, y = variables
    degree = _homogeneous_degree(expr, variables)
    if degree is None or degree % 2:
        return ()
    try:
        poly = sp.Poly(sp.expand(expr), x, y)
    except sp.PolynomialError:
        return ()
    if any(any(power % 2 for power in mon) for mon, coeff in poly.terms() if coeff):
        return ()

    z = sp.Dummy("_angular_z", real=True)
    descended = sp.expand(expr).subs(x**2, z).subs(y**2, 1 - z)
    descended = sp.expand(descended)
    if descended.free_symbols & {x, y}:
        return ()
    derivative = sp.factor(sp.diff(descended, z))
    if derivative == 0:
        value = sp.simplify(descended)
        return (
            AngularParameterStratum(
                sp.sympify(base_condition), value, value, "even_simplex_constant"
            ),
        )
    try:
        roots = tuple(
            map(sp.simplify, bounded_solve_one(derivative, z, allow_general=True) or ())
        )
    except (NotImplementedError, TypeError, ValueError):
        return ()

    parameters = tuple(
        sorted(
            (descended.free_symbols - {z})
            | set().union(*(root.free_symbols for root in roots)),
            key=sp.default_sort_key,
        )
    )
    symbolic = []
    always = []
    for root in roots:
        if not root.free_symbols:
            if (
                bounded_ask(sp.Q.nonnegative(root)) is True
                and bounded_ask(sp.Q.nonpositive(root - 1)) is True
            ):
                always.append(root)
            continue
        membership = root_interval_membership(root, z, 0, 1, parameters)
        if membership is None:
            return ()
        symbolic.append((root, membership))
    if not symbolic:
        return ()

    cell_parameters = tuple(
        sorted(
            set(parameters)
            | (sp.sympify(base_condition).free_symbols - set(variables)),
            key=sp.default_sort_key,
        )
    )
    cells = parameter_truth_cells(
        tuple(membership for _, membership in symbolic),
        cell_parameters,
        base_condition,
    )
    if not cells:
        return ()
    endpoint_values = (
        sp.simplify(descended.subs(z, 0)),
        sp.simplify(descended.subs(z, 1)),
    )
    result = []
    for cell in cells:
        active = list(always)
        active.extend(
            root
            for truth, (root, _) in zip(cell.signature, symbolic, strict=True)
            if truth
        )
        values = endpoint_values + tuple(
            sp.simplify(descended.subs(z, root)) for root in active
        )
        result.append(
            AngularParameterStratum(
                cell.condition,
                sp.Min(*values),
                sp.Max(*values),
                "even_simplex_parameter_cell_qe",
            )
        )
    return tuple(result)


def parameterized_angular_strata(expr, variables, condition=sp.S.true):
    """Refine parameter space where angular-image topology changes.

    For a general real bivariate quadratic form, the only topology-changing
    discriminant is equality of the two eigenvalues.  Splitting on that exact
    algebraic discriminant prevents a generic non-singleton image from hiding
    exceptional scalar-limit strata.
    """
    condition = sp.sympify(condition)
    cells = _even_bivariate_parameter_cells(expr, variables, condition)
    if cells:
        return cells
    result = parameterized_angular_range(expr, variables)
    if not result.certified:
        return ()
    disc = result.parameter_discriminant
    if (
        result.provider == "symmetric_quadratic_eigenvalues"
        and disc is not None
        and disc.free_symbols
    ):
        center = sp.simplify((result.minimum + result.maximum) / 2)
        return (
            AngularParameterStratum(
                sp.simplify_logic(sp.And(condition, sp.Eq(disc, 0))),
                center,
                center,
                "quadratic_repeated_eigenvalue",
            ),
            AngularParameterStratum(
                sp.simplify_logic(sp.And(condition, sp.Ne(disc, 0))),
                result.minimum,
                result.maximum,
                "quadratic_distinct_eigenvalues",
            ),
        )
    return (
        AngularParameterStratum(
            condition, result.minimum, result.maximum, result.provider
        ),
    )


__all__ = [
    "AngularOptimizationResult",
    "AngularParameterStratum",
    "parameterized_angular_range",
    "parameterized_angular_strata",
]
