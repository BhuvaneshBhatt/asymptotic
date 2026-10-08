"""Finite-height log-exp transseries cells for multivariate scale regimes."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from .context import AsymptoticGrowthComparison
from .exp_log_scale import LogExpScale, compare_log_exp_scales
from .proof_obligations import ObligationKind, ProofObligation


def _finite_logexp(node: sp.Expr, u: sp.Symbol) -> bool:
    if u not in node.free_symbols:
        return True
    if node == u:
        return True
    if node.func in (sp.exp, sp.log):
        return _finite_logexp(node.args[0], u)
    if node.is_Add or node.is_Mul:
        return all(_finite_logexp(a, u) for a in node.args)
    if node.is_Pow:
        base, exponent = node.as_base_exp()
        return u not in exponent.free_symbols and _finite_logexp(base, u)
    return False


@dataclass(frozen=True)
class MultivariateGrowthScale:
    """One finite-height LE scale expressed in a certified cell coordinate."""

    expression: sp.Expr
    small_parameter: sp.Symbol
    exponential_height: int
    logarithmic_depth: int

    @classmethod
    def from_expression(cls, expression: sp.Expr, small_parameter: sp.Symbol):
        expression = sp.sympify(expression)
        if not _finite_logexp(expression, small_parameter):
            raise ValueError("expression is not a finite-height log-exp scale")
        # Existing LE comparison is strongest at infinity.  u -> 0+ is mapped
        # to t -> +infinity.
        t = sp.Dummy("scale_t", positive=True)
        at_infinity = sp.simplify(expression.subs(small_parameter, 1 / t))
        scale = LogExpScale.from_expr(at_infinity, t, point=sp.oo)
        return cls(
            expression,
            small_parameter,
            scale.exponential_height,
            scale.logarithmic_depth,
        )

    def compare(self, other: MultivariateGrowthScale) -> AsymptoticGrowthComparison:
        if self.small_parameter != other.small_parameter:
            raise ValueError("growth scales use different cell coordinates")
        t = sp.Dummy("scale_t", positive=True)
        left = self.expression.subs(self.small_parameter, 1 / t)
        right = other.expression.subs(other.small_parameter, 1 / t)
        return compare_log_exp_scales(left, right, t, point=sp.oo)


@dataclass(frozen=True)
class CertifiedTransseriesExpansion:
    expression: sp.Expr
    approximation: sp.Expr
    remainder_scale: sp.Expr
    regime: object
    order: int
    certified: bool
    method: str
    scales: tuple[MultivariateGrowthScale, ...] = ()
    obligations: tuple[ProofObligation, ...] = ()


def _scale_atoms(expr: sp.Expr, u: sp.Symbol) -> tuple[sp.Expr, ...]:
    atoms = set()
    for node in sp.preorder_traversal(expr):
        if u not in getattr(node, "free_symbols", set()):
            continue
        if node.func in (sp.exp, sp.log):
            atoms.add(node)
    if not atoms and u in expr.free_symbols:
        atoms.add(u)
    return tuple(sorted(atoms, key=sp.default_sort_key))


def certified_transseries_expand(
    expr, *, small, large, order=3, statement="explicit log-exp regime"
):
    """Expand one cell under ``small/large=o(1)`` in finite-height LE scales.

    The positive dummy coordinate makes real logarithmic branches unambiguous.
    Unsupported oscillatory/special-function atoms decline conservatively.
    """
    from .stratified_expansion import ExpansionRegime, SmallQuantity

    expr = sp.sympify(expr)
    small = sp.sympify(small)
    large = sp.sympify(large)
    ratio = sp.simplify(small / large)
    u = sp.Dummy("le_u", positive=True)
    transformed = sp.simplify(expr.subs(small, u * large))
    if not _finite_logexp(transformed, u):
        return None

    exact = False
    try:
        series = sp.series(transformed, u, 0, order)
        approximation_u = series.removeO()
        remainder = series.getO()
        if remainder is None:
            exact = sp.simplify(approximation_u - transformed) == 0
            if not exact:
                return None
            remainder_u = sp.S.Zero
        else:
            remainder_u = sp.simplify(remainder.expr)
    except (TypeError, ValueError, NotImplementedError, sp.PoleError):
        approximation_u = transformed
        if sp.simplify(approximation_u - transformed) != 0:
            return None
        exact = True
        remainder_u = sp.S.Zero

    if not _finite_logexp(approximation_u, u):
        return None
    try:
        scales = tuple(
            MultivariateGrowthScale.from_expression(a, u)
            for a in _scale_atoms(approximation_u, u)
        )
    except ValueError:
        return None
    regime = ExpansionRegime((SmallQuantity(ratio),), statement=statement)
    return CertifiedTransseriesExpansion(
        expr,
        sp.simplify(approximation_u.subs(u, ratio)),
        sp.simplify(remainder_u.subs(u, ratio)),
        regime,
        order,
        True,
        "exact_logexp_scale" if exact else "finite_height_logexp_series",
        scales,
    )


def missing_transseries_obligation(expr: sp.Expr) -> ProofObligation:
    return ProofObligation(
        ObligationKind.ORDER_RELATION,
        "the cell is outside the certified finite-height real log-exp transseries class",
        provider="certified_transseries_expand",
        expression=sp.sstr(expr),
    )
