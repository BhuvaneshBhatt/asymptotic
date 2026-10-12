"""Regression corpus for finite-point Laurent/logarithmic multiseries.

The closed-form cutoff inputs are derived from a single shared level cutoff
    |k*y*(y-x**m)| > epsilon
on 0 < x < 1, 0 < y < x**m, so the tests do not import MultipleIntegrate.

The SymPy-only oracle tests run without asymptotic. Integration tests import
asymptotic on demand and expose incomplete scale discovery / term truncation.
"""

from __future__ import annotations

import pytest
import sympy as sp


def canonical_tangent_cutoff(
    epsilon: sp.Symbol, order: int, regulator: int = 1
) -> sp.Expr:
    """Exact cutoff for x**(m-1)/(y*(y-x**m)), shared |k*g| > eps."""
    if order < 2 or regulator == 0:
        raise ValueError("order >= 2 and nonzero regulator required")
    delta = epsilon / abs(regulator)
    smaller = (1 - sp.sqrt(1 - 4 * delta)) / 2
    larger = 1 - smaller
    return (
        sp.log(larger) ** 2
        - sp.log(smaller) ** 2
        - 2 * sp.log(smaller) * sp.log(larger)
        - 4 * sp.polylog(2, smaller)
        + sp.pi**2 / 3
    ) / (2 * order)


def weighted_tangent_cutoff(
    epsilon: sp.Symbol, order: int, regulator: int = 1
) -> sp.Expr:
    """Exact cutoff for x**(3*m-1)/(y*(y-x**m)) with same shared cutoff."""
    if order < 2 or regulator == 0:
        raise ValueError("order >= 2 and nonzero regulator required")
    delta = epsilon / abs(regulator)
    smaller = (1 - sp.sqrt(1 - 4 * delta)) / 2
    larger = 1 - smaller
    return (sp.sqrt(1 - 4 * delta) - (1 - 2 * delta) * sp.log(larger / smaller)) / order


def finite_part_of_series(expression: sp.Expr, epsilon: sp.Symbol) -> sp.Expr:
    """Extract epsilon**0 log(epsilon)**0 after an expansion through order zero.

    This is a test oracle, NOT a new public asymptotic finite-part API.
    A Laurent/logarithmic term with strictly negative epsilon exponent, or
    nonzero log exponent at epsilon**0, is divergent. Terms O(epsilon**a)
    with a>0 are discarded by sp.series(...,0,1).
    """
    series = sp.expand(sp.series(expression, epsilon, 0, 1).removeO())
    divergent = []
    for term in sp.Add.make_args(series):
        exponent = sp.sympify(term.as_powers_dict().get(epsilon, 0))
        if exponent.is_negative is True or (
            exponent.is_zero is True and term.has(sp.log(epsilon))
        ):
            divergent.append(term)
    finite = sp.limit(sp.expand(series - sp.Add(*divergent)), epsilon, 0, dir="+")
    assert finite.is_finite is not False, (expression, series, finite)
    assert not finite.has(epsilon, sp.Integral, sp.Limit)
    return sp.simplify(finite)


@pytest.mark.parametrize("order,regulator", [(2, 1), (3, 1), (4, 1), (2, 2), (2, -3)])
def test_exact_unweighted_cutoff_sympy_oracle(order: int, regulator: int) -> None:
    epsilon = sp.Symbol("epsilon", positive=True)
    cutoff = canonical_tangent_cutoff(epsilon, order, regulator)
    expected = sp.pi**2 / (6 * order) - sp.log(abs(regulator)) ** 2 / (2 * order)
    assert sp.simplify(finite_part_of_series(cutoff, epsilon) - expected) == 0


@pytest.mark.parametrize("order,regulator", [(2, 1), (3, 1), (2, 2), (3, -3)])
def test_exact_weighted_cutoff_sympy_oracle(order: int, regulator: int) -> None:
    epsilon = sp.Symbol("epsilon", positive=True)
    cutoff = weighted_tangent_cutoff(epsilon, order, regulator)
    expected = (1 - sp.log(abs(regulator))) / order
    assert sp.simplify(finite_part_of_series(cutoff, epsilon) - expected) == 0


def _native_multiseries(expr: sp.Expr, epsilon: sp.Symbol, count: int = 8):
    asymptotic = pytest.importorskip("asymptotic")
    return asymptotic.multiseries(
        expr,
        epsilon,
        point=0,
        terms=count,
        allow_series_fallback=True,
        return_result=True,
    )


def _scale_contains(scale, representative: sp.Expr) -> bool:
    return any(
        sp.simplify(item - representative) == 0 for item in getattr(scale, "exprs", ())
    )


def test_finite_origin_scale_discovery_includes_local_uniformizer() -> None:
    """v0.2.0 incorrectly seeds 1/epsilon at point=0 and removes it."""
    epsilon = sp.Symbol("epsilon", positive=True)
    series = _native_multiseries(1 + epsilon + epsilon**2, epsilon)
    assert _scale_contains(series.scale, epsilon), (
        "Missing local epsilon scale at point=0",
        series.scale.exprs,
    )
    assert sp.simplify(series.truncate(8) - (1 + epsilon + epsilon**2)) == 0


def test_finite_shifted_point_uses_x_minus_point_not_inverse_x() -> None:
    x = sp.Symbol("x", real=True)
    # At finite point 2 we must explicitly pass point=2.
    asymptotic = pytest.importorskip("asymptotic")
    series = asymptotic.multiseries(
        2 + (x - 2) + (x - 2) ** 2, x, point=2, terms=8, return_result=True
    )
    assert _scale_contains(series.scale, x - 2), series.scale.exprs


def test_logarithmic_germ_has_both_logarithmic_and_algebraic_scales() -> None:
    epsilon = sp.Symbol("epsilon", positive=True)
    series = _native_multiseries(sp.log(epsilon) ** 2 + 3 + epsilon, epsilon)
    assert _scale_contains(series.scale, epsilon), (
        "Logarithmic terms must not suppress the algebraic local scale",
        series.scale.exprs,
    )


@pytest.mark.parametrize("pole_order", [1, 2])
def test_native_laurent_logarithmic_constant_with_pole(pole_order: int) -> None:
    """An epsilon^0 constant cannot be omitted after higher-order poles."""
    epsilon = sp.Symbol("epsilon", positive=True)
    expr = (
        epsilon ** (-pole_order)
        + 3 * sp.log(epsilon) / epsilon
        - 7 * sp.log(epsilon) ** 2
        + 5 * sp.log(epsilon)
        + 11
        + epsilon * sp.log(epsilon)
    )
    result = _native_multiseries(expr, epsilon, 12)
    assert _scale_contains(result.scale, epsilon), result.scale.exprs
    prefix = sp.sympify(result.truncate(12))
    assert sp.simplify(finite_part_of_series(prefix, epsilon) - 11) == 0


@pytest.mark.parametrize("order,regulator", [(2, 1), (3, 1), (4, 1), (2, 2), (2, -3)])
def test_multiseries_unweighted_cutoff_exposes_finite_constant(
    order: int, regulator: int
) -> None:
    epsilon = sp.Symbol("epsilon", positive=True)
    cutoff = canonical_tangent_cutoff(epsilon, order, regulator)
    # These particular cutoffs have no negative epsilon powers. Their entire
    # finite-part data lives in the epsilon**0 coefficient, which is the first
    # algebraic-scale term (its coefficient may contain log(epsilon)).
    # Asking for eight terms makes the native/SymPy fallback expand unnecessary
    # positive powers and can cause pathological symbolic over-expansion.
    result = _native_multiseries(cutoff, epsilon, 1)
    assert _scale_contains(result.scale, epsilon), result.scale.exprs
    prefix = sp.sympify(result.truncate(1))
    expected = sp.pi**2 / (6 * order) - sp.log(abs(regulator)) ** 2 / (2 * order)
    # The epsilon**0 coefficient must be resolved, not the unexpanded input.
    assert not prefix.has(sp.polylog), (result.scale.exprs, prefix)
    assert sp.simplify(finite_part_of_series(prefix, epsilon) - expected) == 0


@pytest.mark.parametrize("order,regulator", [(2, 1), (3, 1), (2, 2), (3, -3)])
def test_multiseries_weighted_cutoff_exposes_finite_constant(
    order: int, regulator: int
) -> None:
    epsilon = sp.Symbol("epsilon", positive=True)
    cutoff = weighted_tangent_cutoff(epsilon, order, regulator)
    # These particular cutoffs have no negative epsilon powers. Their entire
    # finite-part data lives in the epsilon**0 coefficient, which is the first
    # algebraic-scale term (its coefficient may contain log(epsilon)).
    # Asking for eight terms makes the native/SymPy fallback expand unnecessary
    # positive powers and can cause pathological symbolic over-expansion.
    result = _native_multiseries(cutoff, epsilon, 1)
    assert _scale_contains(result.scale, epsilon), result.scale.exprs
    prefix = sp.sympify(result.truncate(1))
    expected = (1 - sp.log(abs(regulator))) / order
    assert sp.simplify(finite_part_of_series(prefix, epsilon) - expected) == 0
