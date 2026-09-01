from __future__ import annotations

import sympy as sp
from hypothesis import given, settings
from hypothesis import strategies as st

from asymptotic import multiseries, series_reversion, transseries_from_expression
from asymptotic.canonical import canonical_equal, canonical_expr

FAST = settings(max_examples=16, deadline=None, derandomize=True)


@FAST
@given(st.integers(-5, 5), st.integers(-5, 5), st.integers(-5, 5))
def test_canonicalization_is_invariant_to_commutative_argument_order(a, b, c):
    x, y, z = sp.symbols("x y z")
    left = sp.Add(a * x, b * y, c * z, evaluate=False)
    right = sp.Add(c * z, a * x, b * y, evaluate=False)
    assert canonical_equal(left, right)
    assert canonical_expr(left) == canonical_expr(right)


@FAST
@given(st.integers(-3, 3), st.integers(-3, 3))
def test_multiseries_is_invariant_to_expanded_or_factored_input(a, b):
    x = sp.symbols("x", positive=True)
    factored = (1 + a / x) * (1 + b / x)
    expanded = sp.expand(factored)
    lhs = multiseries(factored, x, scale=[1 / x], terms=3).terms(3)
    rhs = multiseries(expanded, x, scale=[1 / x], terms=3).terms(3)
    assert lhs == rhs


@FAST
@given(st.integers(-3, 3).filter(bool))
def test_series_reversion_round_trip_is_stable_under_small_integer_coefficients(a):
    x, y = sp.symbols("x y")
    f = x + a * x**2
    inverse = series_reversion(f, x, y, terms=5, branch=0).truncate()
    round_trip = sp.series(f.subs(x, inverse), y, 0, 5).removeO()
    assert sp.expand(round_trip - y) == 0


@FAST
@given(st.integers(-4, 4), st.integers(-4, 4))
def test_transseries_conversion_is_invariant_to_additive_reassociation(a, b):
    x = sp.symbols("x", positive=True)
    left = sp.Add(1 / x, a / x**2, b / x**3, evaluate=False)
    right = sp.Add(b / x**3, sp.Add(a / x**2, 1 / x, evaluate=False), evaluate=False)
    lhs = transseries_from_expression(left, x, point=sp.oo, complete=True)
    rhs = transseries_from_expression(right, x, point=sp.oo, complete=True)
    assert sp.simplify(lhs.truncate() - rhs.truncate()) == 0
    assert lhs.remainder == rhs.remainder


@FAST
@given(st.integers(1, 4), st.integers(-3, 3))
def test_dummy_symbol_renaming_preserves_multiseries(power, coeff):
    x = sp.symbols("x", positive=True)
    t = sp.symbols("t", positive=True)
    expr_x = sp.exp(1 / x) + coeff / x**power
    expr_t = expr_x.xreplace({x: t})
    lhs = multiseries(expr_x, x, scale=[1 / x], terms=5).truncate(5)
    rhs = multiseries(expr_t, t, scale=[1 / t], terms=5).truncate(5).xreplace({t: x})
    assert sp.expand(lhs - rhs) == 0


@FAST
@given(st.integers(1, 4))
def test_positive_rescaling_of_independent_variable_is_consistent(scale):
    x, t = sp.symbols("x t", positive=True)
    expr = sp.exp(1 / x)
    direct = multiseries(expr, x, scale=[1 / x], terms=5).truncate(5)
    changed = multiseries(expr.subs(x, scale * t), t, scale=[1 / t], terms=5).truncate(5)
    restored = sp.expand(changed.subs(t, x / scale))
    assert sp.simplify(direct - restored) == 0


@FAST
@given(st.integers(2, 5), st.integers(6, 9))
def test_larger_multiseries_budget_preserves_previous_prefix(short_terms, long_terms):
    from asymptotic.canonical import canonical_equal

    if long_terms <= short_terms:
        long_terms = short_terms + 1
    x = sp.symbols("x", positive=True)
    expr = sp.exp(1 / x)
    short = multiseries(expr, x, scale=[1 / x], terms=short_terms).truncate(short_terms)
    long = multiseries(expr, x, scale=[1 / x], terms=long_terms).truncate(short_terms)
    assert canonical_equal(short, long)


@FAST
@given(st.integers(-2, 2), st.integers(-2, 2))
def test_transseries_differentiation_reconstructs_exact_finite_derivative(a, b):
    x = sp.symbols("x", positive=True)
    expr = 1 + a / x + b / x**2
    expansion = transseries_from_expression(expr, x, point=sp.oo, complete=True)
    derivative = expansion.differentiate()
    assert sp.simplify(derivative.truncate() - sp.diff(expr, x)) == 0
    assert derivative.remainder.check() is True


def test_reciprocal_coordinate_change_matches_local_zero_expansion():
    x, h = sp.symbols("x h", positive=True)
    at_infinity = multiseries(sp.exp(1 / x), x, scale=[1 / x], terms=5).truncate(5)
    at_zero = multiseries(sp.exp(h), h, scale=[h], point=0, terms=5).truncate(5)
    assert sp.simplify(at_infinity - at_zero.subs(h, 1 / x)) == 0


def test_asymptotic_integration_then_differentiation_reconstructs_input():
    from asymptotic import asymptotic_integrate

    x = sp.symbols("x", positive=True)
    source = multiseries(1 / x + 1 / x**2, x, scale=[1 / x], terms=4)
    primitive = asymptotic_integrate(source, terms=4)
    assert sp.simplify(sp.diff(primitive.truncate(), x) - source.expr) == 0


def test_parameter_strata_partition_is_disjoint_and_exhaustive():
    from asymptotic.stratification import evaluate_parameter_strata

    a = sp.symbols("a", real=True)
    result = evaluate_parameter_strata(
        (sp.Eq(a, 0), sp.Ne(a, 0)),
        lambda condition: condition,
        parameters=(a,),
        require_exhaustive=True,
    )
    assert result.exhaustive
    assert len(result.strata) == 2
    left, right = (stratum.condition for stratum in result.strata)
    assert sp.simplify(sp.And(left, right)) is sp.S.false
    assert sp.simplify(sp.Or(left, right)) is sp.S.true
