"""Graph substitution must preserve alternative branches and zero fibers."""

import pytest
import sympy as s

from asymptotic import limit
from asymptotic._limit_certificates import _substitute_explicit_graph_auxiliaries
from asymptotic.singular_curve_witnesses import signed_root_pole_witness


def test_alternative_equalities_are_not_global():
    x, v, w = s.symbols("x v w", real=True)
    alternatives = s.Or(s.And(x > 0, s.Eq(w, 1)), s.And(x < 0, s.Eq(w, -1)))
    formula = s.And(s.Eq(v, w + 1), alternatives)
    reduced, remaining = _substitute_explicit_graph_auxiliaries(formula, (v, w))
    assert reduced == alternatives
    assert remaining == (w,)
    assert reduced.subs({x: -1, w: -1}) is s.S.true
    assert reduced.subs({x: 1, w: 1}) is s.S.true


def test_moving_coefficient_preserves_zero_fiber():
    x, v = s.symbols("x v", real=True)
    formula = s.Eq(x * v, 0)
    reduced, remaining = _substitute_explicit_graph_auxiliaries(formula, (v,))
    assert remaining == (v,)
    assert reduced.subs({x: 0, v: 7}) is s.S.true
    assert reduced.subs({x: 1, v: 7}) is s.S.false


@pytest.mark.parametrize("q,n,scale", [(2, 3, 5), (3, 3, 3), (2, 5, 7)])
def test_attained_signed_root_curve(q, n, scale):
    x, y = s.symbols("x y", real=True)
    polynomial = x ** (q * n) - 2 * y**n
    expr = scale * x * y / (s.Abs(polynomial) ** s.Rational(1, n) * s.sign(polynomial))
    certificate = signed_root_pole_witness(expr, (x, y), (0, 0), s.S.true, s.S.true)
    assert certificate[0].name == "DOES_NOT_EXIST"
    axis, curve = certificate[2]
    assert axis.value == 0
    substitutions = dict(curve.substitutions)
    j = next(iter(substitutions[x].free_symbols))
    actual = expr.subs(substitutions, simultaneous=True).subs(j, 1000).evalf(50)
    assert abs(complex(actual - curve.value.evalf(50))) < 1e-7
    assert (
        polynomial.subs(substitutions, simultaneous=True).subs(j, 1000).is_negative
        is True
    )
    assert signed_root_pole_witness(expr, (x, y), (0, 0), y > 0, s.S.true) is None
    assert (
        limit(expr, (x, y), (0, 0), return_result=True).status.name == "DOES_NOT_EXIST"
    )


def test_independent_periodic_tails_are_attained():
    x, y = s.symbols("x y", real=True)
    expr = s.sin(2 * x + 1) - 3 * s.cos(y / 2)
    result = limit(expr, (x, y), (s.oo, -s.oo), return_result=True)
    assert result.status.name == "DOES_NOT_EXIST"
    assert {e.value for e in result.evidence} == {-3, -2}
    for evidence in result.evidence:
        substitutions = dict(evidence.substitutions)
        assert s.simplify(expr.subs(substitutions, simultaneous=True)) == evidence.value
        j = next(iter(substitutions[x].free_symbols))
        assert s.limit(substitutions[x], j, s.oo) == s.oo
        assert s.limit(substitutions[y], j, s.oo) == -s.oo


def test_isolated_piecewise_value_preserves_seams():
    from asymptotic.function_normalization import normalize_functions

    x, y = s.symbols("x y", real=True)
    point = s.Piecewise((99, s.Eq(x, 0) & s.Eq(y, 0)), (x * x + y * y, True))
    assert normalize_functions(point, (x, y), (0, 0))[0] == x * x + y * y
    seam = s.Piecewise((1, s.Eq(y, 0)), (0, True))
    assert normalize_functions(seam, (x, y), (0, 0))[0] == seam
    assert (
        limit(seam, (x, y), (0, 0), return_result=True).status.name == "DOES_NOT_EXIST"
    )


def test_parameter_nonidentity_is_not_inequality():
    from asymptotic._limit_composition import _exact_equal
    from asymptotic._multivariate_germ import _rational_path_limit

    x, y, a = s.symbols("x y a", real=True)
    t = s.Symbol("t", positive=True)
    assert _exact_equal(1, 1 / a) is None
    assert (
        _rational_path_limit((x * x + y * y) / (x * x + a * y * y), (x, y), (0, t))
        is None
    )
    assert _exact_equal(1, 2) is False


@pytest.mark.parametrize("q", [1, 2, 3])
def test_rational_corner_parameter_strata(q):
    x, y, a = s.symbols("x y a", real=True)
    expr = (x ** (2 * q) + y * y) / (x ** (2 * q) + a * y * y)
    result = limit(expr, (x, y), (0, 0), return_result=True)
    assert [st.result.status.name for st in result.strata] == [
        "PROVED",
        "DOES_NOT_EXIST",
    ]
    for st, condition in zip(result.strata, (s.Eq(a, 1), s.Ne(a, 1)), strict=True):
        assert s.simplify(s.Equivalent(st.condition, condition)) is s.S.true
    assert result.strata[0].result.value == 1
    witnesses = result.strata[1].result.evidence
    for parameter in [-1, 0, 2]:
        for witness in witnesses:
            actual = expr.subs(dict(witness.substitutions), simultaneous=True).subs(
                a, parameter
            )
            assert s.simplify(actual - witness.value.subs(a, parameter)) == 0
    assert limit(expr.subs(a, 1), (x, y), (0, 0)) == 1


def test_rational_pole_parameter_strata():
    x, y, a = s.symbols("x y a", real=True)
    result = limit(x * y * y / (x**4 + a * y * y), (x, y), (0, 0), return_result=True)
    positive = next(st for st in result.strata if st.result.status.name == "PROVED")
    nonpositive = next(
        st for st in result.strata if st.result.status.name == "DOES_NOT_EXIST"
    )
    assert s.simplify(s.Equivalent(positive.condition, a > 0)) is s.S.true
    assert s.simplify(s.Equivalent(nonpositive.condition, a <= 0)) is s.S.true
    assert positive.result.value == 0
    witnesses = nonpositive.result.evidence
    assert witnesses[0].value == 0
    assert witnesses[1].value.subs(a, 0) == 1
    assert witnesses[1].value.subs(a, -2) == -s.Rational(1, 4)
