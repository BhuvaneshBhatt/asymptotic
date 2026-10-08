"""Uniform weighted bounds, local perturbations and their proof boundaries."""

import pytest
import sympy as sp

from asymptotic import limit
from asymptotic.limit_models import LimitStatus
from asymptotic.trigonometric_pole_germs import trigonometric_pole_certificate
from asymptotic.weighted_monomial_bounds import (
    _excess_bound,
    weighted_quotient_certificate,
)


@pytest.mark.parametrize(
    "family", ["radical_numerator", "radical_denominator", "expm1", "log1p"]
)
def test_weighted_quotients(family):
    x, y = sp.symbols("x y", real=True)
    expressions = {
        "radical_numerator": sp.sqrt(x * x + y * y)
        * (-(x**6) * y + x * x * y**3)
        / (x**6 + y**4),
        "radical_denominator": x * y * y / (x * x + y * y * sp.sqrt(x * x + y * y)),
        "expm1": (sp.exp(x * y) - 1) / (x * x - x**4 + sp.Abs(y)),
        "log1p": sp.log(1 + x * x * sp.Abs(y - 1)) / sp.sqrt(sp.Abs(x * (y - 1))),
    }
    result = limit(expressions[family], (x, y), (0, 0), return_result=True)
    assert result.status is LimitStatus.PROVED
    assert result.value == 0
    assert result.evidence[0].method == "weighted_monomial_bound"
    mapping = dict(result.evidence[0].substitutions)
    index = next(iter(mapping[x].free_symbols))
    for variable in (x, y):
        assert sp.limit(mapping[variable], index, sp.oo) == 0
    denominator = sp.fraction(expressions[family])[1].subs(mapping)
    for j in (10, 100):
        assert denominator.subs(index, j).is_positive is True


def test_weighted_bound_on_axes_and_anisotropic_scales():
    x, y = sp.symbols("x y", real=True)
    exponents = (sp.Integer(2), sp.Integer(3))
    denominator = [
        (sp.S.One, (sp.Integer(6), sp.S.Zero)),
        (sp.S.One, (sp.S.Zero, sp.Integer(4))),
    ]
    bound = _excess_bound([(sp.S.One, exponents)], denominator, (x, y))
    quotient = sp.Abs(x**2 * y**3) / (x**6 + y**4)
    for n in (10, 100):
        u = sp.Rational(1, n)
        for point in [(0, u), (u, 0), (u, u**2), (-u, u**2), (u**3, -u)]:
            substitutions = dict(zip((x, y), point, strict=True))
            difference = (bound - quotient).subs(substitutions)
            assert difference.evalf(50) >= 0
    assert bound.subs(x, 0) == 0
    assert bound.subs(y, 0) == 0


@pytest.mark.parametrize(
    "expression",
    [
        "x*y/(x**2+y**2)",
        "x**2/(x**2-y**2)",
        "x*y**2/(x**2-y**2*sqrt(x**2+y**2))",
    ],
)
def test_nonvanishing_and_signed_denominators(expression):
    x, y = sp.symbols("x y", real=True)
    expr = sp.sympify(expression, locals={"x": x, "y": y})
    assert (
        weighted_quotient_certificate(expr, (x, y), (0, 0), sp.S.true, sp.S.true)
        is None
    )


@pytest.mark.parametrize("root", [False, True])
@pytest.mark.parametrize("power", [sp.Rational(1, 4), sp.S.Half, sp.S.One])
def test_signed_trigonometric_germs(root, power):
    x, y = sp.symbols("x y", real=True)
    phase = -(x * x + y**4)
    denominator = 1 - (sp.sqrt(sp.cos(phase)) if root else sp.cos(phase))
    expr = -3 * sp.sin(phase) / denominator**power
    expected = 0 if power < sp.S.Half else (6 if root else 3 * sp.sqrt(2))
    if power > sp.S.Half:
        expected = sp.oo
    result = trigonometric_pole_certificate(expr, (x, y), (0, 0), sp.S.true, sp.S.true)
    assert result[0] is LimitStatus.PROVED
    assert result[1] == expected
    mapping = dict(result[2].substitutions)
    index = next(iter(sp.Add(*mapping.values()).free_symbols))
    assert phase.subs(mapping) != 0
    assert sp.limit(phase.subs(mapping), index, sp.oo) == 0
    assert denominator.subs(mapping).subs(index, 10).evalf(50) > 0


@pytest.mark.parametrize("coefficient, expected", [(2, sp.oo), (-2, sp.S.Zero)])
def test_exponential_trigonometric_poles(coefficient, expected):
    x, y = sp.symbols("x y", real=True)
    expr = sp.exp(coefficient / (1 - sp.cos(x - y)))
    assert limit(expr, (x, y), (0, 0)) == expected


def test_changing_phase_sign_is_unresolved():
    x, y = sp.symbols("x y", real=True)
    expr = sp.sin(x - y) / (1 - sp.cos(x - y))
    assert (
        trigonometric_pole_certificate(expr, (x, y), (0, 0), sp.S.true, sp.S.true)
        is None
    )
    # Opposite attained coordinate sequences have opposing infinite limits.
    j = sp.symbols("j", positive=True, integer=True)
    assert sp.limit(expr.subs({x: 1 / j, y: 0}), j, sp.oo) == sp.oo
    assert sp.limit(expr.subs({x: -1 / j, y: 0}), j, sp.oo) == -sp.oo
    assert (1 - sp.cos(1 / j)).subs(j, 10).is_positive is True


def test_logarithmic_cancellation():
    x, y = sp.symbols("x y", real=True)
    expr = (-2 * x**2 + y**2 + 2 * y + sp.log((1 + x**2 - y) ** 2)) / (
        sp.sinh(y**2) + sp.atan(x**2)
    )
    result = limit(expr, (x, y), (0, 0), return_result=True)
    assert result.status is LimitStatus.PROVED
    assert result.value == 0
    assert result.evidence[0].method == "weighted_monomial_bound"
    prefix = 2 * x**2 * y - x**4
    remainder = 2 * (sp.log(1 + x**2 - y) - (x**2 - y) + (x**2 - y) ** 2 / 2)
    for a, b in [
        (sp.Rational(1, 10), sp.Rational(1, 100)),
        (sp.Rational(-1, 20), sp.Rational(-1, 100)),
        (sp.S.Zero, sp.Rational(1, 50)),
    ]:
        mapping = {x: a, y: b}
        assert (sp.fraction(expr)[0] - prefix - remainder).subs(mapping).simplify() == 0
        inner = (x**2 - y).subs(mapping)
        assert abs(remainder.subs(mapping).evalf(50)) <= 4 * abs(inner) ** 3


def test_logarithm_cut_and_nonvanishing_remainder():
    x, y = sp.symbols("x y", real=True)
    expressions = [
        (sp.log(1 + x) - x) / (x**2 + y**2),
        (sp.log((x - 1) ** 2) - 2 * sp.log(x - 1)) / (x**2 + y**2),
    ]
    for expr in expressions:
        assert (
            weighted_quotient_certificate(expr, (x, y), (0, 0), sp.S.true, sp.S.true)
            is None
        )


def test_complex_cosine_phase_is_unresolved():
    x, y = sp.symbols("x y", real=True)
    expr = sp.exp(1 / (1 - sp.cos(sp.I * x)))
    assert (
        trigonometric_pole_certificate(expr, (x, y), (0, 0), sp.S.true, sp.S.true)
        is None
    )


@pytest.mark.parametrize("function", [sp.besselj, sp.besseli])
def test_varying_bessel_order(function):
    from asymptotic._limit_composition import _regular_composition

    x, y = sp.symbols("x y", real=True)
    expression = y * function(y, x) / (1 + y**2)
    assert _regular_composition(expression, (x, y), (0, 0)) is None
    assert (
        weighted_quotient_certificate(expression, (x, y), (0, 0), sp.S.true, sp.S.true)
        is None
    )
    assert _regular_composition(function(0, x), (x,), (0,)) == 1
    assert _regular_composition(y * function(y, 1), (y,), (0,)) == 0
