"""Attained removable trigonometric quotients and uniform inverse poles."""

import pytest
import sympy as sp

from asymptotic import limit
from asymptotic.inverse_pole_composition import inverse_pole_certificate
from asymptotic.limit_models import LimitStatus
from asymptotic.reference_normalization import multivariate_reference_namespace
from asymptotic.trigonometric_removable_germs import trigonometric_removable_certificate

x, y = sp.symbols("x y", real=True)
h = x**3 + y**3


@pytest.mark.parametrize(
    "expr,expected",
    [
        (sp.Abs((sp.sin(x) - sp.sin(y)) / (sp.tan(x) - sp.tan(y)) - 1), 0),
        (3 * x * x * (1 - h * sp.cot(h)) * sp.csc(h), 0),
        ((sp.sin(h) - h) / h**3, -sp.Rational(1, 6)),
        ((1 - sp.cos(h)) / h**2, sp.Rational(1, 2)),
    ],
)
def test_removable_quotient(expr, expected):
    certificate = trigonometric_removable_certificate(
        expr, (x, y), (sp.S.Zero, sp.S.Zero), sp.S.true, sp.S.true
    )
    assert certificate[0] is LimitStatus.PROVED
    assert certificate[1] == expected
    witness = dict(certificate[2].substitutions)
    j = next(iter(witness[x].free_symbols))
    assert sp.limit(expr.subs(witness, simultaneous=True), j, sp.oo) == expected


@pytest.mark.parametrize(
    "inner,expected",
    [
        (1 / (x * x + y * y), sp.pi / 2),
        (-1 / (x * x + y * y), -sp.pi / 2),
        (y * y + x**-2, sp.pi / 2),
    ],
)
def test_inverse_pole(inner, expected):
    certificate = inverse_pole_certificate(
        sp.atan(inner), (x, y), (sp.S.Zero, sp.S.Zero), sp.S.true, sp.S.true
    )
    assert certificate[0] is LimitStatus.PROVED
    assert certificate[1] == expected
    witness = dict(certificate[2].substitutions)
    j = next(iter(witness[x].free_symbols))
    assert (
        sp.limit(sp.atan(inner).subs(witness, simultaneous=True), j, sp.oo) == expected
    )


def test_inverse_declines_mixed_sign():
    assert (
        inverse_pole_certificate(
            sp.atan(1 / (x * x - y * y)),
            (x, y),
            (sp.S.Zero, sp.S.Zero),
            sp.S.true,
            sp.S.true,
        )
        is None
    )


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Re(x+I*y)", x),
        ("Conjugate(x+I*y)", x - sp.I * y),
        ("Log10(x*x+y*y)", sp.log(x * x + y * y) / sp.log(10)),
        ("FresnelC(x)", sp.fresnelc(x)),
    ],
)
def test_exact_source_names(text, expected):
    namespace = multivariate_reference_namespace()
    namespace.update(x=x, y=y)
    assert sp.sympify(text, locals=namespace) == expected


def test_real_root_definition():
    namespace = multivariate_reference_namespace()
    namespace.update(x=x, y=y)
    expr = sp.sympify("RealRoot(x*y,3)", locals=namespace)
    assert expr.subs({x: -8, y: 1}) == -2
    result = limit(expr, (x, y), (0, 0), return_result=True)
    assert result.status is LimitStatus.PROVED
    assert result.value == 0


def test_unknown_function_keeps_definition_gap():
    namespace = multivariate_reference_namespace()
    namespace.update(x=x, y=y)
    expr = sp.sympify("f2(x,y)", locals=namespace)
    assert expr.func == sp.Function("f2")
