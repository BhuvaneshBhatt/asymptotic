import pytest
import sympy as s

from asymptotic import limit
from asymptotic.limit_models import LimitStatus
from asymptotic.variable_order_germs import (
    regular_variable_bessel_certificate as certificate,
)


@pytest.mark.parametrize("f", [s.besselj, s.bessely, s.besseli, s.besselk])
def test_joint_order_argument_continuity_including_integer_order(f):
    x = s.Symbol("x", real=True)
    expr = f(x / (1 + x), 2 + x * x)
    result = limit(expr, x, 0, return_result=True)
    assert result.status is LimitStatus.PROVED and result.value == f(0, 2)
    assert result.evidence[0].method == "regular_variable_order_bessel_continuity"
    # Independent off-center evaluation exercises removable Y/K order strata.
    assert (
        abs(
            complex(expr.subs(x, s.Rational(1, 10**7)).evalf(25))
            - complex(f(0, 2).evalf(25))
        )
        < 1e-6
    )


def test_polynomial_composition_and_rational_tail():
    x = s.Symbol("x", positive=True)
    expr = (1 + 1 / x) * s.besselj(1 + 1 / x, 2 + 1 / x) ** 2 + s.besselk(1 / x, 3)
    r = certificate(expr, x, s.oo, s.true, s.true)
    assert r[0] is LimitStatus.PROVED and r[1] == s.besselj(1, 2) ** 2 + s.besselk(0, 3)


def test_singular_and_large_order_regimes_decline():
    x = s.Symbol("x", positive=True)
    for expr, point in [
        (s.bessely(x, x), 0),
        (s.besselk(x, -2), 0),
        ((s.besselj(x, 2) - s.besselj(0, 2)) / x, 0),
        (1 / s.besselj(x, 2), 0),
        (s.besselj(x, x), s.oo),
    ]:
        assert certificate(expr, x, point, s.true, s.true) is None


def test_complex_order_and_no_parameter_stratification_invention():
    x = s.Symbol("x", real=True)
    a = s.Symbol("a")
    r = certificate(s.besselj(s.I + x, 2 + x), x, 0, s.true, s.true)
    assert r[1] == s.besselj(s.I, 2)
    from asymptotic import complex_ray_limit

    z = s.Symbol("z")
    assert complex_ray_limit(s.besselj(s.I + z, 2 + z), z, 0, ray=s.I) == s.besselj(
        s.I, 2
    )
    assert (
        complex_ray_limit(s.bessely(z, -2), z, 0, ray=s.I, return_result=True).status
        is LimitStatus.UNKNOWN
    )
    assert certificate(s.besselj(a + x, 2), x, 0, s.true, s.true) is None


@pytest.mark.parametrize(
    "f,expected",
    [(s.besselj, 0), (s.besseli, 0), (s.bessely, -s.oo), (s.besselk, s.oo)],
)
def test_positive_large_order_at_fixed_argument(f, expected):
    x = s.Symbol("x", positive=True)
    assert limit(f(x + 1 / x, 2), x, s.oo) == expected
    assert limit(f(1 / x, 2), x, 0) == expected
    assert certificate(f(x, 2 + 1 / x), x, s.oo, s.true, s.true) is None
    assert certificate(f(-x, 2), x, s.oo, s.true, s.true) is None
    y = s.Symbol("y", real=True)
    assert certificate(f(1 / y, 2), y, 0, s.true, s.true) is None
