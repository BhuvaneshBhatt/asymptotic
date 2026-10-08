"""Independent branch samples and attained sequences for recovered source germs."""

import pytest
import sympy as s

from asymptotic import complex_ray_limit, one_sided_limit
from asymptotic.limit_models import LimitStatus
from asymptotic.local_branch_germs import fractional_part_transverse_certificate

x = s.Symbol("x")


@pytest.mark.parametrize("direction,expected", [("+", -s.pi), ("-", -s.pi / 2)])
def test_principal_angle_preserves_approached_quadrant(direction, expected):
    expression = s.arg(-s.sqrt(1 / x) - s.I)
    result = one_sided_limit(expression, x, 0, direction=direction, return_result=True)
    assert result.status is LimitStatus.PROVED and result.value == expected
    point = s.Rational(1, 10000) if direction == "+" else -s.Rational(1, 10000)
    assert abs(complex(s.N(expression.subs(x, point) - expected))) < 0.02


@pytest.mark.parametrize("direction,expected", [("+", s.I * s.pi), ("-", -s.I * s.pi)])
def test_opposite_atanh_cut_limits_agree_with_attained_samples(direction, expected):
    a = s.Symbol("a")
    expression = s.atanh(a + s.I * x) - s.atanh(a - s.I * x)
    result = one_sided_limit(
        expression, x, 0, direction=direction, assumptions=a > 1, return_result=True
    )
    assert result.status is LimitStatus.PROVED and result.value == expected
    point = s.Rational(1, 10000) if direction == "+" else -s.Rational(1, 10000)
    assert abs(complex(s.N(expression.subs({a: 2, x: point}) - expected))) < 0.001


@pytest.mark.parametrize("direction,expected", [("+", -s.oo), ("-", s.oo)])
def test_algebraic_tangent_pole_has_attained_signed_sides(direction, expected):
    p = (5 * s.pi / 2 - 1) ** 2
    expression = s.tan(s.sqrt(x) + 1)
    result = one_sided_limit(expression, x, p, direction=direction, return_result=True)
    assert result.status is LimitStatus.PROVED and result.value == expected
    sample = s.N(
        expression.subs(x, p + (1 if direction == "+" else -1) * s.Rational(1, 1000))
    )
    assert (sample > 1000) if expected is s.oo else (sample < -1000)


@pytest.mark.parametrize("direction,expected", [("+", 0), ("-", s.sin(2))])
def test_fractional_part_crossing_is_one_sided(direction, expected):
    result = one_sided_limit(
        s.frac(x * x) * s.sin(x), x, 2, direction=direction, return_result=True
    )
    assert result.status is LimitStatus.PROVED and result.value == expected


def test_fractional_part_certificate_declines_polynomial_argument():
    expression = s.frac(4 + 4 * (x - 2) + s.I * (x - 2) ** 2)
    assert (
        fractional_part_transverse_certificate(
            expression, x, s.Integer(2), s.S.true, s.S.true
        )
        is None
    )


@pytest.mark.parametrize(
    "ray,expected",
    [
        (-1, s.EulerGamma),
        (1, s.EulerGamma - s.I * s.pi),
        (s.I, s.EulerGamma + s.I * s.pi),
        (-s.I, s.EulerGamma - s.I * s.pi),
    ],
)
def test_li_log_cancellation_retains_boundary_phase(ray, expected):
    expression = s.li(x) - s.log(1 - x)
    result = complex_ray_limit(expression, x, 1, ray=ray, return_result=True)
    assert result.status is LimitStatus.PROVED and result.value == expected
    sample = expression.subs(x, 1 + ray * s.Rational(1, 10000))
    assert abs(complex(s.N(sample - expected))) < 0.001


@pytest.mark.parametrize("a,expected", [(2, -s.I * s.oo), (-2, s.I * s.oo)])
def test_erfc_imaginary_root_has_attained_direction(a, expected):
    expression = s.erfc((x - a) / s.sqrt(x))
    result = one_sided_limit(expression, x, 0, direction="-", return_result=True)
    assert result.status is LimitStatus.PROVED and result.value == expected
    sample = s.N(expression.subs(x, -s.Rational(1, 100)))
    assert s.im(sample).is_negative if a > 0 else s.im(sample).is_positive


@pytest.mark.parametrize(
    "function,center",
    [(s.LambertW, -2), (s.atan, 2 * s.I), (s.atanh, 2), (s.asin, 2), (s.acos, 2)],
)
def test_composition_declines_cut(function, center):
    from asymptotic._limit_composition import _regular_composition
    from asymptotic.multivariate_pole_bounds import removable_germ_certificate

    u, v = s.symbols("u v", real=True)
    argument = center + (u if function is s.atan else s.I * u)
    cofactor = function(argument)
    assert _regular_composition(cofactor, (u, v), (0, 0)) is None
    inner = u * u + v * v
    expression = s.sin(inner) * s.tan(inner) / (1 - s.cos(inner)) * cofactor
    assert (
        removable_germ_certificate(expression, (u, v), (0, 0), s.S.true, s.S.true)
        is None
    )
    # Attained samples approach the cut from both sides and remain in the domain.
    j = s.Integer(100000)
    upper = complex(s.N(cofactor.subs(u, 1 / j), 30))
    lower = complex(s.N(cofactor.subs(u, -1 / j), 30))
    assert abs(upper - lower) > 1


@pytest.mark.parametrize(
    "function,center",
    [
        (s.LambertW, 1),
        (s.atan, 1),
        (s.atanh, s.Rational(1, 2)),
        (s.asin, 0),
        (s.acos, 0),
    ],
)
def test_composition_regular_center(function, center):
    from asymptotic._limit_composition import _regular_composition

    u, v = s.symbols("u v", real=True)
    expression = function(center + u + s.I * v)
    assert _regular_composition(expression, (u, v), (0, 0)) == function(center)


@pytest.mark.parametrize(
    "function,center",
    [(s.LambertW, -2), (s.asin, s.Rational(9, 2)), (s.acos, 2), (s.atanh, 2)],
)
def test_real_cut_continuity(function, center):
    from asymptotic._limit_composition import _regular_composition

    u, v = s.symbols("u v", real=True)
    expression = function(center + u + v)
    assert _regular_composition(expression, (u, v), (0, 0)) == function(center)
    for sign in (-1, 1):
        sample = expression.subs({u: sign * s.Rational(1, 100000), v: 0})
        assert abs(complex(s.N(sample - function(center), 30))) < 0.001


def test_scalar_real_cut_chart():
    from asymptotic import limit

    z = s.Symbol("z")
    result = limit(s.acos(z), z, 2, return_result=True)
    assert result.status is LimitStatus.PROVED
    assert result.value == s.acos(2)
    assert s.simplify(result.value.rewrite(s.log) - s.I * s.log(2 + s.sqrt(3))) == 0


def test_centered_cut_chart():
    from asymptotic._limit_composition import _regular_composition

    z = s.Symbol("z")
    assert _regular_composition(s.atan(z), (z,), (2 * s.I,)) is None
    assert _regular_composition(s.LambertW(-2 + s.I * z), (z,), (0,)) is None
