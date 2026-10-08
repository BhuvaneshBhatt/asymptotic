"""Public mathematical invariants, independent of certificate-provider names."""

import pytest
import sympy as sp

from asymptotic import analytic_limit, complex_ray_limit, limit, one_sided_limit
from asymptotic.function_normalization import RealRoot
from asymptotic.limit_models import LimitStatus


@pytest.mark.parametrize("scale", [sp.Rational(1, 3), 2, 5])
@pytest.mark.parametrize("shift", [-2, 0, sp.Rational(3, 2)])
def test_affine_local_chart(scale, shift):
    x = sp.Symbol("x", real=True)
    argument = scale * (x - shift)
    result = limit(sp.sin(argument) / argument, x, shift, return_result=True)
    assert result.status is LimitStatus.PROVED and result.value == 1
    assert result.variables == (x,) and result.target == (shift,)


@pytest.mark.parametrize("order", [2, 3, sp.Rational(5, 2)])
def test_positive_harmonic_tail(order):
    x = sp.Symbol("x", positive=True)
    assert limit(sp.harmonic(3 * x + 2, order), x, sp.oo) == sp.zeta(order)
    assert (
        limit(
            (3 * x + 3) ** (-order) + (3 * x + 3) ** (1 - order) / (order - 1), x, sp.oo
        )
        == 0
    )


@pytest.mark.parametrize("direction", [sp.I, 2 * sp.I, sp.Rational(1, 3) * sp.I])
def test_positive_ray_rescaling(direction):
    z = sp.Symbol("z")
    assert complex_ray_limit(sp.log(z), z, -1, ray=direction) == sp.I * sp.pi


def test_analytic_value_mode():
    x = sp.Symbol("x")
    f = sp.Function("f")
    assert analytic_limit(f(x), x, 0, analytic_functions=["f"]) == f(0)
    result = analytic_limit(f(x), x, 0, return_result=True)
    assert result.status is LimitStatus.UNKNOWN and result.value is None


@pytest.mark.parametrize("point", [-1, 0])
def test_even_real_root_domain(point):
    x = sp.Symbol("x")
    result = limit(RealRoot(x, 2), x, point, return_result=True)
    assert result.status is LimitStatus.UNKNOWN and result.value is None
    assert one_sided_limit(RealRoot(x, 2), x, 0, direction="+") == 0


@pytest.mark.parametrize("direction", [0, sp.oo, sp.nan])
def test_invalid_infinite_direction(direction):
    from asymptotic import DirectionalInfinity

    with pytest.raises(ValueError, match="finite and nonzero"):
        DirectionalInfinity(direction)


def test_imaginary_ray_domain():
    from asymptotic import DirectionalInfinity

    z = sp.Symbol("z", imaginary=True)
    assert complex_ray_limit(1 / z, z, 0, ray=sp.I) == DirectionalInfinity(-sp.I)
    result = limit(z, z, DirectionalInfinity(1 + sp.I), return_result=True)
    assert result.status is LimitStatus.UNKNOWN
    assert result.value is None


def test_nonreal_variable_excludes_real_ray():
    z = sp.Symbol("z", real=False)
    with pytest.raises(ValueError, match="nonreal variable"):
        complex_ray_limit(z, z, 0, ray=1)


def test_directional_tail_metadata():
    from asymptotic import DirectionalInfinity

    z = sp.Symbol("z")
    result = limit(1 / z, z, DirectionalInfinity(sp.I), return_result=True)
    assert result.status is LimitStatus.PROVED
    assert result.value == 0
    assert result.variables == result.reduced_variables == (z,)
    assert result.target == (DirectionalInfinity(sp.I),)
