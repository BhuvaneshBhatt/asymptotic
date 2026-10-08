import pytest
import sympy as sp

from asymptotic import limit
from asymptotic.attained_ray_germs import logarithmic_ray_value
from asymptotic.limit_models import LimitStatus
from asymptotic.uniform_radial_bounds import (
    radial_limit_certificate,
    radial_vanishing_certificate,
)


def test_finite_radial_center():
    x, y = sp.symbols("x y", real=True)
    expr = (x**2 + y**2) * sp.log(x**2 + y**2) + sp.cos(x)
    assert limit(expr, (x, y), (0, 0)) == 1
    assert (
        radial_vanishing_certificate(
            expr, (x, y), (sp.S.Zero,) * 2, sp.S.true, sp.S.true
        )
        is None
    )


def test_bounded_pole_phases():
    x, y = sp.symbols("x y", real=True)
    assert limit((x**3 + y**3) / (2 + sp.cos(y + 1 / x)), (x, y), (0, 0)) == 0
    assert (
        limit(x**2 * sp.Abs(sp.cos(1 / (x**2 + y**2))) + x + y**2 + 3, (x, y), (0, 0))
        == 3
    )
    assert limit(x * y * sp.cos(x**-2) + (y + 1) * sp.sin(x) / x, (x, y), (0, -1)) == 0


def test_flat_hypersurface():
    x, y = sp.symbols("x y", real=True)
    assert limit(sp.exp(-1 / (x**2 * (y - 1) ** 2)), (x, y), (0, 1)) == 0
    assert limit(sp.exp(-sp.Abs(x - y) / (x - y) ** 2), (x, y), (0, 0)) == 0
    expr = sp.exp(-1 / (x**2 - y**2))
    assert (
        radial_limit_certificate(expr, (x, y), (sp.S.Zero,) * 2, sp.S.true, sp.S.true)
        is None
    )


def test_logarithm_dominates_bounded_terms():
    x, y = sp.symbols("x y", real=True)
    expr = (sp.sin(3 / y) + 3 * sp.cos(x - y)) / (
        sp.log(x**2 + y**2) + 2 * sp.sin(1 / x)
    )
    assert limit(expr, (x, y), (0, 0)) == 0


def test_real_inverse_sine_bound():
    x, y = sp.symbols("x y", real=True)
    expr = x * y * sp.asin((x**2 - y**2) / (x**2 + y**2)) / sp.sqrt(x**2 + y**2)
    assert limit(expr, (x, y), (0, 0)) == 0
    # Without an interval bound the inverse sine can grow on real arguments.
    expr = x * sp.asin(sp.exp(1 / x**2))
    assert (
        radial_limit_certificate(expr, (x, y), (sp.S.Zero,) * 2, sp.S.true, sp.S.true)
        is None
    )


def test_vanishing_oscillation_witnesses():
    x, y = sp.symbols("x y", real=True)
    expr = x * y / (x**2 + y**2) + x * sp.sin(1 / y)
    result = limit(expr, (x, y), (0, 0), return_result=True)
    assert result.status is LimitStatus.DOES_NOT_EXIST
    assert {e.value for e in result.evidence} == {0, sp.Rational(1, 2)}
    for evidence in result.evidence:
        mapping = dict(evidence.substitutions)
        assert mapping[y] != 0
        j = next(iter(mapping[x].free_symbols | mapping[y].free_symbols))
        assert sp.limit(mapping[x], j, sp.oo) == 0
        assert sp.limit(mapping[y], j, sp.oo) == 0
        assert (mapping[x] ** 2 + mapping[y] ** 2).is_positive


def test_logarithmic_ray_orders():
    t = sp.symbols("t", positive=True)
    assert logarithmic_ray_value(t * sp.log(2 * t**2) ** 4, t) == 0
    assert logarithmic_ray_value(sp.log(t) ** 3, t) == -sp.oo
    assert logarithmic_ray_value(sp.log(t) ** 2 / sp.log(2 * t), t) == -sp.oo
    assert logarithmic_ray_value(sp.log(-t), t) is None
    assert logarithmic_ray_value(sp.log(1 + t), t) is None


def test_endpoint_pole_witnesses():
    x, y = sp.symbols("x y", real=True)
    result = limit(
        sp.asin(x * y) / (1 - x * y),
        (x, y),
        (-sp.Rational(1, 2), -2),
        return_result=True,
    )
    assert result.status is LimitStatus.DOES_NOT_EXIST
    assert {e.value for e in result.evidence} == {sp.oo, -sp.oo}


@pytest.mark.parametrize("power", [sp.S.Half, sp.S.One])
def test_positive_denominator_products(power):
    x, y = sp.symbols("x y", real=True)
    numerator = x**2 * y ** (2 * power)
    expr = numerator / (sp.sqrt(x**2 + y**2) * (x**4 + y**2) ** power)
    result = limit(expr, (x, y), (0, 0), return_result=True)
    assert result.status is LimitStatus.PROVED
    assert result.value == 0
    assert any(e.method == "positive_product_bound" for e in result.evidence)


def test_positive_product_rejects_nonvanishing_and_signed_factors():
    from asymptotic.multivariate_pole_bounds import positive_sum_vanishing_certificate

    x, y = sp.symbols("x y", real=True)
    for expr in [
        x * y / (sp.sqrt(x**2 + y**2) * sp.sqrt(x**4 + y**2)),
        x**2 * y**2 / (sp.sqrt(x**2 + y**2) * (x**4 - y**2)),
    ]:
        assert (
            positive_sum_vanishing_certificate(
                expr, (x, y), (0, 0), sp.S.true, sp.S.true
            )
            is None
        )
