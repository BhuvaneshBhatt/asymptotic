"""Complex approach contracts and explicit regular parameter cells."""

import pytest
import sympy as sp

from asymptotic import complex_limit, complex_ray_limit, limit, one_sided_limit
from asymptotic.limit_models import LimitStatus, SimultaneousLimitDoesNotExist
from asymptotic.reference_contracts import parse_reference
from asymptotic.reference_normalization import (
    StepFactorialPower,
    scalar_reference_namespace,
)
from asymptotic.stratification import AsymptoticStratification

z, x, n, h = sp.symbols("z x n h")


@pytest.mark.parametrize(
    "expression,expected",
    [
        (sp.sqrt(z * z), sp.zoo),
        (z**3 + z, sp.zoo),
        ((z * z + 1) ** sp.Rational(1, 3), sp.zoo),
    ],
)
def test_spherical_growth(expression, expected):
    assert complex_limit(expression, z, sp.zoo) == expected
    assert limit(expression, z, sp.zoo) == expected


@pytest.mark.parametrize(
    "expression", [sp.exp(z), sp.sqrt(1 / z), sp.sqrt(z - sp.Symbol("a"))]
)
def test_spherical_decline(expression):
    assert (
        complex_limit(expression, z, sp.zoo, return_result=True).status
        is LimitStatus.UNKNOWN
    )


@pytest.mark.parametrize("ray", [1 + sp.I, 1 - sp.I, 2 + 3 * sp.I, -2 - 3 * sp.I])
def test_li_cut(ray):
    side = 1 if sp.im(ray) > 0 else -1
    assert (
        complex_ray_limit(sp.li(z), z, sp.Rational(1, 3), ray=ray)
        == sp.li(sp.Rational(1, 3)) + side * sp.I * sp.pi
    )


def test_complex_analytic_quotient():
    expression = sp.atan((z * z + 1) ** 2) / sp.sin(z * z + 1) ** 2
    assert limit(expression, z, sp.I) == 1
    assert complex_limit(expression, z, sp.I) == 1
    assert limit((z - sp.I) ** -2, z, sp.I) is sp.zoo
    assert complex_ray_limit((z - sp.I) ** -2, z, sp.I, ray=1) is sp.oo
    assert complex_ray_limit((z - sp.I) ** -2, z, sp.I, ray=sp.I) is -sp.oo


@pytest.mark.parametrize("direction", [sp.I, -sp.I, 2 * sp.I])
def test_imaginary_tail(direction):
    result = limit(sp.cosh(z), z, direction * sp.oo, return_result=True)
    assert result.status is LimitStatus.DOES_NOT_EXIST
    assert [item.value for item in result.evidence] == [1, -1]
    for item in result.evidence:
        variable, sequence = item.substitutions[0]
        assert variable == z
        assert sp.simplify(sp.cosh(sequence) - item.value) == 0
    assert limit(sp.besseli(n, z), z, direction * sp.oo) == 0
    with pytest.raises(SimultaneousLimitDoesNotExist):
        limit(sp.cosh(z), z, direction * sp.oo)


def test_variable_order_decline():
    assert (
        limit(sp.besseli(z, z), z, sp.I * sp.oo, return_result=True).status
        is LimitStatus.UNKNOWN
    )


def test_parameter_ei_tail():
    s = sp.Symbol("s")
    expression = (sp.exp(s * x) * sp.Ei((1 - s) * x) - sp.Ei(x)) * sp.exp(-s * x) / s
    assert limit(expression, x, sp.oo, assumptions=s > 1) == 0
    assert limit(expression, x, sp.oo, return_result=True).status is LimitStatus.UNKNOWN


def test_step_base_cells():
    expression = StepFactorialPower(x, -3, h)
    cells = limit(expression, h, 0, return_result=True)
    assert isinstance(cells, AsymptoticStratification)
    regular = next(c for c in cells.strata if c.condition == sp.Ne(x, 0))
    singular = next(c for c in cells.strata if c.condition == sp.Eq(x, 0))
    assert regular.result.value == x**-3
    assert singular.result.status is LimitStatus.DOES_NOT_EXIST
    assert [e.value for e in singular.result.evidence] == [sp.oo, -sp.oo]
    assert limit(expression, h, 0, assumptions=sp.Ne(x, 0)) == x**-3
    with pytest.raises(SimultaneousLimitDoesNotExist):
        limit(StepFactorialPower(0, -3, h), h, 0)
    assert limit(StepFactorialPower(0, -2, h), h, 0) is sp.oo


def test_step_order_regular():
    base = sp.Symbol("base", nonnegative=True)
    step = sp.Symbol("step", positive=True)
    assert limit(StepFactorialPower(base, n, step), n, 0) == 1
    result = limit(StepFactorialPower(x, n, h), n, 0, return_result=True)
    assert isinstance(result, AsymptoticStratification)
    assert sorted(c.result.status.value for c in result.strata) == ["proved", "unknown"]


def test_step_argument_cells():
    result = one_sided_limit(
        StepFactorialPower(x, n, h), x, 0, direction="+", return_result=True
    )
    assert isinstance(result, AsymptoticStratification)
    regular = next(c for c in result.strata if c.condition == sp.Ne(h, 0))
    assert regular.result.value == 1 / ((1 / h) ** n * sp.gamma(1 - n))
    assert regular.result.variables == (x,)
    assert regular.result.target == (0,)
    assert (
        one_sided_limit(
            StepFactorialPower(x, n, h), x, 0, direction="+", assumptions=sp.Ne(h, 0)
        )
        == regular.result.value
    )


@pytest.mark.parametrize(
    "order,cutoff,expected", [(2, 3, x * x - x), (3, 3, -3 * x * x + 2 * x), (0, 1, 1)]
)
def test_falling_taylor_specialization(order, cutoff, expected):
    expression = parse_reference(
        f"Normal(Series(StepFactorialPower(x,n,1),(x,0,{cutoff})))",
        scalar_reference_namespace(),
    )
    assert sp.expand(limit(expression, n, order) - expected) == 0


def test_whole_plane_parameter_denominator():
    a = sp.Symbol("a")
    assert (
        complex_limit(1 / (z - sp.I + a), z, sp.I, return_result=True).status
        is LimitStatus.UNKNOWN
    )


def test_step_positive_chart():
    cells = one_sided_limit(
        StepFactorialPower(x, -3, h), h, 0, direction="+", return_result=True
    )
    singular = next(c for c in cells.strata if c.condition == sp.Eq(x, 0))
    assert singular.result.status is LimitStatus.PROVED
    assert singular.result.value is sp.oo


def test_step_discrete_coordinate():
    integer_step = sp.Symbol("integer_step", integer=True)
    result = limit(
        StepFactorialPower(x, -3, integer_step), integer_step, 0, return_result=True
    )
    assert result.status is LimitStatus.UNKNOWN


def test_nonfinite_rate_decline():
    from asymptotic.parameter_special_germs import special_parameter_limit

    s = sp.Symbol("s", finite=False)
    expression = (sp.exp(s * x) * sp.Ei((1 - s) * x) - sp.Ei(x)) * sp.exp(-s * x) / s
    assert special_parameter_limit(expression, x, sp.oo, sp.S.true, s > 1) is None


@pytest.mark.parametrize("base,step", [(sp.I, 1), (x, sp.I)])
def test_nonreal_order_cell_decline(base, step):
    result = limit(StepFactorialPower(base, n, step), n, 0, return_result=True)
    assert result.status is LimitStatus.UNKNOWN


def test_nonreal_ei_rate_decline():
    from asymptotic.parameter_special_germs import special_parameter_limit

    s = sp.Symbol("s", imaginary=True)
    expression = (sp.exp(s * x) * sp.Ei((1 - s) * x) - sp.Ei(x)) * sp.exp(-s * x) / s
    assert special_parameter_limit(expression, x, sp.oo, sp.S.true, sp.S.true) is None
