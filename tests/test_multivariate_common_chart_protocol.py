import sympy as sp

from asymptotic.blowup_geometry import (
    CoordinateChart,
    chart_pullback_domain,
    chart_transform,
)
from asymptotic.multivariate_limits_advanced import projective_limit_chart


def test_infinite_end_uses_common_coordinate_chart():
    x = sp.symbols("x", real=True)
    chart = projective_limit_chart(1 / x, (x,), (sp.oo,), domain=x > 0)
    assert isinstance(chart, CoordinateChart)
    assert chart_transform(chart, 1 / x) == chart.expression
    pulled = chart_pullback_domain(chart, x > 0)
    assert pulled.has(chart.variables[0])
    assert chart.domain.has(chart.variables[0])
