import sympy as sp

from asymptotic.limits import LimitStatus, limit


def test_reference_0615_uniform_majorant_parameter_stratification():
    x, y = sp.symbols("x y", real=True)
    expr = (x * y**2 + (1 - sp.cos(x)) ** 2) / sp.sqrt(x**2 + y**2)
    result = limit(expr, (x, y), (0, 0), return_result=True)
    assert result.status is LimitStatus.PROVED
    assert result.value == 0
    assert result.evidence[0].method == "uniform_vanishing_amplitude"
