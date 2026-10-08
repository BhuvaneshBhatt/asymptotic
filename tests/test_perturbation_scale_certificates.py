import sympy as sp

from asymptotic.limits import LimitStatus, limit
from asymptotic.multivariate_limits_advanced import (
    AdvancedLimitStatus,
    weighted_fractional_angular_vanishing_limit,
)


def test_weighted_fractional_angular_residual_0199():
    x, y = sp.symbols("x y", real=True)
    result = limit(x * y / sp.sqrt(x**4 + y**2), (x, y), (0, 0), return_result=True)
    assert result.status is LimitStatus.PROVED
    assert result.value == 0
    certificate = weighted_fractional_angular_vanishing_limit(
        x * y / sp.sqrt(x**4 + y**2), (x, y), (0, 0), domain=sp.S.true
    )
    assert certificate.status is AdvancedLimitStatus.CERTIFIED
    assert certificate.value == 0


def test_weighted_fractional_angular_residual_0554():
    x, y = sp.symbols("x y", real=True)
    expr = x**3 / (sp.sqrt(x**2 + y**2) * (sp.Abs(x) + sp.Abs(y)))
    result = limit(expr, (x, y), (0, 0), return_result=True)
    assert result.status is LimitStatus.PROVED
    assert result.value == 0
    certificate = weighted_fractional_angular_vanishing_limit(
        expr, (x, y), (0, 0), domain=sp.S.true
    )
    assert certificate.status is AdvancedLimitStatus.CERTIFIED
    assert certificate.value == 0
