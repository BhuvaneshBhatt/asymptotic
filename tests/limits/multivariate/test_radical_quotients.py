"""Principal-root polynomial identities preserve complex root branches."""

import sympy as sp

from asymptotic import limit
from asymptotic.limit_models import LimitStatus
from asymptotic.multivariate_pole_bounds import radical_quotient_certificate


def test_radical_quotient():
    x, y = sp.symbols("x y", real=True)
    expr = (2 * sp.sqrt(x) + x - 2 * sp.sqrt(y) - y) / (sp.sqrt(x) - sp.sqrt(y))
    result = limit(expr, (x, y), (0, 0), return_result=True)
    assert result.status is LimitStatus.PROVED
    assert result.value == 2
    assert result.evidence[0].method == "radical_polynomial_quotient"
    j = sp.Symbol("j", positive=True, integer=True)
    for sign in (1, -1):
        path = {x: sign / j**2, y: 4 * sign / j**2}
        assert sp.limit(expr.subs(path, simultaneous=True), j, sp.oo) == 2
        assert (sp.sqrt(x) - sp.sqrt(y)).subs(path, simultaneous=True).is_zero is False


def test_radical_residual_pole():
    x, y = sp.symbols("x y", real=True)
    expr = (sp.sqrt(x) + sp.sqrt(y)) / (sp.sqrt(x) - sp.sqrt(y))
    assert (
        radical_quotient_certificate(expr, (x, y), (0, 0), sp.S.true, sp.S.true) is None
    )
    assert (
        sp.limit(
            expr.subs(
                {
                    x: sp.Symbol("t", positive=True) ** 2,
                    y: 4 * sp.Symbol("t", positive=True) ** 2,
                }
            ),
            sp.Symbol("t", positive=True),
            0,
            dir="+",
        )
        == -3
    )
    assert (
        radical_quotient_certificate(
            (x - y) / (sp.sqrt(x) - sp.sqrt(y)), (x, y), (0, 0), sp.Eq(x, y), sp.S.true
        )
        is None
    )


def test_radical_exactness_budget():
    x, y = sp.symbols("x y", real=True)
    floated = (sp.Float(0.1) * x - sp.Rational(1, 10) * y) / (sp.sqrt(x) - sp.sqrt(y))
    oversized = (sp.sqrt(x) + sp.sqrt(y)) ** 1000000 / (sp.sqrt(x) - sp.sqrt(y))
    for expr in (floated, oversized):
        assert (
            radical_quotient_certificate(expr, (x, y), (0, 0), sp.S.true, sp.S.true)
            is None
        )
