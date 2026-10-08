"""Displaced rational poles retain both parameter cells and attained rays."""

import pytest
import sympy as sp

from asymptotic import limit
from asymptotic.limit_models import LimitStatus
from asymptotic.parameter_conditions import displaced_pole_stratification


@pytest.mark.parametrize(
    "parameter_assumptions", [{}, {"real": True}, {"complex": True}]
)
def test_displaced_cells(parameter_assumptions):
    x, y = sp.symbols("x y", real=True)
    a = sp.Symbol("a", **parameter_assumptions)
    expr = x * y / ((x - a) ** 2 + y**2)
    result = limit(expr, (x, y), (0, 0), return_result=True)
    cells = {cell.condition: cell.result for cell in result.strata}
    assert cells[sp.Ne(a, 0)].status is LimitStatus.PROVED
    assert cells[sp.Ne(a, 0)].value == 0
    singular = cells[sp.Eq(a, 0)]
    assert singular.status is LimitStatus.DOES_NOT_EXIST
    assert {item.value for item in singular.evidence} == {0, sp.Rational(1, 2)}
    for evidence in singular.evidence:
        path = dict(evidence.substitutions)
        actual = expr.subs(a, 0).subs(path, simultaneous=True)
        assert sp.cancel(actual) == evidence.value
        denominator = (x * x + y * y).subs(path, simultaneous=True)
        assert denominator.is_positive is True
        for coordinate in path.values():
            if coordinate != 0:
                index = next(iter(coordinate.free_symbols))
                assert sp.limit(coordinate, index, sp.oo) == 0


@pytest.mark.parametrize("parameter", [-3, sp.I, 2 + 3 * sp.I])
def test_displaced_regular_values(parameter):
    x, y = sp.symbols("x y", real=True)
    result = limit(
        x * y / ((x - parameter) ** 2 + y**2), (x, y), (0, 0), return_result=True
    )
    assert result.status is LimitStatus.PROVED
    assert result.value == 0


def test_displaced_domain():
    x, y = sp.symbols("x y", real=True)
    a = sp.Symbol("a")
    expr = x * y / ((x - a) ** 2 + y**2)
    assert (
        displaced_pole_stratification(expr, (x, y), (0, 0), sp.Eq(y, 0), sp.S.true)
        is None
    )
    infinite = sp.Symbol("infinite", finite=False)
    assert (
        displaced_pole_stratification(
            expr.subs(a, infinite), (x, y), (0, 0), sp.S.true, sp.S.true
        )
        is None
    )
