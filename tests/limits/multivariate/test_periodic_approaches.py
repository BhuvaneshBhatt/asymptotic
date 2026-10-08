"""Exact periodic witnesses, including accumulation and pole avoidance."""

import pytest
import sympy as sp

from asymptotic import SquareWave, limit
from asymptotic.limit_models import LimitStatus, SimultaneousLimitDoesNotExist
from asymptotic.periodic_approaches import (
    hyperbolic_pole_conflict,
    square_wave_infinite_conflict,
)

x, y, z = sp.symbols("x y z", real=True)


@pytest.mark.parametrize(
    "a,b,power", [(2, 1, 2), (-3, 0, 1), (sp.Rational(1, 2), -2, 4)]
)
def test_hyperbolic_periods(a, b, power):
    expr = sp.sech(b + a / (x - 1 + sp.I * (y + 2))) ** power
    result = limit(expr, (x, y), (1, -2), return_result=True)
    assert result.status is LimitStatus.DOES_NOT_EXIST
    assert tuple(e.value for e in result.evidence) == (0, sp.sech(b) ** power)
    for evidence in result.evidence:
        sequence = dict(evidence.substitutions)
        j = next(iter(sequence[x].free_symbols | sequence[y].free_symbols))
        assert sp.limit(sequence[x], j, sp.oo) == 1
        assert sp.limit(sequence[y], j, sp.oo) == -2
        phase = (b + a / (x - 1 + sp.I * (y + 2))).subs(sequence, simultaneous=True)
        for index in (1, 2, 5):
            point = {v: value.subs(j, index) for v, value in sequence.items()}
            assert sp.simplify((x - 1 + sp.I * (y + 2)).subs(point)) != 0
            assert sp.cosh(phase.subs(j, index)).is_zero is False
        if evidence.value == 0:
            assert sp.limit(sp.sech(phase) ** power, j, sp.oo) == 0
        else:
            assert sp.simplify(sp.sech(phase) ** power - evidence.value) == 0
    with pytest.raises(SimultaneousLimitDoesNotExist):
        limit(expr, (x, y), (1, -2))


@pytest.mark.parametrize(
    "variables,target,phase",
    [
        ((x, y), (sp.oo, sp.oo), x + y),
        ((x, y), (sp.oo, -sp.oo), x + y + sp.Rational(1, 3)),
        ((x, y, z), (-sp.oo, sp.oo, sp.oo), x + 2 * y - z),
    ],
)
def test_wave_half_periods(variables, target, phase):
    expr = 3 + 2 * SquareWave(phase)
    result = limit(expr, variables, target, return_result=True)
    assert result.status is LimitStatus.DOES_NOT_EXIST
    assert tuple(e.value for e in result.evidence) == (5, 1)
    for evidence in result.evidence:
        sequence = dict(evidence.substitutions)
        j = next(
            iter(set().union(*(value.free_symbols for value in sequence.values())))
        )
        assert tuple(sp.limit(sequence[v], j, sp.oo) for v in variables) == target
        attained_phase = sp.expand(phase.subs(sequence, simultaneous=True))
        for index in (1, 2, 7):
            value = attained_phase.subs(j, index)
            assert 2 * value != sp.floor(2 * value)
            assert (
                expr.subs({v: value.subs(j, index) for v, value in sequence.items()})
                == evidence.value
            )
    with pytest.raises(SimultaneousLimitDoesNotExist):
        limit(expr, variables, target)


def test_periodic_scope():
    positive = sp.Symbol("positive", positive=True)
    assert (
        square_wave_infinite_conflict(
            SquareWave(positive + y),
            (positive, y),
            (-sp.oo, sp.oo),
            sp.S.true,
            sp.S.true,
        )
        is None
    )
    assert (
        square_wave_infinite_conflict(
            SquareWave(x * y), (x, y), (sp.oo, sp.oo), sp.S.true, sp.S.true
        )
        is None
    )
    assert (
        square_wave_infinite_conflict(
            SquareWave(x + y), (x, y), (sp.oo, sp.oo), x > 0, sp.S.true
        )
        is None
    )
    assert (
        hyperbolic_pole_conflict(
            sp.sech(1 / (x + sp.I * y)),
            (x, y),
            (sp.S.Zero, sp.S.Zero),
            y > 0,
            sp.S.true,
        )
        is None
    )
