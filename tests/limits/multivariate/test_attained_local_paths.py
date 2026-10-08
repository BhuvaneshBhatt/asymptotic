"""Exact subsequences retain domain and pole-avoidance proof obligations."""

import pytest
import sympy as sp

from asymptotic import limit
from asymptotic.limit_models import LimitStatus
from asymptotic.local_path_witnesses import local_path_conflict


@pytest.mark.parametrize("case", [187, 400, 1279])
def test_local_conflict(case):
    x, y, z = sp.symbols("x y z", real=True)
    if case == 187:
        expr = (sp.exp(x**2) - 1) * sp.tanh(y) / sp.sin(x**4 + y**2)
        variables = (x, y)
        expected = {sp.S.Zero, sp.Rational(1, 2)}
    elif case == 400:
        expr = -sp.sin(y - sp.atan(y**2 / (x + y))) / sp.sqrt(x**2 + y**2)
        variables = (x, y)
        expected = {sp.S.Zero, -sp.sqrt(2) / 4}
    else:
        expr = sp.tan(x + y + z) / (
            (sp.Abs(x) + sp.Abs(y) + sp.Abs(z)) * sp.sin(x**2 + y**2 + z**2)
        )
        variables = (x, y, z)
        expected = {sp.oo, -sp.oo}
    result = limit(expr, variables, (0,) * len(variables), return_result=True)
    assert result.status is LimitStatus.DOES_NOT_EXIST
    assert {e.value for e in result.evidence} == expected
    certificate = local_path_conflict(
        expr, variables, (sp.S.Zero,) * len(variables), sp.S.true, sp.S.true
    )
    assert {e.value for e in certificate} == expected
    for e in certificate:
        assert e.method == "attained_local_path"
        substitutions = dict(e.substitutions)
        index = next(
            iter(set().union(*(v.free_symbols for v in substitutions.values())))
        )
        assert all(sp.limit(v, index, sp.oo) == 0 for v in substitutions.values())
        assert (
            sp.limit(expr.subs(substitutions, simultaneous=True), index, sp.oo)
            == e.value
        )
        for base in [p.base for p in expr.atoms(sp.Pow) if p.exp.is_negative is True]:
            along = base.subs(substitutions, simultaneous=True)
            for n in (2, 3, 8, 20):
                assert along.subs(index, n).is_zero is False


def test_local_declines():
    x, y = sp.symbols("x y", real=True)
    expr = sp.sin(x) / (x * x + y * y)
    assert local_path_conflict(expr, (x, y), (0, 0), sp.Eq(y, 0), sp.S.true) is None
    positive = sp.Symbol("p", positive=True)
    assert (
        local_path_conflict(
            expr.subs(y, positive), (x, positive), (0, 0), sp.S.true, sp.S.true
        )
        is None
    )
    # Agreement cannot prove existence, and accumulating poles require another theorem.
    assert (
        local_path_conflict(
            sp.sin(x * x + y * y) / (x * x + y * y),
            (x, y),
            (0, 0),
            sp.S.true,
            sp.S.true,
        )
        is None
    )
    assert (
        local_path_conflict(
            sp.sin(1 / (x * x + y * y)), (x, y), (0, 0), sp.S.true, sp.S.true
        )
        is None
    )


def test_reciprocal_phase_sequences():
    x, y = sp.symbols("x y", real=True)
    radius = sp.sqrt(x * x + y * y)
    expr = (-x - y + radius * sp.sin(1 / radius)) / radius
    result = limit(expr, (x, y), (0, 0), return_result=True)
    assert result.status is LimitStatus.DOES_NOT_EXIST
    assert {e.value for e in result.evidence} == {0, -2}
    for e in result.evidence:
        assert e.method == "attained_reciprocal_phase"
        substitutions = dict(e.substitutions)
        index = next(
            iter(set().union(*(v.free_symbols for v in substitutions.values())))
        )
        assert radius.subs(substitutions, simultaneous=True).is_positive is True
        assert sp.simplify(expr.subs(substitutions, simultaneous=True)) == e.value
        assert all(sp.limit(v, index, sp.oo) == 0 for v in substitutions.values())


def test_reciprocal_phase_poles():
    from asymptotic.local_path_witnesses import reciprocal_phase_conflict

    x, y = sp.symbols("x y", real=True)
    phase = 1 / (x * x + y * y)
    assert (
        reciprocal_phase_conflict(
            1 / sp.sin(phase), (x, y), (0, 0), sp.S.true, sp.S.true
        )
        is None
    )
    assert (
        reciprocal_phase_conflict(
            x * sp.sin(phase), (x, y), (0, 0), sp.S.true, sp.S.true
        )
        is None
    )


def test_scalar_rules_decline_vectors():
    from asymptotic.local_path_witnesses import (
        local_path_conflict,
        reciprocal_phase_conflict,
    )
    from asymptotic.multivariate_pole_bounds import (
        positive_sum_vanishing_certificate,
        radical_quotient_certificate,
        removable_germ_certificate,
    )

    x, y = sp.symbols("x y", real=True)
    expr = sp.Tuple(sp.sin(x) / x, sp.exp(y) / y)
    for certificate in (
        local_path_conflict,
        reciprocal_phase_conflict,
        positive_sum_vanishing_certificate,
        radical_quotient_certificate,
        removable_germ_certificate,
    ):
        assert certificate(expr, (x, y), (0, 0), sp.S.true, sp.S.true) is None


def test_parameter_monomial_paths():
    x, y, a = sp.symbols("x y a", real=True)
    expr = (a * y**2 + x**4) / (x * y**2)
    result = limit(expr, (x, y), (0, 0), return_result=True)
    assert result.status is LimitStatus.DOES_NOT_EXIST
    assert {e.value for e in result.evidence} == {-sp.oo, sp.oo}
    for evidence in result.evidence:
        assert evidence.method == "attained_monomial_parameter_path"
        sequence = dict(evidence.substitutions)
        index = next(iter(sequence[x].free_symbols))
        assert sp.limit(sequence[x], index, sp.oo) == 0
        assert sp.limit(sequence[y], index, sp.oo) == 0
        assert (x * y**2).subs(sequence, simultaneous=True).is_zero is False
        assert (
            sp.limit(expr.subs(sequence, simultaneous=True), index, sp.oo)
            == evidence.value
        )


def test_parameter_paths_decline():
    from asymptotic.local_path_witnesses import parameter_monomial_conflict

    x, y, a = sp.symbols("x y a", real=True)
    for expr in (
        a * x / y,
        a * x / y**2,
        a * x / (x * x + y * y),
        a * (x + y) ** 1000000 / (x * y),
    ):
        assert (
            parameter_monomial_conflict(expr, (x, y), (0, 0), sp.S.true, sp.S.true)
            is None
        )
    assert (
        parameter_monomial_conflict(
            (a * y**2 + x**4) / (x * y**2), (x, y), (0, 0), y > 0, sp.S.true
        )
        is None
    )


def test_parameter_paths_keep_holes():
    from asymptotic.local_path_witnesses import parameter_monomial_conflict

    x, y, a = sp.symbols("x y a", real=True)
    expr = sp.Mul(
        a * y**2 + x**4,
        x + y,
        sp.Pow(x * y**2 * (x + y), -1, evaluate=False),
        evaluate=False,
    )
    assert (
        parameter_monomial_conflict(expr, (x, y), (0, 0), sp.S.true, sp.S.true) is None
    )


def test_local_paths_keep_holes():
    from asymptotic.local_path_witnesses import local_path_conflict

    x, y = sp.symbols("x y", real=True)
    expr = sp.Mul(sp.sin(x), y, sp.Pow(x * y, -1, evaluate=False), evaluate=False)
    assert local_path_conflict(expr, (x, y), (0, 0), sp.S.true, sp.S.true) is None


@pytest.mark.parametrize("function", [sp.sin, sp.tan, sp.sinh, sp.atan])
def test_linear_pole_curves(function):
    from asymptotic.local_path_witnesses import polynomial_pole_conflict

    x, y, z = sp.symbols("x y z", real=True)
    expression = function(x**3 + y**3 + z**3) / (x + y + z)
    evidence = polynomial_pole_conflict(
        expression, (x, y, z), (0, 0, 0), sp.S.true, sp.S.true
    )
    assert {e.value for e in evidence} == {sp.S.Zero, -sp.oo}
    for item in evidence:
        chart = dict(item.substitutions)
        index = next(iter(chart[x].free_symbols))
        assert all(sp.limit(chart[v], index, sp.oo) == 0 for v in (x, y, z))
        assert (x + y + z).subs(chart, simultaneous=True).is_positive is True
        phase = (x**3 + y**3 + z**3).subs(chart, simultaneous=True)
        divisor = (x + y + z).subs(chart, simultaneous=True)
        assert sp.limit(phase, index, sp.oo) == 0
        assert sp.limit(phase / divisor, index, sp.oo) == item.value
        argument = sp.Symbol("argument", real=True)
        assert sp.diff(function(argument), argument).subs(argument, 0) == 1
        if function is sp.tan:
            assert sp.limit(sp.cos(phase), index, sp.oo) == 1
    result = limit(expression, (x, y, z), (0, 0, 0), return_result=True)
    assert result.status is LimitStatus.DOES_NOT_EXIST
    assert {e.value for e in result.evidence} == {sp.S.Zero, -sp.oo}


def test_exponential_pole_curve():
    from asymptotic.local_path_witnesses import polynomial_pole_conflict

    x, y = sp.symbols("x y", real=True)
    expression = sp.exp(x * y) + sp.exp(x**2 / (x + y))
    evidence = polynomial_pole_conflict(
        expression, (x, y), (0, 0), sp.S.true, sp.S.true
    )
    assert {e.value for e in evidence} == {sp.Integer(2), sp.oo}
    for item in evidence:
        chart = dict(item.substitutions)
        index = next(iter(chart[x].free_symbols))
        assert (x + y).subs(chart, simultaneous=True).is_positive is True
        assert (
            sp.limit(expression.subs(chart, simultaneous=True), index, sp.oo)
            == item.value
        )
    result = limit(expression, (x, y), (0, 0), return_result=True)
    assert result.status is LimitStatus.DOES_NOT_EXIST
    assert {e.value for e in result.evidence} == {sp.Integer(2), sp.oo}


def test_shifted_reciprocal_sequences():
    from asymptotic.local_path_witnesses import reciprocal_phase_conflict

    x, y, z = sp.symbols("x y z", real=True)
    expression = sp.sin(x * x + y * y + z * z) * sp.cos(1 / (x + y + z - 2))
    evidence = reciprocal_phase_conflict(
        expression, (x, y, z), (1, 0, 1), sp.S.true, sp.S.true
    )
    assert {e.value for e in evidence} == {sp.sin(2), -sp.sin(2)}
    for item in evidence:
        chart = dict(item.substitutions)
        index = next(iter(chart[x].free_symbols))
        assert tuple(sp.limit(chart[v], index, sp.oo) for v in (x, y, z)) == (1, 0, 1)
        assert (x + y + z - 2).subs(chart, simultaneous=True).is_positive is True
        phase = sp.cancel((1 / (x + y + z - 2)).subs(chart, simultaneous=True))
        oscillator = sp.cos(phase)
        assert oscillator in (sp.S.One, -sp.S.One)
        amplitude = (x * x + y * y + z * z).subs(chart, simultaneous=True)
        assert sp.limit(amplitude, index, sp.oo) == 2
        assert oscillator * sp.sin(2) == item.value
    result = limit(expression, (x, y, z), (1, 0, 1), return_result=True)
    assert result.status is LimitStatus.DOES_NOT_EXIST
    assert {e.value for e in result.evidence} == {sp.sin(2), -sp.sin(2)}


@pytest.mark.parametrize(
    "family",
    [
        "divisible",
        "nonlinear",
        "restricted",
        "degree",
        "oversized_phase",
        "original_pole",
    ],
)
def test_polynomial_pole_declines(family):
    from asymptotic.local_path_witnesses import polynomial_pole_conflict

    x, y = sp.symbols("x y", real=True)
    cases = {
        "divisible": (sp.sin(x**3 + y**3) / (x + y), sp.S.true),
        "nonlinear": (sp.sin(x * x) / (x * x + y * y), sp.S.true),
        "restricted": (sp.sin(x * x) / (x + y), x > 0),
        "degree": (sp.sin(x**5) / (x + y), sp.S.true),
        "oversized_phase": (sp.sin((x + y) ** 1000000) / (x + y), sp.S.true),
        "original_pole": (
            sp.Mul(
                sp.sin(x * x) / (x + y),
                y,
                sp.Pow(y, -1, evaluate=False),
                evaluate=False,
            ),
            sp.S.true,
        ),
    }
    expression, domain = cases[family]
    assert (
        polynomial_pole_conflict(expression, (x, y), (0, 0), domain, sp.S.true) is None
    )


def test_nonzero_path_germs():
    from asymptotic.local_path_witnesses import _nonzero_germ

    t = sp.Symbol("t", positive=True)
    assert _nonzero_germ(t**12, t) is True
    assert _nonzero_germ(sp.exp(t), t) is True
    assert _nonzero_germ(sp.sin(t), t) is True
    assert _nonzero_germ(sp.S.Zero, t) is False
    assert _nonzero_germ(sp.sin(1 / t), t) is False
    assert _nonzero_germ(sp.zoo, t) is False
