"""Exact definitions, uniform bounds and attained discontinuity witnesses."""

import pytest
import sympy as sp

from asymptotic import limit
from asymptotic.discontinuous_functions import SignedFractionalPart
from asymptotic.fixed_ray_branch_germs import DirectionalInfinity
from asymptotic.fractional_pole_witnesses import fractional_pole_conflict
from asymptotic.function_normalization import RealRoot, normalize_functions
from asymptotic.limit_models import LimitStatus
from asymptotic.real_pole_germs import (
    logarithmic_unit_vanishing_certificate,
    positive_logarithmic_pole_certificate,
    signed_real_pole_certificate,
)
from asymptotic.reference_normalization import multivariate_reference_namespace
from asymptotic.uniform_radial_bounds import radial_vanishing_certificate

x, y = sp.symbols("x y", real=True)
zero = (sp.S.Zero, sp.S.Zero)


@pytest.mark.parametrize(
    "value,expected",
    [
        (sp.Rational(12, 5), sp.Rational(2, 5)),
        (-sp.Rational(12, 5), -sp.Rational(2, 5)),
        (0, 0),
        (3, 0),
        (-3, 0),
        (
            sp.Rational(-7, 3) + sp.I * sp.Rational(8, 3),
            -sp.Rational(1, 3) + sp.I * sp.Rational(2, 3),
        ),
    ],
)
def test_signed_fractional_values(value, expected):
    assert SignedFractionalPart(value) == expected


def test_source_definitions():
    vocabulary = multivariate_reference_namespace()
    assert vocabulary["UnitStep"](0) == 1
    assert vocabulary["UnitStep"](-1) == 0
    assert vocabulary["UnitStep"](1) == 1
    assert vocabulary["UnitStep"](0, 1) == 1
    assert vocabulary["UnitStep"](0, -1) == 0
    assert vocabulary["UnitStep"]() == 1
    assert vocabulary["FractionalPart"](-sp.Rational(1, 2)) == -sp.Rational(1, 2)
    assert vocabulary["ArcCot"](-1) == -sp.pi / 4
    assert vocabulary["ArcSec"](-sp.Rational(1, 2)).rewrite(sp.acos) == sp.acos(-2)
    assert vocabulary["Directedoo"](-sp.I) == DirectionalInfinity(-sp.I)


@pytest.mark.parametrize(
    "expr,target,value",
    [
        (RealRoot(x * x, 3) / (2 * x * x), zero, sp.oo),
        (SignedFractionalPart(y) * x, zero, 0),
        (
            SignedFractionalPart(4 - 4 * sp.cos(sp.sqrt(sp.Abs(x * y))))
            * sp.Abs(x * y),
            zero,
            0,
        ),
        (sp.acot(-3 / (x * x + y * y)), zero, 0),
        (sp.asec(-(x * x + y * y)), zero, DirectionalInfinity(-sp.I)),
        (sp.atan(x**4 + y**4) / (x * x + y * y * sp.Abs(x + 1)), zero, 0),
        (x / (sp.sqrt(x * x + y * y) * sp.log(x * x + y * y)), zero, 0),
        (sp.log(1 + 1 / (x * x * y * y)), zero, sp.oo),
        (sp.sign(x - y) / (-x + y), zero, -sp.oo),
        (sp.sin(y) ** 2 * sp.Abs(x - 1) / sp.log(x), (sp.S.One, sp.S.Zero), 0),
        (x * y * sp.exp(-1 / (x * x * y * y)), zero, 0),
        (x * y * sp.exp(-y / x**2), (sp.S.Zero, sp.Integer(2)), 0),
        (x * y * sp.exp(-y / x**2), (sp.S.Zero, sp.Integer(7)), 0),
    ],
)
def test_normalized_values(expr, target, value):
    result = limit(expr, (x, y), target, return_result=True)
    assert result.status is LimitStatus.PROVED
    assert result.value == value


@pytest.mark.parametrize(
    "expr,target",
    [
        (x * y * y / SignedFractionalPart(x * x + y**4), zero),
        (sp.sqrt(x * y * y) / SignedFractionalPart(x * x + y**4), zero),
        (
            RealRoot((x**3 * y + y**5) / (sp.Abs(x) + sp.Abs(y)), 3)
            / sp.sqrt(x * x + y * y),
            zero,
        ),
        (
            sp.Heaviside(x * y, 1) * (x * x - y * y) / (x * x + y * y),
            (sp.S.One, sp.S.Zero),
        ),
        (
            SignedFractionalPart((4 * y * y - x * x) / (x - 2 * y) ** 3),
            (sp.Integer(2), sp.S.One),
        ),
    ],
)
def test_normalized_conflicts(expr, target):
    result = limit(expr, (x, y), target, return_result=True)
    assert result.status is LimitStatus.DOES_NOT_EXIST
    witnesses = [e for e in result.evidence if e.method.startswith("attained_")]
    assert len(witnesses) == 2
    assert witnesses[0].value != witnesses[1].value
    for evidence in witnesses:
        sequence = dict(evidence.substitutions)
        index = next(
            iter(set().union(*(value.free_symbols for value in sequence.values())))
        )
        assert tuple(sp.limit(sequence[v], index, sp.oo) for v in (x, y)) == target
        along = expr.subs(sequence, simultaneous=True)
        if along.has(SignedFractionalPart):
            along = (
                along.replace(
                    lambda a: a.func is SignedFractionalPart,
                    lambda a: SignedFractionalPart(sp.cancel(a.args[0])),
                )
                .rewrite(sp.frac)
                .rewrite(sp.floor)
            )
            along = sp.simplify(along)
        assert sp.limit(along, index, sp.oo) == evidence.value
        for n in (2, 5, 20):
            assert along.subs(index, n).has(sp.nan, sp.zoo) is False


def test_fractional_phase_poles():
    expr = SignedFractionalPart((4 * y * y - x * x) / (x - 2 * y) ** 3)
    result = fractional_pole_conflict(
        expr, (x, y), (sp.Integer(2), sp.S.One), sp.S.true, sp.S.true
    )
    assert tuple(e.value for e in result[2]) == (-sp.Rational(1, 4), -sp.Rational(3, 4))
    for evidence in result[2]:
        sequence = dict(evidence.substitutions)
        index = next(
            iter(set().union(*(value.free_symbols for value in sequence.values())))
        )
        attained = sp.cancel(expr.args[0].subs(sequence, simultaneous=True))
        assert sp.simplify(attained + index - evidence.value) == 0
        assert sp.simplify((x - 2 * y).subs(sequence)) != 0


def test_root_domain():
    argument = (x**3 * y + y**5) / (sp.Abs(x) + sp.Abs(y))
    normalized, evidence = normalize_functions(RealRoot(argument, 3), (x, y), zero)
    assert normalized == sp.sign(argument) * sp.Abs(argument) ** sp.Rational(1, 3)
    assert evidence[0].method == "local_function_normalization"
    unchanged, _ = normalize_functions(RealRoot(x + y, 2), (x, y), zero)
    assert unchanged == RealRoot(x + y, 2)


def test_bound_scope():
    positive = sp.Symbol("p", positive=True)
    assert (
        signed_real_pole_certificate(
            sp.sign(x - y) / (x - y), (x, y), zero, x > y, sp.S.true
        )
        is None
    )
    assert (
        positive_logarithmic_pole_certificate(
            sp.log(1 + 1 / (x * y)), (x, y), zero, sp.S.true, sp.S.true
        )
        is None
    )
    assert (
        radial_vanishing_certificate(
            x * sp.exp(-y / x**2), (x, y), zero, sp.S.true, sp.S.true
        )
        is None
    )
    assert (
        radial_vanishing_certificate(
            x * sp.exp(sp.I * y / x**2), (x, y), zero, sp.S.true, sp.S.true
        )
        is None
    )
    assert (
        logarithmic_unit_vanishing_certificate(
            sp.Abs(x) / sp.log(1 + x), (x, y), zero, sp.S.true, sp.S.true
        )
        is None
    )
    assert (
        fractional_pole_conflict(
            SignedFractionalPart(positive / y),
            (positive, y),
            zero,
            sp.S.true,
            sp.S.true,
        )
        is None
    )


def test_positive_unit_bound():
    result = radial_vanishing_certificate(
        sp.atan(x**4 + y**4) / (x * x + y * y * sp.Abs(1 + x)),
        (x, y),
        zero,
        sp.S.true,
        sp.S.true,
    )
    assert result[0] is LimitStatus.PROVED
    assert result[1] == 0
    assert (
        radial_vanishing_certificate(
            x / (sp.sqrt(x * x + y * y) * sp.log(x * x + y * y)),
            (x, y),
            zero,
            sp.S.true,
            sp.S.true,
        )[1]
        == 0
    )


def test_reciprocal_original_holes():
    from asymptotic.inverse_pole_composition import reciprocal_inverse_pole_certificate

    hole = (x - y) ** 2
    argument = (x * x - 2 * x * y + y * y) * (x * x + y * y) / hole
    result = limit(sp.asec(argument), (x, y), zero, return_result=True)
    assert result.status is LimitStatus.PROVED
    assert result.value == DirectionalInfinity(sp.I)
    for witness in result.evidence:
        if witness.substitutions:
            assert (
                sp.simplify(hole.subs(dict(witness.substitutions))).is_positive is True
            )
    assert (
        reciprocal_inverse_pole_certificate(
            sp.asec((x + y) ** 64), (x, y), zero, sp.S.true, sp.S.true
        )
        is None
    )


def test_cotangent_chart_holes():
    expr = sp.Heaviside(sp.acot(1 / (x - y) ** 2), 0)
    normalized, evidence = normalize_functions(expr, (x, y), zero)
    assert normalized == expr
    assert evidence == ()
    normalized, evidence = normalize_functions(
        sp.acot(-3 / (x * x + y * y)), (x, y), zero
    )
    assert normalized == sp.atan(-(x * x + y * y) / 3)
    assert evidence[0].method == "local_function_normalization"


def test_undefined_ray_declines():
    from asymptotic.attained_ray_germs import ray_leading_term

    t = sp.Symbol("t", positive=True)
    assert ray_leading_term(1 + sp.zoo * (1 + t) ** 2, t) is None
    result = limit(
        y * y * sp.atan(x / y), (x, y), (sp.S.One, sp.S.Zero), return_result=True
    )
    assert result.status is LimitStatus.PROVED
    assert result.value == 0
