"""Local coordinate normalization and uniform elementary quotient bounds."""

import pytest
import sympy as sp

from asymptotic import limit
from asymptotic.limit_models import LimitStatus
from asymptotic.weighted_monomial_bounds import weighted_quotient_certificate


@pytest.mark.parametrize(
    "family",
    [
        "absolute_unit",
        "signed_root",
        "shifted_sine",
        "exponential_logarithm",
        "radial_sine",
        "exponential_pole",
    ],
)
def test_local_bounds(family):
    x, y = sp.symbols("x y", real=True)
    cases = {
        "absolute_unit": ((x * sp.Abs(y - 1) - x) / sp.sqrt(x**2 + y**2), (0, 0), 0),
        "signed_root": (
            7
            * sp.cos(
                sp.Abs(x) ** sp.Rational(4, 3) * sp.sign(x) ** 4 * y**4 / (x**2 + y**8)
            ),
            (0, 0),
            7,
        ),
        "shifted_sine": (
            sp.sin(x**2 * (y + 1)) / (x**2 * sp.Abs(y) + (y + 1) ** 2),
            (0, -1),
            0,
        ),
        "exponential_logarithm": (
            (sp.exp(-2 * x) * sp.log(y - 4) + 5 - y) / (x**2 + sp.Abs(y - 5)),
            (0, 5),
            0,
        ),
        "radial_sine": (
            x**2 + y**2 + sp.log(1 + x**2 * y) / sp.sin(sp.sqrt(x**2 + y**2)),
            (0, 0),
            0,
        ),
        "exponential_pole": (sp.exp(1 / (x**2 + y**2)) / (x**4 + y**4), (0, 0), sp.oo),
    }
    expression, target, expected = cases[family]
    target = tuple(map(sp.sympify, target))
    certificate = weighted_quotient_certificate(
        expression, (x, y), target, sp.S.true, sp.S.true
    )
    assert certificate is not None
    assert certificate[:2] == (LimitStatus.PROVED, expected)
    result = limit(expression, (x, y), target, return_result=True)
    assert result.status is LimitStatus.PROVED
    assert result.value == expected
    assert tuple(v for v, _ in result.evidence[0].substitutions) == (x, y)
    # The certificate must retain a defined approach in the original coordinates.
    along = expression.subs(dict(result.evidence[0].substitutions), simultaneous=True)
    index = next(iter(along.free_symbols))
    assert not along.subs(index, 10).has(sp.nan, sp.zoo)


@pytest.mark.parametrize(
    "family",
    [
        "nonzero_phase",
        "hidden_sign_pole",
        "wrong_unit_center",
        "signed_exponential_phase",
    ],
)
def test_declined_bounds(family):
    x, y = sp.symbols("x y", real=True)
    expressions = {
        "nonzero_phase": sp.cos(x * y / (x**2 + y**2)),
        "hidden_sign_pole": sp.sign(1 / (x - y)) * x**2 * y / (x**2 + y**2),
        "wrong_unit_center": (sp.exp(1 + x) * sp.log(1 + y) - y) / (x**2 + sp.Abs(y)),
        "signed_exponential_phase": sp.exp(1 / (x**2 - y**2)) / (x**4 + y**4),
    }
    assert (
        weighted_quotient_certificate(
            expressions[family], (x, y), (sp.S.Zero, sp.S.Zero), sp.S.true, sp.S.true
        )
        is None
    )


def test_shifted_absolute_branch():
    x, y = sp.symbols("x y", real=True)
    normalized = (x * sp.Abs(y - 1) - x) / sp.sqrt(x**2 + y**2)
    local = -x * y / sp.sqrt(x**2 + y**2)
    for a, b in [
        (sp.Rational(1, 4), sp.Rational(-1, 2)),
        (sp.Rational(-1, 4), sp.Rational(1, 2)),
    ]:
        assert normalized.subs({x: a, y: b}) == local.subs({x: a, y: b})
    # The identity is local: applying it across y=1 would change the expression.
    assert normalized.subs({x: 1, y: 2}) != local.subs({x: 1, y: 2})


def test_exponential_coordinate_pole():
    x, y = sp.symbols("x y", real=True)
    result = limit(sp.exp(1 / x**2) / (x**2 + y**2), (x, y), (0, 0), return_result=True)
    assert result.status is LimitStatus.PROVED
    assert result.value == sp.oo
    assert result.evidence[0].method == "positive_exponential_pole_bound"


def test_approximate_center():
    x, y = sp.symbols("x y", real=True)
    point = sp.Float(0.1)
    exact_point = sp.Rational(point)
    displacement = exact_point - sp.Rational(1, 10)
    assert displacement != 0
    radial = (x - sp.Rational(1, 10)) ** 2 + y**2
    expression = sp.exp(1 / radial) / radial**2
    assert radial.subs({x: exact_point, y: 0}) == displacement**2
    assert (
        weighted_quotient_certificate(
            expression, (x, y), (point, 0), sp.S.true, sp.S.true
        )
        is None
    )


@pytest.mark.parametrize("pole", ["diagonal", "oscillatory"])
def test_removed_poles(pole):
    x, y = sp.symbols("x y", real=True)
    base = x - y if pole == "diagonal" else sp.sin(sp.pi / x)
    cancelled = sp.Mul(base, sp.Pow(base, -1, evaluate=False), evaluate=False)
    expression = sp.Mul(cancelled, x**2 * y / (x**2 + y**2), evaluate=False)
    assert (
        weighted_quotient_certificate(expression, (x, y), (0, 0), sp.S.true, sp.S.true)
        is None
    )


def test_factored_expansion_budget():
    from asymptotic._polynomial_bounds import bounded_expansion_width
    from asymptotic.weighted_monomial_bounds import weighted_quotient_certificate

    x, y = sp.symbols("x y", real=True)
    numerator = sp.prod(x + j * y for j in range(1, 13))
    assert bounded_expansion_width(numerator, 24) == 25
    assert (
        weighted_quotient_certificate(
            numerator / (x * x + y * y), (x, y), (0, 0), sp.S.true, sp.S.true
        )
        is None
    )
