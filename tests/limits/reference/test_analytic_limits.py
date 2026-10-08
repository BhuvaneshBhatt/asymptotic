import pytest
import sympy as s

from asymptotic import limit, one_sided_limit
from asymptotic.analytic_limits import SquareWave, analytic_limit, normalize_direction
from asymptotic.limit_models import LimitStatus

x = s.Symbol("x")
f = s.Function("f")


@pytest.mark.parametrize(
    "name,side", [("above", "+"), ("below", "-"), ("+", "+"), ("-", "-")]
)
def test_named_direction_matches_outward_real_approach(name, side):
    assert SquareWave(0) == 0 and SquareWave(s.S.Half) == 0
    assert normalize_direction(name) == side
    result = analytic_limit(1 / x, x, 0, direction=name, return_result=True)
    assert result.value == (s.oo if side == "+" else -s.oo)


def test_analytic_declaration_is_scoped_and_requires_finite_center():
    p = s.Symbol("p")
    result = analytic_limit(
        f(x),
        x,
        p,
        analytic_functions=["f"],
        return_result=True,
        assumptions=s.Q.finite(p),
    )
    assert result.status is LimitStatus.PROVED and result.value == f(p)
    assert limit(f(x), x, 0, return_result=True).status is LimitStatus.UNKNOWN
    assert (
        analytic_limit(f(x), x, p, analytic_functions=["f"], return_result=True).status
        is LimitStatus.UNKNOWN
    )
    assert (
        analytic_limit(
            f(x), x, 0, analytic_functions=["f"], return_result=True, assumptions=False
        ).status
        is LimitStatus.UNKNOWN
    )
    with pytest.raises(ValueError):
        analytic_limit(
            f(x),
            x,
            0,
            direction="nonsense",
            analytic_functions=["f"],
            return_result=True,
        )
    xr = s.Symbol("xr", real=True)
    xp = s.Symbol("xp", positive=True)
    assert (
        analytic_limit(
            f(xr), xr, s.I, analytic_functions=["f"], return_result=True
        ).status
        is LimitStatus.UNKNOWN
    )
    with pytest.raises(ValueError):
        analytic_limit(
            f(xp), xp, 0, direction="-", analytic_functions=["f"], return_result=True
        )
    assert analytic_limit(
        f(xp), xp, 1, direction="+", analytic_functions=["f"], return_result=True
    ).value == f(1)
    assert analytic_limit(
        f(xp), xp, 0, direction="+", analytic_functions=["f"], return_result=True
    ).value == f(0)


def test_analytic_real_cube_root_continuity_retains_real_branch():
    p = s.Symbol("p")
    a = s.Q.real(p) & s.Q.finite(p)
    result = analytic_limit(
        f(s.real_root(x, 3)),
        x,
        p,
        analytic_functions=["f"],
        return_result=True,
        assumptions=a,
    )
    assert result.value.subs(p, -8) == f(-2)
    assert result.value.subs(p, 8) == f(2)
    assert result.value.subs(p, 0) == f(0)


@pytest.mark.parametrize(
    "p,right,left",
    [(0, 1, -1), (s.S.Half, -1, 1), (1, 1, -1), (s.Rational(1, 4), 1, 1)],
)
def test_square_wave_sides_follow_periodic_definition(p, right, left):
    assert one_sided_limit(SquareWave(x), x, p, direction="+") == right
    assert one_sided_limit(SquareWave(x), x, p, direction="-") == left
    assert SquareWave(p + s.Rational(1, 1000)) == right
    assert SquareWave(p - s.Rational(1, 1000)) == left


def test_named_integer_floor_center_and_membership_translation():
    from asymptotic.reference_normalization import scalar_reference_assumptions

    n = s.Symbol("n")
    a = scalar_reference_assumptions("Q.integer(n)")
    assert a == s.Q.integer(n)
    result = analytic_limit(
        s.floor(x), x, n, direction="above", assumptions=a, return_result=True
    )
    assert result.value == n and result.domain.free_symbols <= {x, n}
    assert (
        analytic_limit(
            s.floor(x), x, n, direction="below", assumptions=a, return_result=True
        ).value
        == n - 1
    )


def test_custom_wave_heights_and_exact_complex_phase_comparison():
    from asymptotic.reference_normalization import scalar_reference_equal

    assert SquareWave(s.Tuple(2, 6), s.Rational(1, 4)) == 6
    assert SquareWave(s.Tuple(2, 6), s.Rational(3, 4)) == 2
    assert scalar_reference_equal(
        s.exp(4 * s.I * s.pi / 5), (-s.S.One) ** s.Rational(4, 5)
    )
