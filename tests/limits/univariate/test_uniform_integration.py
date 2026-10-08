import sympy as sp

from asymptotic import (
    limit,
    local_series,
)
from asymptotic.regime_selection import (
    AsymptoticRegime,
    RegimeGeometry,
    select_asymptotic_regime,
)
from asymptotic.uniform_integration import (
    uniform_composite_expansion,
)


def test_turning_point_regime_is_proved():
    n = sp.symbols("n", positive=True)
    a = sp.symbols("a", finite=True)
    cert = select_asymptotic_regime(sp.besselj(n, n + a * n ** sp.Rational(1, 3)), n)
    assert cert.regime is AsymptoticRegime.TURNING_POINT
    assert cert.transition_parameter == a
    assert cert.verified


def test_outside_transition_window_uses_uniform_scaled_regime():
    n = sp.symbols("n", positive=True)
    atom = sp.besselj(n, n + n ** sp.Rational(1, 2))
    cert = select_asymptotic_regime(atom, n)
    assert cert.regime is AsymptoticRegime.SCALED_ORDER_TAIL
    assert cert.verified
    assert cert.limiting_scaled_variable == 1


def test_scaled_large_order_regime_is_selected():
    n = sp.symbols("n", positive=True)
    cert = select_asymptotic_regime(sp.besselj(n, sp.Rational(4, 5) * n), n)
    assert cert.regime is AsymptoticRegime.SCALED_ORDER_TAIL
    assert cert.scaled_variable == sp.Rational(4, 5)


def test_limit_uses_turning_point_expansion_automatically():
    n = sp.symbols("n", positive=True)
    a = sp.symbols("a", finite=True)
    expr = n ** sp.Rational(1, 3) * sp.besselj(n, n + a * n ** sp.Rational(1, 3))
    expected = 2 ** sp.Rational(1, 3) * sp.airyai(-(2 ** sp.Rational(1, 3)) * a)
    assert sp.simplify(limit(expr, n, sp.oo) - expected) == 0


def test_local_expansion_uses_uniform_provider_at_infinity():
    n = sp.symbols("n", positive=True)
    atom = sp.besselj(n, n)
    expansion = local_series(atom, n, sp.oo, return_result=True)
    assert expansion.provider == "uniform-asymptotic-regime"
    assert expansion.certificate.hypotheses_verified


def test_uniform_cancellation_happens_before_limit():
    n = sp.symbols("n", positive=True)
    a = sp.symbols("a", finite=True)
    j = sp.besselj(n, n + a * n ** sp.Rational(1, 3))
    y = sp.bessely(n, n + a * n ** sp.Rational(1, 3))
    expr = n ** sp.Rational(1, 3) * (j + y) - n ** sp.Rational(1, 3) * j
    expected = -(2 ** sp.Rational(1, 3)) * sp.airybi(-(2 ** sp.Rational(1, 3)) * a)
    assert sp.simplify(limit(expr, n, sp.oo) - expected) == 0


def test_connection_formula_cancels_uniform_prefixes():
    n = sp.symbols("n", positive=True)
    a = sp.symbols("a", finite=True)
    x = n + a * n ** sp.Rational(1, 3)
    expr = n ** sp.Rational(1, 3) * (
        sp.hankel1(n, x) - sp.besselj(n, x) - sp.I * sp.bessely(n, x)
    )
    expansion = uniform_composite_expansion(expr, n)
    assert expansion is not None
    assert sp.simplify(expansion.prefix) == 0


def test_coalescing_saddle_requires_certified_geometry():
    n = sp.symbols("n", positive=True)
    geometry = RegimeGeometry(
        coalescing_saddles=(sp.Integer(-1), sp.Integer(1)),
        saddle_separation_scale=n ** -sp.Rational(1, 3),
        hypotheses_verified=True,
    )
    result = select_asymptotic_regime(sp.exp(-n), n, geometry=geometry)
    assert result.regime is AsymptoticRegime.COALESCING_SADDLES


def test_stokes_boundary_requires_certified_geometry():
    n = sp.symbols("n", positive=True)
    geometry = RegimeGeometry(
        stokes_singulants=(2 * n,),
        stokes_distance_scale=n ** -sp.Rational(1, 2),
        hypotheses_verified=True,
    )
    result = select_asymptotic_regime(sp.exp(-n), n, geometry=geometry)
    assert result.regime is AsymptoticRegime.STOKES_BOUNDARY


def test_uncertified_geometry_does_not_select_theorem():
    n = sp.symbols("n", positive=True)
    geometry = RegimeGeometry(
        coalescing_saddles=(sp.Integer(-1), sp.Integer(1)),
        saddle_separation_scale=n ** -sp.Rational(1, 3),
        hypotheses_verified=False,
    )
    result = select_asymptotic_regime(sp.exp(-n), n, geometry=geometry)
    assert result.regime is AsymptoticRegime.UNKNOWN
