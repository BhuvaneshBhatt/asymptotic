import sympy as sp

from asymptotic import (
    hyperasymptotic_series,
    limit,
)
from asymptotic.hyperasymptotics import (
    ExponentialScale,
    stokes_transition,
)
from asymptotic.olver_coefficients import (
    olver_coefficient_block,
)
from asymptotic.regime_selection import (
    AsymptoticRegime,
    select_asymptotic_regime,
)
from asymptotic.saddle_geometry import (
    analyze_exponential_integral,
    analyze_saddle_geometry,
)
from asymptotic.stokes_geometry import (
    analyze_stokes_geometry,
    select_stokes_regime,
)
from asymptotic.uniform_integration import (
    uniform_composite_expansion,
)
from asymptotic.uniform_special_expansions import (
    bessel_j_large_order,
    bessel_y_large_order,
    hankel_large_order,
)
from asymptotic.uniform_zeros import (
    uniform_zero_approximation,
)


def test_transition_scaling_is_proved_from_limit_not_syntax():
    n = sp.symbols("n", positive=True)
    a, b = sp.symbols("a b", finite=True)
    x = n + a * n ** sp.Rational(1, 3) + b * n ** -sp.Rational(1, 3)
    cert = select_asymptotic_regime(sp.besselj(n, x), n)
    assert cert.regime is AsymptoticRegime.TURNING_POINT
    assert cert.transition_limit == a
    assert cert.transition_parameter.has(n)


def test_scaled_regime_accepts_vanishing_perturbation():
    n = sp.symbols("n", positive=True)
    z = sp.symbols("z", positive=True)
    cert = select_asymptotic_regime(sp.besselj(n, n * z + sp.sqrt(n)), n)
    assert cert.regime is AsymptoticRegime.SCALED_ORDER_TAIL
    assert cert.limiting_scaled_variable == z


def test_transition_limit_retains_subleading_shift():
    n = sp.symbols("n", positive=True)
    a, b = sp.symbols("a b", finite=True)
    x = n + a * n ** sp.Rational(1, 3) + b * n ** -sp.Rational(1, 3)
    expr = n ** sp.Rational(1, 3) * sp.besselj(n, x)
    expected = 2 ** sp.Rational(1, 3) * sp.airyai(-(2 ** sp.Rational(1, 3)) * a)
    assert sp.simplify(limit(expr, n, sp.oo) - expected) == 0


def test_matched_olver_block_has_all_four_families():
    block = olver_coefficient_block(3, sp.S.One)
    assert set(block) == {"A", "B", "C", "D"}
    assert all(len(values) == 3 for values in block.values())
    assert block["A"][0] == block["D"][0] == 1


def test_polynomial_composite_remainder_keeps_cross_term():
    n = sp.symbols("n", positive=True)
    a = sp.symbols("a", finite=True)
    j = sp.besselj(n, n + a * n ** sp.Rational(1, 3))
    expansion = uniform_composite_expansion(j * j, n)
    assert expansion is not None and expansion.certified
    # Squaring (prefix+delta) includes delta**2, hence an n^-2 contribution.
    assert (
        expansion.remainder_scale.has(n**-2)
        or sp.limit(expansion.remainder_scale * n**2, n, sp.oo) != 0
    )


def test_derivative_transition_is_available_to_composite_engine():
    n = sp.symbols("n", positive=True)
    x = sp.Symbol("x")
    derivative = sp.Subs(sp.diff(sp.besselj(n, x), x), x, n)
    expansion = uniform_composite_expansion(n ** sp.Rational(2, 3) * derivative, n)
    assert expansion is not None
    difference = sp.N(
        sp.limit(expansion.prefix, n, sp.oo)
        + 2 ** sp.Rational(2, 3) * sp.airyaiprime(0),
        40,
    )
    assert abs(complex(difference)) < 1e-35


def test_cubic_phase_coalescence_is_detected():
    n = sp.symbols("n", positive=True)
    t = sp.symbols("t", real=True)
    a = sp.symbols("a", positive=True)
    phase = t**3 / 3 - a * n ** -sp.Rational(2, 3) * t
    cert = analyze_saddle_geometry(phase, t, n)
    assert cert.hypotheses_verified
    assert cert.separation_exponent == -sp.Rational(1, 3)
    selected = select_asymptotic_regime(sp.exp(-n), n, geometry=cert.geometry)
    assert selected.regime is AsymptoticRegime.COALESCING_SADDLES


def test_stokes_boundary_scaling_is_detected_and_smoothed():
    n = sp.symbols("n", positive=True)
    c = sp.symbols("c", real=True)
    chi = n * sp.exp(sp.I * c / sp.sqrt(n))
    scale = ExponentialScale(chi, 1, 1, (0,), (sp.pi / 2,), "adjacent")
    geometry = analyze_stokes_geometry(scale, n)
    assert geometry.hypotheses_verified and geometry.stokes
    transition = stokes_transition(scale, n)
    assert transition.hypotheses_verified
    assert transition.smoothing_approximation is not None


def test_uniform_zero_dispatch_uses_shared_turning_map():
    result = uniform_zero_approximation("bessel-j", sp.Integer(40), 1)
    assert result.family == "bessel-j"
    assert result.residual < sp.Rational(1, 1000)


def test_higher_olver_turning_values_match_reference():
    block = olver_coefficient_block(4, sp.S.One)
    assert block["A"][3] == -sp.Rational(887278009, 2504935125000)
    assert block["B"][3] == (
        -sp.Rational(9597171184603, 25476663712500000) * sp.real_root(2, 3)
    )


def test_exponential_integral_feeds_automatic_saddle_analyzer():
    n = sp.symbols("n", positive=True)
    t = sp.symbols("t", real=True)
    phase = t**3 / 3 - n ** -sp.Rational(2, 3) * t
    integral = sp.Integral(sp.exp(-n * phase), (t, -1, 1))
    cert = analyze_exponential_integral(integral, n)
    assert cert is not None and cert.hypotheses_verified
    assert cert.geometry.hypotheses_verified


def test_two_scale_second_level_hyperasymptotics():
    n = sp.symbols("n", positive=True)
    scales = (
        ExponentialScale(n, 1, 1, (0,), (sp.pi / 2,), "s1"),
        ExponentialScale(2 * n, sp.Rational(1, 3), -1, (0,), (sp.pi / 2,), "s2"),
    )
    expansion = hyperasymptotic_series(
        1,
        lambda level, count: tuple(sp.factorial(k + level) for k in range(count)),
        scales,
        levels=2,
        reexpansion_terms=2,
        return_result=True,
    )
    assert len(expansion.levels) == 2
    assert expansion.levels[0].singulant == n
    assert expansion.levels[1].singulant == 2 * n


def test_hankel_uniform_coefficients_share_j_y_convention():
    n = sp.symbols("n", positive=True)
    z = sp.Rational(4, 5)
    j = bessel_j_large_order(n, z, terms=3)
    y = bessel_y_large_order(n, z, terms=3)
    h = hankel_large_order(n, z, kind=1, terms=3)
    difference = sp.N(
        (h.prefix - j.prefix - sp.I * y.prefix).subs(n, sp.Integer(30)),
        40,
    )
    assert abs(complex(difference)) < 1e-35


def test_regime_selector_automatically_analyzes_integral_saddles():
    n = sp.symbols("n", positive=True)
    t = sp.symbols("t", real=True)
    phase = t**3 / 3 - n ** -sp.Rational(2, 3) * t
    integral = sp.Integral(sp.exp(-n * phase), (t, -1, 1))
    result = select_asymptotic_regime(integral, n)
    assert result.regime is AsymptoticRegime.COALESCING_SADDLES


def test_stokes_scale_selects_regime_without_manual_geometry():
    n = sp.symbols("n", positive=True)
    c = sp.symbols("c", real=True)
    scale = ExponentialScale(
        n * sp.exp(sp.I * c / sp.sqrt(n)), 1, 1, (0,), (sp.pi / 2,), "adjacent"
    )
    result = select_stokes_regime(scale, n)
    assert result.regime is AsymptoticRegime.STOKES_BOUNDARY
