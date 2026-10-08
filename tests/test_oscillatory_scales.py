import sympy as sp

from asymptotic.oscillatory_scales import (
    CertifiedOscillatoryExpansion,
    PhaseBehavior,
    certified_oscillatory_expand,
    phase_geometry,
)
from asymptotic.stratified_expansion import stratified_expand

x, e = sp.symbols("x e", positive=True)


def oscillatory_branches(result):
    return [b for b in result.branches if isinstance(b, CertifiedOscillatoryExpansion)]


def test_sine_reciprocal_cell_is_now_certified():
    result = stratified_expand(sp.sin(x / e), (x, e), order=4)
    assert result.certified
    branches = oscillatory_branches(result)
    assert branches
    assert any(
        all(s.behavior is PhaseBehavior.DIVERGENT for s in b.scales) for b in branches
    )


def test_cosine_reciprocal_cell_is_certified():
    result = stratified_expand(sp.cos(x / e), (x, e), order=4)
    assert result.certified
    assert oscillatory_branches(result)


def test_direct_exp_i_phase_is_amplitude_phase_transmonomial():
    result = certified_oscillatory_expand(sp.exp(sp.I * x / e), small=e, large=x)
    assert result is not None and result.certified
    assert len(result.scales) == 1
    assert result.scales[0].behavior is PhaseBehavior.DIVERGENT


def test_amplitude_times_oscillation_is_preserved():
    result = certified_oscillatory_expand(x**2 * sp.exp(sp.I * x / e), small=e, large=x)
    assert result is not None
    assert sp.simplify(result.representation - x**2 * sp.exp(sp.I * x / e)) == 0


def test_nondivegent_phase_declines_oscillatory_provider():
    assert certified_oscillatory_expand(sp.sin(e / x), small=e, large=x) is None


def test_weighted_multivariate_phase_geometry_diverges():
    y = sp.Symbol("y", positive=True)
    geometry = phase_geometry(x**2 / y**3, (x, y), weights=(1, 1))
    assert geometry.certified
    assert geometry.radial_order == -1
    assert geometry.behavior is PhaseBehavior.DIVERGENT


def test_anisotropic_weight_can_change_phase_behavior():
    y = sp.Symbol("y", positive=True)
    geometry = phase_geometry(x**2 / y**3, (x, y), weights=(2, 1))
    assert geometry.certified
    assert geometry.radial_order == 1
    assert geometry.behavior is PhaseBehavior.VANISHING


def test_sum_phase_declines_instead_of_hiding_cancellation_geometry():
    y = sp.Symbol("y", positive=True)
    geometry = phase_geometry(1 / x - 1 / y, (x, y))
    assert not geometry.certified
    assert geometry.obligations


def test_stratifier_routes_pure_imaginary_oscillatory_provider():
    result = stratified_expand(sp.exp(sp.I * x / e), (x, e), order=3)
    assert result.certified
    assert oscillatory_branches(result)
