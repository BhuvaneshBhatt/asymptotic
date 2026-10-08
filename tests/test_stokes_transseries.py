import sympy as sp

from asymptotic.complex_domain import ComplexSector
from asymptotic.sectorial_transseries import SectorialDominance
from asymptotic.stokes_transseries import (
    TransitionKind,
    certify_beyond_all_orders,
    derive_stokes_surfaces,
    dominance_sector_atlas,
    stokes_aware_expansion,
)
from asymptotic.unified_stratification import unified_stratified_expand


def test_stokes_and_antistokes_rays_for_exp_inverse_z():
    z = sp.symbols("z")
    surfaces = derive_stokes_surfaces(sp.exp(1 / z), sp.S.One, z)
    assert sum(s.kind is TransitionKind.ANTI_STOKES for s in surfaces) == 2
    assert sum(s.kind is TransitionKind.STOKES for s in surfaces) == 2


def test_inverse_square_has_four_dominance_sectors():
    z = sp.symbols("z")
    atlas = dominance_sector_atlas(sp.exp(1 / z**2), sp.S.One, z)
    assert atlas.coverage.certified
    assert len(atlas.sectors) == 4
    assert {s.relation for s in atlas.sectors} == {
        SectorialDominance.SMALLER,
        SectorialDominance.LARGER,
    }
    assert len(atlas.connections) == 4
    assert all(c.multiplier is None for c in atlas.connections)


def test_exponentially_small_is_beyond_decay_sector():
    z = sp.symbols("z")
    decay = ComplexSector(0, sp.pi / 2)
    growth = ComplexSector(sp.pi, sp.pi / 2)
    assert certify_beyond_all_orders(sp.exp(-1 / z), z, sector=decay).certified
    assert not certify_beyond_all_orders(sp.exp(-1 / z), z, sector=growth).certified


def test_stokes_aware_expansion_keeps_nonzero_exponential_correction():
    z = sp.symbols("z")
    sector = ComplexSector(0, sp.pi / 2)
    result = stokes_aware_expansion(
        1 + z, ((sp.Symbol("C"), sp.exp(-1 / z)),), z, sector=sector
    )
    assert result.certified
    assert result.corrections[0].beyond_all_orders
    assert result.corrections[0].scale != 0


def test_unified_real_path_retains_oscillatory():
    x, e = sp.symbols("x e", positive=True)
    result = unified_stratified_expand(sp.sin(x / e), (x, e), order=3)
    assert result.certified


def test_unified_sectorial_sum_returns_dominance_atlas():
    z = sp.symbols("z")
    result = unified_stratified_expand(
        sp.exp(1 / z) + sp.exp(-1 / z), z, sector=ComplexSector(0, sp.pi / 4)
    )
    assert result.coverage.certified


def test_unsupported_phase_difference_declines_atlas():
    z = sp.symbols("z")
    atlas = dominance_sector_atlas(sp.exp(sp.sin(1 / z)), sp.S.One, z)
    assert not atlas.coverage.certified
    assert atlas.obligations
