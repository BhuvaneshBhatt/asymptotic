import sympy as sp

from asymptotic.complex_domain import ComplexBranchMetadata, ComplexSector
from asymptotic.sectorial_transseries import (
    SectorialDominance,
    branch_aware_rewrite,
    compare_sectorial_exponentials,
)


def test_exp_inverse_z_small_on_right_sector():
    z = sp.Symbol("z")
    sector = ComplexSector(0, sp.pi / 2)
    result = compare_sectorial_exponentials(
        sp.exp(-1 / z), sp.S.Exp1**0, z, sector=sector
    )
    # Comparator requires exponential scales on both sides.
    result = compare_sectorial_exponentials(sp.exp(-1 / z), sp.exp(0), z, sector=sector)
    assert result.certified
    assert result.relation is SectorialDominance.SMALLER


def test_exp_inverse_z_large_on_left_sector():
    z = sp.Symbol("z")
    sector = ComplexSector(sp.pi, sp.pi / 2)
    result = compare_sectorial_exponentials(sp.exp(-1 / z), sp.exp(0), z, sector=sector)
    assert result.certified
    assert result.relation is SectorialDominance.LARGER


def test_crossing_stokes_rays_declines():
    z = sp.Symbol("z")
    sector = ComplexSector(0, 3 * sp.pi / 2)
    result = compare_sectorial_exponentials(sp.exp(-1 / z), sp.exp(0), z, sector=sector)
    assert not result.certified
    assert result.relation is SectorialDominance.UNKNOWN
    assert result.obligations


def test_inverse_square_has_sector_dependent_dominance():
    z = sp.Symbol("z")
    positive = ComplexSector(0, sp.pi / 4)
    negative = ComplexSector(sp.pi / 2, sp.pi / 4)
    a = compare_sectorial_exponentials(sp.exp(1 / z**2), sp.exp(0), z, sector=positive)
    b = compare_sectorial_exponentials(sp.exp(1 / z**2), sp.exp(0), z, sector=negative)
    assert a.relation is SectorialDominance.LARGER
    assert b.relation is SectorialDominance.SMALLER
    assert a.certified and b.certified


def test_pure_imaginary_inverse_phase_declines_on_open_sector():
    z = sp.Symbol("z")
    sector = ComplexSector(0, sp.pi / 3)
    result = compare_sectorial_exponentials(
        sp.exp(sp.I / z), sp.exp(0), z, sector=sector
    )
    assert not result.certified
    assert result.relation is SectorialDominance.UNKNOWN


def test_branch_aware_log_records_nonprincipal_sheet():
    z = sp.Symbol("z")
    sector = ComplexSector(0, sp.pi / 2)
    branch = ComplexBranchMetadata(
        logarithm_branch=1, branch_cuts=(sp.pi,), principal=False
    )
    result = branch_aware_rewrite(sp.log(z), z, sector=sector, branch=branch)
    assert result.certified
    assert sp.simplify(result.representation - (sp.log(z) + 2 * sp.pi * sp.I)) == 0


def test_fractional_power_uses_selected_log_branch():
    z = sp.Symbol("z")
    sector = ComplexSector(0, sp.pi / 2)
    branch = ComplexBranchMetadata(
        logarithm_branch=1, branch_cuts=(sp.pi,), principal=False
    )
    result = branch_aware_rewrite(
        z ** sp.Rational(1, 3), z, sector=sector, branch=branch
    )
    expected = sp.exp((sp.log(z) + 2 * sp.pi * sp.I) / 3)
    assert result.certified
    assert sp.simplify(result.representation / expected - 1) == 0


def test_sector_intersecting_branch_cut_declines():
    z = sp.Symbol("z")
    sector = ComplexSector(sp.pi, sp.pi / 2)
    branch = ComplexBranchMetadata(branch_cuts=(sp.pi,))
    result = branch_aware_rewrite(sp.log(z), z, sector=sector, branch=branch)
    assert not result.certified
    assert result.obligations[0].kind.value == "complex_branch"


def test_branch_cut_on_sector_boundary_is_allowed_for_open_sector():
    z = sp.Symbol("z")
    sector = ComplexSector(0, sp.pi)
    branch = ComplexBranchMetadata(branch_cuts=(sp.pi / 2,))
    result = branch_aware_rewrite(sp.log(z), z, sector=sector, branch=branch)
    assert result.certified
