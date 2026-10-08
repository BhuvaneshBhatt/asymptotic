import sympy as sp

from asymptotic.complex_domain import ComplexBranchMetadata, ComplexSector
from asymptotic.expansion_contract import (
    AsymptoticExpansion,
    expansion_certificate,
    expansion_is_certified,
)
from asymptotic.multivariate_expansion import multivariate_expansion
from asymptotic.multivariate_transseries import certified_transseries_expand
from asymptotic.oscillatory_scales import certified_oscillatory_expand
from asymptotic.power_expansion import certified_power_expand
from asymptotic.stokes_transseries import stokes_aware_expansion
from asymptotic.stratified_expansion import stratified_expand


def test_power_result_satisfies_unified_contract():
    x, e = sp.symbols("x e", positive=True)
    result = certified_power_expand(1 / (1 + x / e), small=x, large=e, order=3)
    assert isinstance(result, AsymptoticExpansion)
    cert = expansion_certificate(result)
    assert cert.certified
    assert cert.source_kind == "CertifiedPowerExpansion"
    assert cert.representation == result.approximation


def test_logexp_result_satisfies_unified_contract():
    x, e = sp.symbols("x e", positive=True)
    result = certified_transseries_expand(sp.log(1 + x / e), small=x, large=e, order=3)
    cert = expansion_certificate(result)
    assert cert.certified
    assert cert.regime == result.regime


def test_oscillatory_result_satisfies_unified_contract():
    x, e = sp.symbols("x e", positive=True)
    result = certified_oscillatory_expand(sp.sin(x / e), small=e, large=x, order=3)
    cert = expansion_certificate(result)
    assert cert.certified
    assert cert.representation == result.representation


def test_weighted_expansion_carries_uniform_remainder():
    x, y = sp.symbols("x y", real=True)
    result = multivariate_expansion(1 / (1 + x**2 + y**2), (x, y), (0, 0), order=3)
    cert = expansion_certificate(result)
    assert cert.remainder is not None
    assert cert.certified == expansion_is_certified(result)


def test_stratified_container_is_not_misrepresented_as_one_branch():
    x, e = sp.symbols("x e", positive=True)
    result = stratified_expand(sp.sin(x / e), (x, e), order=3)
    # A stratification is an atlas/container, not one expansion branch; its branches
    # individually satisfy the expansion protocol.
    assert result.certified
    assert all(isinstance(branch, AsymptoticExpansion) for branch in result.branches)


def test_stokes_result_exposes_full_represented_expression():
    z = sp.symbols("z")
    sector = ComplexSector(0, sp.pi / 2)
    result = stokes_aware_expansion(
        1 + z, ((2, sp.exp(-1 / z)),), z, sector=sector, branch=ComplexBranchMetadata()
    )
    cert = expansion_certificate(result)
    assert isinstance(result, AsymptoticExpansion)
    assert sp.simplify(cert.expression - (1 + z + 2 * sp.exp(-1 / z))) == 0
    assert cert.certified


def test_stratification_certificate_recurses_into_branches():
    x, e = sp.symbols("x e", positive=True)
    result = stratified_expand(sp.sin(x / e), (x, e), order=3)
    cert = expansion_certificate(result)
    assert len(cert.children) == len(result.branches)
    assert cert.certified
