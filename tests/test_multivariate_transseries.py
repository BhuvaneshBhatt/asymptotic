import sympy as sp

from asymptotic.context import AsymptoticGrowthComparison
from asymptotic.multivariate_transseries import (
    CertifiedTransseriesExpansion,
    MultivariateGrowthScale,
    certified_transseries_expand,
)
from asymptotic.stratified_expansion import stratified_expand

x, e = sp.symbols("x e", positive=True)


def trans_branches(r):
    return [b for b in r.branches if isinstance(b, CertifiedTransseriesExpansion)]


def test_exponential_large_ratio_completes_missing_cell():
    r = stratified_expand(sp.exp(x / e), (x, e), order=4)
    assert r.certified
    ts = trans_branches(r)
    assert ts
    assert any(b.approximation.has(sp.exp(x / e)) for b in ts)


def test_log_large_ratio_has_log_scale_and_power_corrections():
    r = stratified_expand(sp.log(1 + x / e), (x, e), order=4)
    assert r.certified
    expected = sp.log(x / e) + e / x - e**2 / (2 * x**2) + e**3 / (3 * x**3)
    assert any(
        sp.simplify(sp.expand_log(b.approximation - expected, force=True)) == 0
        for b in trans_branches(r)
    )


def test_nested_exponential_is_finite_height_scale():
    r = stratified_expand(sp.exp(sp.exp(x / e)), (x, e), order=3)
    assert r.certified
    b = [b for b in trans_branches(r) if b.approximation.has(sp.exp(sp.exp(x / e)))]
    assert b and max((s.exponential_height for s in b[0].scales), default=0) >= 2


def test_logarithmic_hierarchy_comparison():
    u = sp.Symbol("u", positive=True)
    logscale = MultivariateGrowthScale.from_expression(sp.log(1 / u), u)
    power = MultivariateGrowthScale.from_expression(1 / u, u)
    exponential = MultivariateGrowthScale.from_expression(sp.exp(1 / u), u)
    assert logscale.compare(power) is AsymptoticGrowthComparison.SMALLER
    assert exponential.compare(power) is AsymptoticGrowthComparison.LARGER


def test_nested_exponential_dominates_single_exponential():
    u = sp.Symbol("u", positive=True)
    a = MultivariateGrowthScale.from_expression(sp.exp(sp.exp(1 / u)), u)
    b = MultivariateGrowthScale.from_expression(sp.exp(1 / u), u)
    assert a.compare(b) is AsymptoticGrowthComparison.LARGER


def test_oscillatory_reciprocal_cell_is_completed_by():
    from asymptotic.oscillatory_scales import CertifiedOscillatoryExpansion

    r = stratified_expand(sp.sin(x / e), (x, e), order=3)
    assert r.certified
    assert any(
        isinstance(branch, CertifiedOscillatoryExpansion) for branch in r.branches
    )


def test_direct_transseries_declines_non_le_atom():
    assert (
        certified_transseries_expand(sp.sin(x / e), small=e, large=x, order=3) is None
    )
