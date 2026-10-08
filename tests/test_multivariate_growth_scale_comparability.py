import sympy as sp

from asymptotic.relative_growth import (
    GrowthScaleComparison,
    certify_exp_log_fan,
    prove_growth_comparison,
)


def test_flat_exponential_is_smaller_power_example():
    x = sp.symbols("x", positive=True)
    assert (
        prove_growth_comparison(sp.exp(-1 / x), x**3, x).relation
        == GrowthScaleComparison.LESS
    )


def test_power_is_smaller_than_inverse_log_power():
    x = sp.symbols("x", positive=True)
    assert (
        prove_growth_comparison(x**2, sp.Abs(sp.log(x)) ** -3, x).relation
        == GrowthScaleComparison.LESS
    )


def test_nested_exp_log_comparison():
    x = sp.symbols("x", positive=True)
    a = sp.exp(-sp.exp(1 / x))
    b = sp.exp(-1 / x)
    assert prove_growth_comparison(a, b, x).relation == GrowthScaleComparison.LESS


def test_certified_single_scale_fan_is_exhaustive():
    x = sp.symbols("x", positive=True)
    fan, proofs = certify_exp_log_fan(
        sp.exp(-1 / x) + x**2 + sp.Abs(sp.log(x)) ** -2, (x,)
    )
    assert fan.coverage.certified
    assert proofs and all(p.certified for p in proofs)
