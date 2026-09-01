import sympy as sp

from asymptotic import (
    discover_scale,
    multiseries,
    nested_expansion,
)
from asymptotic.periodic import OscillationKind, periodic_decomposition


def test_periodic_decomposition_x_sin_x():
    x = sp.symbols("x", positive=True)
    dec = periodic_decomposition(x * sp.sin(x), x)
    assert dec.has_oscillation
    assert sp.simplify(dec.envelope - x) == 0
    f = dec.factors[0]
    assert f.period == 2 * sp.pi
    assert (f.lower_bound, f.upper_bound) == (-1, 1)
    assert f.kind is OscillationKind.PERIODIC
    assert sp.simplify(dec.reconstruct() - x * sp.sin(x)) == 0


def test_periodic_decomposition_transformed_phase_and_outer_composition():
    x = sp.symbols("x", positive=True)
    dec = periodic_decomposition(sp.sin(sp.log(x)) / x, x)
    assert sp.simplify(dec.envelope - 1 / x) == 0
    assert sp.simplify(dec.factors[0].phase - sp.log(x)) == 0

    dec2 = periodic_decomposition(sp.exp(sp.sin(x)) / x, x)
    assert sp.simplify(dec2.envelope - 1 / x) == 0
    f = dec2.factors[0]
    assert f.lower_bound == sp.exp(-1)
    assert f.upper_bound == sp.E


def test_scale_discovery_does_not_promote_periodic_phase_to_growth_scale():
    x = sp.symbols("x", positive=True)
    scale = discover_scale(sp.sin(sp.log(x)) / x, x)
    assert any(sp.simplify(e - 1 / x) == 0 for e in scale.exprs)
    assert not any(sp.simplify(e - 1 / sp.log(x)) == 0 for e in scale.exprs)


def test_multiseries_keeps_oscillation_as_coefficient():
    x = sp.symbols("x", positive=True)
    ms = multiseries(sp.sin(sp.log(x)) / x, x)
    lead = ms.leading_term()
    assert sp.simplify(lead - sp.sin(sp.log(x)) / x) == 0


def test_multiseries_differentiate_and_integrate_rediscover_scale():
    x = sp.symbols("x", positive=True)
    ms = multiseries(sp.log(x), x)
    d = ms.differentiate()
    assert sp.simplify(d.expr - 1 / x) == 0
    assert sp.simplify(d.leading_term() - 1 / x) == 0

    one_over_x = multiseries(1 / x, x)
    integ = one_over_x.integrate()
    assert sp.simplify(integ.expr - sp.log(x)) == 0
    assert any(sp.simplify(e - 1 / sp.log(x)) == 0 for e in integ.scale.exprs)


def test_nested_calculus_preserves_exact_expression():
    x = sp.symbols("x", positive=True)
    n = nested_expansion(sp.log(x), x, depth=0)
    assert sp.simplify(n.differentiate().expr - 1 / x) == 0
    m = nested_expansion(1 / x, x, depth=0)
    assert sp.simplify(m.integrate().expr - sp.log(x)) == 0
