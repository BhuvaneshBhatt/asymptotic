import sympy as sp

from asymptotic.branch_monodromy import MonodromyState, WindingState, monodromy_cluster
from asymptotic.complex_cluster_geometry import branched_blowup_atlas
from asymptotic.local_germ_algebra import ComplexLocalGerm


def test_log_repeated_winding_family_is_infinite():
    x, y = sp.symbols("x y", real=True)
    z = x + sp.I * y
    charts, cov = branched_blowup_atlas(sp.log(z), (x, y), (0, 0))
    family, finite, _state = monodromy_cluster(sp.log(z), charts[0].divisors)
    assert cov.certified and not finite
    assert not cov.missing_regimes
    assert any(s in family.free_symbols for s in [sp.Symbol("_winding", integer=True)])


def test_fractional_power_monodromy_multiplier():
    state = MonodromyState((WindingState.turns(0, 1, sp.Rational(1, 2)),))
    assert sp.simplify(state.winding_for(0).multiplier + 1) == 0


def test_complex_germ_carries_monodromy_state():
    state = MonodromyState((WindingState.turns(0, 1, sp.Rational(1, 2)),))
    g = ComplexLocalGerm(sp.Rational(1, 2)).with_monodromy(state)
    assert g.monodromy_state is state
    assert g.branch_index == 1
