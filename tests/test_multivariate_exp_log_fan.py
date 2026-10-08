import sympy as sp

from asymptotic.relative_growth import (
    automatic_exp_log_valuation_fan,
    extract_growth_atoms,
)


def test_nested_exp_log_atoms_are_extracted():
    x, y = sp.symbols("x y", positive=True)
    atoms = extract_growth_atoms(
        sp.exp(-1 / x) + y ** sp.Rational(3, 2) + sp.log(y) + sp.log(sp.exp(-1 / x)),
        (x, y),
    )
    kinds = {a.kind for a in atoms}
    assert {"exp", "power", "log", "variable"} <= kinds


def test_cross_scale_expression_generates_competing_symbolic_cells():
    x, y = sp.symbols("x y", positive=True)
    fan = automatic_exp_log_valuation_fan(sp.exp(-1 / x) + y**2 + sp.log(y), (x, y))
    assert not fan.coverage.certified
    assert "comparability partition" in fan.coverage.statement
    assert len(fan.cells) >= 2
    assert all(c.valuation_cones for c in fan.cells)


def test_rational_specialized_growth_cell_can_feed_blowup():
    x, y = sp.symbols("x y", positive=True)
    fan = automatic_exp_log_valuation_fan(x + y**2, (x, y))
    assert not fan.coverage.certified
    assert fan.cells
