import sympy as sp

from asymptotic.matched import matched_asymptotic_expansion


def test_left_layer_composite_is_certified():
    x = sp.Symbol("x", real=True)
    eps = sp.Symbol("eps", positive=True)
    y = sp.Function("y")
    result = matched_asymptotic_expansion(
        eps * sp.diff(y(x), x, 2) + sp.diff(y(x), x),
        y(x),
        eps,
        conditions=(sp.Eq(y(0), 0), sp.Eq(y(1), 1)),
    )
    branch = result.branches[0]
    assert branch.layer.side == "left"
    assert branch.layer.scale == eps
    assert sp.simplify(branch.composite - (1 - sp.exp(-x / eps))) == 0
    assert branch.residual == 0
    assert branch.condition_residuals[0] == 0
    assert sp.limit(branch.condition_residuals[1], eps, 0, dir="+") == 0
    assert branch.verified is True


def test_right_layer_selected_from_decay_direction():
    x = sp.Symbol("x", real=True)
    eps = sp.Symbol("eps", positive=True)
    y = sp.Function("y")
    result = matched_asymptotic_expansion(
        eps * sp.diff(y(x), x, 2) - sp.diff(y(x), x),
        y(x),
        eps,
        conditions=(sp.Eq(y(0), 0), sp.Eq(y(1), 1)),
    )
    branch = result.branches[0]
    assert branch.layer.side == "right"
    assert sp.simplify(branch.composite - sp.exp((x - 1) / eps)) == 0
    assert branch.verified is True


def test_overlap_is_subtracted_in_composite():
    x = sp.Symbol("x", real=True)
    eps = sp.Symbol("eps", positive=True)
    y = sp.Function("y")
    result = matched_asymptotic_expansion(
        eps * sp.diff(y(x), x, 2) + sp.diff(y(x), x) - 1,
        y(x),
        eps,
        conditions=(sp.Eq(y(0), 0), sp.Eq(y(1), 0)),
        order=1,
    )
    branch = result.branches[0]
    assert branch.common_part != 0
    assert sp.simplify(branch.composite - (x - 1 + sp.exp(-x / eps))) == 0
    assert branch.verified is True
