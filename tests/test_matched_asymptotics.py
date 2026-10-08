import sympy as sp

from asymptotic.matched import MatchedAsymptoticResult, matched_asymptotic_expansion


def test_left_endpoint_layer_is_discovered_and_matched():
    x = sp.Symbol("x", positive=True)
    eps = sp.Symbol("eps", positive=True)
    y = sp.Function("y")

    result = matched_asymptotic_expansion(
        eps * sp.diff(y(x), x, 2) + sp.diff(y(x), x),
        y(x),
        eps,
        conditions=(sp.Eq(y(0), 0), sp.Eq(y(1), 1)),
    )

    assert isinstance(result, MatchedAsymptoticResult)
    assert result.full_order == 2
    assert result.reduced_order == 1
    assert len(result.branches) == 1
    branch = result.branches[0]
    assert branch.layer.side == "left"
    assert branch.layer.exponent == 1
    assert branch.layer.scale == eps
    assert branch.outer_approximation == 1
    assert branch.inner_approximation == 1 - sp.exp(-branch.layer.stretched_variable)
    assert sp.simplify(branch.composite - (1 - sp.exp(-x / eps))) == 0
    assert branch.residual == 0
    assert branch.verified is True


def test_right_endpoint_layer_is_selected_by_decay_direction():
    x = sp.Symbol("x", real=True)
    eps = sp.Symbol("eps", positive=True)
    y = sp.Function("y")

    result = matched_asymptotic_expansion(
        eps * sp.diff(y(x), x, 2) - sp.diff(y(x), x),
        y(x),
        eps,
        conditions=(sp.Eq(y(0), 0), sp.Eq(y(1), 1)),
    )

    assert len(result.branches) == 1
    branch = result.branches[0]
    assert branch.layer.side == "right"
    assert branch.layer.exponent == 1
    assert sp.simplify(branch.composite - sp.exp((x - 1) / eps)) == 0
    assert branch.verified is True


def test_first_correction_uses_overlap_constant_only():
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
    X = branch.layer.stretched_variable
    assert branch.matching_terms == (-1, X)
    assert sp.simplify(branch.inner_approximation - (eps * X - 1 + sp.exp(-X))) == 0
    assert sp.simplify(branch.composite - (x - 1 + sp.exp(-x / eps))) == 0
    assert branch.verified is True


def test_newton_style_balance_discovers_square_root_layer():
    x = sp.Symbol("x", real=True)
    eps = sp.Symbol("eps", positive=True)
    y = sp.Function("y")

    result = matched_asymptotic_expansion(
        eps * sp.diff(y(x), x, 2) + x * sp.diff(y(x), x),
        y(x),
        eps,
        conditions=(sp.Eq(y(0), 0), sp.Eq(y(1), 1)),
    )

    branch = result.branches[0]
    assert branch.layer.side == "left"
    assert branch.layer.exponent == sp.Rational(1, 2)
    assert branch.layer.scale == sp.sqrt(eps)
    expected = sp.erf(x / sp.sqrt(2 * eps))
    assert sp.simplify(branch.composite - expected) == 0
    assert branch.verified is True


def test_nonlinear_problem_is_rejected_heuristically_scaled():
    x = sp.Symbol("x")
    eps = sp.Symbol("eps", positive=True)
    y = sp.Function("y")

    try:
        matched_asymptotic_expansion(
            eps * sp.diff(y(x), x, 2) + y(x) * sp.diff(y(x), x),
            y(x),
            eps,
            conditions=(sp.Eq(y(0), 0), sp.Eq(y(1), 1)),
        )
    except NotImplementedError as exc:
        assert "linear" in str(exc)
    else:
        raise AssertionError("nonlinear singular BVP was accepted")


def test_fractional_layer_uses_ramified_higher_matching():
    x = sp.Symbol("x", real=True)
    eps = sp.Symbol("eps", positive=True)
    y = sp.Function("y")

    result = matched_asymptotic_expansion(
        eps * sp.diff(y(x), x, 2) + x * sp.diff(y(x), x) - x,
        y(x),
        eps,
        conditions=(sp.Eq(y(0), 0), sp.Eq(y(1), 0)),
        order=1,
    )

    branch = result.branches[0]
    X = branch.layer.stretched_variable
    assert branch.layer.exponent == sp.Rational(1, 2)
    assert branch.inner_hierarchy.gauges == (1, sp.sqrt(eps), eps)
    assert branch.matching_terms[0] == -1
    assert sp.simplify(branch.matching_terms[1] - X) == 0
    assert branch.complete is True
