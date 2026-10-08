import sympy as sp

from asymptotic.multivariate_matched import linear_pde_layer_balance


def test_layer_balance_keeps_exact_rational_ordering():
    x = sp.symbols("x")
    eps = sp.symbols("eps", positive=True)
    u = sp.Function("u")(x)
    equation = eps ** sp.Rational(10**20 + 1, 10**20) * sp.diff(u, x) + eps * u
    result = linear_pde_layer_balance(equation, u, (x,), eps, (sp.S.Zero,))
    assert result.dominant_indices == (0,)
    assert not result.certified
    assert (
        "distinguished balance requires at least two dominant terms"
        in result.obligations
    )
