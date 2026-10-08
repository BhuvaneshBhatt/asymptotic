import sympy as sp

from asymptotic.multivariate_matched import (
    composite_expansion,
    linear_pde_layer_balance,
    match_overlap,
)


def test_anisotropic_pde_layer_balance():
    x, y, e = sp.symbols("x y e", positive=True)
    u = sp.Function("u")(x, y)
    # e*u_xx + u_x has alpha_x=1; y remains outer.
    r = linear_pde_layer_balance(
        e * sp.diff(u, x, 2) + sp.diff(u, x) + sp.diff(u, y), (u), (x, y), e, (1, 0)
    )
    assert r.certified
    assert (
        len(r.dominant_indices) == 2
    )  # the x-layer terms dominate the tangential derivative


def test_non_distinguished_scaling_refused():
    x, e = sp.symbols("x e", positive=True)
    u = sp.Function("u")(x)
    r = linear_pde_layer_balance(
        e * sp.diff(u, x, 2) + sp.diff(u, x), u, (x,), e, (sp.Rational(1, 2),)
    )
    assert not r.certified


def test_overlap_matching():
    x, X = sp.symbols("x X", positive=True)
    r = match_overlap(1 + x, 1 + sp.exp(-X), x, X)
    assert r.certified and r.discrepancy == 0


def test_overlap_mismatch_is_explicit():
    x, X = sp.symbols("x X", positive=True)
    r = match_overlap(2 + x, 1 + sp.exp(-X), x, X)
    assert not r.certified and r.discrepancy == 1


def test_composite_subtracts_common_part():
    x, X = sp.symbols("x X", positive=True)
    cert = match_overlap(1 + x, 1 + sp.exp(-X), x, X)
    r = composite_expansion(
        1 + x, (1 + sp.exp(-X),), (1,), matching_certificates=(cert,)
    )
    assert r.certified and sp.simplify(r.expression - (1 + x + sp.exp(-X))) == 0
