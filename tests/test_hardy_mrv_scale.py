def test_gamma_ratio_cancellation_is_seen_as_polynomial_hardy_growth():
    import sympy as sp

    from asymptotic.mrv import mrv_decomposition

    n = sp.symbols("n", positive=True)
    ratio = sp.gamma(n + 3) / sp.gamma(n)
    result = mrv_decomposition(ratio + sp.exp(n), n)
    assert result.representative == sp.exp(n)
    assert any(ratio in cls.members and n in cls.members for cls in result.classes)


def test_binomial_and_factorial_products_use_net_stirling_growth():
    import sympy as sp

    from asymptotic.mrv import mrv_decomposition

    n = sp.symbols("n", positive=True)
    central = sp.binomial(2 * n, n)
    central_result = mrv_decomposition(central + sp.exp(n), n)
    assert any(
        central in cls.members and sp.exp(n) in cls.members for cls in central_result.classes
    )

    product = sp.factorial(n) * sp.factorial(n + 1)
    product_result = mrv_decomposition(product + sp.exp(n), n)
    assert product_result.representative == product


def test_unevaluated_pochhammer_uses_gamma_ratio_normalization():
    import sympy as sp

    from asymptotic.mrv import mrv_decomposition

    n = sp.symbols("n", positive=True)
    pochhammer = sp.RisingFactorial(n, n, evaluate=False)
    result = mrv_decomposition(pochhammer + sp.exp(n), n)
    assert result.representative == pochhammer


def test_fixed_order_binomial_normalizes_to_polynomial_scale():
    import sympy as sp

    from asymptotic.mrv import mrv_decomposition

    n = sp.symbols("n", positive=True)
    value = sp.binomial(n, 3)
    result = mrv_decomposition(value + sp.exp(n), n)
    assert result.representative == sp.exp(n)
