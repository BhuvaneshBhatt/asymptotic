import sympy as sp

from asymptotic import limit, one_sided_limit
from asymptotic.recurrence_simplification import simplify_recurrences


def test_gamma_ratios_and_pole_cancellation():
    x = sp.Symbol("x")
    e = sp.gamma(x + 3) / sp.gamma(x)
    assert simplify_recurrences(e) == x**3 + 3 * x**2 + 2 * x
    r = limit(e, x, -1, return_result=True)
    assert r.value == 0 and r.expression == e
    assert r.evidence[0].method == "gamma_exact_integer_recurrence"


def test_polygamma_exact_cancellations_and_unknown_parameter_strata():
    x = sp.Symbol("x", positive=True)
    a = sp.Symbol("a")
    for m in range(4):
        e = (
            sp.polygamma(m, x + 1)
            - sp.polygamma(m, x)
            - (-1) ** m * sp.factorial(m) / x ** (m + 1)
        )
        assert simplify_recurrences(e) == 0
    e = sp.polygamma(1, a + 1) - sp.polygamma(1, a)
    assert simplify_recurrences(e) == e
    assert simplify_recurrences(e, assumptions=sp.Ne(a, 0)) == -1 / a**2


def test_upper_gamma_same_argument_and_branch_preservation():
    a = sp.Symbol("a")
    x = sp.Symbol("x", nonzero=True)
    e = sp.uppergamma(a + 1, x) - a * sp.uppergamma(a, x)
    assert simplify_recurrences(e) == x**a * sp.exp(-x)
    y = sp.Symbol("y")
    other = sp.uppergamma(a + 1, y) - a * sp.uppergamma(a, x)
    assert simplify_recurrences(other) == other
    # Use the same principal power, never split (-x)**a across its cut.
    e = sp.uppergamma(a + 1, -x) - a * sp.uppergamma(a, -x)
    assert simplify_recurrences(e) == (-x) ** a * sp.exp(x)


def test_expint_order_shift_and_zero_denominator_guard():
    x = sp.Symbol("x", nonzero=True)
    a = sp.Symbol("a", nonzero=True)
    e = a * sp.expint(a + 1, x) + x * sp.expint(a, x) - sp.exp(-x)
    assert simplify_recurrences(e) == 0
    b = sp.Symbol("b")
    e = sp.expint(b + 1, x) + sp.expint(b, x)
    assert simplify_recurrences(e) == e


def test_bessel_hankel_and_modified_signs():
    x = sp.Symbol("x", nonzero=True)
    v = sp.Symbol("v")
    for head in (sp.besselj, sp.bessely, sp.hankel1, sp.hankel2):
        e = head(v, x) + head(v + 2, x) - 2 * (v + 1) * head(v + 1, x) / x
        assert simplify_recurrences(e) == 0
    assert (
        simplify_recurrences(
            sp.besseli(v + 2, x)
            - sp.besseli(v, x)
            + 2 * (v + 1) * sp.besseli(v + 1, x) / x
        )
        == 0
    )
    assert (
        simplify_recurrences(
            sp.besselk(v + 2, x)
            - sp.besselk(v, x)
            - 2 * (v + 1) * sp.besselk(v + 1, x) / x
        )
        == 0
    )
    for head in (sp.jn, sp.yn):
        assert (
            simplify_recurrences(
                head(v + 2, x) + head(v, x) - (2 * v + 3) * head(v + 1, x) / x
            )
            == 0
        )


def test_bessel_punctured_context_and_standalone_guard():
    x = sp.Symbol("x")
    v = sp.Symbol("v")
    e = sp.besselj(v, x) + sp.besselj(v + 2, x) - 2 * (v + 1) * sp.besselj(v + 1, x) / x
    assert simplify_recurrences(e) == e
    assert limit(e, x, 0) == 0
    a = sp.Symbol("a")
    e = sp.besselj(v, a) + sp.besselj(v + 2, a)
    assert simplify_recurrences(e, variables=x, target=0) == e


def test_hurwitz_positive_shift_and_cut_guard():
    s = sp.Symbol("s")
    a = sp.Symbol("a", positive=True)
    e = sp.zeta(s, a + 1) - sp.zeta(s, a) + a ** (-s)
    assert simplify_recurrences(e) == 0
    x = sp.Symbol("x")
    e = sp.zeta(s, x + 1) - sp.zeta(s, x)
    assert simplify_recurrences(e) == e
    assert simplify_recurrences(e, variables=x, target=sp.oo) == -(x ** (-s))
    assert simplify_recurrences(e, variables=x, target=0) == e


def test_structural_budgets_and_large_compact_parameters():
    x = sp.Symbol("x", positive=True)
    e = sp.gamma(x + 1000) / sp.gamma(x)
    assert simplify_recurrences(e) == e
    e = sp.gamma(x + 3) / sp.gamma(x)
    assert simplify_recurrences(e, budget_seconds=0) == e
    assert simplify_recurrences(e, max_steps=1) == e
    assert simplify_recurrences(e, max_ops=1) == e
    z = (x + 1) ** 10000
    e = sp.gamma(z) + sp.gamma(z + 1)
    assert simplify_recurrences(e) == e


def test_loggamma_regularized_heads_and_not_rewritten():
    x = sp.Symbol("x", positive=True)
    for e in (
        sp.loggamma(x + 1) - sp.loggamma(x),
        sp.gamma(x + sp.sin(x)) / sp.gamma(x),
        sp.gamma(x + sp.exp(-x)) - sp.gamma(x),
    ):
        assert simplify_recurrences(e) == e


def test_original_expression_and_identity_sided_limit():
    x = sp.Symbol("x")
    e = sp.gamma(x + 2) / (x * sp.gamma(x))
    r = one_sided_limit(e, x, 0, direction="+", return_result=True)
    assert r.value == 1
    assert any(a.method == "integer_shift_recurrence_gamma" for a in r.evidence)


def test_limit_prepass_preserves_unresolved_fixed_parameter_poles():
    x = sp.Symbol("x")
    a = sp.Symbol("a")
    s = sp.Symbol("s")
    e = sp.gamma(a + 1) - a * sp.gamma(a)
    assert (
        simplify_recurrences(e, variables=x, target=0, require_defined_germ=True) == e
    )
    e = sp.zeta(s, x + 2) - sp.zeta(s, x + 1)
    assert (
        simplify_recurrences(e, variables=x, target=0, require_defined_germ=True) == e
    )
    a = sp.Symbol("a")
    v = sp.Symbol("v")
    z = sp.Symbol("z", nonzero=True)
    e = (
        sp.besselj(v, z) + sp.besselj(v + 2, z) - 2 * (v + 1) * sp.besselj(v + 1, z) / z
    ) / a
    assert (
        simplify_recurrences(e, variables=z, target=1, require_defined_germ=True) == e
    )


def test_powered_gamma_stirling_case_0235_keeps_its_fast_certificate():
    k = sp.Symbol("k", positive=True)
    e = (
        5 ** (-2 * k - 1)
        * sp.gamma(k + 1) ** k
        * sp.gamma(k + sp.Rational(3, 2))
        * sp.gamma(k + 2) ** (-k - 1)
        / sp.gamma(k + sp.Rational(1, 2))
    )
    r = limit(e, k, sp.oo, return_result=True)
    assert r.value == 0
    assert r.evidence[0].method == "shifted_stirling_product_remainder"


def test_coefficient_collection_does_not_term_counts():
    x = sp.Symbol("x", positive=True)
    a = sp.symbols("a:48")
    left = sp.prod(a[2 * j] + a[2 * j + 1] for j in range(12))
    right = sp.prod(a[24 + 2 * j] + a[25 + 2 * j] for j in range(12))
    e = left * sp.gamma(x + 1) + right * sp.gamma(x)
    r = simplify_recurrences(e, return_result=True)
    assert r.expression == e
    assert r.stopped == "coefficient expansion cap"
