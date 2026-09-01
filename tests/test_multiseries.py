import sympy as sp

from asymptotic import (
    AsymptoticScale,
    multiseries,
)
from asymptotic.multiseries import MultiseriesTerm, multiply_term_lists


def native_multiseries(*args, **kwargs):
    """Construct a multiseries with the compatibility fallback disabled."""

    kwargs["allow_series_fallback"] = False
    return multiseries(*args, **kwargs)


def test_two_scale_recursive_expansion():
    x = sp.symbols("x", positive=True)
    scale = AsymptoticScale.from_exprs(x, [1 / sp.log(x), 1 / x])
    ms = multiseries(sp.exp(1 / x + 1 / sp.log(x)), x, scale=scale, terms=4)
    top = ms.terms(3)
    assert [t.exponent for t in top] == [0, 1, 2]
    assert sp.simplify(top[0].coefficient - sp.exp(1 / sp.log(x))) == 0
    assert sp.simplify(top[1].coefficient - sp.exp(1 / sp.log(x))) == 0
    assert sp.simplify(top[2].coefficient - sp.exp(1 / sp.log(x)) / 2) == 0

    coeff = ms.coefficient_series(0).terms(3)
    assert [t.exponent for t in coeff] == [0, 1, 2]
    assert [sp.simplify(t.coefficient) for t in coeff] == [1, 1, sp.Rational(1, 2)]


def test_heap_product_frontier():
    a = [MultiseriesTerm(0, 2), MultiseriesTerm(sp.log(2), 3), MultiseriesTerm(1, 4)]
    b = [MultiseriesTerm(0, 5), MultiseriesTerm(1, 7)]
    out = multiply_term_lists(a, b, 4)
    assert out[0] == MultiseriesTerm(0, 10)
    assert sp.simplify(out[1].exponent - sp.log(2)) == 0
    assert out[1].coefficient == 15


def test_explicit_scale_logarithmic_example():
    x = sp.symbols("x", positive=True)
    t1 = 1 / sp.log(x)
    t2 = 1 / x
    ms = multiseries(sp.log(1 - t1 - t2), x, scale=[t1, t2], terms=3)
    top = ms.terms(2)
    assert top[0].exponent == 0
    assert sp.simplify(top[0].coefficient - sp.log(1 - t1)) == 0
    assert top[1].exponent == 1
    assert sp.simplify(top[1].coefficient + 1 / (1 - t1)) == 0


def test_exponential_scale_alias_is_factored():
    x = sp.symbols("x", positive=True)
    t1 = 1 / x
    t2 = sp.exp(-x)
    ms = multiseries(sp.exp(-2 * x + 1 / x), x, scale=[t1, t2], terms=2)
    top = ms.terms(1)
    assert top[0].exponent == 2
    assert sp.simplify(top[0].coefficient - sp.exp(1 / x)) == 0


def test_book_indefinite_cancellation_example_expands_largest_scale_first():
    x = sp.symbols("x", positive=True)
    t1 = 1 / sp.log(x)
    t2 = 1 / x
    t3 = sp.exp(-x)
    expr = sp.log(1 + 1 / sp.log(x + sp.exp(-x))) - sp.log(1 + 1 / sp.log(x))
    ms = multiseries(expr, x, scale=[t1, t2, t3], terms=2)
    top = ms.terms(1)
    assert len(top) == 1
    assert top[0].exponent == 1
    expected = -1 / (x * (sp.log(x) + sp.log(x) ** 2))
    assert sp.simplify(top[0].coefficient - expected) == 0


def test_native_analytic_composition_uses_native_sparse_backend():
    x = sp.symbols("x", positive=True)
    scale = AsymptoticScale.from_exprs(x, [1 / sp.log(x), 1 / x])
    ms = native_multiseries(sp.exp(1 / x + 1 / sp.log(x)), x, scale=scale, terms=3)

    top = ms.terms(3)
    assert [term.exponent for term in top] == [0, 1, 2]
    assert sp.simplify(top[2].coefficient - sp.exp(1 / sp.log(x)) / 2) == 0


def test_recursive_sparse_backend_handles_deep_expression_tree_with_native_sparse_backend():
    x = sp.symbols("x", positive=True)
    z = sp.symbols("z", positive=True)
    expr = sp.exp(sp.log(2 + 1 / x) / (1 + 1 / x)) * (1 + 1 / x) ** sp.Rational(-3, 2)

    # The result is compared against SymPy, while the package fallback is
    # explicitly disabled so the sparse backend must produce the expansion.
    expected = sp.series(expr.subs(x, 1 / z), z, 0, 4).removeO().expand()

    ms = native_multiseries(expr, x, scale=[1 / x], terms=4)
    got = sum(term.coefficient * z**term.exponent for term in ms.terms(4))
    assert sp.expand(got - expected) == 0


def test_sparse_backend_handles_fractional_laurent_power_with_native_sparse_backend():
    x = sp.symbols("x", positive=True)
    expr = (1 / x + 1 / x**2) ** sp.Rational(-3, 2)

    ms = native_multiseries(expr, x, scale=[1 / x], terms=4)
    got = ms.terms(4)
    assert [term.exponent for term in got] == [
        sp.Rational(-3, 2),
        sp.Rational(-1, 2),
        sp.Rational(1, 2),
        sp.Rational(3, 2),
    ]
    assert [sp.simplify(term.coefficient) for term in got] == [
        1,
        sp.Rational(-3, 2),
        sp.Rational(15, 8),
        sp.Rational(-35, 16),
    ]


def test_sparse_log_of_top_scale_uses_lower_scale_representation():
    x = sp.symbols("x", positive=True)
    expr = sp.log(1 / x + 1 / x**2)

    ms = native_multiseries(expr, x, scale=[1 / sp.log(x), 1 / x], terms=3)
    got = ms.terms(3)
    assert [term.exponent for term in got] == [0, 1, 2]
    assert sp.simplify(got[0].coefficient + sp.log(x)) == 0
    assert got[1].coefficient == 1
    assert got[2].coefficient == sp.Rational(-1, 2)


def test_sparse_failure_dynamically_adds_exponential_scale_with_native_sparse_backend():
    x = sp.symbols("x", positive=True)
    expr = sp.exp(x + 1 / x)

    ms = native_multiseries(expr, x, scale=[1 / x], terms=3)
    top = ms.terms(3)

    assert any(sp.simplify(item - sp.exp(-x)) == 0 for item in ms.scale.exprs)
    assert top[0].exponent == -1
    assert sp.simplify(top[0].coefficient - sp.exp(1 / x)) == 0
    assert any(ext.generator == sp.exp(-x) for ext in ms.tower.extensions)


def test_dynamic_scale_extension_handles_vanishing_exponential():
    x = sp.symbols("x", positive=True)
    expr = sp.exp(-x + 1 / x)

    ms = native_multiseries(expr, x, scale=[1 / x], terms=2)
    top = ms.terms(2)

    assert any(sp.simplify(item - sp.exp(-x)) == 0 for item in ms.scale.exprs)
    assert top[0].exponent == 1
    assert sp.simplify(top[0].coefficient - sp.exp(1 / x)) == 0


def test_dynamic_scale_extension_propagates_through_nested_tree():
    x = sp.symbols("x", positive=True)
    expr = sp.log(1 + sp.exp(-x + 1 / x))

    ms = native_multiseries(expr, x, scale=[1 / x], terms=3)
    top = ms.terms(3)

    assert any(sp.simplify(item - sp.exp(-x)) == 0 for item in ms.scale.exprs)
    assert [term.exponent for term in top] == [1, 2, 3]
    assert sp.simplify(top[0].coefficient - sp.exp(1 / x)) == 0
    assert sp.simplify(top[1].coefficient + sp.exp(2 / x) / 2) == 0
    assert sp.simplify(top[2].coefficient - sp.exp(3 / x) / 3) == 0


def test_dynamic_scale_uses_complete_divergent_part():
    x = sp.symbols("x", positive=True)
    expr = sp.exp(x**2 + x + 1 / x)

    ms = native_multiseries(expr, x, scale=[1 / x], terms=2)
    top = ms.terms(2)
    candidate = sp.exp(-(x**2) - x)

    assert any(sp.simplify(item - candidate) == 0 for item in ms.scale.exprs)
    assert top[0].exponent == -1
    assert sp.simplify(top[0].coefficient - sp.exp(1 / x)) == 0


def test_dynamic_scale_extension_can_chain_through_exp_log_tower():
    x = sp.symbols("x", positive=True)
    expr = sp.exp(sp.exp(x) + 1 / x)

    ms = native_multiseries(expr, x, scale=[1 / x], terms=2)
    top = ms.terms(2)

    assert any(sp.simplify(item - sp.exp(-x)) == 0 for item in ms.scale.exprs)
    assert any(sp.simplify(item - sp.exp(-sp.exp(x))) == 0 for item in ms.scale.exprs)
    assert top[0].exponent == -1
    assert sp.simplify(top[0].coefficient - sp.exp(1 / x)) == 0


def test_log_obligation_dynamically_adds_reciprocal_log_scale():
    x = sp.symbols("x", positive=True)
    expr = sp.log(1 / x + 1 / x**2)

    ms = native_multiseries(expr, x, scale=[1 / x], terms=3)
    got = ms.terms(3)

    assert any(sp.simplify(item - 1 / sp.log(x)) == 0 for item in ms.scale.exprs)
    assert [term.exponent for term in got] == [0, 1, 2]
    assert sp.simplify(got[0].coefficient + sp.log(x)) == 0
    assert got[1].coefficient == 1
    assert got[2].coefficient == sp.Rational(-1, 2)


def test_log_obligation_adds_lower_exponential_scale():
    x = sp.symbols("x", positive=True)
    t = sp.exp(-x)
    expr = sp.log(t * (1 + t))

    ms = native_multiseries(expr, x, scale=[t], terms=3)
    got = ms.terms(3)

    assert any(sp.simplify(item - 1 / x) == 0 for item in ms.scale.exprs)
    assert [term.exponent for term in got] == [0, 1, 2]
    assert sp.simplify(got[0].coefficient + x) == 0
    assert got[1].coefficient == 1
    assert got[2].coefficient == sp.Rational(-1, 2)


def test_log_obligation_can_chain_to_iterated_exponential_lower_scale():
    x = sp.symbols("x", positive=True)
    t = sp.exp(-sp.exp(x))
    expr = sp.log(t * (1 + t))

    ms = native_multiseries(expr, x, scale=[t], terms=2)
    got = ms.terms(2)

    assert any(sp.simplify(item - sp.exp(-x)) == 0 for item in ms.scale.exprs)
    assert got[0].exponent == 0
    assert sp.simplify(got[0].coefficient + sp.exp(x)) == 0
    assert got[1].exponent == 1
    assert got[1].coefficient == 1


def test_obligation_history_records_typed_recoverable_obligations():
    from asymptotic.obligations import ObligationKind

    x = sp.symbols("x", positive=True)

    ms = native_multiseries(sp.log(1 / x + 1 / x**2), x, scale=[1 / x], terms=2)
    ms.terms(2)

    assert ms.obligation_history
    assert ms.obligation_history[0].kind is ObligationKind.LOGARITHMIC_SCALE
    assert ms.obligation_history[0].recoverable is True


def test_terminal_unsupported_obligation_is_recorded_before_fallback():
    from asymptotic.obligations import ObligationKind

    x = sp.symbols("x", positive=True)
    ms = multiseries(sp.gamma(1 + 1 / x), x, scale=[1 / x], terms=3)
    got = ms.terms(3)

    assert got[0].exponent == 0
    assert ms.obligation_history
    assert ms.obligation_history[0].kind is ObligationKind.UNSUPPORTED_NODE
    assert ms.obligation_history[0].recoverable is False


def test_native_analytic_unary_functions_use_native_sparse_backend():
    x = sp.symbols("x", positive=True)
    expr = sp.sin(1 / x) + sp.cos(1 / x) + sp.atan(1 / x)

    z = sp.symbols("z")
    expected = sp.series(sp.sin(z) + sp.cos(z) + sp.atan(z), z, 0, 5).removeO().subs(z, 1 / x)

    ms = native_multiseries(expr, x, scale=[1 / x], terms=5)
    got = ms.truncate(5)
    assert sp.expand(got - expected) == 0


def test_native_hyperbolic_and_erf_composition_with_native_sparse_backend():
    x = sp.symbols("x", positive=True)
    expr = sp.sinh(1 / x) + sp.cosh(1 / x) + sp.erf(1 / x)
    reference = sp.series(expr, x, sp.oo, 5).removeO()

    got = native_multiseries(expr, x, scale=[1 / x], terms=5).truncate(5)
    assert sp.expand(got - reference) == 0
