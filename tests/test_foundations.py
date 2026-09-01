import sympy as sp

from asymptotic import (
    AsymptoticContext,
    mrv_decomposition,
    nested_expansion,
)
from asymptotic.decomposition import (
    maximal_univariate_decomposition,
    rational_decomposition,
)
from asymptotic.scale import ScaleDiscovery
from asymptotic.sparse import LazySparseSeries


def test_lazy_stream_crosses_large_exact_cancellation_without_oversampling():
    z = sp.symbols("z")
    cutoff = 36
    polynomial = sum(z**k / sp.factorial(k) for k in range(cutoff + 1))
    expr = sp.exp(z) - polynomial
    sparse = LazySparseSeries(expr, z, AsymptoticContext(z))

    stream = sparse.stream()
    term = stream.next_term()
    assert term is not None
    assert term.exponent == cutoff + 1
    assert sp.simplify(term.coefficient - 1 / sp.factorial(cutoff + 1)) == 0
    # The merge and analytic nodes retain one persistent frontier rather than
    # rebuilding a guessed n, 2n, 4n prefix sequence.
    root = next(state for state in sparse.node_states if state.expr == expr)
    assert "add" in root.payload


def test_maximal_univariate_decomposition_reconstructs_chain():
    x = sp.symbols("x", positive=True)
    expr = sp.log(1 + sp.sin(sp.exp(-x)))
    layers = maximal_univariate_decomposition(expr, x)
    assert len(layers) >= 4
    # Applying every outer layer to its recorded inner is an exact identity.
    for layer in layers:
        assert (
            sp.simplify(
                layer.apply(layer.inner) - layer.outer.xreplace({layer.symbol: layer.inner})
            )
            == 0
        )
    assert layers[0].apply(layers[0].inner) == expr


def test_rational_decomposition_uses_exact_trig_reconstruction():
    x = sp.symbols("x", real=True)
    expr = sp.sin(x) + sp.cos(x) / (1 + sp.tan(x))
    rationalized, substitutions = rational_decomposition(expr, x)
    assert substitutions
    t, back = substitutions[0]
    expected = (2 * t / (1 + t**2)) + ((1 - t**2) / (1 + t**2)) / (1 + 2 * t / (1 - t**2))
    assert sp.cancel(rationalized - expected) == 0
    assert back == sp.tan(x / 2)


def test_mrv_decomposition_identifies_exponential_class():
    x = sp.symbols("x", positive=True)
    expr = sp.log(sp.log(x)) + sp.exp(-x)
    dec = mrv_decomposition(expr, x)
    assert dec.representative is not None
    assert sp.simplify(dec.representative - sp.exp(-x)) == 0


def test_scale_discovery_exposes_structural_and_mrv_preprocessing():
    x = sp.symbols("x", positive=True)
    expr = sp.log(1 + sp.exp(-x)) + 1 / x
    discovery = ScaleDiscovery(expr, x)
    assert discovery.decomposition.original == expr
    assert discovery.mrv is not None
    scale = discovery.discover()
    assert len(scale) >= 2


def test_nested_expansion_is_resumable_and_arithmetic_is_exact():
    x = sp.symbols("x", positive=True)
    left = nested_expansion(x + 1 / x, x, depth=1)
    right = nested_expansion(-x + 1 / x**2, x, depth=1)
    combined = left + right
    assert sp.simplify(combined.expr - (1 / x + 1 / x**2)) == 0
    combined.refine(2)
    assert combined.forms
    assert sp.simplify(combined.forms[0].reconstruct() - combined.expr) == 0

    product = left * right
    power = left**2
    expn = left.exp()
    logn = nested_expansion(1 + 1 / x, x, depth=1).log()
    for obj in (product, power, expn, logn):
        obj.refine(1)
        assert obj.forms
        assert sp.simplify(obj.forms[0].reconstruct() - obj.expr) == 0


def test_structural_decomposition_is_attached_to_nested_forms():
    x = sp.symbols("x", positive=True)
    ne = nested_expansion(sp.exp(x) * (1 + 1 / x), x, depth=1)
    form = ne.forms[0]
    assert form.structural is not None
    assert form.mrv is not None


def test_maximal_univariate_decomposition_finds_shared_inner_branch():
    x = sp.symbols("x", real=True)
    inner = sp.exp(x)
    expr = sp.sin(inner) + sp.cos(inner)
    layers = maximal_univariate_decomposition(expr, x)
    assert layers
    assert layers[0].inner == inner
    assert layers[0].apply(inner) == expr


def test_mrv_decomposition_uses_shared_composition_inner():
    x = sp.symbols("x", positive=True)
    inner = sp.exp(x)
    expr = sp.sin(inner) + sp.cos(inner)
    dec = mrv_decomposition(expr, x)
    assert any(inner in cls.members for cls in dec.classes)


def test_rational_decomposition_groups_exact_exponential_powers():
    x = sp.symbols("x", real=True)
    expr = sp.exp(2 * x) + sp.exp(3 * x) + sp.exp(-x)
    rationalized, substitutions = rational_decomposition(expr, x)
    assert len(substitutions) == 1
    e, back = substitutions[0]
    assert back == sp.exp(x)
    assert sp.cancel(rationalized - (e**2 + e**3 + e**-1)) == 0
    assert sp.simplify(rationalized.xreplace(dict(substitutions)) - expr) == 0


def test_rational_decomposition_groups_fractional_argument_coefficients_safely():
    x = sp.symbols("x", real=True)
    expr = sp.exp(x / 2) + sp.exp(3 * x / 2)
    rationalized, substitutions = rational_decomposition(expr, x)
    assert len(substitutions) == 1
    e, back = substitutions[0]
    assert back == sp.exp(x / 2)
    assert sp.cancel(rationalized - (e + e**3)) == 0
