import sympy as sp
from hypothesis import given
from hypothesis import strategies as st

from asymptotic import (
    asymptotic_big_o,
    asymptotic_equivalent,
    asymptotic_little_o,
)
from asymptotic.relations import (
    asymptotic_equal,
    asymptotic_less,
    asymptotic_less_equal,
    asymptotic_same_order,
)


@given(st.integers(min_value=1, max_value=8), st.integers(min_value=1, max_value=8))
def test_relation_aliases_and_growth_lattice(power, extra):
    x = sp.symbols("x", positive=True)
    f = x**power
    g = x ** (power + extra)
    assert asymptotic_less(f, g, x) is True
    assert asymptotic_little_o(f, g, x) is True
    assert asymptotic_less_equal(f, g, x) is True
    assert asymptotic_big_o(f, g, x) is True
    assert asymptotic_equal(f, g, x) is False
    assert asymptotic_same_order(f, g, x) is False


def test_equivalent_is_stronger_than_equal():
    x = sp.symbols("x", positive=True)
    assert asymptotic_equal(2 * x, x, x) is True
    assert asymptotic_equivalent(2 * x, x, x) is False


def test_valid_assumptions_can_resolve_unknown_without_reversing_certainty():
    x, a = sp.symbols("x a")
    undecided = asymptotic_equal(a * x, x, x)
    resolved = asymptotic_equal(a * x, x, x, assumptions=sp.Q.positive(a))
    assert undecided is None
    assert resolved is True
