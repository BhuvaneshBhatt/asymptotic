import sympy as sp
from hypothesis import given
from hypothesis import strategies as st

from asymptotic import (
    big_o,
    equivalent,
    little_o,
)
from asymptotic.relations import (
    equal,
    less,
    less_equal,
    same_order,
)


@given(st.integers(min_value=1, max_value=8), st.integers(min_value=1, max_value=8))
def test_relation_aliases_and_growth_lattice(power, extra):
    x = sp.symbols("x", positive=True)
    f = x**power
    g = x ** (power + extra)
    assert less(f, g, x) is True
    assert little_o(f, g, x) is True
    assert less_equal(f, g, x) is True
    assert big_o(f, g, x) is True
    assert equal(f, g, x) is False
    assert same_order(f, g, x) is False


def test_equivalent_is_stronger_than_equal():
    x = sp.symbols("x", positive=True)
    assert equal(2 * x, x, x) is True
    assert equivalent(2 * x, x, x) is False


def test_valid_assumptions_can_resolve_reversing_certainty():
    x, a = sp.symbols("x a")
    undecided = equal(a * x, x, x)
    resolved = equal(a * x, x, x, assumptions=sp.Q.positive(a))
    assert undecided.value is True
    assert undecided.condition == sp.Ne(a, 0, evaluate=False)
    assert resolved is True
