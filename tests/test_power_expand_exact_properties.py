import sympy as sp
from hypothesis import given
from hypothesis import strategies as st

from asymptotic._power_simplify import power_expand_exact


@given(st.integers(min_value=1, max_value=8))
def test_power_expand_exact_positive_square_powers_are_equal(power):
    x = sp.symbols("x", positive=True)
    exponent = sp.Rational(1, power)
    original = (x**power) ** exponent
    expanded = power_expand_exact(original, sp.Q.positive(x))
    assert sp.simplify(expanded - x) == 0


def test_power_expand_exact_does_not_drop_sqrt_branch_correction():
    z = sp.symbols("z", nonzero=True)
    expanded = power_expand_exact(sp.sqrt(z**2))
    # A generic complex z cannot be replaced by z globally.
    assert expanded != z
    # On the positive real stratum the correction must collapse.
    positive = power_expand_exact(sp.sqrt(z**2), sp.Q.positive(z))
    assert sp.refine(positive, sp.Q.positive(z)) == z
