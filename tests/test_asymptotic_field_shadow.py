import sympy as sp

from asymptotic import RemainderKind
from asymptotic.asymptotic_field import (
    asymptotic_differential_field,
    infinitesimal_ideal_decision,
    moderate_growth_decision,
)
from asymptotic.remainder_theorems import certify_frechet_inverse_operator_remainder


def test_rt_it_and_shadow_ghost_projection():
    x = sp.symbols("x", positive=True)
    t = 1 / x
    assert moderate_growth_decision(1 / x, t, x).verdict is True
    assert moderate_growth_decision(x, t, x).verdict is False
    assert infinitesimal_ideal_decision(1 / x, t, x).verdict is True
    assert infinitesimal_ideal_decision(sp.log(x), t, x).verdict is False
    field = asymptotic_differential_field(x, (1 / sp.log(x), 1 / x))
    dec = field.projection(1, sp.log(x) + 1 / x)
    assert dec.shadow == sp.log(x)
    assert dec.ghost == 1 / x
    assert dec.replay() is True


def test_shadow_projection_is_field_homomorphic_on_supported_example():
    x = sp.symbols("x", positive=True)
    field = asymptotic_differential_field(x, (1 / x,))
    a = 2 + 1 / x
    b = 3 + 2 / x
    assert field.shadow(0, a + b) == field.shadow(0, a) + field.shadow(0, b)
    assert sp.simplify(field.shadow(0, a * b) - field.shadow(0, a) * field.shadow(0, b)) == 0


def test_first_order_frechet_inverse_operator_certificate():
    x = sp.symbols("x", positive=True)
    d = sp.Function("d")
    cert = certify_frechet_inverse_operator_remainder(x**-2, sp.diff(d(x), x) + d(x), d, x, sp.oo)
    assert cert.certified
    assert cert.conclusion.kind is RemainderKind.BIG_O
    assert cert.hypotheses[0].verdict is True
    assert cert.hypotheses[1].verdict is True
    assert cert.hypotheses[2].verdict is True


def test_exponential_shadow_extension_rule():
    x = sp.symbols("x", positive=True)
    field = asymptotic_differential_field(x, (1 / x,))
    dec = field.projection(0, sp.exp(1 / x))
    assert dec.shadow == 1
    assert sp.simplify(dec.ghost - (sp.exp(1 / x) - 1)) == 0
    assert dec.certified
