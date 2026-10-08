"""Principal power magnitude bounds and excluded oscillatory denominators."""

import pytest
import sympy as sp

from asymptotic import limit
from asymptotic.complex_power_germs import (
    decaying_power_log_certificate,
    exponentially_small_denominator_certificate,
)


@pytest.mark.parametrize("base", [-sp.E, 2])
def test_small_power_log(base):
    x = sp.Symbol("x", positive=True)
    e = x**3 * sp.log((-3) ** (base ** (-2 * x + 1)))
    assert limit(e, x, sp.oo) == 0


def test_unit_modulus_declines():
    x = sp.Symbol("x", positive=True)
    e = sp.log((-3) ** ((-1) ** (-x))) / x
    assert decaying_power_log_certificate(e, x, sp.oo) is None


def test_denominator_decay():
    x = sp.Symbol("x", positive=True)
    delta = (
        sp.sqrt((1 - sp.E) ** (-x)) * sp.cos(1 / x) ** 2
        + sp.sqrt((1 + sp.E) ** (-x)) * sp.sin(1 / x) ** 2
    )
    assert limit(sp.exp(x / (1 - delta)), x, sp.oo) is sp.zoo


def test_denominator_poles_decline():
    x = sp.Symbol("x", positive=True)
    e = sp.exp(x / (1 - (-sp.E) ** (-x) / sp.sin(x)))
    assert exponentially_small_denominator_certificate(e, x, sp.oo) is None


def test_nondecaying_root_declines():
    x = sp.Symbol("x", positive=True)
    e = sp.exp(x / (1 - sp.sqrt((-1) ** (-x))))
    assert exponentially_small_denominator_certificate(e, x, sp.oo) is None


def test_real_denominator_direction():
    x = sp.Symbol("x", positive=True)
    assert limit(sp.exp(x / (1 - 2 ** (-x))), x, sp.oo) is sp.oo
