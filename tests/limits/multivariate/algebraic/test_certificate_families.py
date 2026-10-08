import sympy as sp

from asymptotic.limits import LimitStatus, limit
from asymptotic.multivariate_certificates import (
    general_isolated_zero_certificate,
    general_lojasiewicz_exponent,
)

x, y = sp.symbols("x y", real=True)


def test_general_lojasiewicz_exact_certificate():
    r = general_lojasiewicz_exponent(x**4 + y**6, (x, y), (0, 0))
    assert r.certified and r.exponent == 6


def test_general_isolated_zero_exact_certificate():
    r = general_isolated_zero_certificate(x**2 + y**2, (x, y), (0, 0))
    assert r.certified and r.value is sp.S.true


def test_rational_germ_cancellation():
    r = limit(2 * (x**2 + y**2) / (x**2 + y**2), (x, y), (0, 0), return_result=True)
    assert r.status is LimitStatus.PROVED and r.value == 2


def test_radial_log_limit():
    r = limit((x**2 + y**2) * sp.log(x**2 + y**2), (x, y), (0, 0), return_result=True)
    assert r.status is LimitStatus.PROVED and r.value == 0
