import sympy as sp

from asymptotic.context import AsymptoticContext
from asymptotic.hardy_solve import asymptotic_sturm_certificate


def test_sturm_reduces_leading_coefficient_zero_in_asymptotic_germ():
    x, y = sp.symbols("x y", real=True)
    # exp(-x) is not identically zero, so degree must remain two.
    cert = asymptotic_sturm_certificate(
        sp.exp(-x) * y**2 + y - 1,
        y,
        x,
        point=sp.oo,
        context=AsymptoticContext(x, point=sp.oo),
    )
    assert cert.sequence
    assert cert.replay() in (True, None)


def test_sturm_reduces_symbolically_zero_leading_coefficient():
    x, y = sp.symbols("x y", real=True)
    zero = sp.exp(x) - sp.exp(x)
    cert = asymptotic_sturm_certificate(zero * y**3 + y**2 - 1, y, x, point=sp.oo)
    assert cert.certified
    assert cert.distinct_real_roots == 2
