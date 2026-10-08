import sympy as sp

from asymptotic import limit
from asymptotic.limit_models import LimitStatus
from asymptotic.univariate_complex_rational import finite_complex_rational_certificate


def test_complex_continuity_and_removable_zeros():
    z = sp.Symbol("z")
    for p in (sp.I, 1 + sp.I, sp.exp(sp.I * sp.pi / 3)):
        assert limit((z - p) / (z * z - p * p), z, p) == sp.simplify(
            sp.expand_complex(1 / (2 * p))
        )
        assert limit((z - p) ** 3 / (z - p) ** 2, z, p) == 0
        assert limit(z**3 + 2 * z, z, p) == sp.simplify(sp.expand_complex(p**3 + 2 * p))


def test_poles_and_unknown_parameter_strata_are_not_certified():
    z, a = sp.symbols("z a")
    for e in (1 / (z - sp.I + a), sp.log(z), sp.sqrt(z)):
        assert limit(e, z, sp.I, return_result=True).status is LimitStatus.UNKNOWN
    assert limit(1 / (z - sp.I), z, sp.I) is sp.zoo
    assert (
        finite_complex_rational_certificate(
            1 / (z - sp.I), z, sp.I, sp.S.true, sp.S.true
        )
        is None
    )


def test_complex_target_needs_accumulating_complex_variable():
    z = sp.Symbol("z")
    x = sp.Symbol("x", real=True)
    for domain in (sp.S.false, sp.Eq(z, sp.I)):
        assert (
            limit(z, z, sp.I, domain=domain, return_result=True).status
            is LimitStatus.UNKNOWN
        )
    assert limit(x, x, sp.I, return_result=True).status is LimitStatus.UNKNOWN
    y = sp.Symbol("y", imaginary=True)
    assert limit(y, y, 1 + sp.I, return_result=True).status is LimitStatus.UNKNOWN


def test_complex_fast_path_does_not_intercept_real_infinity():
    z = sp.Symbol("z")
    assert limit(1 / z, z, sp.oo) == 0
    assert limit(1 / z, z, -sp.oo) == 0
