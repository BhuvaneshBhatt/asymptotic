import sympy as sp

from asymptotic import limit, one_sided_limit
from asymptotic.limit_models import LimitStatus
from asymptotic.reference_normalization import scalar_reference_namespace
from asymptotic.stratification import AsymptoticStratification
from asymptotic.weierstrass_origin import (
    WeierstrassSigma,
    WeierstrassZeta,
    weierstrass_origin_certificate,
)


def test_weierstrass_origin_germs_and_rational_composition():
    z = sp.Symbol("z")
    g2, g3 = sp.symbols("g2 g3")
    v = sp.Tuple(g2, g3)
    for u in (z, 2 * z, z / (1 + z)):
        assert limit(WeierstrassZeta(u, v) - 1 / u, z, 0) == 0
        assert limit(WeierstrassSigma(u, v) / u, z, 0) == 1
        assert limit((WeierstrassSigma(u, v) / u + 2) / (1 + z), z, 0) == 3
        assert one_sided_limit(WeierstrassZeta(u, v) - 1 / u, z, 0, direction="-") == 0


def test_remainders_are_not_discarded_amplifies_them():
    z = sp.Symbol("z")
    v = sp.Tuple(1, 2)
    for e in (
        (WeierstrassZeta(z, v) - 1 / z) / z**3,
        (WeierstrassSigma(z, v) - z) / z**5,
        1 / (WeierstrassSigma(z, v) - z),
    ):
        assert limit(e, z, 0, return_result=True).status is LimitStatus.UNKNOWN


def test_weierstrass_parameters_domains_and_are_guarded():
    z = sp.Symbol("z")
    v = sp.Tuple(1, 2)
    for e, p, d in [
        (WeierstrassSigma(z, sp.Tuple(z, 1)) / z, 0, True),
        (WeierstrassSigma(z, sp.Tuple(sp.oo, 1)) / z, 0, True),
        (WeierstrassSigma(z, v) / z, 1, True),
        (WeierstrassSigma(z, v) / z, 0, False),
        (sp.Function("UnknownSigma")(z) / z, 0, True),
        (
            WeierstrassSigma(z, sp.Tuple(sp.Symbol("infinite", infinite=True), 1)) / z,
            0,
            True,
        ),
    ]:
        assert (
            limit(e, z, p, domain=d, return_result=True).status is LimitStatus.UNKNOWN
        )


def test_negative_step_factorial_cells():
    ns = scalar_reference_namespace()
    a, h = sp.symbols("a h")
    e = ns["StepFactorialPower"](a, -3, h)
    cells = limit(e, h, 0, return_result=True)
    assert isinstance(cells, AsymptoticStratification)
    assert {c.result.status for c in cells.strata} == {
        LimitStatus.PROVED,
        LimitStatus.DOES_NOT_EXIST,
    }
    r = limit(e, h, 0, assumptions=sp.Ne(a, 0), return_result=True)
    assert r.status is LimitStatus.PROVED and r.value == a**-3
    assert "step_factorial_parameter_germ" in [e.method for e in r.evidence]
    assert (
        limit(e.subs(a, 0), h, 0, return_result=True).status is not LimitStatus.PROVED
    )


def test_defined_function_contracts():
    z = sp.Symbol("z")
    g2, g3 = sp.symbols("g2 g3")
    e = scalar_reference_namespace()["WeierstrassSigma"](z, sp.Tuple(g2, g3)) / z
    r = limit(e, z, 0, return_result=True)
    assert r.status is LimitStatus.PROVED and r.value == 1
    assert r.variables == (z,) and r.target == (0,)


def test_origin_certificate_requires_an_accumulating_symbol_domain():
    v = sp.Tuple(1, 2)
    for z, p in (
        (sp.Symbol("z", positive=True), -1),
        (sp.Symbol("z", integer=True), 0),
        (sp.Symbol("z", imaginary=True), 1 + sp.I),
    ):
        e = WeierstrassSigma(z - p, v) / (z - p)
        assert limit(e, z, p, return_result=True).status is LimitStatus.UNKNOWN


def test_parameter_singular_coefficients_and_nonscalar_inputs_decline():
    z, a = sp.symbols("z a")
    v = sp.Tuple(1, 2)
    germ = WeierstrassSigma(z, v) / z - 1
    assert (
        limit(sp.log(a) * germ, z, 0, return_result=True).status is LimitStatus.UNKNOWN
    )
    assert weierstrass_origin_certificate(sp.Tuple(germ), z, 0, True) is None
