import sympy as sp

from asymptotic.limits import LimitStatus, limit
from asymptotic.local_strata import LocalStratum
from asymptotic.mapped_strata import StratumMap, mapped_local_stratum
from asymptotic.multivariate_certificates import (
    growth_scale_limit,
    isolated_zero_certificate,
    lojasiewicz_exponent_bound,
    parameter_stratified_limit,
    sertoz_rational_limit,
)
from asymptotic.stratified_mapping import check_whitney_conditions, tangent_space_limit


def test_lojasiewicz_and_isolated_zero_positive_form():
    x, y = sp.symbols("x y", real=True)
    f = x**4 + y**6
    assert isolated_zero_certificate(f, (x, y), (0, 0)).certified
    c = lojasiewicz_exponent_bound(f, (x, y), (0, 0))
    assert c.certified and c.value == 6


def test_sertoz_rational_limit():
    x, y = sp.symbols("x y", real=True)
    f = (x**8 + y**8) / (x**4 + y**6)
    c = sertoz_rational_limit(f, (x, y), (0, 0))
    assert c.certified and c.value == 0
    r = limit(f, (x, y), (0, 0), return_result=True)
    assert r.status is LimitStatus.PROVED and r.value == 0


def test_growth_scale_radial_log():
    x, y = sp.symbols("x y", real=True)
    c = growth_scale_limit((x * x + y * y) * sp.log(x * x + y * y), (x, y), (0, 0))
    assert c.certified and c.value == 0


def test_parameter_regular_stratum():
    x, y, a = sp.symbols("x y a", real=True)
    strat = parameter_stratified_limit((x + a * y) / (1 + a * x), (x, y), (0, 0), (a,))
    assert strat.exhaustive and len(strat.strata) == 1
    assert strat.strata[0].result.status is LimitStatus.PROVED
    assert sp.simplify(strat.strata[0].result.value) == 0


def test_general_plucker_whitney_plane_to_point():
    u, v = sp.symbols("u v", real=True)
    X, Y, Z = sp.symbols("X Y Z", real=True)
    mapping = StratumMap((u, v), (u, v, sp.S.Zero), (X, Y, Z))
    mapped = mapped_local_stratum(LocalStratum("surface"), mapping)
    tl = tangent_space_limit(mapping, parameter_target=(0, 0))
    assert tl.certified and tl.plane_dimension == 2
    w = check_whitney_conditions(mapped, parameter_target=(0, 0))
    assert w.certified


def test_reciprocal_pole_sign_family_is_certified_before_cad():
    x, y = sp.symbols("x y", real=True)
    from asymptotic.limits import LimitStatus, limit

    cases = [
        ((x**2 + y**2) / (x**4 + y**4), LimitStatus.PROVED, sp.oo),
        (-(x**2 + y**2) / (x**4 + y**4), LimitStatus.PROVED, -sp.oo),
        ((x**2 - y**2) / (x**4 + y**4), LimitStatus.DOES_NOT_EXIST, None),
        (1 / (x**4 + x**2 * y**2 + y**4), LimitStatus.PROVED, sp.oo),
        (-1 / (x**4 + x**2 * y**2 + y**4), LimitStatus.PROVED, -sp.oo),
        (1 / (x**2 - y**2), LimitStatus.DOES_NOT_EXIST, None),
    ]
    for expr, status, value in cases:
        r = limit(expr, (x, y), (0, 0), return_result=True)
        assert r.status is status
        assert r.value == value


def test_analytic_difference_and_singleton_fast_paths():
    x, y = sp.symbols("x y", real=True)
    from asymptotic.limits import limit

    cases = [
        ((sp.sin(x) - sp.sin(y)) / (x - y), sp.Integer(1)),
        ((sp.cos(x) - sp.cos(y)) / (x**2 - y**2), -sp.Rational(1, 2)),
        ((sp.sin(x) ** 4 + sp.sin(y) ** 4) / (x**2 + y**2), sp.Integer(0)),
        ((sp.sin(x) - x + sp.sin(y) - y) / (x**2 + y**2), sp.Integer(0)),
        (1 / (sp.sin(x) ** 2 + sp.sin(y) ** 2), sp.oo),
        ((x**2 + y**2) ** 2 * sp.log(x**2 + y**2) ** 3, sp.Integer(0)),
    ]
    for expr, value in cases:
        r = limit(expr, (x, y), (0, 0), return_result=True)
        assert r.value == value


def test_parameter_direction_preserves_conditional_dne_leaf():
    x, y, a = sp.symbols("x y a", real=True)
    from asymptotic.limits import LimitStatus
    from asymptotic.multivariate_certificates import (
        recursive_parameter_stratified_limit,
    )

    r = recursive_parameter_stratified_limit(
        (a * x**2 + y**2) / (x**2 + y**2), (x, y), (0, 0), (a,)
    )
    assert r.exhaustive
    statuses = {s.result.status for s in r.strata}
    assert statuses == {LimitStatus.PROVED, LimitStatus.DOES_NOT_EXIST}
    proved = next(s for s in r.strata if s.result.status is LimitStatus.PROVED)
    assert proved.result.value == 1


def test_reciprocal_power_and_extremal_limits():
    x, y = sp.symbols("x y", real=True)
    from asymptotic.multivariate_certificates import extremal_limit

    inverse = limit((x**4 + y**6) / (x**8 + y**12), (x, y), (0, 0), return_result=True)
    assert inverse.status is LimitStatus.PROVED and inverse.value == sp.oo

    ratpow = limit(
        (x**4 + y**4) / sp.sqrt(x**2 + y**2),
        (x, y),
        (0, 0),
        return_result=True,
    )
    assert ratpow.status is LimitStatus.PROVED and ratpow.value == 0

    minimum = extremal_limit(x**4 / (x**4 + y**6), (x, y), (0, 0), kind="min")
    assert minimum.certified and minimum.value == 0


def test_semidefinite_product_is_not_misclassified_as_definite():
    x, y = sp.symbols("x y", real=True)
    from asymptotic.multivariate_certificates import local_sign_certificate

    assert not local_sign_certificate(x**2 * y**2, (x, y), (0, 0)).certified
    assert not local_sign_certificate(-(x**2) * y**2, (x, y), (0, 0)).certified
