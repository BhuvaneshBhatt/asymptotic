import sympy as sp

from asymptotic.limits import LimitStatus, limit
from asymptotic.multivariate_certificates import (
    angular_cluster_path_certificate,
    generalized_log_product_order_certificate,
    inverse_branch_limit_certificate,
    parameter_weighted_order_certificate,
    singular_polylog_special_germ_certificate,
)


def test_parameter_exponent_is_not_collapsed_unconditionally():
    x, y, p = sp.symbols("x y p", real=True)
    e = x**16 * y**22 / (x**2 + y**2) ** p
    c = parameter_weighted_order_certificate(e, (x, y), (0, 0))
    assert not c.certified
    assert c.data[1] == (p < 19)
    from asymptotic.stratification import AsymptoticStratification

    r = limit(e, (x, y), (0, 0), return_result=True)
    assert isinstance(r, AsymptoticStratification)
    assert r.exhaustive
    assert not any(
        s.result.status is LimitStatus.PROVED and s.result.value == 0
        for s in r.strata
        if s.condition != (p < 19)
    )


def test_general_log_products():
    x, y = sp.symbols("x y", real=True)
    for e in (
        x * y * sp.log(sp.Abs(x * y)),
        x**2 * sp.sqrt(sp.Abs(y)) * sp.log(x * x + y * y),
    ):
        if not e.has(sp.sqrt(sp.Abs(y))):
            c = generalized_log_product_order_certificate(e, (x, y), (0, 0))
            assert c.certified and c.value == 0
    r = limit(x * y * sp.log(sp.Abs(x * y)), (x, y), (0, 0), return_result=True)
    assert r.status is LimitStatus.PROVED and r.value == 0


def test_angular_cluster_dne_fast_path():
    x, y, z = sp.symbols("x y z", real=True)
    rho = sp.sqrt(x * x + y * y + z * z)
    e = x / (sp.exp(rho) * rho)
    c = angular_cluster_path_certificate(e, (x, y, z), (0, 0, 0))
    assert c.certified and c.method.endswith("_dne")
    r = limit(e, (x, y, z), (0, 0, 0), return_result=True)
    assert r.status is LimitStatus.DOES_NOT_EXIST


def test_atanh_endpoint_branch():
    x, y = sp.symbols("x y", real=True)
    e = sp.atanh(2 * x / (1 + x * x + y * y))
    c = inverse_branch_limit_certificate(e, (x, y), (1, 0))
    assert c.certified and c.value == sp.oo
    r = limit(e, (x, y), (1, 0), return_result=True)
    assert r.status is LimitStatus.PROVED and r.value == sp.oo


def test_bessely_polylog_germs():
    x, y = sp.symbols("x y", real=True)
    a = sp.cosh(y) - y**2 / 16
    e = -sp.Rational(1, 2) * sp.pi * x * a * sp.bessely(1, x)
    c = singular_polylog_special_germ_certificate(e, (x, y), (0, 0))
    assert c.certified and c.value == 1
    e2 = -1 / x**2 + e
    c2 = singular_polylog_special_germ_certificate(e2, (x, y), (0, 0))
    assert c2.certified and c2.value == -sp.oo


def test_fractional_order_log_and_shifted_radial_pole():
    x, y = sp.symbols("x y", real=True)
    e = x**2 * sp.sqrt(sp.Abs(y)) * sp.log(x * x + y * y)
    c = generalized_log_product_order_certificate(e, (x, y), (0, 0))
    assert c.certified and c.value == 0
    assert limit(e, (x, y), (0, 0), return_result=True).value == 0
    pole = (1 - 3 * x + x * y) / sp.sqrt((x - 1) ** 2 + (y - 1) ** 2)
    r = limit(pole, (x, y), (1, 1), return_result=True)
    assert r.status is LimitStatus.PROVED and r.value == -sp.oo


def test_fractional_part_rational_pole_dne():
    x, y = sp.symbols("x y", real=True)
    e = sp.frac((4 * y**2 - x**2) / (x - 2 * y) ** 3)
    r = limit(e, (x, y), (0, 0), return_result=True)
    assert r.status is LimitStatus.DOES_NOT_EXIST
    assert any(ev.method == "fractional_part_dne" for ev in r.evidence)


def test_rational_exceptional_direction_detects_cancellation_line():
    x, y = sp.symbols("x y", real=True)
    result = limit(x**2 / (4 * y - 5 * x), (x, y), (0, 0), return_result=True)
    assert result.status.name == "DOES_NOT_EXIST"
    from asymptotic._limit_paths import _exceptional_rational_paths

    paths = list(_exceptional_rational_paths(x**2 / (4 * y - 5 * x), (x, y), (0, 0)))
    assert any(label.startswith("exceptional_direction:") for _, _, label in paths)
    values = set()
    for t, path, _ in paths:
        substitution = dict(zip((x, y), path, strict=True))
        assert sp.expand((4 * y - 5 * x).subs(substitution)) != 0
        values.add(sp.limit((x**2 / (4 * y - 5 * x)).subs(substitution), t, 0, dir="+"))
    assert sp.Rational(1, 4) in values and -sp.Rational(1, 4) in values


def test_rational_exceptional_direction_checks_reciprocal_chart():
    x, y = sp.symbols("x y", real=True)
    result = limit(y**2 / (4 * x - 5 * y), (x, y), (0, 0), return_result=True)
    assert result.status.name == "DOES_NOT_EXIST"
    from asymptotic._limit_paths import _exceptional_rational_paths

    paths = list(_exceptional_rational_paths(y**2 / (4 * x - 5 * y), (x, y), (0, 0)))
    assert any("x_over_y" in label for _, _, label in paths)
    values = set()
    for t, path, _ in paths:
        substitution = dict(zip((x, y), path, strict=True))
        assert sp.expand((4 * x - 5 * y).subs(substitution)) != 0
        values.add(sp.limit((y**2 / (4 * x - 5 * y)).subs(substitution), t, 0, dir="+"))
    assert sp.Rational(1, 4) in values and -sp.Rational(1, 4) in values


def test_exceptional_curve_lifting_finds_higher_order_divisor_contact():
    x, y = sp.symbols("x y", real=True)
    result = limit(x**4 / (y - x - x**3), (x, y), (0, 0), return_result=True)
    assert result.status.name == "DOES_NOT_EXIST"
    assert any(e.method.startswith("exceptional_curve:") for e in result.evidence)
