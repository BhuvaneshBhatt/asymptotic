import sympy as sp

from asymptotic.limits import LimitStatus, limit
from asymptotic.multivariate_certificates import (
    logarithmic_germ_order_certificate,
    radial_norm_canonicalization_certificate,
    real_projection_simplification_certificate,
    special_function_local_germ_certificate,
)


def assert_proved(expr, variables, value):
    result = limit(expr, variables, (0,) * len(variables), return_result=True)
    assert result.status is LimitStatus.PROVED
    assert result.value == value or sp.simplify(result.value - value) == 0


def test_radial_norm_failure_family():
    x, y = sp.symbols("x y", real=True)
    rho = sp.sqrt(x**2 + y**2)
    assert radial_norm_canonicalization_certificate(rho, (x, y), (0, 0)).value == 0
    assert (
        radial_norm_canonicalization_certificate(x * y / rho, (x, y), (0, 0)).value == 0
    )
    assert_proved(rho, (x, y), 0)
    assert_proved(x * y / rho, (x, y), 0)
    assert_proved(rho**4 / sp.sin(rho**2), (x, y), 0)


def test_logarithmic_failure_family():
    x, y = sp.symbols("x y", real=True)
    r2 = x**2 + y**2

    def log10(z):
        return sp.log(z) / sp.log(10)

    cases = [
        3 * r2 * log10(r2),
        r2 * log10(r2),
        x * y * log10(r2),
        x**2 * y**2 * log10(r2),
    ]
    for expr in cases:
        cert = logarithmic_germ_order_certificate(expr, (x, y), (0, 0))
        assert cert.certified and cert.value == 0
        assert_proved(expr, (x, y), 0)
    assert_proved(log10(sp.sqrt(r2)), (x, y), -sp.oo)


def test_real_projection_failure_family():
    x, y = sp.symbols("x y", real=True)
    expr = sp.re(x - 1 + sp.I * y - sp.I)
    cert = real_projection_simplification_certificate(expr, (x, y), (0, 0))
    # SymPy may simplify Re eagerly; either the projection route or ordinary route is valid.
    if cert.certified:
        assert cert.value == -1
    assert_proved(expr, (x, y), -1)


def test_special_function_registry_zero_germs():
    x, y = sp.symbols("x y", real=True)
    expr = x**2 * sp.fresnelc(y)
    cert = special_function_local_germ_certificate(expr, (x, y), (0, 0))
    assert cert.certified and cert.value == 0
    assert_proved(expr, (x, y), 0)


def test_bessely_one_singular_product_germ():
    x, y = sp.symbols("x y", real=True)
    expr = (
        -sp.Rational(1, 2) * (x * sp.pi) * (sp.cosh(y) - y**2 / 16) * sp.bessely(1, x)
    )
    cert = special_function_local_germ_certificate(expr, (x, y), (0, 0))
    assert cert.certified and cert.value == 1
    assert_proved(expr, (x, y), 1)
