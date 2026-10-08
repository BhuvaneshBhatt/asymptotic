import sympy as sp

from asymptotic.fixed_ray_branch_germs import DirectionalInfinity
from asymptotic.limits import LimitStatus, limit
from asymptotic.multivariate_certificates import (
    complex_sech_phase_certificate,
    exponential_small_power_certificate,
    generalized_log_product_order_certificate,
    variable_power_branch_certificate,
)


def test_log10_products_ignore_constant_log_denominator():
    x, y = sp.symbols("x y", real=True)
    cases = (
        x * y * sp.log(sp.Abs(x * y)) / sp.log(10),
        x**2 * sp.sqrt(sp.Abs(y)) * sp.log(x**2 + y**2) / sp.log(10),
    )
    for expr in cases:
        cert = generalized_log_product_order_certificate(expr, (x, y), (0, 0))
        assert cert.certified and cert.value == 0
        result = limit(expr, (x, y), (0, 0), return_result=True)
        assert result.status is LimitStatus.PROVED and result.value == 0


def test_small_exponential_power_tends_to_one():
    x, y = sp.symbols("x y", real=True)
    expr = (1 + x * y) ** (1 / (sp.Abs(x) + sp.Abs(y)))
    cert = exponential_small_power_certificate(expr, (x, y), (0, 0))
    assert cert.certified and cert.value == 1


def test_signed_variable_power_has_principal_branch_conflict():
    x, y = sp.symbols("x y", real=True)
    expr = (2 * x + y) ** sp.cot(2 * x + y - 1)
    cert = variable_power_branch_certificate(expr, (x, y), (0, 0))
    assert cert.certified and cert.method == "variable_power_branch_dne"
    assert (
        limit(expr, (x, y), (0, 0), return_result=True).status
        is LimitStatus.DOES_NOT_EXIST
    )


def test_complex_sech_reciprocal_phase_is_dne():
    x, y = sp.symbols("x y", real=True)
    expr = sp.sech((x + 1 + sp.I * y) / (x - 1 + sp.I * y)) ** 2
    cert = complex_sech_phase_certificate(expr, (x, y), (1, 0))
    assert cert.certified and cert.method == "complex_sech_phase_dne"


def test_angular_atlas_does_not_finite_cluster():
    x, y = sp.symbols("x y", real=True)
    expr = sp.asec(-(x * x + y * y))
    result = limit(expr, (x, y), (0, 0), return_result=True)
    assert result.status is LimitStatus.PROVED
    assert result.value == DirectionalInfinity(-sp.I)


def test_radial_angular_case_reaches_expensive_routes():
    x, y, z = sp.symbols("x y z", real=True)
    rho = sp.sqrt(x * x + y * y + z * z)
    result = limit(x * sp.exp(-rho) / rho, (x, y, z), (0, 0, 0), return_result=True)
    assert result.status is LimitStatus.DOES_NOT_EXIST
    assert result.evidence


def test_radial_oscillation_with_vanishing_perturbation_is_dne_0721():
    x, y = sp.symbols("x y", real=True)
    q = x * x + y * y
    expr = 2 * q ** sp.Rational(3, 2) * sp.cos(1 / q) - sp.sin(1 / q)
    result = limit(expr, (x, y), (0, 0), return_result=True)
    assert result.status is LimitStatus.DOES_NOT_EXIST
    assert any(ev.method == "vanishing_perturbation_cluster" for ev in result.evidence)


def test_reference_approach_variable():
    h, x = sp.symbols("h x", real=True)
    expr = (sp.exp(x) * sp.cos(2 * sp.sqrt(5) * h / 5) - 1) / h
    result = limit(expr, (h, x), (0, 0), return_result=True)
    assert result.status is LimitStatus.DOES_NOT_EXIST
