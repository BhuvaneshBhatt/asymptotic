import sympy as sp

from asymptotic.local_expansion import (
    SectorCertificate,
    adaptive_path_limit,
    local_series,
)
from asymptotic.special_functions import StruveH, StruveL


def test_bessel_j_fixed_order_hankel_expansion():
    x = sp.symbols("x", positive=True)
    expansion = local_series(sp.besselj(0, x), x, sp.oo, depth=3)
    assert expansion is not None
    phase = x - sp.pi / 4
    expected = sp.sqrt(2 / (sp.pi * x)) * (
        sp.cos(phase) + sp.sin(phase) / (8 * x) - 9 * sp.cos(phase) / (128 * x**2)
    )
    assert sp.simplify(expansion.prefix - expected) == 0
    assert expansion.certificate.hypotheses_verified
    assert expansion.certificate.sector == SectorCertificate(-sp.pi, sp.pi, True)


def test_bessel_parameter_condition_is_not_assumed():
    x = sp.symbols("x", positive=True)
    nu = sp.symbols("nu")
    expansion = local_series(sp.besselj(nu, x), x, sp.oo, depth=2)
    assert expansion is not None
    assert not expansion.certificate.hypotheses_verified


def test_modified_bessel_k_has_wide_principal_sector():
    x = sp.symbols("x", positive=True)
    expansion = local_series(sp.besselk(2, x), x, sp.oo, depth=2)
    assert expansion is not None
    assert expansion.certificate.sector == SectorCertificate(
        -3 * sp.pi / 2, 3 * sp.pi / 2, True
    )
    assert sp.limit(expansion.order, x, sp.oo) == 0


def test_airy_ai_coefficients_and_sector():
    x = sp.symbols("x", positive=True)
    expansion = local_series(sp.airyai(x), x, sp.oo, depth=2)
    assert expansion is not None
    zeta = sp.Rational(2, 3) * x ** sp.Rational(3, 2)
    expected = (
        sp.exp(-zeta)
        / (2 * sp.sqrt(sp.pi) * x ** sp.Rational(1, 4))
        * (1 - sp.Rational(5, 72) / zeta)
    )
    assert sp.simplify(expansion.prefix - expected) == 0
    assert expansion.certificate.sector == SectorCertificate(-sp.pi, sp.pi, True)


def test_airy_ai_prime_uses_derivative_coefficients():
    x = sp.symbols("x", positive=True)
    expansion = local_series(sp.airyaiprime(x), x, sp.oo, depth=2)
    assert expansion is not None
    zeta = sp.Rational(2, 3) * x ** sp.Rational(3, 2)
    expected = (
        -(x ** sp.Rational(1, 4))
        * sp.exp(-zeta)
        / (2 * sp.sqrt(sp.pi))
        * (1 + sp.Rational(7, 72) / zeta)
    )
    assert sp.simplify(expansion.prefix - expected) == 0


def test_struve_h_origin_series_and_branch_certificate():
    z = sp.symbols("z", positive=True)
    expansion = local_series(StruveH(0, z), z, 0, depth=2)
    assert expansion is not None
    expected = 2 * z / sp.pi - 2 * z**3 / (9 * sp.pi)
    assert sp.simplify(expansion.prefix - expected) == 0
    assert expansion.certificate.branch == "principal power z**nu"
    assert expansion.certificate.hypotheses_verified


def test_modified_struve_origin_signs():
    z = sp.symbols("z", positive=True)
    expansion = local_series(StruveL(0, z), z, 0, depth=2)
    assert expansion is not None
    expected = 2 * z / sp.pi + 2 * z**3 / (9 * sp.pi)
    assert sp.simplify(expansion.prefix - expected) == 0


def test_unverified_parameter_certificate_cannot_drive_path_proof():
    t = sp.symbols("t", positive=True)
    nu = sp.symbols("nu")
    assert adaptive_path_limit(sp.besselk(nu, t), t, sp.oo, depths=(2,)) is None


def test_struve_h_infinity_combines_bessel_and_algebraic_parts():
    x = sp.symbols("x", positive=True)
    expansion = local_series(StruveH(0, x), x, sp.oo, depth=1)
    assert expansion is not None
    bessel = local_series(sp.bessely(0, x), x, sp.oo, depth=1)
    assert sp.simplify(expansion.prefix - bessel.prefix - 2 / (sp.pi * x)) == 0
    assert expansion.certificate.sector == SectorCertificate(-sp.pi, sp.pi, True)


def test_modified_struve_infinity_combines_i_and_algebraic_parts():
    x = sp.symbols("x", positive=True)
    expansion = local_series(StruveL(0, x), x, sp.oo, depth=1)
    assert expansion is not None
    bessel = local_series(sp.besseli(0, x), x, sp.oo, depth=1)
    assert sp.simplify(expansion.prefix - bessel.prefix + 2 / (sp.pi * x)) == 0
    assert expansion.certificate.sector == SectorCertificate(
        -sp.pi / 2, sp.pi / 2, True
    )


def test_hankel_provider_reuses_fixed_order_bessel_expansions():
    from sympy.functions.special.bessel import hankel1

    x = sp.symbols("x", positive=True)
    expansion = local_series(hankel1(0, x), x, sp.oo, depth=2)
    assert expansion is not None
    j = local_series(sp.besselj(0, x), x, sp.oo, depth=2)
    y = local_series(sp.bessely(0, x), x, sp.oo, depth=2)
    assert sp.simplify(expansion.prefix - j.prefix - sp.I * y.prefix) == 0
