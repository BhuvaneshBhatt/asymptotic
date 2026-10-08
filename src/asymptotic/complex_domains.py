"""Branch and domain data for uniform Bessel asymptotics."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


@dataclass(frozen=True)
class ComplexDomainCertificate:
    """A proved or stated complex domain for a branch-sensitive expansion."""

    cut: str
    sector: sp.Expr
    branch: str
    component: str
    boundary_margin: sp.Expr | None
    conjugation_symmetric: bool
    hypotheses_verified: bool
    obligations: tuple[str, ...] = ()


def principal_cut_component(z: sp.Expr) -> str:
    """Classify a concrete point relative to the negative-real branch cut."""
    z = sp.sympify(z)
    if z.is_positive is True:
        return "positive-real"
    if z.is_real is True and z.is_negative is True:
        return "cut"
    im = sp.im(z)
    if im.is_positive is True:
        return "upper"
    if im.is_negative is True:
        return "lower"
    return "symbolic"


def principal_sector_verified(z: sp.Expr, half_angle: sp.Expr) -> bool:
    """Prove sector membership for concrete/numerically decidable points."""
    z = sp.sympify(z)
    if z == 0:
        return False
    if z.is_positive is True:
        return True
    if z.is_number:
        angle = complex(sp.N(z, 30))
        import cmath

        return abs(cmath.phase(angle)) < float(sp.N(half_angle, 30))
    return False


def bessel_complex_domain(
    z: sp.Expr, margin: sp.Expr = sp.S.Zero
) -> ComplexDomainCertificate:
    """Principal Olver domain inside the negative-axis cut."""
    z = sp.sympify(z)
    margin = sp.sympify(margin)
    half = sp.pi - margin
    component = principal_cut_component(z)
    verified = component != "cut" and principal_sector_verified(z, half)
    return ComplexDomainCertificate(
        "negative real axis",
        sp.Abs(sp.arg(z)) <= half,
        "analytic continuation from positive real axis",
        component,
        margin,
        True,
        verified,
    )


def modified_bessel_complex_domain(
    z: sp.Expr, margin: sp.Expr = sp.S.Zero
) -> ComplexDomainCertificate:
    """Principal Debye domain for I/K uniform large-order expansions."""
    z = sp.sympify(z)
    margin = sp.sympify(margin)
    half = sp.pi / 2 - margin
    component = principal_cut_component(z)
    verified = component != "cut" and principal_sector_verified(z, half)
    return ComplexDomainCertificate(
        "negative real axis and turning points z=+/-i",
        sp.Abs(sp.arg(z)) <= half,
        "principal sqrt/log continued from positive real axis",
        component,
        margin,
        True,
        verified,
    )
