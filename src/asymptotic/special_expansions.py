"""Certified primitive expansions for special functions."""

from __future__ import annotations

from types import MappingProxyType

import sympy as sp
from sympy.functions.special.bessel import hankel1, hankel2

from ._symbolic_policy import bounded_ask
from .local_expansion import (
    ExpansionCertificate,
    LocalExpansion,
    SectorCertificate,
)
from .special_functions import StruveH, StruveL


def _certificate(
    *,
    sector=None,
    conditions=(),
    branch="principal",
    remainder_bound=None,
    uniform=False,
    verified=True,
):
    return ExpansionCertificate(
        sector=sector,
        parameter_conditions=tuple(conditions),
        branch=branch,
        remainder_bound=remainder_bound,
        closed_subsector_uniform=uniform,
        hypotheses_verified=verified,
    )


def _at_infinity(expr, variable, point, depth, prefix, order, certificate=None):
    if point != sp.oo or expr.args[-1] != variable:
        return None
    return LocalExpansion(
        expr,
        variable,
        point,
        sp.expand(prefix),
        order,
        depth,
        "special-function",
        certificate,
    )


def _real_axis_certificate(order):
    return _certificate(
        sector=SectorCertificate(0, 0),
        remainder_bound=order,
    )


def _bessel_coefficient(nu, k):
    """Coefficient a_k(nu) in the fixed-order Hankel expansions."""
    if k == 0:
        return sp.S.One
    factors = (4 * nu**2 - (2 * j - 1) ** 2 for j in range(1, k + 1))
    return sp.prod(factors) / (sp.factorial(k) * 8**k)


def _fixed_parameter(nu, variable):
    return not nu.has(variable) and nu.is_finite is not False


def _bessel_jy(expr, variable, point, depth, *, second_kind):
    nu, z = expr.args
    if z != variable or point != sp.oo or not _fixed_parameter(nu, variable):
        return None
    count = max(1, depth)
    phase = z - sp.pi * nu / 2 - sp.pi / 4
    even = sp.S.Zero
    odd = sp.S.Zero
    for k in range(count):
        coefficient = _bessel_coefficient(nu, k) / z**k
        if k % 2:
            odd += (-1) ** ((k - 1) // 2) * coefficient
        else:
            even += (-1) ** (k // 2) * coefficient
    scale = sp.sqrt(2 / (sp.pi * z))
    if second_kind:
        prefix = scale * (sp.sin(phase) * even + sp.cos(phase) * odd)
    else:
        prefix = scale * (sp.cos(phase) * even - sp.sin(phase) * odd)
    order = z ** (-sp.Rational(1, 2) - count)
    sector = SectorCertificate(-sp.pi, sp.pi, boundary_margin_required=True)
    cert = _certificate(
        sector=sector,
        conditions=(sp.Q.finite(nu),),
        remainder_bound=sp.exp(sp.Abs(sp.im(z)))
        * sp.Abs(z) ** (-sp.Rational(1, 2) - count),
        uniform=True,
        verified=nu.is_finite is True,
    )
    return _at_infinity(expr, z, point, depth, prefix, order, cert)


def _bessel_j(expr, variable, point, depth):
    return _bessel_jy(expr, variable, point, depth, second_kind=False)


def _bessel_y(expr, variable, point, depth):
    return _bessel_jy(expr, variable, point, depth, second_kind=True)


def _modified_bessel(expr, variable, point, depth, *, growing):
    nu, z = expr.args
    if z != variable or point != sp.oo or not _fixed_parameter(nu, variable):
        return None
    count = max(1, depth)
    terms = sp.S.Zero
    for k in range(count):
        a = _bessel_coefficient(nu, k) / z**k
        terms += (-1) ** k * a if growing else a
    if growing:
        scale = sp.exp(z) / sp.sqrt(2 * sp.pi * z)
        sector = SectorCertificate(-sp.pi / 2, sp.pi / 2, boundary_margin_required=True)
    else:
        scale = sp.sqrt(sp.pi / (2 * z)) * sp.exp(-z)
        sector = SectorCertificate(
            -3 * sp.pi / 2, 3 * sp.pi / 2, boundary_margin_required=True
        )
    prefix = scale * terms
    order = sp.Abs(scale) * z ** (-count)
    cert = _certificate(
        sector=sector,
        conditions=(sp.Q.finite(nu),),
        remainder_bound=sp.Abs(scale) * sp.Abs(z) ** (-count),
        uniform=True,
        verified=nu.is_finite is True,
    )
    return _at_infinity(expr, z, point, depth, prefix, order, cert)


def _bessel_i(expr, variable, point, depth):
    return _modified_bessel(expr, variable, point, depth, growing=True)


def _bessel_k(expr, variable, point, depth):
    return _modified_bessel(expr, variable, point, depth, growing=False)


def _airy_u(k):
    return sp.gamma(3 * k + sp.Rational(1, 2)) / (
        54**k * sp.factorial(k) * sp.gamma(k + sp.Rational(1, 2))
    )


def _airy(expr, variable, point, depth, *, bi, derivative):
    if expr.args != (variable,) or point != sp.oo:
        return None
    z = variable
    count = max(1, depth)
    zeta = sp.Rational(2, 3) * z ** sp.Rational(3, 2)
    series = sp.S.Zero
    for k in range(count):
        coefficient = _airy_u(k)
        if derivative and k:
            coefficient *= -sp.Rational(6 * k + 1, 6 * k - 1)
        sign = 1 if bi else (-1) ** k
        series += sign * coefficient / zeta**k
    if derivative:
        scale = z ** sp.Rational(1, 4) / sp.sqrt(sp.pi)
        if not bi:
            scale = -scale / 2
    else:
        scale = z ** (-sp.Rational(1, 4)) / sp.sqrt(sp.pi)
        if not bi:
            scale /= 2
    exponential = sp.exp(zeta if bi else -zeta)
    prefix = scale * exponential * series
    order = sp.Abs(scale * exponential) * zeta ** (-count)
    # The single-exponential Poincare forms are uniform away from their Stokes boundaries.
    if bi:
        sector = SectorCertificate(-sp.pi / 3, sp.pi / 3, boundary_margin_required=True)
    else:
        sector = SectorCertificate(-sp.pi, sp.pi, boundary_margin_required=True)
    cert = _certificate(
        sector=sector,
        remainder_bound=sp.Abs(scale * exponential) * sp.Abs(zeta) ** (-count),
        uniform=True,
    )
    return _at_infinity(expr, z, point, depth, prefix, order, cert)


def _airy_ai(expr, variable, point, depth):
    return _airy(expr, variable, point, depth, bi=False, derivative=False)


def _airy_ai_prime(expr, variable, point, depth):
    return _airy(expr, variable, point, depth, bi=False, derivative=True)


def _airy_bi(expr, variable, point, depth):
    return _airy(expr, variable, point, depth, bi=True, derivative=False)


def _airy_bi_prime(expr, variable, point, depth):
    return _airy(expr, variable, point, depth, bi=True, derivative=True)


def _struve_origin(expr, variable, point, depth, *, modified):
    nu, z = expr.args
    if z != variable or point != 0 or not _fixed_parameter(nu, variable):
        return None
    count = max(1, depth)
    terms = []
    for k in range(count):
        sign = 1 if modified else (-1) ** k
        terms.append(
            sign
            * (z / 2) ** (nu + 2 * k + 1)
            / (sp.gamma(k + sp.Rational(3, 2)) * sp.gamma(k + nu + sp.Rational(3, 2)))
        )
    prefix = sum(terms, sp.S.Zero)
    next_power = nu + 2 * count + 1
    order = sp.Abs(z) ** sp.re(next_power)
    cert = _certificate(
        sector=SectorCertificate(-sp.pi, sp.pi, boundary_margin_required=True),
        conditions=(sp.Q.finite(nu), sp.Q.positive(sp.re(nu) + 2 * count + 1)),
        branch="principal power z**nu",
        remainder_bound=order,
        uniform=True,
        verified=(
            nu.is_finite is True
            and bounded_ask(sp.Q.positive(sp.re(nu) + 2 * count + 1)) is True
        ),
    )
    return LocalExpansion(
        expr, z, point, prefix, order, depth, "special-function", cert
    )


def _struve_algebraic_terms(nu, z, count, *, modified):
    terms = sp.S.Zero
    for k in range(count):
        sign = (-1) ** (k + 1) if modified else 1
        terms += (
            sign
            * sp.gamma(k + sp.Rational(1, 2))
            * (z / 2) ** (nu - 2 * k - 1)
            / (sp.pi * sp.gamma(nu + sp.Rational(1, 2) - k))
        )
    return terms


def _struve_infinity(expr, variable, point, depth, *, modified):
    nu, z = expr.args
    if z != variable or point != sp.oo or not _fixed_parameter(nu, variable):
        return None
    count = max(1, depth)
    base_expr = sp.besseli(nu, z) if modified else sp.bessely(nu, z)
    base = (
        _bessel_i(base_expr, z, point, depth)
        if modified
        else _bessel_y(base_expr, z, point, depth)
    )
    if base is None:
        return None
    algebraic = _struve_algebraic_terms(nu, z, count, modified=modified)
    next_algebraic = sp.Abs(z) ** (sp.re(nu) - 2 * count - 1)
    remainder = sp.Abs(base.certificate.remainder_bound) + next_algebraic
    sector = (
        SectorCertificate(-sp.pi / 2, sp.pi / 2, boundary_margin_required=True)
        if modified
        else SectorCertificate(-sp.pi, sp.pi, boundary_margin_required=True)
    )
    cert = _certificate(
        sector=sector,
        conditions=(sp.Q.finite(nu),),
        branch="principal power z**nu",
        remainder_bound=remainder,
        uniform=True,
        verified=nu.is_finite is True,
    )
    # The real-axis order is the sum of the independently bounded Bessel and
    # algebraic remainders.
    order = sp.Abs(base.order) + z ** (sp.re(nu) - 2 * count - 1)
    return _at_infinity(expr, z, point, depth, base.prefix + algebraic, order, cert)


def _struve_h(expr, variable, point, depth):
    if point == 0:
        return _struve_origin(expr, variable, point, depth, modified=False)
    return _struve_infinity(expr, variable, point, depth, modified=False)


def _struve_l(expr, variable, point, depth):
    if point == 0:
        return _struve_origin(expr, variable, point, depth, modified=True)
    return _struve_infinity(expr, variable, point, depth, modified=True)


def _hankel(expr, variable, point, depth, *, first):
    nu, z = expr.args
    if z != variable or point != sp.oo:
        return None
    j = _bessel_j(sp.besselj(nu, z), z, point, depth)
    y = _bessel_y(sp.bessely(nu, z), z, point, depth)
    if j is None or y is None:
        return None
    sign = sp.I if first else -sp.I
    prefix = j.prefix + sign * y.prefix
    order = sp.Abs(j.order) + sp.Abs(y.order)
    cert = _certificate(
        sector=SectorCertificate(-sp.pi, sp.pi, boundary_margin_required=True),
        conditions=(sp.Q.finite(nu),),
        remainder_bound=j.certificate.remainder_bound + y.certificate.remainder_bound,
        uniform=True,
        verified=(
            j.certificate.hypotheses_verified and y.certificate.hypotheses_verified
        ),
    )
    return _at_infinity(expr, z, point, depth, prefix, order, cert)


def _hankel1(expr, variable, point, depth):
    return _hankel(expr, variable, point, depth, first=True)


def _hankel2(expr, variable, point, depth):
    return _hankel(expr, variable, point, depth, first=False)


def _erf(expr, variable, point, depth):
    if expr.args != (variable,):
        return None
    x = variable
    if point == sp.oo:
        n = max(1, depth // 2)
        corr = sum(
            (-1) ** k * sp.rf(sp.Rational(1, 2), k) / x ** (2 * k) for k in range(n)
        )
        tail = sp.exp(-(x**2)) / (sp.sqrt(sp.pi) * x ** (2 * n + 1))
        cert = _certificate(
            sector=SectorCertificate(
                -sp.pi / 4, sp.pi / 4, boundary_margin_required=True
            ),
            remainder_bound=sp.Abs(tail),
            uniform=True,
        )
        return _at_infinity(
            expr,
            x,
            point,
            depth,
            1 - sp.exp(-(x**2)) * corr / (sp.sqrt(sp.pi) * x),
            tail,
            cert,
        )
    return None


def _fresnel_c(expr, variable, point, depth):
    if expr.args != (variable,) or point != sp.oo:
        return None
    x = variable
    prefix = sp.Rational(1, 2) + sp.sin(sp.pi * x**2 / 2) / (sp.pi * x)
    return _at_infinity(
        expr, x, point, depth, prefix, x**-3, _real_axis_certificate(x**-3)
    )


def _fresnel_s(expr, variable, point, depth):
    if expr.args != (variable,) or point != sp.oo:
        return None
    x = variable
    prefix = sp.Rational(1, 2) - sp.cos(sp.pi * x**2 / 2) / (sp.pi * x)
    return _at_infinity(
        expr, x, point, depth, prefix, x**-3, _real_axis_certificate(x**-3)
    )


def _si(expr, variable, point, depth):
    if expr.args != (variable,) or point != sp.oo:
        return None
    x = variable
    prefix = sp.pi / 2 - sp.cos(x) / x - sp.sin(x) / x**2
    return _at_infinity(
        expr, x, point, depth, prefix, x**-3, _real_axis_certificate(x**-3)
    )


def _ci(expr, variable, point, depth):
    if expr.args != (variable,) or point != sp.oo:
        return None
    x = variable
    prefix = sp.sin(x) / x - sp.cos(x) / x**2
    return _at_infinity(
        expr, x, point, depth, prefix, x**-3, _real_axis_certificate(x**-3)
    )


_BUILTIN_PROVIDERS = MappingProxyType(
    {
        sp.besselj: _bessel_j,
        sp.bessely: _bessel_y,
        sp.besseli: _bessel_i,
        sp.besselk: _bessel_k,
        hankel1: _hankel1,
        hankel2: _hankel2,
        sp.airyai: _airy_ai,
        sp.airyaiprime: _airy_ai_prime,
        sp.airybi: _airy_bi,
        sp.airybiprime: _airy_bi_prime,
        StruveH: _struve_h,
        StruveL: _struve_l,
        sp.erf: _erf,
        sp.fresnelc: _fresnel_c,
        sp.fresnels: _fresnel_s,
        sp.Si: _si,
        sp.Ci: _ci,
    }
)
