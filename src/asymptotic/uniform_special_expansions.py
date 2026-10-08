"""Uniform and beyond-all-orders special-function asymptotics.

This module keeps large-parameter turning-point expansions separate from the
fixed-parameter local-expansion registry.  Its result objects record the
turning-point coordinate, branch convention, uniformity region, Stokes data,
and the scale of the first omitted contribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from .complex_domains import (
    ComplexDomainCertificate,
    bessel_complex_domain,
    modified_bessel_complex_domain,
)
from .hyperasymptotics import rescaled_terminant
from .olver_coefficients import airy_u, debye_u, debye_v, olver_coefficient, olver_zeta
from .variation_bounds import olver_error_control, turning_point_variation


@dataclass(frozen=True)
class TurningPointCertificate:
    """Geometry supporting an Airy-uniform simple-turning-point expansion."""

    turning_point: sp.Expr
    coordinate: sp.Expr
    branch: str
    domain: sp.Expr
    turning_point_uniform: bool
    hypotheses_verified: bool
    obligations: tuple[str, ...] = ()
    complex_domain: ComplexDomainCertificate | None = None


@dataclass(frozen=True)
class StokesCertificate:
    """Stokes geometry for a pair of competing exponential scales."""

    singulant: sp.Expr
    stokes_rays: tuple[sp.Expr, ...]
    anti_stokes_rays: tuple[sp.Expr, ...]
    multiplier: sp.Expr | None
    smoothing_variable: sp.Expr | None
    convention: str
    hypotheses_verified: bool


@dataclass(frozen=True)
class RemainderBound:
    """Remainder statement distinguishing asymptotic and numerical bounds."""

    constant: sp.Expr
    scale: sp.Expr
    bound: sp.Expr
    theorem: str
    order_certified: bool
    enclosure_certified: bool
    hypotheses_verified: bool


@dataclass(frozen=True)
class ExponentialRemainderCertificate:
    """Theorem-backed remainder after a terminant re-expansion."""

    truncation_index: sp.Expr
    reexpansion_terms: int
    sector_condition: sp.Expr
    remainder_scale: sp.Expr
    theorem: str
    hypotheses_verified: bool


@dataclass(frozen=True)
class UniformExpansion:
    """Finite large-parameter expansion with turning-point metadata."""

    expression: sp.Expr
    large_parameter: sp.Symbol
    scaled_variable: sp.Expr
    prefix: sp.Expr
    order: sp.Expr
    method: str
    turning_point: TurningPointCertificate | None
    stokes: tuple[StokesCertificate, ...] = ()
    exponentially_improved: bool = False
    optimal_truncation_index: sp.Expr | None = None
    exponential_error_scale: sp.Expr | None = None
    certified: bool = False
    remainder_certificate: ExponentialRemainderCertificate | None = None
    remainder_bound: RemainderBound | None = None


def _positive_large_parameter(parameter: sp.Symbol) -> bool:
    return bool(parameter.is_positive is True and parameter.is_real is True)


def _bessel_turning_coordinate(z: sp.Expr) -> sp.Expr:
    return olver_zeta(z)


def _bessel_prefactor(z: sp.Expr, zeta: sp.Expr) -> sp.Expr:
    if z == 1:
        return sp.real_root(2, 3)
    return (4 * zeta / (1 - z**2)) ** sp.Rational(1, 4)


def _turning_point_variation_constant(order: sp.Expr, z: sp.Expr) -> sp.Expr:
    """Olver variation constant without an unspecified theorem parameter."""
    if z == 1:
        return sp.S(2)
    variation = turning_point_variation("bessel_A1_B0", z)
    return 2 * sp.exp(2 * variation / sp.Abs(order) ** 2)


def bessel_j_large_order(
    order: sp.Symbol,
    z: sp.Expr,
    *,
    terms: int = 1,
) -> UniformExpansion:
    """Airy-uniform expansion of ``J_order(order*z)`` through ``z=1``.

    The implemented certified route is the real transition region
    ``order > 0`` and ``0 < z <= 1``.  The leading Olver term is uniform at
    the turning point. Higher coefficient functions are left as an explicit
    obligation until their removable singularities and remainder bounds are
    represented symbolically.
    """
    order = sp.sympify(order)
    z = sp.sympify(z)
    if terms < 1:
        raise ValueError("terms must be positive")
    zeta = _bessel_turning_coordinate(z)
    airy_arg = order ** sp.Rational(2, 3) * zeta
    prefactor = _bessel_prefactor(z, zeta)
    if z == 1:
        from .olver_coefficients import olver_coefficient_block

        coefficients = olver_coefficient_block(terms, z)
        a_coefficients = coefficients["A"]
        b_coefficients = coefficients["B"]
    else:
        a_coefficients = tuple(olver_coefficient("A", k, z) for k in range(terms))
        b_coefficients = tuple(olver_coefficient("B", k, z) for k in range(terms))
    a_series = sp.Add(
        *(
            coefficient / order ** (2 * k)
            for k, coefficient in enumerate(a_coefficients)
        )
    )
    b_series = sp.Add(
        *(
            coefficient / order ** (2 * k)
            for k, coefficient in enumerate(b_coefficients)
        )
    )
    prefix = prefactor * (
        sp.airyai(airy_arg) * a_series / order ** sp.Rational(1, 3)
        + sp.airyaiprime(airy_arg) * b_series / order ** sp.Rational(5, 3)
    )
    complex_domain = bessel_complex_domain(z)
    domain = sp.And(order > 0, complex_domain.sector)
    verified = _positive_large_parameter(order) and complex_domain.hypotheses_verified
    cert = TurningPointCertificate(
        sp.S.One,
        zeta,
        "Olver continuation on C minus the negative real axis",
        domain,
        True,
        verified,
        complex_domain=complex_domain,
    )
    remainder = sp.Abs(prefactor) * (
        sp.Abs(sp.airyai(airy_arg)) / order ** (2 * terms + sp.Rational(1, 3))
        + sp.Abs(sp.airyaiprime(airy_arg)) / order ** (2 * terms + sp.Rational(5, 3))
    )
    return UniformExpansion(
        sp.besselj(order, order * z),
        order,
        z,
        prefix,
        remainder,
        "bessel-airy-uniform",
        cert,
        certified=cert.hypotheses_verified,
        remainder_bound=RemainderBound(
            _turning_point_variation_constant(order, z),
            remainder,
            _turning_point_variation_constant(order, z) * remainder,
            "Olver Airy-uniform remainder",
            True,
            False,
            cert.hypotheses_verified,
        ),
    )


def bessel_y_large_order(
    order: sp.Symbol,
    z: sp.Expr,
    *,
    terms: int = 1,
) -> UniformExpansion:
    """Airy-uniform expansion of ``Y_order(order*z)`` through ``z=1``."""
    order = sp.sympify(order)
    z = sp.sympify(z)
    if terms < 1:
        raise ValueError("terms must be positive")
    zeta = _bessel_turning_coordinate(z)
    airy_arg = order ** sp.Rational(2, 3) * zeta
    prefactor = _bessel_prefactor(z, zeta)
    a_series = sp.Add(
        *(olver_coefficient("A", k, z) / order ** (2 * k) for k in range(terms))
    )
    b_series = sp.Add(
        *(olver_coefficient("B", k, z) / order ** (2 * k) for k in range(terms))
    )
    prefix = -prefactor * (
        sp.airybi(airy_arg) * a_series / order ** sp.Rational(1, 3)
        + sp.airybiprime(airy_arg) * b_series / order ** sp.Rational(5, 3)
    )
    complex_domain = bessel_complex_domain(z)
    domain = sp.And(order > 0, complex_domain.sector)
    verified = _positive_large_parameter(order) and complex_domain.hypotheses_verified
    cert = TurningPointCertificate(
        sp.S.One,
        zeta,
        "Olver continuation on C minus the negative real axis",
        domain,
        True,
        verified,
        complex_domain=complex_domain,
    )
    remainder = sp.Abs(prefactor) * (
        sp.Abs(sp.airybi(airy_arg)) / order ** (2 * terms + sp.Rational(1, 3))
        + sp.Abs(sp.airybiprime(airy_arg)) / order ** (2 * terms + sp.Rational(5, 3))
    )
    return UniformExpansion(
        sp.bessely(order, order * z),
        order,
        z,
        prefix,
        remainder,
        "bessel-airy-uniform",
        cert,
        certified=cert.hypotheses_verified,
        remainder_bound=RemainderBound(
            _turning_point_variation_constant(order, z),
            remainder,
            _turning_point_variation_constant(order, z) * remainder,
            "Olver Airy-uniform remainder",
            True,
            False,
            cert.hypotheses_verified,
        ),
    )


def hankel_large_order(
    order: sp.Symbol,
    z: sp.Expr,
    *,
    kind: int = 1,
    terms: int = 1,
) -> UniformExpansion:
    """Airy-uniform Hankel expansion with arbitrary Olver coefficient pairs."""
    if kind not in (1, 2):
        raise ValueError("kind must be 1 or 2")
    if terms < 1:
        raise ValueError("terms must be positive")
    order = sp.sympify(order)
    z = sp.sympify(z)
    zeta = _bessel_turning_coordinate(z)
    sign = 1 if kind == 1 else -1
    rotation = sp.exp(sign * 2 * sp.pi * sp.I / 3)
    outer = 2 * sp.exp(-sign * sp.pi * sp.I / 3)
    prefactor = _bessel_prefactor(z, zeta)
    airy_arg = rotation * order ** sp.Rational(2, 3) * zeta
    a_series = sp.Add(
        *(olver_coefficient("A", k, z) / order ** (2 * k) for k in range(terms))
    )
    b_series = sp.Add(
        *(olver_coefficient("B", k, z) / order ** (2 * k) for k in range(terms))
    )
    prefix = (
        outer
        * prefactor
        * (
            sp.airyai(airy_arg) * a_series / order ** sp.Rational(1, 3)
            + rotation
            * sp.airyaiprime(airy_arg)
            * b_series
            / order ** sp.Rational(5, 3)
        )
    )
    complex_domain = bessel_complex_domain(z)
    verified = _positive_large_parameter(order) and complex_domain.hypotheses_verified
    cert = TurningPointCertificate(
        sp.S.One,
        zeta,
        "Olver continuation on C minus the negative real axis",
        sp.And(order > 0, complex_domain.sector),
        True,
        verified,
        complex_domain=complex_domain,
    )
    func = sp.hankel1 if kind == 1 else sp.hankel2
    scale = sp.Abs(outer * prefactor) * (
        sp.Abs(sp.airyai(airy_arg)) / order ** (2 * terms + sp.Rational(1, 3))
        + sp.Abs(sp.airyaiprime(airy_arg)) / order ** (2 * terms + sp.Rational(5, 3))
    )
    constant = _turning_point_variation_constant(order, z)
    return UniformExpansion(
        func(order, order * z),
        order,
        z,
        prefix,
        scale,
        "hankel-airy-uniform",
        cert,
        certified=verified,
        remainder_bound=RemainderBound(
            constant,
            scale,
            constant * scale,
            "Olver Hankel Airy-uniform variation bound",
            True,
            False,
            verified,
        ),
    )


def hankel_prime_large_order(
    order: sp.Symbol,
    z: sp.Expr,
    *,
    kind: int = 1,
    terms: int = 1,
) -> UniformExpansion:
    """Airy-uniform expansion of the derivative of H^(1/2)_order(order*z)."""
    if kind not in (1, 2):
        raise ValueError("kind must be 1 or 2")
    if terms < 1:
        raise ValueError("terms must be positive")
    order = sp.sympify(order)
    z = sp.sympify(z)
    zeta = _bessel_turning_coordinate(z)
    sign = 1 if kind == 1 else -1
    rotation = sp.exp(sign * 2 * sp.pi * sp.I / 3)
    if z == 1:
        derivative_prefactor = sp.real_root(2, 3) ** -1
    else:
        derivative_prefactor = ((1 - z**2) / (4 * zeta)) ** sp.Rational(1, 4)
    c_series = sp.Add(
        *(olver_coefficient("C", k, z) / order ** (2 * k) for k in range(terms))
    )
    d_series = sp.Add(
        *(olver_coefficient("D", k, z) / order ** (2 * k) for k in range(terms))
    )
    airy_arg = rotation * order ** sp.Rational(2, 3) * zeta
    outer = 4 * sp.exp(-sign * 2 * sp.pi * sp.I / 3) / z
    prefix = (
        outer
        * derivative_prefactor
        * (
            sp.exp(-sign * 2 * sp.pi * sp.I / 3)
            * sp.airyai(airy_arg)
            * c_series
            / order ** sp.Rational(4, 3)
            + sp.airyaiprime(airy_arg) * d_series / order ** sp.Rational(2, 3)
        )
    )
    domain = bessel_complex_domain(z)
    verified = _positive_large_parameter(order) and domain.hypotheses_verified
    cert = TurningPointCertificate(
        sp.S.One,
        zeta,
        "Olver continuation on C minus the negative real axis",
        sp.And(order > 0, domain.sector),
        True,
        verified,
        complex_domain=domain,
    )
    x = sp.Symbol("_x")
    func = sp.hankel1 if kind == 1 else sp.hankel2
    scale = sp.Abs(outer * derivative_prefactor) * (
        sp.Abs(sp.airyai(airy_arg)) / order ** (2 * terms + sp.Rational(4, 3))
        + sp.Abs(sp.airyaiprime(airy_arg)) / order ** (2 * terms + sp.Rational(2, 3))
    )
    return UniformExpansion(
        sp.Subs(sp.diff(func(order, x), x), x, order * z),
        order,
        z,
        prefix,
        scale,
        "hankel-derivative-airy-uniform",
        cert,
        certified=verified,
        remainder_bound=RemainderBound(
            _turning_point_variation_constant(order, z),
            scale,
            _turning_point_variation_constant(order, z) * scale,
            "Olver Hankel derivative remainder",
            True,
            False,
            verified,
        ),
    )


def bessel_j_prime_large_order(
    order: sp.Symbol,
    z: sp.Expr,
    *,
    terms: int = 1,
) -> UniformExpansion:
    """Airy-uniform expansion of J'_order(order*z), including C_k and D_k."""
    order = sp.sympify(order)
    z = sp.sympify(z)
    if terms < 1:
        raise ValueError("terms must be positive")
    zeta = _bessel_turning_coordinate(z)
    airy_arg = order ** sp.Rational(2, 3) * zeta
    if z == 1:
        derivative_prefactor = sp.real_root(2, 3) ** -1
    else:
        derivative_prefactor = ((1 - z**2) / (4 * zeta)) ** sp.Rational(1, 4)
    c_series = sp.Add(
        *(olver_coefficient("C", k, z) / order ** (2 * k) for k in range(terms))
    )
    d_series = sp.Add(
        *(olver_coefficient("D", k, z) / order ** (2 * k) for k in range(terms))
    )
    prefix = -(
        2
        / z
        * derivative_prefactor
        * (
            sp.airyai(airy_arg) * c_series / order ** sp.Rational(4, 3)
            + sp.airyaiprime(airy_arg) * d_series / order ** sp.Rational(2, 3)
        )
    )
    complex_domain = bessel_complex_domain(z)
    verified = _positive_large_parameter(order) and complex_domain.hypotheses_verified
    cert = TurningPointCertificate(
        sp.S.One,
        zeta,
        "Olver continuation on C minus the negative real axis",
        sp.And(order > 0, complex_domain.sector),
        True,
        verified,
        complex_domain=complex_domain,
    )
    remainder = sp.Abs(2 / z * derivative_prefactor) * (
        sp.Abs(sp.airyai(airy_arg)) / order ** (2 * terms + sp.Rational(4, 3))
        + sp.Abs(sp.airyaiprime(airy_arg)) / order ** (2 * terms + sp.Rational(2, 3))
    )
    return UniformExpansion(
        sp.Subs(
            sp.diff(sp.besselj(order, sp.Symbol("_x")), sp.Symbol("_x")),
            sp.Symbol("_x"),
            order * z,
        ),
        order,
        z,
        prefix,
        remainder,
        "bessel-derivative-airy-uniform",
        cert,
        certified=verified,
        remainder_bound=RemainderBound(
            _turning_point_variation_constant(order, z),
            remainder,
            _turning_point_variation_constant(order, z) * remainder,
            "Olver derivative Airy-uniform remainder",
            True,
            False,
            verified,
        ),
    )


def bessel_y_prime_large_order(
    order: sp.Symbol,
    z: sp.Expr,
    *,
    terms: int = 1,
) -> UniformExpansion:
    """Airy-uniform expansion of Y'_order(order*z), including C_k and D_k."""
    order = sp.sympify(order)
    z = sp.sympify(z)
    if terms < 1:
        raise ValueError("terms must be positive")
    zeta = _bessel_turning_coordinate(z)
    airy_arg = order ** sp.Rational(2, 3) * zeta
    if z == 1:
        derivative_prefactor = sp.real_root(2, 3) ** -1
    else:
        derivative_prefactor = ((1 - z**2) / (4 * zeta)) ** sp.Rational(1, 4)
    c_series = sp.Add(
        *(olver_coefficient("C", k, z) / order ** (2 * k) for k in range(terms))
    )
    d_series = sp.Add(
        *(olver_coefficient("D", k, z) / order ** (2 * k) for k in range(terms))
    )
    prefix = (
        2
        / z
        * derivative_prefactor
        * (
            sp.airybi(airy_arg) * c_series / order ** sp.Rational(4, 3)
            + sp.airybiprime(airy_arg) * d_series / order ** sp.Rational(2, 3)
        )
    )
    complex_domain = bessel_complex_domain(z)
    verified = _positive_large_parameter(order) and complex_domain.hypotheses_verified
    cert = TurningPointCertificate(
        sp.S.One,
        zeta,
        "Olver continuation on C minus the negative real axis",
        sp.And(order > 0, complex_domain.sector),
        True,
        verified,
        complex_domain=complex_domain,
    )
    remainder = sp.Abs(2 / z * derivative_prefactor) * (
        sp.Abs(sp.airybi(airy_arg)) / order ** (2 * terms + sp.Rational(4, 3))
        + sp.Abs(sp.airybiprime(airy_arg)) / order ** (2 * terms + sp.Rational(2, 3))
    )
    return UniformExpansion(
        sp.Subs(
            sp.diff(sp.bessely(order, sp.Symbol("_x")), sp.Symbol("_x")),
            sp.Symbol("_x"),
            order * z,
        ),
        order,
        z,
        prefix,
        remainder,
        "bessel-derivative-airy-uniform",
        cert,
        certified=verified,
        remainder_bound=RemainderBound(
            _turning_point_variation_constant(order, z),
            remainder,
            _turning_point_variation_constant(order, z) * remainder,
            "Olver derivative Airy-uniform remainder",
            True,
            False,
            verified,
        ),
    )


def _modified_eta(z: sp.Expr) -> sp.Expr:
    root = sp.sqrt(1 + z**2)
    return root + sp.log(z / (1 + root))


def _modified_series(
    z: sp.Expr, terms: int, derivative: bool, alternating: bool
) -> sp.Expr:
    p = 1 / sp.sqrt(1 + z**2)
    poly = debye_v if derivative else debye_u
    return sp.Add(
        *(
            (-1 if alternating else 1) ** k
            * poly(k).subs(sp.Symbol("p"), p)
            / sp.Symbol("_nu") ** k
            for k in range(terms)
        )
    )


def _modified_bessel_large_order(
    order: sp.Symbol,
    z: sp.Expr,
    *,
    kind: str,
    derivative: bool,
    terms: int,
) -> UniformExpansion:
    if terms < 1:
        raise ValueError("terms must be positive")
    order = sp.sympify(order)
    z = sp.sympify(z)
    eta = _modified_eta(z)
    p = 1 / sp.sqrt(1 + z**2)
    poly = debye_v if derivative else debye_u
    alternating = kind == "K"
    series = sp.Add(
        *(
            (-1 if alternating else 1) ** k * poly(k).subs(sp.Symbol("p"), p) / order**k
            for k in range(terms)
        )
    )
    if derivative:
        amplitude = (1 + z**2) ** sp.Rational(1, 4) / z
    else:
        amplitude = (1 + z**2) ** -sp.Rational(1, 4)
    if kind == "I":
        normalization = 1 / sp.sqrt(2 * sp.pi * order)
        exponential = sp.exp(order * eta)
        prefix = normalization * amplitude * exponential * series
        func = sp.besseli
    else:
        normalization = sp.sqrt(sp.pi / (2 * order))
        exponential = sp.exp(-order * eta)
        prefix = (
            (-1 if derivative else 1) * normalization * amplitude * exponential * series
        )
        func = sp.besselk
    x = sp.Symbol("_x")
    expression = (
        sp.Subs(sp.diff(func(order, x), x), x, order * z)
        if derivative
        else func(order, order * z)
    )
    domain = modified_bessel_complex_domain(z)
    verified = _positive_large_parameter(order) and domain.hypotheses_verified
    next_term = sp.Abs(
        normalization
        * amplitude
        * exponential
        * poly(terms).subs(sp.Symbol("p"), p)
        / order**terms
    )
    if verified:
        error_control = olver_error_control(
            poly(1),
            poly(terms),
            p,
            order,
            terms,
            amplitude=normalization * amplitude * exponential,
        )
        bound = RemainderBound(
            error_control.constant,
            error_control.omitted_variation.bound
            * sp.Abs(normalization * amplitude * exponential)
            / sp.Abs(order) ** terms,
            error_control.bound,
            "Olver total-variation bound for the modified-Bessel Debye expansion",
            True,
            bool(error_control.hypotheses_verified and z.is_positive is True),
            bool(error_control.hypotheses_verified),
        )
    else:
        bound = RemainderBound(
            sp.oo,
            next_term,
            sp.oo,
            "Olver total-variation bound outside a verified admissible path",
            True,
            False,
            False,
        )
    stokes = StokesCertificate(
        2 * order * eta,
        (-sp.pi / 2, sp.pi / 2),
        (sp.S.Zero, sp.pi),
        None,
        None,
        "principal modified-Bessel Debye branch",
        verified,
    )
    return UniformExpansion(
        expression,
        order,
        z,
        prefix,
        next_term,
        f"modified-bessel-{kind.lower()}-uniform"
        + ("-derivative" if derivative else ""),
        None,
        (stokes,),
        certified=verified,
        remainder_bound=bound,
    )


def modified_bessel_i_large_order(
    order: sp.Symbol, z: sp.Expr, *, terms: int = 4
) -> UniformExpansion:
    """Uniform Debye expansion of I_order(order*z)."""
    return _modified_bessel_large_order(
        order, z, kind="I", derivative=False, terms=terms
    )


def modified_bessel_k_large_order(
    order: sp.Symbol, z: sp.Expr, *, terms: int = 4
) -> UniformExpansion:
    """Uniform Debye expansion of K_order(order*z)."""
    return _modified_bessel_large_order(
        order, z, kind="K", derivative=False, terms=terms
    )


def modified_bessel_i_prime_large_order(
    order: sp.Symbol, z: sp.Expr, *, terms: int = 4
) -> UniformExpansion:
    """Uniform Debye expansion of I'_order(order*z)."""
    return _modified_bessel_large_order(
        order, z, kind="I", derivative=True, terms=terms
    )


def modified_bessel_k_prime_large_order(
    order: sp.Symbol, z: sp.Expr, *, terms: int = 4
) -> UniformExpansion:
    """Uniform Debye expansion of K'_order(order*z)."""
    return _modified_bessel_large_order(
        order, z, kind="K", derivative=True, terms=terms
    )


def bessel_j_transition(
    order: sp.Symbol,
    offset: sp.Expr,
) -> UniformExpansion:
    """Large-order transition expansion for ``J_order(order+a*order**(1/3))``."""
    order = sp.sympify(order)
    offset = sp.sympify(offset)
    airy_arg = -sp.real_root(2, 3) * offset
    prefix = sp.real_root(2, 3) * sp.airyai(airy_arg) / order ** sp.Rational(1, 3)
    verified = _positive_large_parameter(order) and offset.is_real is True
    cert = TurningPointCertificate(
        sp.S.One,
        -sp.real_root(2, 3) * offset / order ** sp.Rational(2, 3),
        "real transition scaling",
        sp.And(order > 0, sp.Q.real(offset)),
        True,
        verified,
    )
    return UniformExpansion(
        sp.besselj(order, order + offset * order ** sp.Rational(1, 3)),
        order,
        offset,
        prefix,
        order**-1,
        "bessel-transition",
        cert,
        certified=verified,
    )


def _airy_u_symbolic(k: sp.Expr) -> sp.Expr:
    return (
        sp.rf(sp.Rational(1, 6), k)
        * sp.rf(sp.Rational(5, 6), k)
        / (sp.factorial(k) * 2**k)
    )


def airy_ai_exponentially_improved(
    z: sp.Expr,
    *,
    reexpansion_terms: int = 2,
) -> UniformExpansion:
    """Exponentially improved Ai expansion with its terminant remainder.

    The optimal Poincare truncation is n=floor(2*|xi|). The retained terminant
    re-expansion has a theorem-backed residual
    O(exp(-2*|xi|)*xi**(-m)) for |arg(z)| <= 2*pi/3.
    """
    z = sp.sympify(z)
    if reexpansion_terms < 1:
        raise ValueError("reexpansion_terms must be positive")
    xi = sp.Rational(2, 3) * z ** sp.Rational(3, 2)
    n = sp.floor(2 * sp.Abs(xi))
    k = sp.Symbol("_k", integer=True, nonnegative=True)
    poincare = sp.Sum(
        (-1) ** k * _airy_u_symbolic(k) / xi**k,
        (k, 0, n - 1),
    )
    terminant_sum = sp.Add(
        *(
            (-1) ** (n + j) * airy_u(j) * rescaled_terminant(n - j, 2 * xi) / xi**j
            for j in range(reexpansion_terms)
        )
    )
    scale = sp.exp(-xi) / (2 * sp.sqrt(sp.pi) * z ** sp.Rational(1, 4))
    prefix = scale * (poincare + terminant_sum)
    sector_condition = sp.Abs(sp.arg(z)) <= 2 * sp.pi / 3
    verified = z.is_positive is True
    residual_scale = (
        sp.Abs(scale) * sp.exp(-2 * sp.Abs(xi)) * sp.Abs(xi) ** (-reexpansion_terms)
    )
    remainder = ExponentialRemainderCertificate(
        n,
        reexpansion_terms,
        sector_condition,
        residual_scale,
        "Airy terminant remainder",
        verified,
    )
    stokes = StokesCertificate(
        2 * xi,
        (-2 * sp.pi / 3, 2 * sp.pi / 3),
        (-sp.pi / 3, sp.pi / 3, sp.pi),
        None,
        None,
        "principal Airy branch; terminant carries Stokes smoothing",
        verified,
    )
    return UniformExpansion(
        sp.airyai(z),
        sp.Symbol("_airy_scale", positive=True),
        z,
        prefix,
        residual_scale,
        "airy-terminant",
        None,
        (stokes,),
        True,
        n,
        residual_scale,
        verified,
        remainder,
    )


def airy_stokes_expansion(
    z: sp.Expr,
    *,
    kind: str = "Ai",
    terms: int = 4,
    exponentially_improved: bool = False,
) -> UniformExpansion:
    """Airy large-argument expansion with explicit Stokes geometry."""
    z = sp.sympify(z)
    if kind != "Ai":
        raise NotImplementedError("the Stokes-aware layer currently implements Ai")
    if terms < 1:
        raise ValueError("terms must be positive")
    if exponentially_improved:
        return airy_ai_exponentially_improved(z, reexpansion_terms=terms)
    xi = sp.Rational(2, 3) * z ** sp.Rational(3, 2)
    series = sp.Add(*((-1) ** k * airy_u(k) / xi**k for k in range(terms)))
    prefix = sp.exp(-xi) * series / (2 * sp.sqrt(sp.pi) * z ** sp.Rational(1, 4))
    stokes = StokesCertificate(
        2 * xi,
        (-2 * sp.pi / 3, 2 * sp.pi / 3),
        (-sp.pi / 3, sp.pi / 3, sp.pi),
        None,
        None,
        "principal Airy branch",
        True,
    )
    return UniformExpansion(
        sp.airyai(z),
        sp.Symbol("_airy_scale", positive=True),
        z,
        prefix,
        sp.Abs(prefix / xi**terms),
        "airy-stokes",
        None,
        (stokes,),
        certified=True,
    )


def optimally_truncated_airy_ai(z: sp.Expr) -> UniformExpansion:
    """Return the certified first-level terminant improvement of Airy Ai."""
    return airy_ai_exponentially_improved(z, reexpansion_terms=2)


def _hankel_a(index: sp.Expr, nu: sp.Expr) -> sp.Expr:
    return (
        sp.rf(sp.Rational(1, 2) - nu, index)
        * sp.rf(sp.Rational(1, 2) + nu, index)
        / ((-2) ** index * sp.factorial(index))
    )


def hankel_exponentially_improved(
    nu: sp.Expr,
    z: sp.Expr,
    *,
    kind: int = 1,
    reexpansion_terms: int = 2,
) -> UniformExpansion:
    """Exponentially improved fixed-order Hankel expansion using terminants."""
    if kind not in (1, 2):
        raise ValueError("kind must be 1 or 2")
    if reexpansion_terms < 1:
        raise ValueError("reexpansion_terms must be positive")
    nu = sp.sympify(nu)
    z = sp.sympify(z)
    ell = sp.floor(2 * sp.Abs(z))
    k = sp.Symbol("_k", integer=True, nonnegative=True)
    sign = 1 if kind == 1 else -1
    phase = z - sp.pi * nu / 2 - sp.pi / 4
    poincare = sp.Sum(
        (sign * sp.I) ** k * _hankel_a(k, nu) / z**k,
        (k, 0, ell - 1),
    )
    terminant_sum = sp.Add(
        *(
            (sign * sp.I) ** j
            * _hankel_a(j, nu)
            * rescaled_terminant(ell - j, -sign * 2 * sp.I * z)
            / z**j
            for j in range(reexpansion_terms)
        )
    )
    correction = (-1) ** ell * 2 * sp.cos(sp.pi * nu) * terminant_sum
    scale = sp.sqrt(2 / (sp.pi * z)) * sp.exp(sign * sp.I * phase)
    prefix = scale * (poincare + correction)
    # This is the sector of the residual R_{m,ell}^{+/-}; positive real z is
    # a directly verified subset for both signs.
    sector_condition = sp.Abs(sp.arg(z * sp.exp(-sign * sp.pi * sp.I / 2))) <= sp.pi
    verified = z.is_positive is True and nu.is_finite is True
    residual_scale = (
        sp.Abs(scale)
        * 2
        * sp.Abs(sp.cos(sp.pi * nu))
        * sp.exp(-2 * sp.Abs(z))
        * sp.Abs(z) ** (-reexpansion_terms)
    )
    remainder = ExponentialRemainderCertificate(
        ell,
        reexpansion_terms,
        sector_condition,
        residual_scale,
        "Hankel terminant remainder",
        verified,
    )
    func = sp.hankel1 if kind == 1 else sp.hankel2
    stokes = StokesCertificate(
        -sign * 2 * sp.I * z,
        (sign * sp.pi / 2,),
        (sp.S.Zero, sp.pi),
        None,
        None,
        "principal square-root branch; terminant carries Stokes smoothing",
        verified,
    )
    return UniformExpansion(
        func(nu, z),
        sp.Symbol("_hankel_scale", positive=True),
        z,
        prefix,
        residual_scale,
        "hankel-terminant",
        None,
        (stokes,),
        True,
        ell,
        residual_scale,
        verified,
        remainder,
    )


def bessel_j_exponentially_improved(
    nu: sp.Expr,
    z: sp.Expr,
    *,
    reexpansion_terms: int = 2,
) -> UniformExpansion:
    """Exponentially improved fixed-order J_nu from the two Hankel sectors."""
    first = hankel_exponentially_improved(
        nu, z, kind=1, reexpansion_terms=reexpansion_terms
    )
    second = hankel_exponentially_improved(
        nu, z, kind=2, reexpansion_terms=reexpansion_terms
    )
    prefix = (first.prefix + second.prefix) / 2
    remainder_scale = (first.order + second.order) / 2
    verified = first.certified and second.certified
    cert = ExponentialRemainderCertificate(
        first.optimal_truncation_index,
        reexpansion_terms,
        sp.And(
            first.remainder_certificate.sector_condition,
            second.remainder_certificate.sector_condition,
        ),
        remainder_scale,
        "Bessel J from certified Hankel terminant remainders",
        verified,
    )
    return UniformExpansion(
        sp.besselj(sp.sympify(nu), sp.sympify(z)),
        first.large_parameter,
        sp.sympify(z),
        prefix,
        remainder_scale,
        "bessel-j-terminant",
        None,
        first.stokes + second.stokes,
        True,
        first.optimal_truncation_index,
        remainder_scale,
        verified,
        cert,
    )


def bessel_y_exponentially_improved(
    nu: sp.Expr,
    z: sp.Expr,
    *,
    reexpansion_terms: int = 2,
) -> UniformExpansion:
    """Exponentially improved fixed-order Y_nu from the two Hankel sectors."""
    first = hankel_exponentially_improved(
        nu, z, kind=1, reexpansion_terms=reexpansion_terms
    )
    second = hankel_exponentially_improved(
        nu, z, kind=2, reexpansion_terms=reexpansion_terms
    )
    prefix = (first.prefix - second.prefix) / (2 * sp.I)
    remainder_scale = (first.order + second.order) / 2
    verified = first.certified and second.certified
    cert = ExponentialRemainderCertificate(
        first.optimal_truncation_index,
        reexpansion_terms,
        sp.And(
            first.remainder_certificate.sector_condition,
            second.remainder_certificate.sector_condition,
        ),
        remainder_scale,
        "Bessel Y from certified Hankel terminant remainders",
        verified,
    )
    return UniformExpansion(
        sp.bessely(sp.sympify(nu), sp.sympify(z)),
        first.large_parameter,
        sp.sympify(z),
        prefix,
        remainder_scale,
        "bessel-y-terminant",
        None,
        first.stokes + second.stokes,
        True,
        first.optimal_truncation_index,
        remainder_scale,
        verified,
        cert,
    )
