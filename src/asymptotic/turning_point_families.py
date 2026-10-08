"""Shared architecture for Airy-uniform simple-turning-point families."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from ._symbolic_policy import bounded_ask


@dataclass(frozen=True)
class TurningPointFamily:
    """Liouville--Green data needed by a simple-turning-point expansion."""

    name: str
    large_parameter: sp.Expr
    variable: sp.Expr
    turning_points: tuple[sp.Expr, ...]
    coordinate: sp.Expr
    prefactor: sp.Expr
    airy_argument: sp.Expr
    branch: str
    domain: sp.Expr
    coefficient_family: str
    hypotheses_verified: bool


@dataclass(frozen=True)
class TurningPointAdapter:
    """A special-function expression attached to generic turning-point data."""

    expression: sp.Expr
    family: TurningPointFamily
    leading_approximation: sp.Expr
    remainder_scale: sp.Expr
    certified: bool


def parabolic_cylinder_turning_point(mu: sp.Expr, t: sp.Expr) -> TurningPointAdapter:
    """Leading Airy-uniform adapter for U(-mu**2/2, mu*t*sqrt(2))."""
    mu = sp.sympify(mu)
    t = sp.sympify(t)
    if t == 1:
        zeta = sp.S.Zero
        phi = sp.real_root(2, 3) ** -sp.Rational(1, 4)
    elif t.is_real is True and bounded_ask(sp.Q.nonpositive(t - 1)) is True:
        eta = sp.acos(t) / 2 - t * sp.sqrt(1 - t**2) / 2
        zeta = -((sp.Rational(3, 2) * eta) ** sp.Rational(2, 3))
        phi = (zeta / (t**2 - 1)) ** sp.Rational(1, 4)
    else:
        xi = t * sp.sqrt(t**2 - 1) / 2 - sp.acosh(t) / 2
        zeta = (sp.Rational(3, 2) * xi) ** sp.Rational(2, 3)
        phi = (zeta / (t**2 - 1)) ** sp.Rational(1, 4)
    g = sp.gamma(sp.Rational(1, 2) + mu**2 / 2) ** sp.Rational(1, 2) / (
        2 ** (mu**2 / 4 + sp.Rational(1, 4)) * sp.sqrt(sp.pi)
    )
    leading = (
        2
        * sp.sqrt(sp.pi)
        * mu ** sp.Rational(1, 3)
        * g
        * phi
        * sp.airyai(mu ** sp.Rational(4, 3) * zeta)
    )
    verified = (
        mu.is_positive is True
        and t.is_real is True
        and bounded_ask(sp.Q.positive(t + 1)) is True
    )
    family = TurningPointFamily(
        "parabolic-cylinder-U",
        mu,
        t,
        (sp.S.One,),
        zeta,
        phi,
        mu ** sp.Rational(4, 3) * zeta,
        "real continuation through t=1",
        sp.And(mu > 0, t > -1),
        "Airy A/B",
        bool(verified),
    )
    expression = sp.Function("ParabolicCylinderU")(-(mu**2) / 2, mu * t * sp.sqrt(2))
    return TurningPointAdapter(
        expression, family, leading, sp.Abs(leading) / mu**4, bool(verified)
    )


def whittaker_turning_point(
    kappa: sp.Expr, mu: sp.Expr, x: sp.Expr
) -> TurningPointFamily:
    """Return the shared turning-point geometry descriptor for large-kappa Whittaker problems."""
    kappa, mu, x = map(sp.sympify, (kappa, mu, x))
    discriminant = sp.sqrt(kappa**2 - mu**2)
    left = 2 * kappa - 2 * discriminant
    right = 2 * kappa + 2 * discriminant
    verified = kappa.is_positive is True and mu.is_nonnegative is True
    return TurningPointFamily(
        "whittaker-large-kappa",
        kappa,
        x,
        (left, right),
        sp.Symbol("zeta"),
        sp.Symbol("Phi", positive=True),
        sp.Symbol("zeta") * kappa ** sp.Rational(2, 3),
        "principal Whittaker/Liouville branch",
        sp.And(kappa > 0, mu >= 0, mu < kappa),
        "Airy A/B",
        bool(verified),
    )


def associated_legendre_turning_point(
    nu: sp.Expr, mu: sp.Expr, x: sp.Expr
) -> TurningPointFamily:
    """Return a common large-degree/order turning-point descriptor for associated Legendre functions."""
    nu, mu, x = map(sp.sympify, (nu, mu, x))
    lam = nu + sp.Rational(1, 2)
    alpha = mu / lam
    turning = sp.sqrt(1 - alpha**2)
    verified = lam.is_positive is True and mu.is_nonnegative is True
    return TurningPointFamily(
        "associated-legendre",
        lam,
        x,
        (-turning, turning),
        sp.Symbol("zeta"),
        sp.Symbol("Phi", positive=True),
        sp.Symbol("zeta") * lam ** sp.Rational(2, 3),
        "principal Ferrers/Legendre branch",
        sp.And(lam > 0, mu >= 0, mu < lam),
        "Airy/Bessel transition",
        bool(verified),
    )


def whittaker_m_large_kappa(
    kappa: sp.Expr, mu: sp.Expr, x: sp.Expr
) -> TurningPointAdapter:
    """Uniform leading large-kappa approximation of M_(kappa,mu)(x)."""
    family = whittaker_turning_point(kappa, mu, x)
    kappa, mu, x = map(sp.sympify, (kappa, mu, x))
    leading = (
        sp.sqrt(x)
        * sp.gamma(2 * mu + 1)
        * kappa ** (-mu)
        * sp.besselj(2 * mu, 2 * sp.sqrt(x * kappa))
    )
    expression = sp.Function("WhittakerM")(kappa, mu, x)
    scale = sp.Abs(leading) / sp.sqrt(kappa)
    return TurningPointAdapter(
        expression, family, leading, scale, family.hypotheses_verified
    )


def associated_legendre_large_degree(
    nu: sp.Expr, mu: sp.Expr, theta: sp.Expr
) -> TurningPointAdapter:
    """Uniform fixed-order large-degree Ferrers-P approximation."""
    x = sp.cos(theta)
    family = associated_legendre_turning_point(nu, mu, x)
    nu, mu, theta = map(sp.sympify, (nu, mu, theta))
    leading = (
        nu ** (-mu)
        * sp.sqrt(theta / sp.sin(theta))
        * sp.besselj(mu, (nu + sp.Rational(1, 2)) * theta)
    )
    expression = sp.assoc_legendre(nu, -mu, sp.cos(theta))
    scale = sp.Abs(leading) / nu
    verified = family.hypotheses_verified and theta.is_positive is True
    return TurningPointAdapter(expression, family, leading, scale, bool(verified))
