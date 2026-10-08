"""Certified leading asymptotics for multidimensional Laplace and saddle integrals.

This module complements :mod:`multivariate_laplace` with explicit interior
stationary-phase, orthant boundary/corner Laplace, and Morse--Bott geometry.
The implementation is theorem-shaped: it certifies only cases
whose local hypotheses can be proved symbolically.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import sympy as sp

SaddleKind = Literal["interior", "boundary", "corner", "morse-bott"]


@dataclass(frozen=True)
class SaddleCertificate:
    """Local geometric evidence for a multidimensional saddle or minimum."""

    kind: SaddleKind
    point: tuple[sp.Expr, ...]
    stationary_directions: tuple[int, ...]
    active_boundaries: tuple[int, ...]
    hessian: sp.Matrix
    determinant: sp.Expr
    signature: int | None
    certified: bool
    obligations: tuple[str, ...] = ()


@dataclass(frozen=True)
class SaddleAsymptoticResult:
    """Leading asymptotic term together with its local saddle certificate."""

    expression: sp.Expr | None
    parameter: sp.Symbol
    certificate: SaddleCertificate
    relative_remainder_order: sp.Expr | None


def _point_subs(variables, point):
    variables, point = (tuple(variables), tuple(map(sp.sympify, point)))
    if len(variables) != len(point):
        raise ValueError("point dimension mismatch")
    return (variables, point, dict(zip(variables, point, strict=True)))


def stationary_phase_leading(
    amplitude, phase, variables: Sequence[sp.Symbol], point, parameter: sp.Symbol
):
    """Leading real stationary-phase term for an isolated nondegenerate critical point."""
    variables, point, subs = _point_subs(variables, point)
    phase, amplitude, parameter = map(sp.sympify, (phase, amplitude, parameter))
    grad = tuple(sp.simplify(sp.diff(phase, x).subs(subs)) for x in variables)
    H = sp.hessian(phase, variables).subs(subs).applyfunc(sp.simplify)
    det = sp.simplify(H.det())
    obligations = []
    if any(g != 0 for g in grad):
        obligations.append("phase gradient vanishes at the saddle")
    if det.is_zero is not False:
        obligations.append("nonzero Hessian determinant")
    sig = None
    if not obligations:
        try:
            eig = H.eigenvals()
            pos = neg = 0
            for val, mult in eig.items():
                if val.is_positive is True:
                    pos += mult
                elif val.is_negative is True:
                    neg += mult
                else:
                    raise ValueError
            sig = pos - neg
        except (ValueError, NotImplementedError):
            obligations.append("real Hessian signature")
    cert = SaddleCertificate(
        "interior",
        point,
        tuple(range(len(variables))),
        (),
        H,
        det,
        sig,
        not obligations,
        tuple(obligations),
    )
    if obligations:
        return SaddleAsymptoticResult(None, parameter, cert, None)
    n = len(variables)
    a0 = sp.simplify(amplitude.subs(subs))
    phi0 = sp.simplify(phase.subs(subs))
    lead = sp.simplify(
        a0
        * sp.exp(sp.I * parameter * phi0)
        * (2 * sp.pi / parameter) ** sp.Rational(n, 2)
        * sp.exp(sp.I * sp.pi * sig / 4)
        / sp.sqrt(sp.Abs(det))
    )
    return SaddleAsymptoticResult(lead, parameter, cert, 1 / parameter)


def orthant_laplace_leading(
    amplitude,
    phase,
    variables: Sequence[sp.Symbol],
    point,
    parameter: sp.Symbol,
    *,
    active_boundaries: Sequence[int],
):
    """Leading Laplace term at a smooth orthant boundary/corner minimum.

    Active coordinates must have strictly positive inward phase derivative;
    tangent coordinates must be stationary with positive-definite Hessian.
    """
    variables, point, subs = _point_subs(variables, point)
    active = tuple(sorted(set(active_boundaries)))
    if any(i < 0 or i >= len(variables) for i in active):
        raise ValueError("invalid active boundary index")
    tangent = tuple(i for i in range(len(variables)) if i not in active)
    phase, amplitude, parameter = map(sp.sympify, (phase, amplitude, parameter))
    obligations = []
    slopes = []
    for i in active:
        s = sp.simplify(sp.diff(phase, variables[i]).subs(subs))
        slopes.append(s)
        if s.is_positive is not True:
            obligations.append(f"positive inward derivative in coordinate {i}")
    for i in tangent:
        if sp.simplify(sp.diff(phase, variables[i]).subs(subs)) != 0:
            obligations.append(f"stationarity in tangent coordinate {i}")
    H = (
        sp.hessian(phase, [variables[i] for i in tangent])
        .subs(subs)
        .applyfunc(sp.simplify)
        if tangent
        else sp.zeros(0)
    )
    det = sp.simplify(H.det()) if tangent else sp.S.One
    if tangent:
        try:
            minors = [sp.simplify(H[:k, :k].det()) for k in range(1, len(tangent) + 1)]
            if not all(m.is_positive is True for m in minors):
                obligations.append("positive-definite tangent Hessian")
        except (TypeError, ValueError, NotImplementedError):
            obligations.append("positive-definite tangent Hessian")
    kind = "corner" if len(active) > 1 else "boundary"
    cert = SaddleCertificate(
        kind, point, tangent, active, H, det, None, not obligations, tuple(obligations)
    )
    if obligations:
        return SaddleAsymptoticResult(None, parameter, cert, None)
    phi0 = sp.simplify(phase.subs(subs))
    a0 = sp.simplify(amplitude.subs(subs))
    m = len(tangent)
    k = len(active)
    gaussian = (2 * sp.pi / parameter) ** sp.Rational(m, 2) / sp.sqrt(det)
    endpoint = parameter ** (-k) / sp.prod(slopes)
    return SaddleAsymptoticResult(
        sp.simplify(a0 * sp.exp(-parameter * phi0) * gaussian * endpoint),
        parameter,
        cert,
        parameter ** (-sp.Rational(1, 2)),
    )


def morse_bott_laplace_leading(
    amplitude,
    phase,
    variables,
    point,
    parameter,
    *,
    normal_directions,
    manifold_measure=1,
):
    """Leading local Morse--Bott Laplace factor with supplied critical-manifold measure.

    The caller supplies the normal coordinate indices and the already-computed
    leading integral of the amplitude over the critical manifold.  This keeps
    global manifold integration separate from local normal certification.
    """
    variables, point, subs = _point_subs(variables, point)
    normal = tuple(normal_directions)
    phase, parameter = (sp.sympify(phase), sp.sympify(parameter))
    obligations = []
    grad = tuple(sp.simplify(sp.diff(phase, x).subs(subs)) for x in variables)
    if any(g != 0 for g in grad):
        obligations.append("critical manifold stationarity")
    H = (
        sp.hessian(phase, [variables[i] for i in normal])
        .subs(subs)
        .applyfunc(sp.simplify)
    )
    det = sp.simplify(H.det())
    principal_minors = tuple(
        sp.simplify(H[:i, :i].det()) for i in range(1, len(normal) + 1)
    )
    if not principal_minors or not all(
        minor.is_positive is True for minor in principal_minors
    ):
        obligations.append("positive-definite normal Hessian")
    cert = SaddleCertificate(
        "morse-bott",
        point,
        normal,
        (),
        H,
        det,
        None,
        not obligations,
        tuple(obligations),
    )
    if obligations:
        return SaddleAsymptoticResult(None, parameter, cert, None)
    phi0 = sp.simplify(phase.subs(subs))
    lead = (
        sp.sympify(manifold_measure)
        * sp.exp(-parameter * phi0)
        * (2 * sp.pi / parameter) ** sp.Rational(len(normal), 2)
        / sp.sqrt(det)
    )
    return SaddleAsymptoticResult(
        sp.simplify(lead), parameter, cert, parameter ** (-sp.Rational(1, 2))
    )
