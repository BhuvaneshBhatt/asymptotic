"""Continuous compositions after certified local sign and zero-germ reductions."""

import sympy as sp

from ._limit_composition import _regular_composition
from ._polynomial_bounds import bounded_degree
from .limit_models import LimitEvidence, LimitStatus
from .uniform_radial_bounds import radial_vanishing_certificate


def _eligible(expr, variables, target, domain, assumptions):
    return (
        isinstance(expr, sp.Expr)
        and len(variables) in (2, 3)
        and domain is sp.S.true
        and assumptions is sp.S.true
        and not expr.free_symbols - set(variables)
        and not expr.has(sp.Float)
        and sp.count_ops(expr) <= 45
        and all(
            v.assumptions0 == sp.Symbol(v.name, real=True).assumptions0
            for v in variables
        )
        and all(p.is_Rational is True for p in target)
    )


def nonzero_sign_composition(expr, variables, target, domain, assumptions):
    """Replace locally constant real signs, preserving original denominator guards.

    Continuity of a real inner argument with nonzero center gives a neighborhood
    of fixed sign. Absolute values use the same local sign identity. The
    remaining expression must be continuous, and every original negative-power
    base must stay nonzero locally; a cancelled, empty-domain germ is rejected.
    """
    if not _eligible(expr, variables, target, domain, assumptions) or not expr.has(
        sp.sign
    ):
        return None
    replacements = {}
    for atom in sorted(
        expr.atoms(sp.sign, sp.Abs), key=lambda a: sp.count_ops(a.args[0])
    ):
        inner = atom.args[0].xreplace(replacements)
        if inner.is_real is not True:
            continue
        center = _regular_composition(inner, variables, target)
        if center is None or center.is_finite is not True:
            continue
        if center.is_positive is True:
            orientation = sp.S.One
        elif center.is_negative is True:
            orientation = -sp.S.One
        else:
            continue
        replacements[atom] = (
            orientation if atom.func is sp.sign else orientation * inner
        )
    if not any(atom.func is sp.sign for atom in replacements):
        return None
    normalized = expr.xreplace(replacements)
    value = _regular_composition(normalized, variables, target)
    if value is None:
        return None
    for power in expr.atoms(sp.Pow):
        if power.exp.is_negative is True:
            guard = _regular_composition(
                power.base.xreplace(replacements), variables, target
            )
            if guard is None or guard.is_zero is not False:
                return None
    j = sp.Dummy("attained_sign_index", positive=True, integer=True)
    sequence = tuple((v, p + 1 / j) for v, p in zip(variables, target, strict=True))
    evidence = LimitEvidence(
        "nonzero_sign_composition",
        "Each replaced real argument is continuous with a nonzero center, so its sign is fixed in a full neighborhood. The original negative-power bases stay nonzero there. The normalized expression is continuous, and the displayed ray eventually lies in that neighborhood.",
        sequence,
        value,
    )
    return LimitStatus.PROVED, value, evidence


def rational_zero_composition(expr, variables, target, domain, assumptions):
    """Compose a uniformly vanishing real rational germ with a regular scalar head.

    The admitted scalar heads are real and analytic near zero. A fresh real
    coordinate verifies continuity of the entire outer expression at the head's
    endpoint; substitution into an outer discontinuity is not a certificate.
    """
    heads = (
        sp.sin,
        sp.cos,
        sp.exp,
        sp.sinh,
        sp.cosh,
        sp.erf,
        sp.atan,
        sp.asin,
        sp.acos,
        sp.atanh,
    )
    if not _eligible(expr, variables, target, domain, assumptions) or not expr.has(
        *heads
    ):
        return None
    for atom in sorted(expr.atoms(*heads), key=sp.default_sort_key):
        inner = atom.args[0]
        numerator, denominator = inner.as_numer_denom()
        if any(
            bounded_degree(p, variables, 32) is None for p in (numerator, denominator)
        ):
            continue
        try:
            sp.Poly(numerator, *variables, domain=sp.QQ)
            sp.Poly(denominator, *variables, domain=sp.QQ)
        except (sp.PolynomialError, sp.polys.polyerrors.CoercionFailed):
            continue
        zero = radial_vanishing_certificate(
            inner, variables, target, domain, assumptions
        )
        if zero is None:
            continue
        z = sp.Dummy("scalar_endpoint", real=True)
        outer = expr.xreplace({atom: z})
        endpoint = atom.func(0)
        value = _regular_composition(outer, (*variables, z), (*target, endpoint))
        if value is None:
            continue
        evidence = (
            zero[2],
            LimitEvidence(
                "rational_zero_composition",
                "The inner rational expression is real on its defined domain and tends uniformly to zero. The scalar head is real analytic near zero, and the complete outer expression is continuous in an independent real endpoint coordinate. The inner certificate supplies an attained, pole-avoiding approach.",
                zero[2].substitutions,
                value,
            ),
        )
        return LimitStatus.PROVED, value, evidence
    return None


def signed_root_composition(expr, variables, target, domain, assumptions):
    """Continue real signed powers through a zero of their continuous argument.

    For positive r and a nonnegative integer k, |h|**r sign(h)**k tends
    to zero whenever real continuous h tends to zero. An independent real
    endpoint verifies continuity of the complete outer expression.
    """
    if not _eligible(expr, variables, target, domain, assumptions):
        return None
    for atom in sorted(expr.atoms(sp.Mul), key=sp.default_sort_key):
        factors = atom.as_powers_dict()
        for sign in atom.atoms(sp.sign):
            h = sign.args[0]
            k = factors.get(sign, sp.S.Zero)
            r = factors.get(sp.Abs(h), sp.S.Zero)
            if (
                k.is_Integer is not True
                or not 0 <= k <= 4
                or r.is_Rational is not True
                or not 0 < r <= 4
                or h.is_real is not True
                or set(factors) != {sign, sp.Abs(h)}
            ):
                continue
            if _regular_composition(h, variables, target) != 0:
                continue
            z = sp.Dummy("signed_power_endpoint", real=True)
            value = _regular_composition(
                expr.xreplace({atom: z}), (*variables, z), (*target, sp.S.Zero)
            )
            if value is None:
                continue
            j = sp.Dummy("attained_root_index", positive=True, integer=True)
            sequence = tuple(
                (v, p + 1 / j) for v, p in zip(variables, target, strict=True)
            )
            evidence = LimitEvidence(
                "signed_root_composition",
                "The real inner argument is continuous and tends to zero. The magnitude of its signed positive power is bounded by |h|**r, which tends to zero. The outer expression is continuous in an independent real endpoint coordinate; the displayed ray eventually lies in its defined neighborhood.",
                sequence,
                value,
            )
            return LimitStatus.PROVED, value, evidence
    return None
