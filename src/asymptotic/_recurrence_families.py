"""Curated exact recurrences; no general contiguous-relation search.

Identities: DLMF 18.9, 11.4.23/25, 13.3.1/2, 15.5.11.
Harmonic/Lerch shifts follow their defining series, then analytic continuation.
"""

import sympy as sp

from .reference_normalization import RegularizedHypergeometric0F1
from .special_functions import (
    HypergeometricU,
    KelvinBei,
    KelvinBer,
    KelvinKei,
    KelvinKer,
    LegendreQ,
    ParabolicCylinderD,
    RegularizedHypergeometric1F1,
    RegularizedHypergeometric2F1,
    StruveH,
    StruveL,
    WhittakerM,
    WhittakerW,
)

POLYNOMIALS = (
    sp.hermite,
    sp.hermite_prob,
    sp.chebyshevt,
    sp.chebyshevu,
    sp.legendre,
    sp.assoc_legendre,
    sp.laguerre,
    sp.assoc_laguerre,
    sp.gegenbauer,
    sp.jacobi,
)
KELVIN = (KelvinBer, KelvinBei, KelvinKer, KelvinKei)
EXTRA_HEADS = (
    LegendreQ,
    *KELVIN,
    sp.Ynm,
    sp.lowergamma,
    sp.harmonic,
    sp.lerchphi,
    sp.factorial,
    sp.rf,
    sp.ff,
    sp.subfactorial,
    sp.bernoulli,
    sp.euler,
    StruveH,
    StruveL,
    RegularizedHypergeometric0F1,
    ParabolicCylinderD,
    HypergeometricU,
    WhittakerM,
    WhittakerW,
    RegularizedHypergeometric1F1,
    RegularizedHypergeometric2F1,
    *POLYNOMIALS,
    sp.hyper,
)
TWO_BASIS = (
    LegendreQ,
    *KELVIN,
    sp.Ynm,
    ParabolicCylinderD,
    HypergeometricU,
    WhittakerM,
    WhittakerW,
    RegularizedHypergeometric1F1,
    RegularizedHypergeometric2F1,
    sp.subfactorial,
    *POLYNOMIALS,
    StruveH,
    StruveL,
    RegularizedHypergeometric0F1,
    sp.hyper,
)


def signatures(atom):
    h = atom.func
    a = atom.args
    if h is sp.factorial or h is sp.subfactorial:
        return [((), a[0])]
    if h in (sp.rf, sp.ff):
        return [((a[0],), a[1])]
    if h in (sp.bernoulli, sp.euler):
        if len(a) != 2 or a[0].is_Integer is not True or not 0 <= a[0] <= 16:
            return []
        return [((a[0],), a[1])]
    if h in (HypergeometricU, RegularizedHypergeometric1F1):
        return [(("a", a[1], a[2]), a[0]), (("b", a[0], a[2]), a[1])]
    if h is RegularizedHypergeometric2F1:
        return [
            ((a[1], a[2], a[3]), a[0]),
            ((a[0], a[2], a[3]), a[1]),
            (("c", a[0], a[1], a[3]), a[2]),
        ]
    if h is sp.harmonic:
        return [((a[1] if len(a) == 2 else sp.S.One,), a[0])]
    if h is sp.lerchphi:
        return [((a[0], a[1]), a[2])]
    if h in POLYNOMIALS:
        return [(a[1:], a[0])]
    if h is sp.hyper:
        upper, lower, z = a
        if len(lower) != 1:
            return []
        b = lower[0]
        if len(upper) == 0:
            return [(("0f1", z), b)]
        if len(upper) == 1:
            return [(("1f1_a", b, z), upper[0]), (("1f1_b", upper[0], z), b)]
        if len(upper) == 2:
            # SymPy sorts the upper parameters. Match either slot so that a
            # harmless sorting permutation does not change congruence groups.
            return [
                (("2f1_a", upper[1], b, z), upper[0]),
                (("2f1_a", upper[0], b, z), upper[1]),
                (("2f1_c", *upper, z), b),
            ]
        return []
    return [(a[1:], a[0])]


def function_at(head, key, parameter):
    if head is RegularizedHypergeometric2F1 and key[0] == "c":
        return head(key[1], key[2], parameter, key[3])
    if head in (HypergeometricU, RegularizedHypergeometric1F1):
        return (
            head(parameter, key[1], key[2])
            if key[0] == "a"
            else head(key[1], parameter, key[2])
        )
    if head in (sp.rf, sp.ff, sp.bernoulli, sp.euler):
        return head(*key, parameter, evaluate=False)
    if head is not sp.hyper:
        return head(parameter, *key, evaluate=False)
    kind = key[0]
    if kind == "0f1":
        return sp.hyper((), (parameter,), key[1])
    if kind == "1f1_a":
        return sp.hyper((parameter,), (key[1],), key[2])
    if kind == "1f1_b":
        return sp.hyper((key[1],), (parameter,), key[2])
    if kind == "2f1_c":
        return sp.hyper((key[1], key[2]), (parameter,), key[3])
    return sp.hyper((parameter, key[1]), (key[2],), key[3])


def next_basis(head, key, a0, k, basis, guard):
    """Return one upward step, raising ValueError if a denominator is unproved."""
    p = a0 + k - 1

    def divide(num, den):
        if not guard(den):
            raise ValueError("unresolved denominator")
        return num / den

    if head is sp.factorial:
        return (p + 1) * basis[k - 1]
    if head is sp.rf:
        return (key[0] + p) * basis[k - 1]
    if head is sp.ff:
        return (key[0] - p) * basis[k - 1]
    if head is sp.bernoulli:
        return basis[k - 1] + (key[0] * p ** (key[0] - 1) if key[0] else 0)
    if head is sp.euler:
        return 2 * p ** key[0] - basis[k - 1]
    if head is sp.lowergamma:
        z = key[0]
        if not guard(z):
            raise ValueError("zero argument")
        return p * basis[k - 1] - z**p * sp.exp(-z)
    if head is sp.harmonic:
        if not guard(a0 + k):
            raise ValueError("harmonic pole")
        return basis[k - 1] + (a0 + k) ** (-key[0])
    if head is sp.lerchphi:
        z, s = key
        return divide(basis[k - 1] - p ** (-s), z)
    if k == 1:
        return function_at(head, key, a0 + 1)
    left, middle = basis[k - 2], basis[k - 1]
    if head is sp.subfactorial:
        return p * (middle + left)
    if head is sp.Ynm:
        m, theta, phi = key
        high = sp.sqrt(((p + 1) ** 2 - m**2) / ((2 * p + 1) * (2 * p + 3)))
        low = sp.sqrt((p**2 - m**2) / ((2 * p - 1) * (2 * p + 1)))
        return divide(sp.cos(theta) * middle - low * left, high)
    if head in KELVIN:
        z = key[0]
        real, imag = (
            (KelvinBer, KelvinBei)
            if head in (KelvinBer, KelvinBei)
            else (KelvinKer, KelvinKei)
        )
        R = {0: real(a0, z), 1: real(a0 + 1, z)}
        image = {0: imag(a0, z), 1: imag(a0 + 1, z)}
        for j in range(2, k + 1):
            c = divide(sp.sqrt(2) * (a0 + j - 1), z)
            R[j] = -R[j - 2] - c * (R[j - 1] - image[j - 1])
            image[j] = -image[j - 2] - c * (R[j - 1] + image[j - 1])
        return R[k] if head is real else image[k]
    if head is ParabolicCylinderD:
        return key[0] * middle - p * left
    if head is LegendreQ:
        m, z = key
        return divide((2 * p + 1) * z * middle - (p + m) * left, p - m + 1)
    if head in (WhittakerM, WhittakerW):
        mu, z = key
        if head is WhittakerM:
            return divide(
                (mu - p + sp.Rational(1, 2)) * left + (2 * p - z) * middle,
                mu + p + sp.Rational(1, 2),
            )
        return (mu**2 - (p - sp.Rational(1, 2)) ** 2) * left + (z - 2 * p) * middle
    if head is HypergeometricU:
        fixed, z = key[1:]
        if key[0] == "a":
            return divide((2 * p + z - fixed) * middle - left, p * (p - fixed + 1))
        return divide((p + z - 1) * middle - (p - fixed - 1) * left, z)
    if head is RegularizedHypergeometric1F1:
        fixed, z = key[1:]
        if key[0] == "a":
            return divide((fixed - p) * left + (2 * p - fixed + z) * middle, p)
        return divide(left + (1 - p - z) * middle, z * (fixed - p))
    if head is RegularizedHypergeometric2F1:
        if key[0] == "c":
            a, b, z = key[1:]
            return divide(
                (1 - z) * left - (p - 1 - (2 * p - a - b - 1) * z) * middle,
                (p - a) * (p - b) * z,
            )
        b, c, z = key
        return divide((c - p) * left + (2 * p - c + (b - p) * z) * middle, p * (1 - z))
    if head in (sp.hermite, sp.hermite_prob):
        return (2 if head is sp.hermite else 1) * (key[0] * middle - p * left)
    if head in (sp.chebyshevt, sp.chebyshevu):
        return 2 * key[0] * middle - left
    if head is sp.legendre:
        return divide((2 * p + 1) * key[0] * middle - p * left, p + 1)
    if head is sp.assoc_legendre:
        m, z = key
        return divide((2 * p + 1) * z * middle - (p + m) * left, p - m + 1)
    if head in (sp.laguerre, sp.assoc_laguerre):
        alpha, z = (sp.S.Zero, key[0]) if head is sp.laguerre else key
        return divide((2 * p + alpha + 1 - z) * middle - (p + alpha) * left, p + 1)
    if head is sp.gegenbauer:
        alpha, z = key
        return divide(2 * (p + alpha) * z * middle - (p + 2 * alpha - 1) * left, p + 1)
    if head is sp.jacobi:
        a, b, z = key
        t = 2 * p + a + b
        return divide(
            (t + 1) * (t * (t + 2) * z + a * a - b * b) * middle
            - 2 * (p + a) * (p + b) * (t + 2) * left,
            2 * (p + 1) * (p + a + b + 1) * t,
        )
    if head in (StruveH, StruveL):
        z = key[0]
        # Use the same principal power (z/2)**p; never split complex powers.
        source = (z / 2) ** p / (sp.sqrt(sp.pi) * sp.gamma(p + sp.Rational(3, 2)))
        term = divide(2 * p * middle, z) + source
        return term - left if head is StruveH else left - term
    if head is RegularizedHypergeometric0F1:
        return divide(left - p * middle, key[0])
    kind = key[0]
    if kind == "0f1":
        return divide(p * (p - 1) * (left - middle), key[1])
    if kind == "1f1_a":
        b, z = key[1:]
        return divide((b - p) * left + (2 * p - b + z) * middle, p)
    if kind == "1f1_b":
        a, z = key[1:]
        return divide(p * (p - 1) * left + p * (1 - p - z) * middle, z * (a - p))
    if kind == "2f1_c":
        a, b, z = key[1:]
        return divide(
            p * (p - 1) * (1 - z) * left
            - p * (p - 1 - (2 * p - a - b - 1) * z) * middle,
            (p - a) * (p - b) * z,
        )
    b, c, z = key[1:]
    return divide((c - p) * left + (2 * p - c + (b - p) * z) * middle, p * (1 - z))


def uncertified_germ_reason(expr, variables, target):
    """Prevent generic native/composition fallback from inventing new germs.

    Called only after the certified first pass and recurrence retry. Exposing a
    standard function's definition does not certify a parameter-dependent limit.
    """
    if not isinstance(expr, sp.Expr):
        return None
    opaque = (
        ParabolicCylinderD,
        HypergeometricU,
        WhittakerM,
        WhittakerW,
        RegularizedHypergeometric1F1,
        RegularizedHypergeometric2F1,
        LegendreQ,
        *KELVIN,
    )
    for atom in expr.atoms(*opaque):
        if atom.has(*variables):
            return "a translated special-function germ needs a dedicated certified provider after exact recurrence reduction"
    for atom in expr.atoms(*POLYNOMIALS):
        if not atom.has(*variables):
            continue
        degree = atom.args[0]
        if degree.has(*variables):
            return "a varying polynomial degree needs a controlled large-degree asymptotic theorem"
        if degree.is_integer is not True or degree.is_nonnegative is not True:
            return "a non-polynomial degree or associated-function endpoint needs a branch-aware germ theorem"
    for atom in expr.atoms(StruveH, StruveL):
        if not atom.has(*variables):
            continue
        if any(p in (sp.oo, -sp.oo) for p in target):
            return "a Struve infinity germ or cancellation needs a propagated asymptotic remainder"
        argument = atom.args[-1]
        if sp.count_ops(argument) > 32:
            return "a composed Struve germ needs a controlled argument chart"
        boundary = argument.subs(dict(zip(variables, target)))
        if boundary.has(sp.oo, -sp.oo, sp.zoo, sp.nan):
            return "a composed Struve infinity germ needs a propagated asymptotic remainder"
    return None
