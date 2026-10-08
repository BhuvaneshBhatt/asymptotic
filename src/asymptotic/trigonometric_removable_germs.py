"""Exact trigonometric identities and bounded removable scalar germs."""

import sympy as sp

from ._limit_composition import _regular_composition
from ._polynomial_bounds import bounded_degree
from .limit_models import LimitEvidence, LimitStatus
from .local_path_witnesses import _nonzero_germ
from .uniform_radial_bounds import radial_vanishing_certificate


def trigonometric_removable_certificate(expr, variables, target, domain, assumptions):
    """Certify a finite extension while retaining original pole avoidance.

    Difference quotients use the exact half-angle identities. The scalar
    kernels have holomorphic extensions established by their Taylor germs;
    their cofactors need continuity or an independent uniform zero bound.
    """
    if (
        not isinstance(expr, sp.Expr)
        or len(variables) not in (2, 3)
        or domain is not sp.S.true
        or assumptions is not sp.S.true
        or expr.free_symbols - set(variables)
        or expr.has(sp.Float)
        or sp.count_ops(expr) > 45
        or any(p.exp.is_Integer and abs(p.exp) > 16 for p in expr.atoms(sp.Pow))
        or any(
            v.assumptions0 != sp.Symbol(v.name, real=True).assumptions0
            for v in variables
        )
        or any(
            p.is_real is not True or p.is_finite is not True or p.free_symbols
            for p in target
        )
    ):
        return None
    replacements = {}
    for denominator in expr.atoms(sp.Add):
        tangents = list(denominator.atoms(sp.tan))
        if len(tangents) != 2:
            continue
        a, b = (t.args[0] for t in sorted(tangents, key=sp.default_sort_key))
        numerator = sp.sin(a) - sp.sin(b)
        quotient = numerator / (sp.tan(a) - sp.tan(b))
        extension = sp.cos((a + b) / 2) * sp.cos(a) * sp.cos(b) / sp.cos((a - b) / 2)
        for term in expr.atoms(sp.Mul):
            if term == quotient:
                replacements[term] = extension
    transformed = expr.xreplace(replacements)
    value = (
        _regular_composition(transformed, variables, target) if replacements else None
    )
    kernels = []
    if value is None:
        for atom in sorted(
            expr.atoms(sp.sin, sp.cos, sp.cot, sp.csc), key=sp.default_sort_key
        ):
            inner = atom.args[0]
            if (
                bounded_degree(inner, variables, 8) is None
                or _regular_composition(inner, variables, target) != 0
            ):
                continue
            if expr.has(sp.csc(inner)):
                if expr.has(sp.cot(inner)):
                    kernels.append(
                        (inner, (1 - inner * sp.cot(inner)) * sp.csc(inner), sp.S.Zero)
                    )
                kernels.append((inner, inner * sp.csc(inner), sp.S.One))
            if expr.has(sp.sin(inner) - inner):
                kernels.append(
                    (inner, (sp.sin(inner) - inner) / inner**3, -sp.Rational(1, 6))
                )
            if expr.has(1 - sp.cos(inner)) or expr.has(sp.cos(inner) - 1):
                kernels.append(
                    (inner, (1 - sp.cos(inner)) / inner**2, sp.Rational(1, 2))
                )
        for inner, kernel, extension in kernels:
            # Cancellation is bounded by the small admitted expression. A
            # residual scalar kernel is rejected by continuity and norm checks.
            cofactor = sp.cancel(expr / kernel)
            coefficient = _regular_composition(cofactor, variables, target)
            if coefficient is None:
                bound = radial_vanishing_certificate(
                    cofactor, variables, target, domain, assumptions
                )
                coefficient = sp.S.Zero if bound is not None else None
            if coefficient is not None and coefficient.is_finite is True:
                value = extension * coefficient
                break
    if value is None or value.is_finite is not True:
        return None
    guards = [p.base for p in expr.atoms(sp.Pow) if p.exp.is_negative is True]
    guards.extend(sp.sin(a.args[0]) for a in expr.atoms(sp.cot, sp.csc))
    guards.extend(sp.cos(a.args[0]) for a in expr.atoms(sp.tan, sp.sec))
    t = sp.Dummy("trig_ray_parameter", positive=True)
    chosen = None
    for multipliers in ((1,) * len(variables), tuple(range(1, len(variables) + 1))):
        paths = tuple(p + c * t for p, c in zip(target, multipliers, strict=True))
        along = [
            g.subs(dict(zip(variables, paths, strict=True)), simultaneous=True)
            for g in guards
        ]
        if all(
            g.is_positive is True or g.is_negative is True or _nonzero_germ(g, t)
            for g in along
        ):
            chosen = multipliers
            break
    if chosen is None:
        return None
    j = sp.Dummy("trig_index", positive=True, integer=True)
    sequence = tuple(
        (v, p + sp.Integer(c) / j)
        for v, p, c in zip(variables, target, chosen, strict=True)
    )
    evidence = LimitEvidence(
        "trigonometric_removable_germ",
        "Exact half-angle identities or holomorphic scalar Taylor germs give a finite extension. The cofactor is continuous or independently bounded uniformly by a vanishing norm power. An attained ray avoids every original trigonometric and algebraic denominator.",
        sequence,
        value,
    )
    return LimitStatus.PROVED, value, evidence
