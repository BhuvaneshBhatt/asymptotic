"""One-sided inverse tangent poles with exact algebraic coefficients."""

import sympy as sp

from .compact_limit_germs import answer


def inverse_tangent_pole_certificate(expr, x):
    """Resolve a simple real pole inside atan and regular logarithmic factors."""
    if x.is_positive is not True or expr.free_symbols - {x}:
        return None
    atoms = expr.atoms(sp.atan)
    if len(atoms) != 1 or sp.count_ops(expr) > 600:
        return None
    if any(
        atom.func not in (sp.atan, sp.sin, sp.cos, sp.log)
        for atom in expr.atoms(sp.Function)
    ):
        return None
    atom = next(iter(atoms))
    numerator, denominator = sp.fraction(atom.args[0])
    if any(
        a.func is not sp.sin and a.func is not sp.cos
        for a in atom.args[0].atoms(sp.Function)
    ):
        return None
    n = sp.cancel(numerator.subs(x, 0))
    d = sp.cancel(denominator.subs(x, 0))
    derivative = sp.cancel(sp.diff(denominator, x).subs(x, 0))
    if d != 0 or derivative.is_zero is not False:
        return None
    coefficient = sp.cancel(n / derivative)
    if coefficient.is_positive is True:
        value = sp.pi / 2
    elif coefficient.is_negative is True:
        value = -sp.pi / 2
    else:
        return None
    replacements = {atom: value}
    for logarithm in expr.atoms(sp.log):
        argument = logarithm.args[0]
        if argument.has(sp.atan, sp.log):
            return None
        at = sp.cancel(argument.subs(x, 0))
        if at.is_positive is not True:
            return None
        replacements[logarithm] = sp.log(at)
    reduced = expr.xreplace(replacements)
    # No singular composition remains; coefficients must be a regular
    # trigonometric polynomial, not an unexamined rational denominator.
    if any(p.exp.is_negative is True and p.base.has(x) for p in reduced.atoms(sp.Pow)):
        return None
    value = sp.cancel(reduced.subs(x, 0))
    if value.has(sp.nan, sp.zoo, sp.oo) or value.is_finite is not True:
        return None
    return answer(
        value,
        "algebraic_trigonometric_atan_pole",
        "Exact algebraic cancellation proves that the inner denominator "
        "has a simple zero and nonzero derivative, while its numerator "
        "is nonzero. On the positive chart its quotient has the certified "
        "sign and atan tends to the corresponding +/-pi/2 boundary. "
        "The original denominator is nonzero throughout a sufficiently "
        "small punctured chart. Other trigonometric polynomial factors "
        "are entire and each logarithm has a strictly positive endpoint.",
    )
