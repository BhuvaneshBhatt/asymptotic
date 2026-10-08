"""Logarithmic growth of rational magnitudes with bounded principal phase."""

import sympy as sp

from .limit_models import LimitEvidence, LimitStatus


def principal_unit_phase_log_certificate(expr, x, point, domain, assumptions):
    if (
        point is not sp.oo
        or x.is_positive is not True
        or domain is not sp.S.true
        or assumptions is sp.S.false
        or assumptions.has(x)
        or sp.count_ops(expr) > 65
    ):
        return None
    for logarithm in expr.atoms(sp.log):
        argument = logarithm.args[0]
        phases = []
        for atom in argument.atoms(sp.exp, sp.Pow):
            if atom.func is sp.exp:
                phase = atom.args[0] / sp.I
                if phase.is_real is True and phase.is_rational_function(x):
                    phases.append(atom)
            elif (
                atom.base == -1
                and atom.exp.is_real is True
                and atom.exp.is_rational_function(x)
            ):
                phases.append(atom)
        if len(phases) != 1:
            continue
        magnitude = sp.cancel(argument / phases[0])
        if not magnitude.is_rational_function(x):
            continue
        if any(p.exp.is_Integer and abs(p.exp) > 8 for p in magnitude.atoms(sp.Pow)):
            continue

        def degree(e):
            try:
                num, den = (sp.Poly(v, x) for v in sp.fraction(sp.cancel(e)))
            except sp.PolynomialError:
                return None
            if num.is_zero or den.is_zero or max(num.degree(), den.degree()) > 8:
                return None
            if any(
                c.is_real is not True or c.is_finite is not True
                for q in (num, den)
                for c in q.all_coeffs()
            ):
                return None
            leading = num.LC() / den.LC()
            if leading.is_positive is not True or leading.is_finite is not True:
                return None
            return num.degree() - den.degree()

        d = degree(magnitude)
        if d is None:
            continue
        if expr == logarithm and d != 0:
            value = sp.oo if d > 0 else -sp.oo
        else:
            others = [a for a in expr.atoms(sp.log) if a != logarithm]
            if len(others) != 1 or expr != logarithm / others[0]:
                continue
            other = others[0]
            b = degree(other.args[0])
            if b is None or b <= 0:
                continue
            value = sp.Rational(d, b)
        return (
            LimitStatus.PROVED,
            value,
            LimitEvidence(
                "principal_unit_phase_log_growth",
                "The principal exponential or (-1)^real-exponent factor has modulus one. The rational magnitude is eventually positive, with log(magnitude)=degree*log(x)+O(1); principal imaginary parts stay bounded by pi even across the cut. Division by a growing real logarithm therefore has the stated degree ratio. A nonzero degree without division gives a dominant signed real divergence. Rational phase and magnitude poles are eventually avoided.",
                value=value,
            ),
        )
    return None
