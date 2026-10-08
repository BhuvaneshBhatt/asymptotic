"""Bounded Scorer tail and inverse-nome certificates."""

import sympy as sp

from .compact_limit_germs import answer
from .elementary_limit_germs import rational_value
from .special_functions import (
    InverseEllipticNome,
    JacobiAmplitude,
    ScorerGi,
    ScorerGiPrime,
    ScorerHi,
    ScorerHiPrime,
)


def special_function_certificate(expr, x, point):
    """Use fixed real-tail expansions only when their remainder vanishes."""
    from .local_tail_germs import budget

    if not budget(expr, 70):
        return None
    if point == 0 and x.is_positive is True:
        from .complex_ray_germs import special_function_ray_certificate

        ray = special_function_ray_certificate(expr, x)
        if ray is not None:
            return ray
    if point == 0 and expr.has(JacobiAmplitude):
        from .local_tail_germs import fixed_finite

        substitutions = {}
        for atom in expr.atoms(JacobiAmplitude):
            argument, parameter = atom.args
            if parameter.has(x) or not fixed_finite(parameter):
                return None
            if argument.func is not sp.elliptic_f or argument.args[1] != parameter:
                return None
            phase = argument.args[0]
            if rational_value(phase, x, 0) != 0:
                return None
            substitutions[atom] = phase
        value = rational_value(expr.xreplace(substitutions), x, 0)
        if value is None or value.is_finite is not True:
            return None
        return answer(
            value,
            "elliptic_amplitude_local_inverse",
            "For fixed finite m, F(phi|m) is analytic with derivative 1 at phi=0. Its principal inverse germ satisfies am(F(phi|m)|m)=phi on a sufficiently small disc. The rational substitution tends to zero, avoids the local branch singularities and makes the displayed cancellation exact before division.",
        )
    if point == 0 and expr == sp.Derivative(InverseEllipticNome(x), x):
        return answer(
            sp.Integer(16),
            "inverse_nome_origin",
            "K(m) expansions give q(m)=m/16+O(m**2). The analytic inverse function theorem gives m(q)=16*q+O(q**2), hence m'(q)->16.",
        )
    atoms = expr.atoms(ScorerGi, ScorerGiPrime, ScorerHi, ScorerHiPrime)
    if point not in (sp.oo, -sp.oo) or len(atoms) != 1:
        return None
    atom = next(iter(atoms))
    coefficient = expr.coeff(atom)
    remainder = sp.expand(expr - coefficient * atom)
    if not coefficient.is_rational_function(x) or not remainder.is_rational_function(x):
        return None
    t = sp.Dummy("positive_tail", positive=True)
    argument = atom.args[0].subs(x, (1 if point == sp.oo else -1) / t)
    if not argument.is_rational_function(t) or sp.count_ops(argument) > 25:
        return None
    lead = argument.as_leading_term(t)
    factor, order = lead.as_coeff_exponent(t)
    if order.is_negative is not True or factor.is_real is not True:
        return None
    gi = atom.func in (ScorerGi, ScorerGiPrime)
    if (gi and factor.is_positive is not True) or (
        not gi and factor.is_negative is not True
    ):
        return None
    derivative = atom.func in (ScorerGiPrime, ScorerHiPrime)
    power = 2 if derivative else 1
    sign = -1 if atom.func in (ScorerGiPrime, ScorerHi) else 1
    chart = (1 if point == sp.oo else -1) / t
    leading = coefficient.subs(x, chart) * sign / (sp.pi * argument**power)
    value = rational_value(leading + remainder.subs(x, chart), t, 0)
    error = rational_value(coefficient.subs(x, chart) / argument ** (power + 3), t, 0)
    if error != 0 or value is None or value.is_finite is not True:
        return None
    return answer(
        value,
        "scorer_real_tail",
        "DLMF 9.12 gives the selected real-tail leading term with error O(abs(argument)**(-power-3)); its derivative expansion has the same relative cubic remainder. The rational weight times that bound tends to zero. Rational argument and coefficient denominators avoid zero on the eventual tail. Opposite tails and complex sectors require separate certificates.",
    )
