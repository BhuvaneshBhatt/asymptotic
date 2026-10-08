"""Exact, bounded normalization of principal constant unit roots."""

import sympy as sp


def normalize_unit_roots(expr):
    replacements = {}
    for a in expr.atoms(sp.exp):
        phase = sp.cancel(a.args[0] / (sp.I * sp.pi))
        if phase.is_Rational and phase.q <= 12 and abs(phase) <= 12:
            replacements[a] = sp.cos(sp.pi * phase) + sp.I * sp.sin(sp.pi * phase)
    expr = expr.xreplace(replacements)
    return expr.xreplace(
        {
            p: sp.cos(sp.pi * p.exp) + sp.I * sp.sin(sp.pi * p.exp)
            for p in expr.atoms(sp.Pow)
            if p.base == -1 and p.exp.is_Rational and p.exp.q <= 12 and abs(p.exp) <= 12
        }
    )


def normalize_logarithmic_angles(expr, variable):
    """Reduce trigonometric constants whose exponential is exactly algebraic.

    exp(i*c) converts integer combinations of constant principal logarithms
    into algebraic products. The exponential identity requires no logarithm
    merging or branch assumptions. This exposes exact poles hidden by a
    serialized angle before continuity or numerical zero tests are attempted.
    """
    replacements = {}
    for atom in expr.atoms(sp.sin, sp.cos):
        argument = atom.args[0]
        if not argument.has(sp.log) or sp.count_ops(argument) > 100:
            continue
        expanded = sp.expand(argument)
        slope = expanded.coeff(variable)
        constant = expanded - slope * variable
        if constant.free_symbols or constant.has(variable):
            continue
        if not slope.is_Rational or abs(slope) > 8:
            continue
        phase = sp.expand_power_exp(sp.exp(sp.expand(sp.I * constant)))
        if phase.has(sp.log, sp.exp) or phase.is_algebraic is not True:
            continue
        if phase.is_zero is not False:
            continue
        sine = sp.cancel((phase - 1 / phase) / (2 * sp.I))
        cosine = sp.cancel((phase + 1 / phase) / 2)
        local = slope * variable
        replacements[atom] = (
            sine * sp.cos(local) + cosine * sp.sin(local)
            if atom.func is sp.sin
            else cosine * sp.cos(local) - sine * sp.sin(local)
        )
    return expr.xreplace(replacements)
