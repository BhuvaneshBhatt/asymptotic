"""Compact perturbation certificates; no expansion of the original large scale.

Gamma remainders follow DLMF 5.11.13; principal W tails follow 4.13.10.
"""

import sympy as sp

from .compact_limit_germs import answer


def gamma_root_correction_certificate(expr, x, point):
    if point is not sp.oo or sp.count_ops(expr) > 60:
        return None
    for root in expr.atoms(sp.Pow):
        if root.base.func is not sp.rf:
            continue
        z, a = root.base.args
        if a.has(x) or a.is_positive is not True or root.exp != 1 / a:
            continue
        # Fixed finite parameters and a positive polynomial tail keep both
        # gamma arguments positive and the real power on its principal sheet.
        if a.is_finite is not True:
            continue
        try:
            p = sp.Poly(z, x)
        except sp.PolynomialError:
            continue
        if (
            p.degree() < 1
            or p.degree() > 4
            or p.LC().is_positive is not True
            or any(
                c.is_real is not True or c.is_finite is not True for c in p.all_coeffs()
            )
        ):
            continue
        remainder = sp.expand(expr - root + z)
        if remainder.has(x) or remainder.is_finite is not True:
            continue
        value = remainder + (a - 1) / 2
        return answer(
            value,
            "gamma_ratio_root_first_correction",
            "For fixed finite a>0 and real z->+infinity, rf(z,a)=z**a*(1+a*(a-1)/(2*z)+O(z**-2)); its positive 1/a power is z+(a-1)/2+O(1/z). Only an exact additive cancellation of z is accepted, so the remainder is not amplified. Both gamma arguments and the powered ratio are positive on the tail.",
        )
    return None


def principal_lambert_analytic_scale_certificate(expr, x, point):
    from .fixed_ray_branch_germs import DirectionalInfinity

    directional = isinstance(point, DirectionalInfinity)
    if directional:
        ray = point.args[0]
        if ray.free_symbols or ray.is_finite is not True or ray.is_zero is not False:
            return None
    if (point is not sp.oo and not directional) or sp.count_ops(expr) > 80:
        return None
    atoms = expr.atoms(sp.LambertW)
    if len(atoms) != 1:
        return None
    atom = next(iter(atoms))
    if len(atom.args) > 1 and atom.args[1] != 0:
        return None
    g = atom.args[0]
    if g not in (x, -x):
        return None
    # t is complex: positivity would incorrectly erase the negative-tail cut.
    t = sp.Dummy("complex_W_scale")
    transformed = sp.cancel(expr.xreplace({atom: g * t * t}))
    # The coordinate t was defined as the principal square root of W(g)/g.
    # Its range is the closed right half-plane, where sqrt(t**2)=t.
    transformed = transformed.xreplace({sp.sqrt(t * t): t, sp.sqrt(t**-2): 1 / t})
    if transformed.has(x) or sp.count_ops(transformed) > 80:
        return None
    if any(
        f.func not in (sp.exp, sp.sin, sp.cos, sp.sinh, sp.cosh)
        for f in transformed.atoms(sp.Function)
    ):
        return None
    if any(
        p.exp.is_Integer is not True or abs(p.exp) > 16
        for p in transformed.atoms(sp.Pow)
    ):
        return None
    try:
        expansion = sp.series(transformed, t, 0, 2)
        polynomial = expansion.removeO()
        p = sp.Poly(polynomial, t)
    except (ValueError, NotImplementedError, sp.PolynomialError):
        return None
    if p.degree() > 1:
        return None
    value = p.nth(0)
    if value.is_finite is not True:
        return None
    return answer(
        value,
        "principal_lambert_complex_removable_scale",
        "On a fixed nonzero real or complex infinite ray the principal W(g)=O(log(abs(g))) (DLMF 4.13.10, retaining the negative-real boundary value), hence W(g)/g->0. Exact substitution t=sqrt(W(g)/g), t**2=W(g)/g, removes the large coordinate. The remaining entire-function Laurent germ has no negative powers and a holomorphic removable limit with O(t) remainder. On fixed nonzero rays W(g)/g is eventually off the negative real axis: nonreal rays have a leading off-cut phase, positive real rays give a positive ratio, and negative real rays retain the nonzero principal W imaginary part. Thus the principal square root has positive real part and its reciprocal is the principal square root of g/W(g). The original denominators g and W(g) are nonzero on these tails; no positivity or power splitting is assumed.",
    )


def real_log_exp_tower_certificate(expr, x, point):
    if (
        point != 0
        or x.is_positive is not True
        or sp.count_ops(expr) > 60
        or not expr.has(sp.log, sp.exp)
    ):
        return None
    if any(f.func not in (sp.log, sp.exp) for f in expr.atoms(sp.Function)) or any(
        p.exp.is_Integer is not True or abs(p.exp) > 16 for p in expr.atoms(sp.Pow)
    ):
        return None
    reduced = expr
    y = sp.Dummy("real_log_coordinate", real=True)

    def real_germ(h):
        if h.is_real is True:
            return True
        if h.func is sp.exp:
            return real_germ(h.args[0])
        rational = h.xreplace({sp.log(x): y})
        if rational.has(x):
            return False
        n, d = sp.fraction(sp.cancel(rational))
        try:
            polys = (sp.Poly(n, y), sp.Poly(d, y))
        except sp.PolynomialError:
            return False
        return not polys[1].is_zero and all(
            c.is_real is True and c.is_finite is True
            for p in polys
            for c in p.all_coeffs()
        )

    for _ in range(4):
        replacements = {
            a: a.args[0].args[0]
            for a in reduced.atoms(sp.log)
            if a.args[0].func is sp.exp and real_germ(a.args[0].args[0])
        }
        if not replacements:
            break
        reduced = reduced.xreplace(replacements)
    if reduced == expr or reduced.func is not sp.exp:
        return None
    exponent = reduced.args[0].xreplace({sp.log(x): y})
    if exponent.has(x):
        return None
    try:
        p = sp.Poly(exponent, y)
    except sp.PolynomialError:
        return None
    if (
        p.degree() < 1
        or p.degree() > 4
        or any(c.is_finite is not True or c.is_real is not True for c in p.all_coeffs())
    ):
        return None
    sign = p.LC() * (-1) ** p.degree()
    if sign.is_positive is True:
        value = sp.oo
    elif sign.is_negative is True:
        value = sp.S.Zero
    else:
        return None
    return answer(
        value,
        "real_log_exp_tower_polynomial_tail",
        "Each log(exp(h)) cancellation has a proved real h on the positive punctured germ, so principal logarithms introduce no winding term. The reduced exponential has a real polynomial exponent in log(x); its signed leading term determines the limit as log(x)->-infinity. Rational intermediate denominators have only finitely many zeros in this coordinate and are excluded by a sufficiently small punctured neighbourhood.",
    )
