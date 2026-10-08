"""Sound local certificates exposed by reconstructed source applications."""

import sympy as sp

from .algebraic_limit_germs import _stratum_sign
from .limit_models import LimitEvidence, LimitStatus
from .local_nonexistence_families import _sides
from .special_function_limit_germs import linear_in


def _answer(items):
    if len(items) == 1 or all(e.value == items[0].value for e in items):
        return LimitStatus.PROVED, items[0].value, tuple(items)
    return LimitStatus.DOES_NOT_EXIST, None, tuple(items)


def signed_erfc_root_certificate(expr, x, point, domain, assumptions):
    if point != 0 or sp.count_ops(expr) > 30:
        return None
    atoms = expr.atoms(sp.erfc)
    if len(atoms) != 1:
        return None
    atom = next(iter(atoms))
    outer = linear_in(expr, atom)
    if outer is None or any(v.has(x) for v in outer):
        return None
    multiplier, offset = outer
    if multiplier not in (1, -1) or offset not in (0, 2):
        return None
    sides = _sides(x, sp.S.Zero, domain, assumptions)
    if not sides:
        return None
    arg = atom.args[0]
    roots = [
        a
        for a in arg.atoms(sp.Pow)
        if a.exp in (sp.S.Half, -sp.S.Half) and a.base.has(x)
    ]
    if len(roots) != 1:
        return None
    base = roots[0].base
    rate = sp.cancel(base / x)
    if rate not in (sp.S.One, -sp.S.One):
        return None
    numerator = sp.expand(arg * sp.sqrt(base))
    constant = numerator.subs(x, 0)
    if not numerator.is_polynomial(x) or sp.Poly(numerator, x).degree() > 1:
        return None
    slope = sp.diff(numerator, x)
    imaginary = False
    if constant.is_imaginary is True and slope.is_imaginary is True:
        constant, slope = sp.im(constant), sp.im(slope)
        imaginary = True
    if (
        constant.is_real is not True
        or constant.is_finite is not True
        or slope.is_real is not True
        or slope.is_finite is not True
    ):
        return None
    sign = _stratum_sign(constant, assumptions)
    if sign not in (1, -1):
        return None
    n = sp.Dummy("attained_erfc_root_n", positive=True, integer=True)
    items = []
    for side in sides:
        if imaginary:
            if side * rate < 0:
                return None
            value = -sign * sp.I * sp.oo
        else:
            value = (
                (sp.S.Zero if sign == 1 else sp.Integer(2))
                if side * rate > 0
                else sign * sp.I * sp.oo
            )
        value = (
            multiplier * value
            if value.has(sp.oo, -sp.oo)
            else multiplier * value + offset
        )
        items.append(
            LimitEvidence(
                "signed_real_imaginary_erfc_root_germ",
                "Along x=+/-1/n^2 the affine real numerator has its certified nonzero constant sign. A positive square-root base gives a signed real infinite argument, where erfc tends to 0 or 2. A negative base gives the principal imaginary square root; erfc(i*y)=1-i*erfi(y), and erfi is odd and diverges positively at positive infinity. These sequences avoid the zero denominator and retain its principal branch.",
                ((x, side / n**2),),
                value,
            )
        )
    return _answer(items)


def principal_arg_root_certificate(expr, x, point, domain, assumptions):
    if point != 0 or expr.func is not sp.arg or sp.count_ops(expr) > 25:
        return None
    sides = _sides(x, sp.S.Zero, domain, assumptions)
    roots = [
        p
        for p in expr.atoms(sp.Pow)
        if p.exp in (sp.S.Half, -sp.S.Half) and p.base.has(x)
    ]
    if not sides or len(roots) != 1:
        return None
    root = roots[0]
    c = sp.cancel(root.base * x if root.exp == sp.S.Half else x / root.base)
    if (
        c.has(x)
        or c.is_finite is not True
        or (c.is_positive is not True and c.is_negative is not True)
        or expr.args[0] != -root - sp.I
    ):
        return None
    n = sp.Dummy("attained_arg_root_n", positive=True, integer=True)
    items = []
    for side in sides:
        value = -sp.pi if (c * side).is_positive is True else -sp.pi / 2
        items.append(
            LimitEvidence(
                "principal_arg_signed_root_chart",
                "On the side with c/x>0 the argument has large negative real part and fixed negative imaginary part, so its principal angle tends to -pi from above. On the other side the principal square root is positive imaginary and the whole argument is strictly negative imaginary, with angle -pi/2. Nonzero sequences stay in their certified quadrants and avoid zero.",
                ((x, side / n**2),),
                value,
            )
        )
    return _answer(items)


def _root_constants(expr):
    return expr.xreplace(
        {
            p: sp.sqrt(sp.factor(p.base))
            for p in expr.atoms(sp.Pow)
            if p.exp == sp.S.Half and not p.base.free_symbols
        }
    )


def algebraic_tangent_pole_certificate(expr, x, point, domain, assumptions):
    if expr.func is not sp.tan or sp.count_ops(expr) > 45:
        return None
    sides = _sides(x, sp.sympify(point), domain, assumptions)
    if not sides:
        return None
    arg = expr.args[0]
    if arg.atoms(sp.Function):
        return None
    for p in arg.atoms(sp.Pow):
        center = _root_constants(p.base.subs(x, point))
        if p.exp.is_Rational is not True or abs(p.exp) > 8:
            return None
        if p.exp.is_Integer is not True and center.is_positive is not True:
            return None
        if p.exp.is_negative and center.is_zero is not False:
            return None
    center = sp.simplify(_root_constants(arg.subs(x, point)))
    derivative = sp.simplify(_root_constants(sp.diff(arg, x).subs(x, point)))
    if (
        sp.simplify(sp.cos(center)) != 0
        or derivative.is_finite is not True
        or (derivative.is_positive is not True and derivative.is_negative is not True)
    ):
        return None
    n = sp.Dummy("attained_algebraic_tangent_n", positive=True, integer=True)
    items = []
    for side in sides:
        value = -sp.oo if (derivative * side).is_positive is True else sp.oo
        items.append(
            LimitEvidence(
                "attained_algebraic_tangent_pole",
                "All noninteger powers have strictly positive center bases, giving real analytic local branches. Exact factorization of fixed radical constants identifies the tangent pole. The argument derivative is nonzero, so tan(arg)=-1/(arg-center)+O(arg-center). The sequences x=point+/-1/n attain the signed infinities and eventually avoid all other poles and radical boundaries.",
                ((x, point + side / n),),
                value,
            )
        )
    return _answer(items)


def opposite_atanh_cut_certificate(expr, x, point, domain, assumptions):
    if point != 0 or sp.count_ops(expr) > 50:
        return None
    sides = _sides(x, sp.S.Zero, domain, assumptions)
    atoms = expr.atoms(sp.atanh)
    if not sides or len(atoms) != 2:
        return None
    plus = minus = None
    center = None
    terms = []
    for a in atoms:
        c = a.args[0].subs(x, 0)
        rate = sp.diff(a.args[0], x)
        if rate.has(x) or rate.is_finite is not True or rate.is_imaginary is not True:
            return None
        orientation = (
            1
            if _stratum_sign(c - 1, assumptions) == 1
            else -1
            if _stratum_sign(-c - 1, assumptions) == 1
            else None
        )
        if orientation is None:
            return None
        # Strict real inequalities certify the finite real center, including
        # parameters whose real assumption is expressed by the inequality.
        c = orientation * c
        rate = orientation * rate
        if center is not None and center != c:
            return None
        center = c
        coefficient = expr.coeff(a)
        if coefficient not in (1, -1):
            return None
        terms.append(coefficient * a)
        weight = coefficient * orientation
        if sp.im(rate).is_positive is True:
            plus = weight
        elif sp.im(rate).is_negative is True:
            minus = weight
    if plus is None or minus is None or plus != -minus or expr != sum(terms):
        return None
    sign = plus
    n = sp.Dummy("attained_atanh_cut_n", positive=True, integer=True)
    return _answer(
        [
            LimitEvidence(
                "attained_opposite_atanh_cut_germ",
                "For real a>1, atanh(z)=(Log(1+z)-Log(1-z))/2. At z=a+i*epsilon the negative-real logarithm is approached from below, giving imaginary part +pi/2; the conjugate approach gives -pi/2. Subtraction cancels the common real part. The explicit real sequences stay away from z=+/-1 and attain the opposite branch limits.",
                ((x, side / n),),
                sign * side * sp.I * sp.pi,
            )
            for side in sides
        ]
    )


def fractional_part_transverse_certificate(expr, x, point, domain, assumptions):
    atoms = expr.atoms(sp.frac)
    if not atoms or len(atoms) != 1 or sp.count_ops(expr) > 50:
        return None
    sides = _sides(x, sp.sympify(point), domain, assumptions)
    if not sides:
        return None
    atom = next(iter(atoms))
    arg = atom.args[0]
    if not arg.is_polynomial(x) or sp.Poly(arg, x).degree() > 4:
        return None
    if any(
        c.is_real is not True or c.is_finite is not True
        for c in sp.Poly(arg, x).all_coeffs()
    ):
        return None
    center = arg.subs(x, point)
    slope = sp.diff(arg, x).subs(x, point)
    if (
        center.is_integer is not True
        or center.is_positive is not True
        or (slope.is_positive is not True and slope.is_negative is not True)
    ):
        return None
    data = linear_in(expr, atom)
    if data is None:
        return None
    a, b = data
    for coefficient in (a, b):
        if any(
            f.func not in (sp.sin, sp.cos, sp.exp)
            for f in coefficient.atoms(sp.Function)
        ):
            return None
        if any(p.exp.is_Integer is not True for p in coefficient.atoms(sp.Pow)):
            return None
    av, bv = a.subs(x, point), b.subs(x, point)
    if av.is_finite is not True or bv.is_finite is not True:
        return None
    n = sp.Dummy("attained_fractional_part_n", positive=True, integer=True)
    items = []
    for side in sides:
        value = bv if (slope * side).is_positive is True else av + bv
        items.append(
            LimitEvidence(
                "attained_transverse_fractional_part",
                "The real polynomial argument crosses a strictly positive integer with nonzero derivative. Above that integer its fractional part tends to zero; below it tends to one. Restriction to positive arguments makes the source and native fractional-part conventions agree. Outer coefficients are continuous rational/entire expressions with finite center values. The sequences eventually stay between adjacent integers and avoid all original denominator zeros.",
                ((x, point + side / n),),
                value,
            )
        )
    return _answer(items)


def logarithmic_integral_cancellation_certificate(expr, x, point, domain, assumptions):
    atoms = expr.atoms(sp.li)
    if not atoms or len(atoms) != 1 or sp.count_ops(expr) > 40:
        return None
    sides = _sides(x, sp.sympify(point), domain, assumptions)
    if not sides:
        return None
    atom = next(iter(atoms))
    arg = atom.args[0]
    c = sp.diff(arg, x)
    if (
        c.has(x)
        or c.is_finite is not True
        or c.is_zero is not False
        or arg.subs(x, point) != 1
        or expr != atom - sp.log(1 - arg)
    ):
        return None
    n = sp.Dummy("attained_log_integral_n", positive=True, integer=True)
    items = []
    for side in sides:
        rate = sp.expand_complex(side * c)
        if rate.is_negative is True:
            value = sp.EulerGamma
        elif rate.is_positive is True or sp.im(rate).is_negative is True:
            value = sp.EulerGamma - sp.I * sp.pi
        elif sp.im(rate).is_positive is True:
            value = sp.EulerGamma + sp.I * sp.pi
        else:
            return None
        items.append(
            LimitEvidence(
                "principal_logarithmic_integral_cancelled_germ",
                "Write delta=arg-1=c*(x-point). Log(arg)=delta*(1+O(delta)). The local Ei germ in li(arg)=Ei(Log(arg)) gives EulerGamma plus the matching logarithm and O(delta), retaining the principal boundary phase. On the negative real delta ray Ei takes its real boundary value, so the logarithms cancel without an imaginary constant; off that ray the difference Log(delta)-Log(-delta) is the certified signed i*pi. Attained sequences stay off arg=1 and retain the chosen ray.",
                ((x, point + side / n),),
                value,
            )
        )
    return _answer(items)
