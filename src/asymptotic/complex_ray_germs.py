"""Finite principal-cut integral germs on a fixed real-parameter ray.

DLMF 6.4: off-cut Ei has +/-i*pi relative to its real negative
axis value; Ci/Chi have +/-i*pi relative to their positive-axis value.
Only affine argument germs and regular linear outer compositions are used.
"""

import sympy as sp

from .elementary_limit_germs import rational_value
from .limit_models import LimitEvidence, LimitStatus
from .special_function_limit_germs import linear_in


def ray_integral_cut_certificate(expr, t):
    atoms = expr.atoms(sp.Ei, sp.Ci, sp.Chi, sp.li)
    if len(atoms) != 1 or sp.count_ops(expr) > 70:
        return None
    atom = next(iter(atoms))
    u = atom.args[0]
    if not u.is_rational_function(t) or sp.count_ops(u) > 24:
        return None
    if any(p.exp.is_Integer and abs(p.exp) > 8 for p in u.atoms(sp.Pow)):
        return None
    center = rational_value(u, t, 0)
    if center is None or center.is_finite is not True:
        return None
    if atom.func is sp.li:
        if center.is_positive is not True or (center - 1).is_negative is not True:
            return None
    elif center.is_negative is not True:
        return None
    # Rational imaginary parts have an eventually fixed sign on t>0.
    imaginary = sp.cancel(sp.im(u))
    if imaginary == 0:
        sign = 0
    else:
        if not imaginary.is_rational_function(t):
            return None
        try:
            num, den = (sp.Poly(v, t) for v in sp.fraction(imaginary))
        except sp.PolynomialError:
            return None
        if num.is_zero or den.is_zero:
            return None
        coefficient = min(num.terms())[1] / min(den.terms())[1]
        if coefficient.is_positive is True:
            sign = 1
        elif coefficient.is_negative is True:
            sign = -1
        else:
            return None
    data = linear_in(expr, atom)
    if data is None:
        return None
    a, b = (rational_value(c, t, 0) for c in data)
    if a is None or b is None or a.is_finite is not True or b.is_finite is not True:
        return None
    if sign == 0:
        boundary = atom.func(center)
    elif atom.func in (sp.Ei, sp.li):
        boundary = atom.func(center) + sign * sp.I * sp.pi
    else:
        boundary = atom.func(-center) + sign * sp.I * sp.pi
    value = sp.simplify(a * boundary + b)
    return (
        LimitStatus.PROVED,
        value,
        LimitEvidence(
            "principal_integral_fixed_ray_boundary",
            "DLMF 6.4 principal Ei/Ci/Chi negative-axis boundary or li(z)=Ei(Log(z)) at 0<z<1 on a rational argument germ. The ray stays in its selected half-plane (or on the real cut); both rational outer coefficients have finite limits and eventually nonzero denominators, so bounded local errors vanish.",
            value=value,
        ),
    )


def elementary_ray_branch_certificate(expr, t):
    """Finite inverse-function cuts and fixed-sector log-gamma poles."""
    from .fixed_ray_branch_germs import DirectionalInfinity

    if sp.count_ops(expr) > 60 or expr.free_symbols - {t}:
        return None
    if expr.func is sp.loggamma:
        argument = expr.args[0]
        if argument.is_rational_function(t):
            leading = argument.as_leading_term(t)
            coefficient, order = leading.as_coeff_exponent(t)
            if (
                order.is_negative is True
                and coefficient.is_finite is True
                and coefficient.is_zero is False
                and coefficient.is_negative is False
            ):
                value = DirectionalInfinity(
                    sp.simplify(coefficient / sp.Abs(coefficient))
                )
                return (
                    LimitStatus.PROVED,
                    value,
                    LimitEvidence(
                        "fixed_sector_loggamma_pole",
                        "Stirling's logarithmic remainder is smaller than w*log(Abs(w)) in a fixed sector off the negative real axis. The rational pole has a fixed nonnegative-axis phase; its nonreal ray or positive tail avoids gamma poles, and its normalized direction tends to coefficient/Abs(coefficient).",
                        ((t, 1 / sp.Dummy("ray_index", positive=True, integer=True)),),
                        value,
                    ),
                )
        center = argument.subs(t, 0)
        imaginary = sp.im(argument).expand(complex=True)
        slope = imaginary.diff(t).subs(t, 0)
        if (
            center.is_Rational
            and center < 0
            and center.is_integer is False
            and abs(center) < 8
            and slope.is_zero is False
        ):
            side = (
                1
                if slope.is_positive is True
                else -1
                if slope.is_negative is True
                else 0
            )
            if side and sp.expand(argument - center - sp.diff(argument, t) * t) == 0:
                shifts = int(sp.ceiling(-center))
                value = sp.log(sp.Abs(sp.gamma(center))) - side * shifts * sp.I * sp.pi
                return (
                    LimitStatus.PROVED,
                    value,
                    LimitEvidence(
                        "loggamma_recurrence_cut",
                        "The exact logarithmic gamma recurrence shifts to a positive regular center. Each of the negative factors has the selected principal argument +/-pi; their sum gives the signed branch value. The affine ray avoids every pole.",
                        value=value,
                    ),
                )
    if expr.func not in (sp.acosh, sp.asech, sp.asec):
        return None
    argument = expr.args[0]
    if not argument.is_rational_function(t):
        return None
    center = argument.subs(t, 0)
    slope = sp.im(argument).diff(t).subs(t, 0)
    side = 1 if slope.is_positive is True else -1 if slope.is_negative is True else 0
    if not side:
        return None
    if expr.func is sp.asec and center == 0 and argument == slope * sp.I * t:
        value = DirectionalInfinity(side * sp.I)
        return (
            LimitStatus.PROVED,
            value,
            LimitEvidence(
                "inverse_secant_imaginary_pole",
                "For a pure imaginary nonzero argument tending to zero, acos(1/z) has logarithmically diverging imaginary part of sign Im(z); the affine ray avoids zero and the inverse-function cuts.",
                value=value,
            ),
        )
    if expr.func is sp.asech:
        if center.is_positive is not True or (center - 1).is_positive is not True:
            return None
        center = 1 / center
        side = -side
    if center.is_real is True and (-1 < center < 1) is sp.S.true:
        value = side * sp.I * sp.acos(center)
        return (
            LimitStatus.PROVED,
            value,
            LimitEvidence(
                "inverse_hyperbolic_cut",
                "The two boundary values of acosh on (-1,1) are +/-I*acos(center). The rational imaginary germ selects the half-plane with nonzero first derivative, so the ray remains off the cut locally.",
                value=value,
            ),
        )
    return None


def special_function_ray_certificate(expr, t):
    """Signed nome/Owen branches and affine circular Piecewise boundaries."""
    from .fixed_ray_branch_germs import DirectionalInfinity
    from .local_tail_germs import budget, fixed_finite
    from .special_functions import EllipticNome, OwenT

    if not budget(expr, 80) or t.is_positive is not True:
        return None
    if expr.func is EllipticNome:
        argument = expr.args[0]
        slope = sp.diff(argument, t)
        if slope.has(t) or sp.expand(argument - 2 - slope * t) != 0:
            return None
        height = sp.im(slope)
        side = (
            1 if height.is_positive is True else -1 if height.is_negative is True else 0
        )
        if not side or not fixed_finite(slope):
            return None
        value = side * sp.I * sp.exp(-sp.pi / 2)
        statement = "The parameter nome is exp(-pi*K(1-m)/K(m)). At m=2, K(1-m)->A=K(-1)>0 and K(m)->A*(1+side*I). Their nonzero ratio gives the signed nome value. The affine ray stays off the cut and the denominator is bounded away from zero."
        method = "elliptic_nome_cut"
    elif expr.func is OwenT:
        h, argument = expr.args
        center = argument.subs(t, 0)
        if center not in (sp.I, -sp.I) or h.has(t) or not fixed_finite(h):
            return None
        slope = sp.diff(argument, t)
        if slope.has(t) or sp.expand(argument - center - slope * t) != 0:
            return None
        if not fixed_finite(slope) or slope.is_zero is not False:
            return None
        # A transverse ray avoids the cut. A tangent ray is supported only
        # from between the two endpoint poles, where continuation is fixed.
        transverse = sp.re(slope).is_zero is False
        inward = (sp.im(center) * sp.im(slope)).is_negative is True
        if not transverse and not inward:
            return None
        value = DirectionalInfinity(center)
        method = "owen_endpoint_log_pole"
        statement = "The exact integrand exp(-h**2*(1+a**2)/2)/(2*pi*(1+a**2)) has residue 1/(4*pi*center) at center=+/-I. Its primitive is that residue times log(a-center), plus a bounded holomorphic germ. log(t)->-infinity fixes the displayed imaginary direction. The affine ray avoids the endpoint, the other pole and the selected cuts."
    elif isinstance(expr, sp.Piecewise) and not expr.free_symbols - {t}:
        for branch, condition in expr.args:
            if condition is sp.S.true:
                active = True
            elif (
                isinstance(
                    condition,
                    (
                        sp.LessThan,
                        sp.StrictLessThan,
                        sp.GreaterThan,
                        sp.StrictGreaterThan,
                    ),
                )
                and condition.lhs.is_nonnegative is True
            ):
                bound = condition.lhs
                radius = condition.rhs
                if radius.is_nonnegative is not True or radius.is_finite is not True:
                    return None
                signed = sp.expand_complex(bound**2 - radius**2).expand()
                if not signed.is_rational_function(t):
                    return None
                leading = signed.as_leading_term(t)
                if leading == 0:
                    active = isinstance(condition, (sp.LessThan, sp.GreaterThan))
                else:
                    coefficient, _ = leading.as_coeff_exponent(t)
                    if (
                        coefficient.is_positive is not True
                        and coefficient.is_negative is not True
                    ):
                        return None
                    active = (
                        coefficient.is_negative
                        if isinstance(condition, (sp.LessThan, sp.StrictLessThan))
                        else coefficient.is_positive
                    )
            else:
                return None
            if active:
                value = rational_value(branch, t, 0)
                if value is None or value.is_finite is not True:
                    return None
                method = "circular_piecewise_ray"
                statement = "On t>0 the sign of |argument|**2-radius**2 selects the first active branch, with original priority preserved. Its rational leading coefficient proves the eventual sign even on tangent rays, and the selected rational branch has a finite limit with eventual denominator avoidance."
                break
        else:
            return None
    else:
        return None
    n = sp.Dummy("ray_index", integer=True, positive=True)
    return (
        LimitStatus.PROVED,
        value,
        LimitEvidence(method, statement, ((t, 1 / n),), value),
    )
