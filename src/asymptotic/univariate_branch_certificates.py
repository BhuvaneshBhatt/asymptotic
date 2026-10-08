"""Narrow branch-aware scalar certificates, with explicit side semantics."""

import sympy as sp

from ._symbolic_policy import bounded_limit
from .limit_models import LimitEvidence, LimitStatus


def complex_exponential_certificate(expr, variable, point, domain):
    """Distinguish rotating complex infinity from positive real infinity.

    |exp(g)| = exp(Re(g)); Re(g)->+infinity and |Im(g)|->infinity
    give infinity on the Riemann sphere, without a positive-real direction.
    Two real sides are checked unless the coordinate/domain selects one side.
    """
    if expr.func is sp.exp and point in (sp.oo, -sp.oo):
        from .fixed_ray_branch_germs import DirectionalInfinity
        from .local_tail_germs import fixed_finite

        exponent = expr.args[0]
        rate = sp.diff(exponent, variable)
        offset = exponent - rate * variable
        signed_rate = rate if point == sp.oo else -rate
        if (
            not offset.has(variable)
            and signed_rate.is_positive is True
            and fixed_finite(offset)
        ):
            value = DirectionalInfinity(sp.exp(sp.I * sp.im(offset)))
            return (
                LimitStatus.PROVED,
                value,
                LimitEvidence(
                    "fixed_exponential_direction",
                    "A real positive affine growth rate makes the modulus diverge; the fixed finite offset gives the exact constant normalized phase exp(I*im(offset)).",
                    value=value,
                ),
            )
    if expr.func is not sp.exp or point != 0 or expr.args[0].is_real is True:
        return None
    if expr.free_symbols - {variable} or sp.count_ops(expr) > 40:
        return None
    if domain is not sp.S.true and domain != (variable > 0):
        return None
    sides = (
        (1,) if variable.is_positive is True or domain == (variable > 0) else (1, -1)
    )
    t = sp.Dummy("exp_side", positive=True)
    values = []
    for side in sides:
        exponent = expr.args[0].subs(variable, side * t)
        parts = sp.expand_complex(exponent)
        real = bounded_limit(sp.re(parts), t, 0, allow_general=True)
        imaginary = bounded_limit(sp.im(parts), t, 0, allow_general=True)
        if real is -sp.oo:
            values.append(sp.S.Zero)
        elif real is sp.oo and imaginary in (sp.oo, -sp.oo):
            values.append(sp.zoo)
        elif (
            real is not None
            and imaginary is not None
            and not real.has(sp.AccumBounds, sp.Limit)
            and not imaginary.has(sp.AccumBounds, sp.Limit)
            and real.is_finite is True
            and imaginary.is_finite is True
        ):
            values.append(sp.exp(real + sp.I * imaginary))
        elif real is sp.oo and imaginary is not None and imaginary.is_finite is True:
            from .fixed_ray_branch_germs import DirectionalInfinity

            values.append(DirectionalInfinity(sp.exp(sp.I * imaginary)))
        else:
            return None
    same = all(v == values[0] for v in values)
    status = LimitStatus.PROVED if same else LimitStatus.DOES_NOT_EXIST
    value = values[0] if same else None
    return (
        status,
        value,
        LimitEvidence(
            "complex_exponential_modulus",
            "the real and imaginary exponent germs certify the modulus and distinguish the real sides",
            value=value,
        ),
    )


def reciprocal_log_inverse_certificate(expr, variable, point, domain):
    """Certify the branch side before analyzing an iterated logarithmic pole."""
    if (
        point is not -sp.oo
        or domain is not sp.S.true
        or expr.func not in (sp.asec, sp.acsc)
    ):
        return None
    if expr.args[0] != sp.acos(1 / sp.log(variable)) - sp.pi / 2:
        return None
    from .fixed_ray_branch_germs import DirectionalInfinity

    value = DirectionalInfinity(sp.I if expr.func is sp.asec else -sp.I)
    n = sp.Dummy("negative_tail_index", positive=True, integer=True)
    return (
        LimitStatus.PROVED,
        value,
        LimitEvidence(
            "reciprocal_log_inverse_branch",
            "On the negative real tail log(x)=L+I*pi. The convergent local acos expansion gives 1/(acos(1/log(x))-pi/2)=-log(x)+O(1/log(x)); its imaginary part tends to -pi and stays strictly below the negative-real cut. Principal acos and asin logarithmic definitions give respectively positive and negative diverging imaginary parts of size log(L). The finite real parts are negligible; the displayed tail avoids logarithmic and reciprocal poles.",
            ((variable, -sp.exp(n)),),
            value,
        ),
    )
