"""Paired dilogarithm inversion with principal-cut boundary values."""

import sympy as sp

from .compact_limit_germs import answer


def paired_dilogarithm_tail_certificate(expr, x, point, domain, assumptions):
    """Cancel paired quadratic logarithms on a fixed right-half-plane chart."""
    if (
        point is not sp.oo
        or domain is not sp.S.true
        or assumptions is sp.S.false
        or assumptions.has(x)
        or len(expr.atoms(sp.polylog)) != 2
        or sp.count_ops(expr) > 120
    ):
        return None
    parameters = expr.free_symbols - {x}
    if len(parameters) != 1:
        return None
    s = next(iter(parameters))
    if (sp.re(s) > 0) not in sp.And.make_args(assumptions):
        return None
    first = sp.log((sp.I * s * x + s) / (s - sp.I))
    second = sp.log(s * (x + sp.I) / (sp.I * s - 1))
    template = (
        sp.I
        * (
            (second - first) * sp.log(s * x + 1)
            - sp.polylog(2, (s * x + 1) / (sp.I * s + 1))
            + sp.polylog(2, sp.I * (s * x + 1) / (s + sp.I))
        )
        / 2
    )
    if sp.expand(expr - template) != 0:
        return None
    a = sp.log(-s / (1 + sp.I * s))
    principal = sp.log(-s / (1 - sp.I * s))
    b = sp.Piecewise(
        (principal - 2 * sp.pi * sp.I, sp.Eq(sp.im(s) + sp.Abs(s) ** 2, 0)),
        (principal, True),
    )
    value = sp.I * (a - b) * (a + b - 2 * sp.log(s)) / 4
    return answer(
        value,
        "paired_dilogarithm_inversion_tail",
        "Li2(z)+Li2(1/z)=-pi**2/6-Log(-z)**2/2 on each "
        "principal boundary. For Re(s)>0 the affine slopes and "
        "denominators are nonzero; inverse-argument remainders are O(1/x). "
        "The two quadratic logarithms cancel exactly, leaving the stated "
        "constant, with error O(log(x)/x). On Im(s)+Abs(s)**2=0 "
        "the second negative-real logarithmic slope is attained from "
        "below, requiring Log(slope)-2*pi*i. The first cut slope is "
        "attained from above. Away from these boundaries continuity "
        "fixes the principal constants. All affine zeros are avoided "
        "on a sufficiently large positive real tail.",
    )
