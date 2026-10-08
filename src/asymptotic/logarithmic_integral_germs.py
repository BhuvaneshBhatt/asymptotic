"""Positive logarithmic-integral tails with bounded exponent perturbations."""

import sympy as sp

from .compact_limit_germs import answer


def logarithmic_integral_power_certificate(expr, x, point, domain, assumptions):
    """Compare li(x) and li(exp(x)) without expanding nested slow exponents."""
    if (
        point is not sp.oo
        or domain is not sp.S.true
        or assumptions is sp.S.false
        or assumptions.has(x)
        or not expr.has(sp.li)
        or expr.free_symbols - {x}
        or sp.count_ops(expr) > 100
    ):
        return None
    positive = sp.Dummy("positive_li_tail", positive=True)
    expr = expr.subs(x, positive)
    x = positive
    powers = expr.as_powers_dict()
    left, right = sp.li(x), sp.li(sp.exp(x))
    if left not in powers or right not in powers or powers.get(sp.E) != x:
        return None
    remainder = sp.Mul(
        *(
            base**exponent
            for base, exponent in powers.items()
            if base not in (left, right, sp.E)
        )
    )
    if remainder.has(x) or remainder.is_positive is not True:
        return None

    def bounded(node):
        if not node.has(x):
            return node.is_finite is True and node.is_real is True
        if node.is_Add or node.is_Mul:
            return all(bounded(a) for a in node.args)
        if node.func is sp.exp:
            z = node.args[0]
            return any(
                z == chart ** (1 / chart) for chart in (sp.log(x), sp.log(sp.log(x)))
            )
        return False

    first = sp.expand((powers[left] - 1) * sp.log(x))
    second = sp.expand((-powers[right] - 1) * x)
    if not bounded(first) or not bounded(second):
        return None
    return answer(
        sp.oo,
        "logarithmic_integral_bounded_power_tail",
        "On the positive tail li(x)=x/log(x)*(1+O(1/log(x))) and "
        "li(exp(x))=exp(x)/x*(1+O(1/x)). The two scaled exponent "
        "perturbations are bounded: for z->+infinity, z**(1/z) "
        "lies between 1 and e, so exp(z**(1/z)) is bounded. "
        "Taking real logarithms gives 2*log(x)-log(log(x))+O(1), "
        "which diverges. Both bases are positive and nonzero eventually; "
        "all powers use that real logarithm, with no cut or pole crossings.",
    )
