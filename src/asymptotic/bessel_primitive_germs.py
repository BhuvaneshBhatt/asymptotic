"""Finite-part limits of paired hypergeometric Bessel primitives."""

import sympy as sp

from .compact_limit_germs import answer


def bessel_primitive_certificate(expr, x, point, domain, assumptions):
    """Recognize an exact primitive of K_nu before expanding growing I terms.

    For rational noninteger |nu|<2, each 1F2 primitive has a convergent
    Frobenius series without a constant term. Their paired derivative is K_nu.
    The finite part at infinity follows from the continued Mellin integral;
    its exponentially small tail cannot reintroduce an uncancelled I term.
    """
    if (
        point is not sp.oo
        or domain is not sp.S.true
        or assumptions is sp.S.false
        or assumptions.has(x)
        or expr.free_symbols - {x}
        or sp.count_ops(expr) > 160
    ):
        return None
    if x.is_positive is not True:
        positive = sp.Dummy("positive_bessel_tail", positive=True)
        expr = expr.subs(x, positive)
        x = positive
    atoms = expr.atoms(sp.hyper)
    if len(atoms) != 2:
        return None
    orders = []
    for atom in atoms:
        if len(atom.ap) != 1 or len(atom.bq) != 2 or atom.argument != x**2 / 4:
            return None
        a = atom.ap[0]
        nu = 2 * a - 1
        if (
            not nu.is_Rational
            or nu.is_Integer
            or abs(nu) >= 2
            or sorted(atom.bq, key=sp.default_sort_key)
            != sorted((a + 1, nu + 1), key=sp.default_sort_key)
        ):
            return None
        orders.append(nu)
    if sum(orders) != 0:
        return None
    nu = max(orders)
    markers = {atom: sp.Dummy("bessel_primitive") for atom in atoms}
    reduced = expr.xreplace(markers)
    if not reduced.is_polynomial(*markers.values()):
        return None
    polynomial = sp.Poly(sp.expand(reduced), *markers.values())
    if polynomial.total_degree() != 1:
        return None
    constant = polynomial.coeff_monomial(1)
    if constant.has(x) or constant.is_finite is not True:
        return None
    weights = {}
    for atom, order in zip(atoms, orders, strict=True):
        coefficient = polynomial.coeff_monomial(markers[atom])
        weight = sp.simplify(
            sp.expand_func(
                coefficient
                * 2**order
                * (order + 1)
                * sp.gamma(order + 1)
                / x ** (order + 1)
            )
        )
        weight = sp.simplify(
            weight.replace(
                lambda node: (
                    node.func is sp.gamma
                    and node.args[0].is_Rational
                    and node.args[0] < 0
                ),
                lambda node: (
                    sp.gamma(node.args[0] + sp.ceiling(-node.args[0]))
                    / sp.rf(node.args[0], sp.ceiling(-node.args[0]))
                ),
            )
        )
        if weight.has(x) or weight.is_finite is not True:
            return None
        weights[order] = weight
    if sp.simplify(weights[nu] + weights[-nu]) != 0:
        return None
    scale = sp.simplify(-2 * weights[nu] * sp.sin(sp.pi * nu) / sp.pi)
    value = sp.simplify(constant + scale * sp.pi / (2 * sp.cos(sp.pi * nu / 2)))
    return answer(
        value,
        "paired_bessel_primitive_tail",
        "The exact Frobenius primitives differentiate to opposite multiples of "
        "I_nu and I_-nu. I_-nu-I_nu=(2/pi)*sin(pi*nu)*K_nu cancels the "
        "entire growing solution, not just its leading term. Their zero "
        "Frobenius constant fixes the finite part pi/(2*cos(pi*nu/2)) by "
        "analytic continuation of the K Mellin integral. The remaining "
        "integral from x to infinity is O(exp(-x)/sqrt(x)). Rational "
        "noninteger |nu|<2 excludes resonant logarithms and parameter poles; "
        "the positive real tail avoids the origin and every defining pole.",
    )
