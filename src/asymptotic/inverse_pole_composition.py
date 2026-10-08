"""Uniform inverse-trigonometric endpoints with attained pole approaches."""

import sympy as sp

from ._limit_composition import _regular_composition
from .limit_models import LimitEvidence, LimitStatus
from .local_path_witnesses import _nonzero_germ
from .multivariate_pole_bounds import regular_numerator_pole_certificate


def inverse_pole_certificate(expr, variables, target, domain, assumptions):
    """Compose atan with a uniformly signed real pole and regular cofactors.

    The inner certificate ranges over the full real domain. Its attained ray
    must also avoid every denominator in the original outer expression.
    """
    if (
        not isinstance(expr, sp.Expr)
        or sp.count_ops(expr) > 60
        or expr.free_symbols - set(variables)
        or not expr.has(sp.atan)
        or any(p.exp.is_Integer and abs(p.exp) > 16 for p in expr.atoms(sp.Pow))
    ):
        return None
    replacements = {}
    approaches = []
    for atom in sorted(expr.atoms(sp.atan), key=sp.default_sort_key):
        inner = regular_numerator_pole_certificate(
            atom.args[0], variables, target, domain, assumptions
        )
        if inner is None:
            continue
        _, value, evidence = inner
        replacements[atom] = sp.pi / 2 if value is sp.oo else -sp.pi / 2
        approaches.append(evidence)
    if not replacements:
        return None
    value = _regular_composition(expr.xreplace(replacements), variables, target)
    if value is None or value.is_finite is not True:
        return None
    guards = [p.base for p in expr.atoms(sp.Pow) if p.exp.is_negative is True]
    for approach in approaches:
        sequence = approach.substitutions
        j = next(
            iter(
                set().union(*(p.free_symbols for _, p in sequence)) - expr.free_symbols
            ),
            None,
        )
        if j is None:
            continue
        t = sp.Dummy("inverse_pole_parameter", positive=True)
        along = [
            g.subs(dict(sequence), simultaneous=True).subs(j, 1 / t) for g in guards
        ]
        if not all(
            g.is_positive is True or g.is_negative is True or _nonzero_germ(g, t)
            for g in along
        ):
            continue
        evidence = LimitEvidence(
            "uniform_inverse_pole",
            "The inner positive-denominator pole certificate gives a uniform signed infinity. The real atan endpoint is the corresponding signed pi/2. Remaining cofactors are continuous; the attained pole ray also avoids the original outer denominators.",
            sequence,
            value,
        )
        return LimitStatus.PROVED, value, evidence
    return None


def reciprocal_inverse_pole_certificate(expr, variables, target, domain, assumptions):
    """Transfer a uniformly signed real reciprocal pole through principal secant.

    For R>1, acos(R)=I*acosh(R), while acos(-R)=pi-I*acosh(R).
    Thus asec(u)=acos(1/u) has the corresponding directed imaginary
    infinity when the reciprocal tends uniformly to a signed real infinity.
    """
    if (
        expr.func is not sp.asec
        or len(variables) not in (2, 3)
        or domain is not sp.S.true
        or assumptions is not sp.S.true
        or expr.free_symbols - set(variables)
        or expr.has(sp.Float)
        or sp.count_ops(expr) > 45
        or any(
            v.assumptions0 != sp.Symbol(v.name, real=True).assumptions0
            for v in variables
        )
    ):
        return None
    from ._polynomial_bounds import bounded_degree, bounded_expansion_width
    from .fixed_ray_branch_germs import DirectionalInfinity
    from .real_pole_germs import _attained_guards

    argument = expr.args[0]
    numerator, denominator = argument.as_numer_denom()
    if bounded_expansion_width(argument, 64) > 64 or any(
        bounded_degree(part, variables, 16) is None for part in (numerator, denominator)
    ):
        return None
    reciprocal = sp.cancel(1 / argument)
    inner = regular_numerator_pole_certificate(
        reciprocal, variables, target, domain, assumptions
    )
    if inner is None or inner[1] not in (sp.oo, -sp.oo):
        return None
    sequence = _attained_guards(sp.Tuple(expr, reciprocal), variables, target)
    if sequence is None:
        return None
    value = DirectionalInfinity(sp.I if inner[1] is sp.oo else -sp.I)
    evidence = (
        LimitEvidence(
            inner[2].method,
            inner[2].statement
            + " The displayed sequence also avoids the original inverse-function denominator holes.",
            sequence,
            inner[1],
        ),
        LimitEvidence(
            "principal_secant_reciprocal_pole",
            "The reciprocal has a uniformly signed real pole. Principal acos is I*acosh(R) for R>1 and pi-I*acosh(R) on the negative real ray. Its modulus diverges with the displayed imaginary direction. The inner pole certificate supplies an attained approach and denominator avoidance.",
            sequence,
            value,
        ),
    )
    return LimitStatus.PROVED, value, evidence
