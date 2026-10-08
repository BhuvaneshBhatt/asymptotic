"""Limits from convergent integral definitions, with explicit real domains."""

import sympy as sp

from .elementary_limit_germs import answer, rational_value
from .special_function_limit_germs import linear_in
from .special_functions import FresnelF, FresnelG


def _positive_real_rational_tail(argument, x):
    if not argument.is_rational_function(x):
        return False
    try:
        polynomials = tuple(sp.Poly(p, x) for p in sp.fraction(sp.cancel(argument)))
    except sp.PolynomialError:
        return False
    return (
        all(
            coefficient.is_real is True
            for polynomial in polynomials
            for coefficient in polynomial.all_coeffs()
        )
        and rational_value(argument, x, sp.oo) is sp.oo
    )


def gaussian_log_moment_certificate(expr, x, point, domain, assumptions):
    """Evaluate the erf/log cancellation as a convergent Gaussian log moment."""
    if (
        (point is not sp.oo and not (point == 0 and x.is_positive is True))
        or domain is not sp.S.true
        or assumptions is sp.S.false
        or assumptions.has(x)
        or x.is_real is False
        or sp.count_ops(expr) > 65
    ):
        return None
    atoms = expr.atoms(sp.hyper)
    if len(atoms) != 1:
        return None
    atom = next(iter(atoms))
    if tuple(map(tuple, atom.args[:2])) != (
        (sp.S.Half, sp.S.Half),
        (sp.Rational(3, 2), sp.Rational(3, 2)),
    ):
        return None
    errors = expr.atoms(sp.erf, sp.erfinv)
    if len(errors) != 1:
        return None
    error = next(iter(errors))
    argument = error if error.func is sp.erfinv else error.args[0]
    inverse = argument.func is sp.erfinv
    if inverse:
        from .local_tail_germs import budget

        if not budget(expr, 65) or point != 0 or x.is_positive is not True:
            return None
        delta = 1 - argument.args[0]
        if rational_value(delta, x, 0) != 0:
            return None
        leading = delta.as_leading_term(x)
        factor, order = leading.as_coeff_exponent(x)
        if factor.is_negative is True and order.is_positive is True:
            from .limit_models import LimitEvidence, LimitStatus

            return (
                LimitStatus.UNKNOWN,
                None,
                LimitEvidence(
                    "inverse_error_complex_branch_required",
                    "The inverse-error argument approaches 1 from outside its real interval. The exact Gaussian-moment identity is available, but the complex inverse branch and its tail sector require a separate certificate. The real inverse theorem cannot be used on this chart.",
                ),
            )
        if factor.is_positive is not True or order.is_positive is not True:
            return None
    elif point is not sp.oo or not _positive_real_rational_tail(argument, x):
        return None
    if sp.cancel(atom.args[2] + argument**2) != 0:
        return None
    data = linear_in(expr, atom)
    if data is None:
        return None
    coefficient, rest = data
    scale = sp.cancel(-coefficient * sp.sqrt(sp.pi) / (2 * argument))
    remainder = sp.expand_mul(
        rest
        - scale * sp.log(argument) * (argument.args[0] if inverse else sp.erf(argument))
    )
    amplitude = rational_value(scale, x, point)
    baseline = rational_value(remainder, x, point)
    if any(
        value is None or value.is_finite is not True for value in (amplitude, baseline)
    ):
        return None
    value = baseline - amplitude * (sp.log(2) + sp.EulerGamma / 2)
    return answer(
        value,
        "inverse_gaussian_log_moment"
        if inverse
        else "gaussian_log_moment_cancellation",
        "Termwise integration of the entire erf series gives 2*y*2F2(1/2,1/2;3/2,3/2;-y^2)/sqrt(pi)=integral_0^y erf(t)/t dt. Integration by parts changes the cancellation into 2/sqrt(pi)*integral_0^y exp(-t^2)*log(t) dt. The integrand is absolutely integrable at zero and infinity. Its full moment is digamma(1/2)/2=-log(2)-EulerGamma/2. The rational argument tends to positive infinity, or erfinv(1-delta) does so with positive rational delta tending to zero; erf(erfinv(u))=u on this real interval. Finite rational coefficient limits preserve this value.",
    )


def fresnel_auxiliary_tail_certificate(expr, x, point, domain, assumptions):
    """Transfer the real Fresnel auxiliary tail through finite rational weights."""
    if (
        point is not sp.oo
        or domain is not sp.S.true
        or assumptions is sp.S.false
        or assumptions.has(x)
        or x.is_real is False
        or sp.count_ops(expr) > 50
    ):
        return None
    atoms = expr.atoms(FresnelF, FresnelG)
    if not atoms:
        return None
    transformed = expr
    for atom in atoms:
        if not _positive_real_rational_tail(atom.args[0], x):
            return None
        data = linear_in(transformed, atom)
        if data is None:
            return None
        coefficient, transformed = data
        value = rational_value(coefficient, x, sp.oo)
        if value is None or value.is_finite is not True:
            return None
    value = rational_value(transformed, x, sp.oo)
    if value is None or value.is_finite is not True:
        return None
    return answer(
        value,
        "fresnel_auxiliary_real_tail",
        "DLMF 7.5.3-7.5.4 express each auxiliary function as a bounded real sine/cosine combination of C(y)-1/2 and S(y)-1/2. Both differences tend to zero as real y tends to +infinity (DLMF 7.12). Positive rational argument tails and finite rational weights preserve convergence; their poles are eventually avoided.",
    )


def appell_integral_continuity_certificate(expr, x, point, domain, assumptions):
    """Use Euler's F1 integral on real argument charts strictly below one."""
    point = sp.sympify(point)
    if (
        point.is_finite is not True
        or point.is_real is not True
        or domain is not sp.S.true
        or assumptions is sp.S.false
        or assumptions.has(x)
        or x.is_real is False
        or sp.count_ops(expr) > 60
    ):
        return None
    atoms = expr.atoms(sp.appellf1)
    if len(atoms) != 1:
        return None
    atom = next(iter(atoms))
    a, b1, b2, c, u, v = atom.args
    if (
        any(
            p.has(x) or p.is_real is not True or p.is_finite is not True
            for p in (a, b1, b2, c)
        )
        or a.is_positive is not True
        or (c - a).is_positive is not True
    ):
        return None
    targets = []
    for argument in (u, v):
        if not argument.is_rational_function(x):
            return None
        numerator, denominator = sp.fraction(sp.cancel(argument))
        try:
            polynomials = (sp.Poly(numerator, x), sp.Poly(denominator, x))
        except sp.PolynomialError:
            return None
        if any(
            coefficient.is_real is not True
            for p in polynomials
            for coefficient in p.all_coeffs()
        ):
            return None
        target = argument.subs(x, point)
        if target.is_finite is not True or (1 - target).is_positive is not True:
            return None
        targets.append(target)
    data = linear_in(expr, atom)
    if data is None:
        return None
    coefficient, rest = data
    # Outside F1 only rational expressions and positive-centered radicals are
    # admitted; substituting arbitrary functions could conceal a discontinuity.
    for term in (coefficient, rest):
        for function in term.atoms(sp.Function):
            if function.has(x):
                return None
        for power in term.atoms(sp.Pow):
            if power.exp.is_Integer is not True and power.has(x):
                if (
                    power.exp.has(x)
                    or power.base.subs(x, point).is_positive is not True
                ):
                    return None
        value = term.subs(x, point)
        if value.is_finite is not True:
            return None
    value = coefficient.subs(x, point) * sp.appellf1(
        a, b1, b2, c, *targets
    ) + rest.subs(x, point)
    return answer(
        value,
        "appell_euler_integral_continuity",
        "Euler's F1 integral (DLMF 16.15.1) has weight t^(a-1)*(1-t)^(c-a-1), integrable for c>a>0. Both real arguments stay strictly below one, so their factors are uniformly bounded near the target. Dominated convergence proves continuity. The outside rational and positive-centered radical factors are continuous and finite.",
    )
