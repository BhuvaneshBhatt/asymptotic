"""Bounded finite complex rational germs, without branch or infinity guesses."""

import sympy as sp

from .limit_models import LimitEvidence, LimitStatus


def finite_complex_rational_certificate(expr, x, point, domain, assumptions):
    if (
        not isinstance(expr, sp.Expr)
        or point.has(x)
        or point.is_finite is not True
        or point.is_real is not False
    ):
        return None
    if (
        assumptions is sp.S.false
        or assumptions.has(x)
        or domain is not sp.S.true
        or x.is_real is True
        or sp.count_ops(expr) > 120
    ):
        return None
    if x.is_imaginary is True and point.is_imaginary is not True:
        return None
    if not expr.is_rational_function(x):
        return None
    try:
        numerator, denominator = sp.fraction(expr)
        polynomials = (sp.Poly(numerator, x), sp.Poly(denominator, x))
        if any(
            p.degree() > 32 or any(c.is_finite is not True for c in p.all_coeffs())
            for p in polynomials
        ):
            return None
        # Numeric roots of unity must share the same algebraic representation
        # in the source coefficients and the target before common factors cancel.
        canonical = []
        for p in polynomials:
            canonical.append(
                sp.Add(
                    *(
                        sp.expand_complex(c) * x ** monomial[0]
                        if c.is_number is True
                        else c * x ** monomial[0]
                        for monomial, c in p.terms()
                    )
                )
            )
        algebraic = all(
            c.is_algebraic is True for p in polynomials for c in p.all_coeffs()
        )
        reduced = (
            sp.cancel(canonical[0] / canonical[1], extension=True)
            if algebraic
            else sp.cancel(canonical[0] / canonical[1])
        )
        point = sp.expand_complex(point) if point.is_number is True else point
        numerator, denominator = sp.fraction(reduced)
        at_denominator = sp.simplify(denominator.subs(x, point))
        from ._symbolic_policy import bounded_ask

        if (
            at_denominator.is_zero is not False
            and bounded_ask(sp.Q.nonzero(at_denominator), assumptions) is not True
        ):
            return None
        value = sp.simplify(numerator.subs(x, point) / at_denominator)
        if value.is_number is True:
            value = sp.simplify(sp.expand_complex(value))
        if value.is_finite is not True:
            return None
    except (sp.PolynomialError, TypeError, ValueError):
        return None
    return (
        LimitStatus.PROVED,
        value,
        LimitEvidence(
            "finite_complex_rational_germ",
            "Exact polynomial cancellation preserves the punctured complex germ; the reduced denominator is nonzero at the target, so continuity proves the limit in every complex approach direction.",
            value=value,
        ),
    )
