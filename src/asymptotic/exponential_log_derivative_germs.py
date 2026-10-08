"""Small exponential increments with rational, possibly periodic amplitudes."""

import sympy as sp

from .compact_limit_germs import answer


def exponential_log_derivative_certificate(expr, x, point, domain, assumptions):
    """Resolve f'*exp(f)/(exp(f)-1) for a decaying rational unit-phase germ."""
    if (
        point is not sp.oo
        or domain is not sp.S.true
        or assumptions is sp.S.false
        or x.is_real is False
        or expr.free_symbols - {x}
        or not expr.has(sp.exp)
        or sp.count_ops(expr) > 130
    ):
        return None
    for clause in sp.And.make_args(assumptions):
        if clause is sp.S.true:
            continue
        if not (
            isinstance(clause, (sp.GreaterThan, sp.StrictGreaterThan))
            and clause.lhs == x
            and clause.rhs.is_real is True
            and clause.rhs.is_finite is True
        ):
            return None
    for outer in sorted(expr.atoms(sp.exp), key=sp.default_sort_key):
        f = outer.args[0]
        phases = f.atoms(sp.exp)
        if len(phases) > 1:
            continue
        omega = sp.S.Zero
        amplitude = f
        if phases:
            phase = next(iter(phases))
            try:
                polynomial = sp.Poly(phase.args[0], x)
            except sp.PolynomialError:
                continue
            if polynomial.degree() != 1 or polynomial.nth(0) != 0:
                continue
            omega = polynomial.nth(1) / sp.I
            if (
                omega.free_symbols
                or omega.is_real is not True
                or omega.is_finite is not True
            ):
                continue
            amplitude = sp.cancel(f / phase)
        try:
            numerator, denominator = (
                sp.Poly(p, x) for p in sp.fraction(sp.cancel(amplitude))
            )
        except sp.PolynomialError:
            continue
        if (
            numerator.is_zero
            or denominator.is_zero
            or numerator.degree() >= denominator.degree()
        ):
            continue
        if max(numerator.degree(), denominator.degree()) > 8 or any(
            c.free_symbols or c.is_finite is not True
            for c in (*numerator.all_coeffs(), *denominator.all_coeffs())
        ):
            continue
        if numerator.LC().is_zero is not False or denominator.LC().is_zero is not False:
            continue
        expected = sp.diff(f, x) * outer / (outer - 1)
        if sp.cancel(expr - expected) != 0:
            continue
        return answer(
            sp.I * omega,
            "small_exponential_log_derivative",
            "For f=r(x)*exp(i*omega*x), r is a nonzero fixed rational function "
            "of negative degree and omega is finite real. Thus f tends uniformly "
            "to zero, is eventually nonzero, and f'/f=i*omega+r'/r tends to "
            "i*omega. The analytic germ f*exp(f)/(exp(f)-1) tends to one. "
            "Rational poles and zeros occur only finitely often, and for "
            "0<abs(f)<2*pi the exponential denominator has no zero. These "
            "bounds apply on the whole positive tail and every admitted lower-bound constraint.",
        )
    return None
