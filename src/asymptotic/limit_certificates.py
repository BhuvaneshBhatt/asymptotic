"""Certified local branches, cancellation scales and oscillatory limits.

Branch jumps use local Taylor germs. Polylogarithmic cancellations use the
inversion formula (DLMF 25.12) with an exponentially vanishing remainder.
Exponential-integral constants follow DLMF 6.4 and 6.6; gamma ratios use 5.11.
"""

import sympy as sp

from ._symbolic_policy import bounded_ask, bounded_limit
from .limit_models import LimitEvidence, LimitStatus


def _answer(value, method, statement):
    return LimitStatus.PROVED, value, LimitEvidence(method, statement, value=value)


def _sign(expr, assumptions=True):
    expr = sp.sympify(expr)
    if expr.has(sp.nan, sp.zoo) or expr.is_real is False:
        return None
    clauses = sp.And.make_args(sp.sympify(assumptions))
    if sp.Gt(expr, 0) in clauses or sp.Lt(0, expr) in clauses:
        return 1
    if sp.Lt(expr, 0) in clauses or sp.Gt(0, expr) in clauses:
        return -1
    if bounded_ask(sp.Q.positive(expr), assumptions) is True:
        return 1
    if bounded_ask(sp.Q.negative(expr), assumptions) is True:
        return -1
    if expr == 0:
        return 0
    return None


def _leading_sign(expr, t, assumptions=True):
    try:
        leading = sp.simplify(expr.as_leading_term(t))
        coefficient, power = leading.as_coeff_exponent(t)
        if coefficient.has(t):
            return None
        return _sign(coefficient, assumptions), power
    except (ValueError, TypeError, NotImplementedError, sp.PoleError):
        return None


def branch_boundary_certificate(expr, x, point, domain, assumptions):
    """Evaluate finite elementary cut germs, including higher-order tangencies."""
    if point in (sp.oo, -sp.oo) or not expr.has(sp.log, sp.arg, sp.atanh, sp.Pow):
        return None
    if domain is not sp.S.true and domain != (x > point):
        return None
    if any(
        a.func
        not in (
            sp.log,
            sp.arg,
            sp.atanh,
            sp.exp,
            sp.sin,
            sp.cos,
            sp.tan,
            sp.re,
            sp.im,
            sp.Abs,
        )
        for a in expr.atoms(sp.Function)
    ) or any(a.exp.has(x) for a in expr.atoms(sp.Pow)):
        return None
    # The logarithmic definition preserves the principal atanh branch.
    expression = expr.replace(
        lambda a: a.func is sp.atanh,
        lambda a: (sp.log(1 + a.args[0]) - sp.log(1 - a.args[0])) / 2,
    )
    t = sp.Dummy("cut_t", positive=True)
    signs = (
        (1,)
        if (x.is_positive is True and point == 0)
        or (domain is not sp.S.true and domain == (x > point))
        else (1, -1)
    )
    values = []
    detected = False
    for side in signs:
        along = expression.subs(x, point + side * t)
        replacements = {}
        for atom in sp.preorder_traversal(along):
            if atom.func in (sp.log, sp.arg):
                argument, exponent = atom.args[0], None
            elif (
                atom.is_Pow and atom.exp.is_integer is not True and not atom.exp.has(t)
            ):
                argument, exponent = atom.base, atom.exp
            else:
                continue
            at = sp.simplify(sp.expand_complex(argument.subs(t, 0)))
            if _sign(at, assumptions) != -1:
                continue
            imaginary = sp.refine(sp.im(sp.expand_complex(argument)), assumptions)
            data = _leading_sign(imaginary, t, assumptions)
            if data is None or data[0] is None:
                return None
            branch_sign, power = data
            if branch_sign != 0 and (power.is_positive is not True):
                return None
            phase = sp.pi if branch_sign in (0, 1) else -sp.pi
            radius = -at
            replacements[atom] = (
                phase
                if atom.func is sp.arg
                else sp.log(radius) + sp.I * phase
                if exponent is None
                else radius**exponent * sp.exp(sp.I * phase * exponent)
            )
            detected = True
        if not detected:
            return None
        # Boundary constants cannot replace a germ inside a singular outer
        # quotient: (log(-1+i*t)-i*pi)/t needs a derivative, not continuity.
        for power_atom in along.atoms(sp.Pow):
            if power_atom.exp.is_negative is True:
                base = power_atom.base.xreplace(replacements).subs(t, 0)
                if base.is_zero is not False:
                    return None
        candidate = sp.simplify(along.xreplace(replacements).subs(t, 0))
        if candidate.has(t, sp.nan, sp.zoo, sp.oo, -sp.oo, sp.AccumBounds, sp.Limit):
            return None
        values.append(candidate)
    if len(values) == 2 and sp.simplify(values[0] - values[1]) != 0:
        if (values[0] - values[1]).equals(0) is not False:
            return None
        n = sp.Dummy("attained_cut_n", integer=True, positive=True)
        return (
            LimitStatus.DOES_NOT_EXIST,
            None,
            tuple(
                LimitEvidence(
                    "principal_branch_taylor_jump",
                    "The real sequence x=point+/-1/n attains the signed cut germ; outer denominators are nonzero at the boundary and remain nonzero on its tail.",
                    ((x, point + side / n),),
                    value,
                )
                for side, value in zip(signs, values)
            ),
        )
    return _answer(
        values[0],
        "principal_branch_taylor_boundary",
        "the signed imaginary Taylor germ selects the principal-cut boundary",
    )


def quadratic_root_quotient_certificate(expr, x, point, assumptions):
    """Derivative of a principal root off its cut, divided by a simple zero."""
    n, d = sp.fraction(expr)
    roots = [a for a in n.atoms(sp.Pow) if a.exp == sp.Rational(1, 2) and a.has(x)]
    if len(roots) != 1:
        return None
    root = roots[0]
    coefficient = sp.expand(n).coeff(root)
    rest = sp.simplify(n - coefficient * root)
    if coefficient.has(x) or rest.has(x) or coefficient == 0:
        return None
    b = sp.simplify(-rest / coefficient)
    if _sign(sp.re(b), assumptions) != 1:
        return None
    q = root.base
    if sp.simplify(q.subs(x, point) - b * b) != 0 or sp.simplify(d.subs(x, point)) != 0:
        return None
    derivative = sp.simplify(sp.diff(d, x).subs(x, point))
    if derivative == 0 or derivative.is_zero is not False:
        return None
    value = sp.simplify(
        coefficient * sp.diff(q, x).subs(x, point) / (2 * b * derivative)
    )
    return _answer(
        value,
        "principal_root_derivative_quotient",
        "Re(b)>0 fixes sqrt(b**2)=b; root differentiation and a simple denominator zero give the quotient",
    )


def gamma_ratio_certificate(expr, x, point, domain, assumptions):
    n, d = sp.fraction(expr)
    if n.func is not sp.gamma or d.func is not sp.gamma:
        return None
    difference = sp.simplify(n.args[0] - d.args[0])
    if difference.has(x) or difference.is_real is not True:
        return None
    g = d.args[0]
    if not (
        point is sp.oo
        or (x.is_positive is True and point == 0)
        or (domain is not sp.S.true and domain == (x > point))
    ):
        return None
    gv = bounded_limit(g, x, point, allow_general=True)
    if gv is not sp.oo:
        return None
    sign = _sign(difference, assumptions)
    if sign is None:
        return None
    return _answer(
        sp.S.Zero if sign == -1 else sp.S.One if sign == 0 else sp.oo,
        "gamma_positive_ratio",
        "Gamma(g+a)/Gamma(g) ~ g**a on the positive real tail",
    )


def periodic_subsequence_certificate(expr, x, point):
    """Prove DNE by exact arithmetic subsequences; never infer agreement."""
    atoms = expr.atoms(sp.sin, sp.cos, sp.tan, sp.cot, sp.sec, sp.csc)
    if point is not sp.oo or not atoms or expr.free_symbols - {x}:
        return None
    if any(
        a.func
        not in (
            sp.sin,
            sp.cos,
            sp.tan,
            sp.cot,
            sp.sec,
            sp.csc,
            sp.exp,
            sp.log,
            sp.atan,
            sp.Abs,
        )
        for a in expr.atoms(sp.Function)
    ):
        return None
    slopes = []
    for atom in atoms:
        a = sp.diff(atom.args[0], x)
        b = sp.simplify(atom.args[0] - a * x)
        if a.is_Rational is not True or a == 0 or b.has(x) or b.is_real is not True:
            return None
        slopes.append(abs(a))
    slope = sp.Rational(
        sp.igcd(*[a.p for a in slopes]) if len(slopes) > 1 else slopes[0].p,
        sp.ilcm(*[a.q for a in slopes]) if len(slopes) > 1 else slopes[0].q,
    )
    n = sp.Dummy("phase_n", integer=True, positive=True)
    values = []
    witnesses = []
    for phase in (
        0,
        sp.pi / 2,
        sp.pi,
        3 * sp.pi / 2,
        sp.pi / 4,
        sp.pi * slope / (2 * max(slopes)),
    ):
        along = sp.simplify(expr.subs(x, (2 * sp.pi * n + phase) / slope))
        if along.has(sp.nan, sp.zoo, sp.oo, -sp.oo) or any(
            a.has(n)
            for a in along.atoms(sp.sin, sp.cos, sp.tan, sp.cot, sp.sec, sp.csc)
        ):
            continue
        # Complex moving tails and residual parity powers need their own
        # branch/attainment theorem, rather than a generic backend limit.
        if along.has(n) and along.is_real is not True:
            continue
        if any(
            p.exp.has(n) and p.base.is_positive is not True for p in along.atoms(sp.Pow)
        ):
            continue
        value = bounded_limit(along, n, sp.oo, allow_general=True)
        if value is None or value.has(n, sp.nan, sp.zoo, sp.AccumBounds, sp.Limit):
            continue
        for previous in values:
            if previous == value:
                continue
            if (
                previous in (sp.oo, -sp.oo)
                or value in (sp.oo, -sp.oo)
                or sp.simplify(previous - value).is_zero is False
            ):
                return (
                    LimitStatus.DOES_NOT_EXIST,
                    None,
                    (
                        LimitEvidence(
                            "periodic_arithmetic_subsequences",
                            "two exact periodic arithmetic subsequences have incompatible scalar limits",
                            ((x, (2 * sp.pi * n + phase) / slope),),
                            value,
                        ),
                        witnesses[values.index(previous)],
                    ),
                )
        values.append(value)
        witnesses.append(
            LimitEvidence(
                "periodic_arithmetic_subsequences",
                "Exact arithmetic phase avoids all periodic poles; the substituted expression has the recorded attained limit.",
                ((x, (2 * sp.pi * n + phase) / slope),),
                value,
            )
        )
    return None


def polylog_inversion_certificate(expr, x, point):
    """Integer polylogarithms on real exponential tails, with controlled errors."""
    atoms = expr.atoms(sp.polylog)
    if point is not sp.oo or not atoms:
        return None
    replacements = {}
    residuals = []
    for atom in atoms:
        order, z = atom.args
        if order.is_Integer is not True or order < 1 or order > 12:
            return None
        exponentials = tuple(z.atoms(sp.exp))
        if len(exponentials) == 1:
            exponential = exponentials[0]
            rate = sp.simplify(exponential.args[0] / x)
            coefficient = sp.simplify(z / exponential)
            if (
                rate.has(x)
                or rate.is_positive is not True
                or coefficient.has(x)
                or coefficient.is_real is not True
                or (
                    coefficient.is_positive is not True
                    and coefficient.is_negative is not True
                )
            ):
                return None
            log_minus = (
                rate * x
                + sp.log(abs(coefficient))
                + (sp.I * sp.pi if coefficient.is_positive else 0)
            )
        elif z.is_rational_function(x) and z.is_real is True:
            tail = bounded_limit(z, x, sp.oo, allow_general=True)
            if tail not in (sp.oo, -sp.oo):
                return None
            log_minus = sp.log(-z) if tail is -sp.oo else sp.log(z) + sp.I * sp.pi
        else:
            return None
        polynomial = sp.expand(
            -((2 * sp.pi * sp.I) ** order)
            / sp.factorial(order)
            * sp.bernoulli(order, sp.Rational(1, 2) + log_minus / (2 * sp.pi * sp.I))
        )
        # The inverse argument is real and eventually inside (-1,1), so
        # the inversion remainder is real, including inside re(...).
        residual = sp.Dummy("polylog_remainder", real=True)
        replacements[atom] = polynomial + residual
        residuals.append((residual, z))
    transformed = sp.expand(expr.xreplace(replacements))
    # For n>=1, |Li_n(w)|<=|w|/(1-|w|) on |w|<1. Only polynomial
    # coefficients are discarded, so each remainder vanishes exponentially.
    symbols = [r for r, z in residuals]
    for residual, z in residuals:
        coefficient = transformed.coeff(residual)
        if coefficient.has(*symbols):
            return None
        if any(a.func not in (sp.log, sp.exp) for a in coefficient.atoms(sp.Function)):
            return None
        if bounded_limit(coefficient / z, x, sp.oo, allow_general=True) != 0:
            return None
    if sp.expand(transformed - sum(transformed.coeff(r) * r for r in symbols)).has(
        *symbols
    ):
        return None
    transformed = transformed.xreplace(dict.fromkeys(symbols, 0))
    # Principal log(1-c exp(kx)) for real c on an exponential tail.
    for atom in transformed.atoms(sp.log):
        argument = atom.args[0]
        exps = tuple(argument.atoms(sp.exp))
        if len(exps) != 1:
            continue
        e = exps[0]
        c = sp.simplify((argument - 1) / e)
        k = sp.simplify(e.args[0] / x)
        coefficient = sp.expand(transformed).coeff(atom)
        symbol = sp.Dummy("log_tail")
        dependence = transformed.xreplace({atom: symbol})
        if (
            not c.has(x)
            and c.is_real is True
            and c.is_zero is False
            and k.is_positive is True
            and coefficient.is_polynomial(x)
            and dependence.is_polynomial(symbol)
            and sp.Poly(dependence, symbol).degree() <= 1
        ):
            leading = k * x + sp.log(abs(c)) + (sp.I * sp.pi if c.is_negative else 0)
            transformed = transformed.xreplace({atom: leading})
    # Positive exponential polynomials include (1-exp(x))**2. Expanding
    # that square exposes several exponentials, but its leading log is still
    # exact up to an exponentially small term times any polynomial multiplier.
    for atom in transformed.atoms(sp.log):
        coefficient = sp.expand(transformed).coeff(atom)
        symbol = sp.Dummy("log_polynomial_tail")
        dependence = transformed.xreplace({atom: symbol})
        if (
            not coefficient.is_polynomial(x)
            or not dependence.is_polynomial(symbol)
            or sp.Poly(dependence, symbol).degree() > 1
        ):
            continue
        terms = []
        for term in sp.Add.make_args(sp.expand(atom.args[0])):
            exps = tuple(term.atoms(sp.exp))
            if not exps:
                if term.has(x):
                    break
                terms.append((sp.S.Zero, term))
                continue
            if len(exps) != 1:
                break
            exponential = exps[0]
            rate = sp.simplify(exponential.args[0] / x)
            c = sp.cancel(term / exponential)
            if (
                rate.is_number is not True
                or rate.is_real is not True
                or c.has(x)
                or c.is_real is not True
                or c.is_finite is not True
            ):
                break
            terms.append((rate, c))
        else:
            if not terms:
                continue
            rate = max(k for k, c in terms)
            leading = sp.simplify(sum(c for k, c in terms if k == rate))
            if rate.is_positive is True and leading.is_positive is True:
                transformed = transformed.xreplace({atom: rate * x + sp.log(leading)})
    value = bounded_limit(sp.simplify(transformed), x, sp.oo, allow_general=True)
    if value is None or value.has(x, sp.nan, sp.zoo, sp.AccumBounds, sp.Limit):
        return None
    return _answer(
        value,
        "integer_polylog_inversion",
        "the integer inversion polynomial is exact up to exponentially small polylog/log tails with polynomial coefficients",
    )


def exponential_integral_tail_certificate(expr, x, point, assumptions):
    """Negative-real small germs and bounded-imaginary left-half-plane tails."""
    atoms = expr.atoms(sp.Ei, sp.Ci)
    if not atoms:
        return None
    if point is not sp.oo and not (point == 0 and x.is_positive is True):
        return None
    transformed = expr
    for atom in atoms:
        coefficient = sp.expand(transformed).coeff(atom)
        if coefficient.has(x) and not coefficient.is_polynomial(x):
            return None
        argument = atom.args[0]
        locally_real = argument.is_real is True
        if argument.func is sp.log and argument.args[0].is_rational_function(x):
            at = sp.simplify(argument.args[0].subs(x, point))
            locally_real = at.is_positive is True and all(
                a.is_real is True for a in argument.args[0].free_symbols
            )
        real = argument if locally_real else sp.re(sp.expand_complex(argument))
        imaginary = sp.S.Zero if locally_real else sp.im(sp.expand_complex(argument))
        real_limit = bounded_limit(real, x, point, allow_general=True)
        imag_limit = bounded_limit(imaginary, x, point, allow_general=True)
        if (
            real_limit is -sp.oo
            and imag_limit is not None
            and (
                imag_limit.is_finite is True
                or (atom.func is sp.Ei and imag_limit in (sp.oo, -sp.oo))
            )
        ):
            if coefficient.has(x):
                return None
            t = sp.Dummy("integral_tail", positive=True)
            along = (
                imaginary.subs(x, 1 / t)
                if point is sp.oo
                else imaginary.subs(x, point + t)
            )
            data = _leading_sign(along, t, assumptions)
            if data is None or data[0] is None:
                return None
            value = data[0] * sp.I * sp.pi
            transformed = transformed.xreplace({atom: value})
        elif atom.func is sp.Ei and imaginary == 0 and real_limit == 0:
            if point is sp.oo:
                t = sp.Dummy("small_integral", positive=True)
                data = _leading_sign(argument.subs(x, 1 / t), t, assumptions)
                # Exponentially small arguments have an essential singularity;
                # their sign can instead be proved structurally.
                negative = _sign(argument, assumptions) == -1
            else:
                negative = None
                t = sp.Dummy("small_integral", positive=True)
                data = _leading_sign(argument.subs(x, point + t), t, assumptions)
                negative = data is not None and data[0] == -1
            if not negative:
                return None
            # The convergent Ei series remainder is O(argument). Ensure its
            # coefficient times argument tends to zero before dropping it.
            error = bounded_limit(coefficient * argument, x, point, allow_general=True)
            if error != 0:
                return None
            transformed = transformed.xreplace(
                {atom: sp.EulerGamma + sp.expand_log(sp.log(-argument), force=False)}
            )
        else:
            return None
    value = bounded_limit(sp.simplify(transformed), x, point, allow_general=True)
    if value is None or value.has(x, sp.nan, sp.zoo, sp.AccumBounds, sp.Limit):
        return None
    return _answer(
        value,
        "exponential_integral_principal_germ",
        "the convergent real Ei series or the signed Ci/Ei continuation constant fixes the principal branch",
    )


def scalar_limit_certificate(expr, x, point, domain, assumptions):
    point = sp.sympify(point)
    expr_original_variable = x
    if point in (sp.oo, -sp.oo) and (
        x.is_real is False
        or (point is sp.oo and x.is_nonpositive is True)
        or (point is -sp.oo and x.is_nonnegative is True)
    ):
        return (
            LimitStatus.UNKNOWN,
            None,
            LimitEvidence(
                "incompatible_real_infinity_chart",
                "The default real infinity chart is incompatible with the explicit variable type or sign. No real attained witnesses are certified; complex directions require an explicit supported chart.",
            ),
        )
    orientation = -1 if point is -sp.oo else 1
    domain = sp.sympify(domain)
    if domain is not sp.S.true and not (
        isinstance(domain, sp.StrictGreaterThan)
        and domain.lhs == x
        and domain.rhs == point
    ):
        return None
    from .exponential_log_derivative_germs import exponential_log_derivative_certificate

    logarithmic_derivative = exponential_log_derivative_certificate(
        expr, x, point, domain, assumptions
    )
    if logarithmic_derivative is not None:
        return logarithmic_derivative
    from .bessel_primitive_germs import bessel_primitive_certificate

    primitive = bessel_primitive_certificate(expr, x, point, domain, assumptions)
    if primitive is not None:
        return primitive
    from .logarithmic_integral_germs import logarithmic_integral_power_certificate

    integral_power = logarithmic_integral_power_certificate(
        expr, x, point, domain, assumptions
    )
    if integral_power is not None:
        return integral_power
    from .dilogarithm_tail_germs import paired_dilogarithm_tail_certificate

    dilogarithm = paired_dilogarithm_tail_certificate(
        expr, x, point, domain, assumptions
    )
    if dilogarithm is not None:
        return dilogarithm
    from .parameter_tail_germs import (
        binomial_ratio_certificate,
        hyperbolic_parameter_tail_certificate,
    )

    for provider in (binomial_ratio_certificate, hyperbolic_parameter_tail_certificate):
        parameter_tail = provider(expr, x, point, domain, assumptions)
        if parameter_tail is not None:
            return parameter_tail
    from .integral_ray_germs import (
        gaussian_root_side_certificate,
        vertical_expint_certificate,
    )

    for provider in (gaussian_root_side_certificate, vertical_expint_certificate):
        ray = provider(expr, x, point, domain, assumptions)
        if ray is not None:
            return ray
    from .joint_phase_germs import joint_phase_certificate

    joint = joint_phase_certificate(expr, x, point, domain, assumptions)
    if joint is not None:
        return joint
    from .definition_limit_germs import (
        axis_argument_certificate,
        divergent_integer_certificate,
        local_integral_certificate,
        logarithmic_ei_certificate,
        stirling_remainder_certificate,
        unresolved_gamma_parameter_certificate,
    )

    for provider in (
        axis_argument_certificate,
        divergent_integer_certificate,
        local_integral_certificate,
        logarithmic_ei_certificate,
        unresolved_gamma_parameter_certificate,
    ):
        result = provider(expr, x, point, domain, assumptions)
        if result is not None:
            return result
    from .integral_definition_germs import gaussian_log_moment_certificate

    result = gaussian_log_moment_certificate(expr, x, point, domain, assumptions)
    if result is not None:
        return result
    if assumptions is not sp.S.false and not assumptions.has(x):
        result = stirling_remainder_certificate(expr, x, point)
        if result is not None:
            return result
    from .attained_oscillatory_germs import (
        nested_secant_certificate,
        secant_uniform_quotient_certificate,
    )

    for provider in (nested_secant_certificate, secant_uniform_quotient_certificate):
        oscillation = provider(expr, x, point, assumptions)
        if oscillation is not None:
            return oscillation
    from .special_function_germs import special_function_certificate

    special = special_function_certificate(expr, x, point)
    if special is not None:
        return special
    from .univariate_branch_certificates import reciprocal_log_inverse_certificate

    inverse = reciprocal_log_inverse_certificate(expr, x, point, domain)
    if inverse is not None:
        return inverse
    from .tail_cancellation_germs import upper_gamma_cut_certificate

    cut = upper_gamma_cut_certificate(expr, x, point, domain)
    if cut is not None:
        return cut
    complex_norm = point is sp.oo and any(
        a.args[0].is_Pow
        and a.args[0].base == x
        and a.args[0].exp.free_symbols
        and a.args[0].exp.is_real is not True
        for a in expr.atoms(sp.Abs)
    )
    if point in (sp.oo, -sp.oo) and (point is -sp.oo or x.is_positive is not True):
        coordinate = sp.Dummy("real_positive_tail", positive=True)
        replacement = coordinate if point is sp.oo else -coordinate
        expr = expr.subs(x, replacement)
        domain = sp.sympify(domain).subs(x, replacement)
        assumptions = sp.sympify(assumptions).subs(x, replacement)
        x = coordinate
        point = sp.oo
    original_variable = expr_original_variable

    if point is sp.oo and expr.is_Pow and expr.exp.has(sp.binomial):
        from ._symbolic_policy import bounded_ask

        parameters = set().union(
            *(atom.free_symbols for atom in expr.exp.atoms(sp.binomial))
        ) - {x}
        if any(
            parameter.is_real is not True
            and bounded_ask(sp.Q.real(parameter), assumptions) is not True
            for parameter in parameters
        ):
            return (
                LimitStatus.UNKNOWN,
                None,
                LimitEvidence(
                    "binomial_power_parameter_contract",
                    "A complex parameter in the binomial exponent can change its real growth sign. A real-parameter or explicit complex-sector theorem is required before positive-base domination is used.",
                ),
            )

    def restore(result):
        if result is None or x == original_variable:
            return result
        from dataclasses import replace

        status, value, evidence = result
        items = evidence if isinstance(evidence, tuple) else (evidence,)
        items = tuple(
            replace(
                ev,
                substitutions=tuple(
                    (original_variable, orientation * sequence)
                    if variable == x
                    else (variable, sequence)
                    for variable, sequence in ev.substitutions
                ),
            )
            for ev in items
        )
        return status, value, items

    from .integral_definition_germs import (
        appell_integral_continuity_certificate,
        fresnel_auxiliary_tail_certificate,
        gaussian_log_moment_certificate,
    )

    for provider in (
        appell_integral_continuity_certificate,
        fresnel_auxiliary_tail_certificate,
        gaussian_log_moment_certificate,
    ):
        certificate = provider(expr, x, point, domain, assumptions)
        if certificate is not None:
            return restore(certificate)

    from .discontinuous_tail_germs import discontinuous_periodic_tail_certificate
    from .flat_exponential_germs import flat_exponential_certificate

    for provider in (
        discontinuous_periodic_tail_certificate,
        flat_exponential_certificate,
    ):
        certificate = provider(expr, x, point, domain, assumptions)
        if certificate is not None:
            return restore(certificate)

    from .complex_power_germs import (
        decaying_power_log_certificate,
        exponentially_small_denominator_certificate,
    )

    for provider in (
        decaying_power_log_certificate,
        exponentially_small_denominator_certificate,
    ):
        certificate = provider(expr, x, point)
        if certificate is not None:
            return restore(certificate)

    from .analytic_cancellation_germs import (
        acos_endpoint_ratio_certificate,
        argument_step_germ_certificate,
        complex_logarithmic_real_pole_certificate,
        gamma_analytic_cancellation_certificate,
        inverse_sech_projective_pole_certificate,
        logarithmic_power_sum_certificate,
        vanishing_elliptic_amplitude_certificate,
    )

    certificate = inverse_sech_projective_pole_certificate(
        expr, x, point, domain, assumptions
    )
    if certificate is not None:
        return restore(certificate)

    certificate = argument_step_germ_certificate(expr, x, point, domain, assumptions)
    if certificate is not None:
        return restore(certificate)

    certificate = complex_logarithmic_real_pole_certificate(
        expr, x, point, domain, assumptions
    )
    if certificate is not None:
        return restore(certificate)

    for provider in (
        gamma_analytic_cancellation_certificate,
        acos_endpoint_ratio_certificate,
        logarithmic_power_sum_certificate,
        vanishing_elliptic_amplitude_certificate,
    ):
        certificate = provider(expr, x, point)
        if certificate is not None:
            return restore(certificate)

    from .harmonic_cancellation_germs import (
        circular_segment_endpoint_certificate,
        harmonic_convergent_tail_certificate,
        logarithmic_opposite_pole_ratio_certificate,
    )

    for provider in (
        harmonic_convergent_tail_certificate,
        logarithmic_opposite_pole_ratio_certificate,
        circular_segment_endpoint_certificate,
    ):
        certificate = provider(expr, x, point, domain, assumptions)
        if certificate is not None:
            return restore(certificate)
    from .attained_oscillatory_germs import (
        attained_local_phase_certificate,
        exponential_tangent_tail_certificate,
    )

    for provider in (
        attained_local_phase_certificate,
        exponential_tangent_tail_certificate,
    ):
        certificate = provider(expr, x, point, domain, assumptions)
        if certificate is not None:
            return restore(certificate)
    from .analytic_limits import square_wave_certificate

    certificate = square_wave_certificate(expr, x, point, domain, assumptions)
    if certificate is not None:
        return restore(certificate)
    from .local_branch_germs import (
        algebraic_tangent_pole_certificate,
        fractional_part_transverse_certificate,
        logarithmic_integral_cancellation_certificate,
        opposite_atanh_cut_certificate,
        principal_arg_root_certificate,
        signed_erfc_root_certificate,
    )

    for provider in (
        signed_erfc_root_certificate,
        principal_arg_root_certificate,
        algebraic_tangent_pole_certificate,
        opposite_atanh_cut_certificate,
        fractional_part_transverse_certificate,
        logarithmic_integral_cancellation_certificate,
    ):
        certificate = provider(expr, x, point, domain, assumptions)
        if certificate is not None:
            return restore(certificate)
    from .lambert_small_germs import reciprocal_lambert_germ_certificate

    certificate = reciprocal_lambert_germ_certificate(
        expr, x, point, domain, assumptions
    )
    if certificate is not None:
        return restore(certificate)
    from .exponential_integral_germs import (
        bounded_ei_composition_certificate,
        expint_exponential_inner_certificate,
    )

    for provider in (
        bounded_ei_composition_certificate,
        expint_exponential_inner_certificate,
    ):
        certificate = provider(expr, x, point, domain, assumptions)
        if certificate is not None:
            return restore(certificate)
    from .algebraic_limit_germs import (
        assumed_atan_parameter_certificate,
        assumed_parameter_power_certificate,
        complex_near_one_power_certificate,
        coupled_modulo_envelope_certificate,
        exponential_identity_power_certificate,
        finite_inverse_trig_rational_germ_certificate,
        inverse_trig_cut_certificate,
        literal_sine_decimal_pole_certificate,
        local_real_root_pullback_certificate,
        parity_subsequence_certificate,
        radical_secant_tail_certificate,
        rational_exponential_dominance_certificate,
        real_tangent_log_boundary_certificate,
        reciprocal_sine_amplitude_certificate,
    )

    for provider in (
        finite_inverse_trig_rational_germ_certificate,
        coupled_modulo_envelope_certificate,
        parity_subsequence_certificate,
        reciprocal_sine_amplitude_certificate,
        rational_exponential_dominance_certificate,
        complex_near_one_power_certificate,
        exponential_identity_power_certificate,
        local_real_root_pullback_certificate,
        inverse_trig_cut_certificate,
        radical_secant_tail_certificate,
        real_tangent_log_boundary_certificate,
        assumed_parameter_power_certificate,
        assumed_atan_parameter_certificate,
        literal_sine_decimal_pole_certificate,
    ):
        certificate = provider(expr, x, point, domain, assumptions)
        if certificate is not None:
            return restore(certificate)
    from .local_nonexistence_families import (
        attained_tangent_pole_certificate,
        local_analytic_pole_certificate,
        local_rational_pole_certificate,
        reciprocal_exponential_sides_certificate,
    )

    for provider in (
        local_rational_pole_certificate,
        reciprocal_exponential_sides_certificate,
        attained_tangent_pole_certificate,
        local_analytic_pole_certificate,
    ):
        certificate = provider(expr, x, point, domain, assumptions)
        if certificate is not None:
            return restore(certificate)
    from .attained_periodic_families import (
        imaginary_root_phase_cancellation_certificate,
        periodic_composition_certificate,
    )
    from .principal_branch_logs import principal_unit_phase_log_certificate

    for provider in (
        periodic_composition_certificate,
        imaginary_root_phase_cancellation_certificate,
        principal_unit_phase_log_certificate,
    ):
        certificate = provider(expr, x, point, domain, assumptions)
        if certificate is not None:
            return restore(certificate)
    from .local_radical_branch_germs import (
        reciprocal_root_argument_certificate,
        signed_radical_chart_certificate,
    )

    for provider in (
        signed_radical_chart_certificate,
        reciprocal_root_argument_certificate,
    ):
        certificate = provider(expr, x, point, domain, assumptions)
        if certificate is not None:
            return restore(certificate)
    from .large_order_bessel_germs import proportional_bessel_certificate

    large_order = proportional_bessel_certificate(expr, x, point, domain, assumptions)
    if large_order is not None:
        return restore(large_order)
    from .variable_order_germs import regular_variable_bessel_certificate

    variable_order = regular_variable_bessel_certificate(
        expr, x, point, domain, assumptions
    )
    if variable_order is not None:
        return restore(variable_order)
    from .rational_pullback_germs import (
        finite_error_germ_certificate,
        local_real_sign_certificate,
        rounded_tail_certificate,
    )

    for provider in (
        lambda: local_real_sign_certificate(expr, x, point, domain, assumptions),
        lambda: rounded_tail_certificate(expr, x, point),
        lambda: finite_error_germ_certificate(expr, x, point, domain),
    ):
        result = provider()
        if result is not None and expr_original_variable.is_real is not False:
            return restore(result)
    from .special_function_limit_germs import (
        airy_positive_remainder_certificate,
        erfi_sector_certificate,
        fixed_bessel_tail_certificate,
        fixed_order_subtracted_tail_certificate,
        fresnel_signed_pole_certificate,
        lambert_minus_one_branchpoint_certificate,
        lambert_positive_chart_certificate,
        li_signed_boundary_certificate,
        oscillatory_integral_tail_certificate,
        spherical_tail_bound_certificate,
    )

    for provider in (
        lambda: li_signed_boundary_certificate(expr, x, point, domain),
        lambda: fresnel_signed_pole_certificate(expr, x, point, domain),
        lambda: erfi_sector_certificate(expr, x, point),
        lambda: oscillatory_integral_tail_certificate(expr, x, point),
        lambda: fixed_bessel_tail_certificate(expr, x, point),
        lambda: spherical_tail_bound_certificate(expr, x, point),
        lambda: airy_positive_remainder_certificate(expr, x, point),
        lambda: fixed_order_subtracted_tail_certificate(expr, x, point),
        lambda: lambert_minus_one_branchpoint_certificate(expr, x, point),
        lambda: lambert_positive_chart_certificate(expr, x, point),
    ):
        result = provider()
        if result is not None:
            if expr_original_variable.is_real is False:
                continue
            if result[0] is LimitStatus.DOES_NOT_EXIST and (
                expr_original_variable.is_integer is True or assumptions.has(x)
            ):
                continue
            return restore(result)
    from .elementary_limit_germs import (
        exponential_phase_subsequences,
        gamma_ratio_cancellation_certificate,
        near_one_cancellation_certificate,
        nonpositive_radical_subsequences,
        unit_phase_envelope_certificate,
        zeta_laurent_cancellation_certificate,
    )

    for provider in (
        lambda: unit_phase_envelope_certificate(expr, x, point, domain, assumptions),
        lambda: (
            exponential_phase_subsequences(expr, x, point, domain)
            if not assumptions.has(x) and expr_original_variable.is_integer is not True
            else None
        ),
        lambda: nonpositive_radical_subsequences(expr, x, point, domain),
        lambda: near_one_cancellation_certificate(expr, x, point),
        lambda: gamma_ratio_cancellation_certificate(expr, x, point),
        lambda: zeta_laurent_cancellation_certificate(expr, x, point, domain),
    ):
        result = provider()
        if result is not None and expr_original_variable.is_real is not False:
            return restore(result)
    from .rational_pullback_germs import (
        gamma_rational_pullback_certificate,
        integral_rational_adaptive_tail_certificate,
        near_one_adaptive_certificate,
        real_error_tail_certificate,
    )

    for provider in (
        gamma_rational_pullback_certificate,
        near_one_adaptive_certificate,
        real_error_tail_certificate,
        integral_rational_adaptive_tail_certificate,
    ):
        result = provider(expr, x, point)
        if result is not None and expr_original_variable.is_real is not False:
            return restore(result)
    from .local_tail_germs import (
        ci_trig_cancellation_certificate,
        radical_trig_tail_certificate,
        reciprocal_gamma_exponential_domination_certificate,
        secant_odd_pi_pole_certificate,
        upper_gamma_fixed_argument_tail_certificate,
    )

    for provider in (
        ci_trig_cancellation_certificate,
        secant_odd_pi_pole_certificate,
        upper_gamma_fixed_argument_tail_certificate,
        reciprocal_gamma_exponential_domination_certificate,
        radical_trig_tail_certificate,
    ):
        result = provider(expr, x, point)
        if result is not None:
            return restore(result)
    from .perturbation_scale_germs import (
        gamma_root_correction_certificate,
        principal_lambert_analytic_scale_certificate,
        real_log_exp_tower_certificate,
    )

    for provider in (
        gamma_root_correction_certificate,
        principal_lambert_analytic_scale_certificate,
        real_log_exp_tower_certificate,
    ):
        result = provider(expr, x, point)
        if result is not None:
            return restore(result)
    from .tail_cancellation_germs import (
        dirichlet_cancellation_certificate,
        gamma_ratio_scale_certificate,
        harmonic_tail_certificate,
        monomial_phase_subsequences,
        quadratic_radical_perturbation_certificate,
        small_log_power_certificate,
        tangent_pole_subsequences,
        zeta_finite_pole_certificate,
    )

    for provider in (
        lambda: quadratic_radical_perturbation_certificate(expr, x, point),
        lambda: zeta_finite_pole_certificate(expr, x, point, assumptions, domain),
        lambda: harmonic_tail_certificate(expr, x, point),
        lambda: small_log_power_certificate(expr, x, point),
        lambda: gamma_ratio_scale_certificate(expr, x, point),
    ):
        result = provider()
        if result is not None:
            return restore(result)
    from .gamma_exponential_germs import (
        exponential_envelope_certificate,
        positive_gamma_tail_certificate,
        signed_gamma_germ_certificate,
    )

    for provider in (
        signed_gamma_germ_certificate,
        positive_gamma_tail_certificate,
        exponential_envelope_certificate,
    ):
        result = provider(expr, x, point, assumptions)
        if result is not None:
            return restore(result)
    from .compact_limit_germs import (
        bounded_oscillatory_envelope_certificate,
        compact_rational_certificate,
        gamma_stirling_product_certificate,
        integer_floor_side_certificate,
        lambert_small_scale_certificate,
        positive_lambert_composition_certificate,
        real_monotone_composition_certificate,
    )
    from .univariate_perturbation_limits import (
        gamma_exponential_difference_certificate,
        gamma_small_shift_certificate,
        iterated_digamma_exponential_certificate,
        nested_gamma_growth_certificate,
        regularized_hypergeometric_decay_certificate,
        shifted_gamma_ratio_cancellation_certificate,
        zeta_pole_projection_certificate,
        zeta_tail_certificate,
    )

    for provider in (
        shifted_gamma_ratio_cancellation_certificate,
        zeta_pole_projection_certificate,
        zeta_tail_certificate,
        gamma_small_shift_certificate,
        gamma_exponential_difference_certificate,
        nested_gamma_growth_certificate,
        iterated_digamma_exponential_certificate,
        regularized_hypergeometric_decay_certificate,
        shifted_gamma_ratio_cancellation_certificate,
    ):
        result = provider(expr, x, point)
        if result is not None:
            return restore(result)
    from .compact_limit_germs import (
        gamma_reciprocal_shift_certificate,
        minmax_interval_dominance_certificate,
    )

    for provider in (
        gamma_reciprocal_shift_certificate,
        minmax_interval_dominance_certificate,
    ):
        result = provider(expr, x, point)
        if result is not None:
            return restore(result)
    for provider in (
        compact_rational_certificate,
        lambert_small_scale_certificate,
        positive_lambert_composition_certificate,
        gamma_stirling_product_certificate,
        real_monotone_composition_certificate,
        bounded_oscillatory_envelope_certificate,
        integer_floor_side_certificate,
    ):
        result = provider(expr, x, point)
        if result is not None:
            return restore(result)
    for provider in (
        lambda: dirichlet_cancellation_certificate(expr, x, point),
        lambda: (
            monomial_phase_subsequences(expr, x, point, domain)
            if not assumptions.has(x)
            else None
        ),
        lambda: (
            tangent_pole_subsequences(expr, x, point, domain)
            if not assumptions.has(x)
            else None
        ),
    ):
        result = provider()
        if result is not None:
            return restore(result)
    if complex_norm and all(
        a.func in (sp.log, sp.re, sp.Abs) for a in expr.atoms(sp.Function)
    ):
        norm_replacements = {
            a: a.args[0].base ** sp.re(a.args[0].exp)
            for a in expr.atoms(sp.Abs)
            if a.args[0].is_Pow and a.args[0].base.is_positive is True
        }
        expr = expr.xreplace(norm_replacements)
        value = bounded_limit(expr, x, point, allow_general=True)
        if value is not None and not value.has(
            x, sp.nan, sp.zoo, sp.AccumBounds, sp.Limit, sp.Subs, sp.Derivative
        ):
            return _answer(
                value,
                "positive_base_complex_power_norm",
                "on the positive tail, |x**s|=x**Re(s), so the real-exponent algebraic/logarithmic germ is normalized before evaluation",
            )
    for provider in (
        lambda: nested_log_cotangent_certificate(expr, x, point),
        lambda: complex_parameter_exponential_tail_certificate(expr, x, point),
        lambda: rootsum_log_trace_certificate(expr, x, point),
        lambda: real_log_norm_certificate(expr, x, point, domain),
        lambda: constant_positive_power_certificate(expr, x, point, domain),
        lambda: fibonacci_tail_certificate(expr, x, point),
        lambda: expint_integer_certificate(expr, x, point, domain),
        lambda: principal_radical_infinity_certificate(expr, x, point),
        lambda: hankel_integer_origin_certificate(expr, x, point),
        lambda: primepi_expansion_certificate(expr, x, point),
        lambda: lambert_positive_tail_certificate(expr, x, point),
        lambda: gamma_positive_sum_certificate(expr, x, point, assumptions),
        lambda: neighboring_sine_certificate(expr, x, point),
        lambda: branch_boundary_certificate(expr, x, point, domain, assumptions),
        lambda: quadratic_root_quotient_certificate(expr, x, point, assumptions),
        lambda: gamma_ratio_certificate(expr, x, point, domain, assumptions),
        lambda: (
            periodic_subsequence_certificate(expr, x, point)
            if not assumptions.has(x)
            else None
        ),
        lambda: polylog_inversion_certificate(expr, x, point),
        lambda: exponential_integral_tail_certificate(expr, x, point, assumptions),
    ):
        result = provider()
        if result is not None:
            return restore(result)
    return None


def nested_log_cotangent_certificate(expr, x, point):
    """Logarithmic quotient whose cotangent argument tends to zero."""
    if point != 0 or not expr.has(sp.cot):
        return None
    n, d = sp.fraction(expr)
    denominator_coefficient = sp.cancel(d / sp.log(sp.log(x)))
    if denominator_coefficient.has(x) or denominator_coefficient == 0:
        return None
    cotangent = sp.cot(x / sp.log(x))
    for atom in n.atoms(sp.log):
        factor = sp.cancel(atom.args[0] / cotangent)
        if factor.has(x) or factor.is_zero is not False:
            continue
        coefficient = sp.expand(n).coeff(atom)
        rest = sp.simplify(n - coefficient * atom + coefficient * sp.log(1 / x))
        if coefficient.has(x) or rest.has(x):
            continue
        return _answer(
            sp.simplify(coefficient / denominator_coefficient),
            "nested_log_cotangent_quotient",
            "cot(u)~1/u with u=x/log(x); the principal log branch shifts stay bounded, while log(log(x)) diverges, so the quotient has the stated leading logarithmic coefficient",
        )
    return None


def complex_parameter_exponential_tail_certificate(expr, x, point):
    if point is not sp.oo:
        return None
    for power in expr.atoms(sp.Pow):
        if (
            power.base != x
            or power.exp.has(x)
            or not power.exp.free_symbols
            or power.exp.is_real is True
        ):
            continue
        coefficient = sp.cancel(expr / power)
        exponentials = tuple(coefficient.atoms(sp.exp))
        if len(exponentials) != 1:
            continue
        exponential = exponentials[0]
        rate = sp.simplify(exponential.args[0] / x)
        if rate.has(x) or rate.is_positive is not True:
            continue
        prefactor = sp.cancel(coefficient / exponential)
        admissible = True
        for factor in sp.Mul.make_args(prefactor):
            base, exponent = factor.as_base_exp()
            if (
                factor.is_number
                and factor.is_finite is True
                and factor.is_zero is False
            ):
                continue
            if (
                base not in (x, sp.log(x))
                or exponent.is_number is not True
                or exponent.is_real is not True
            ):
                admissible = False
        if admissible:
            return _answer(
                sp.zoo,
                "complex_parameter_exponential_modulus",
                "for each finite complex parameter, the positive exponential dominates the polynomial/logarithmic modulus; no positive-real direction is asserted for its unrestricted phase",
            )
    return None


def rootsum_log_trace_certificate(expr, x, point):
    """Sum weighted logarithms with an exact Lagrange-interpolation trace."""
    atoms = expr.atoms(sp.RootSum)
    if point is not sp.oo or not atoms:
        return None
    transformed = expr
    for atom in atoms:
        if len(atom.fun.variables) != 1:
            return None
        r = atom.fun.variables[0]
        polynomial = sp.Poly(atom.poly.as_expr().subs(atom.poly.gen, r), r)
        if (
            any(not c.is_number for c in polynomial.all_coeffs())
            or sp.gcd(polynomial, polynomial.diff()).degree() != 0
        ):
            return None
        weight = sp.cancel(atom.fun.expr / sp.log(x - r))
        if weight.has(x) or not weight.is_rational_function(r):
            return None
        numerator, denominator = sp.fraction(weight)
        scale = sp.cancel(denominator / polynomial.diff().as_expr())
        if scale.has(r) or scale == 0:
            return None
        remainder = sp.rem(numerator, polynomial.as_expr(), r)
        trace = sp.Poly(remainder, r).nth(polynomial.degree() - 1) / (
            scale * polynomial.LC()
        )
        symbol = sp.Dummy("trace_sum")
        dependence = transformed.xreplace({atom: symbol})
        if (
            not dependence.is_polynomial(symbol)
            or sp.Poly(dependence, symbol).degree() > 1
        ):
            return None
        coefficient = sp.expand(dependence).coeff(symbol)
        if bounded_limit(coefficient / x, x, sp.oo, allow_general=True) != 0:
            return None
        transformed = transformed.xreplace({atom: trace * sp.log(x)})
    value = bounded_limit(sp.simplify(transformed), x, point, allow_general=True)
    if value is None or value.has(x, sp.nan, sp.zoo, sp.AccumBounds, sp.Limit):
        return None
    return _answer(
        value,
        "rootsum_log_lagrange_trace",
        "the simple-root trace of N(r)/P'(r) is the x**(degree(P)-1) remainder coefficient divided by LC(P); each fixed-root log(x-r) differs from log(x) by O(1/x)",
    )


def real_log_norm_certificate(expr, x, point, domain):
    """Normalize absolute values of negative-real principal logarithms locally."""
    if point in (sp.oo, -sp.oo) or not expr.has(sp.Abs, sp.atan2, sp.log):
        return None
    r = sp.Dummy("real_norm_coordinate", real=True)
    transformed = expr.subs(x, r)
    changed = False
    for piecewise in transformed.atoms(sp.Piecewise):
        stable = True
        for value, condition in piecewise.args:
            for relation in condition.atoms(sp.Rel):
                delta = sp.simplify((relation.lhs - relation.rhs).subs(r, point))
                if delta == 0 or delta.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
                    stable = False
            if not stable:
                break
            at = condition.subs(r, point)
            if at is sp.S.true:
                transformed = transformed.xreplace({piecewise: value})
                changed = True
                break
            if at is not sp.S.false:
                break
    for atom in transformed.atoms(sp.Abs):
        if atom.args[0].func is not sp.log:
            continue
        q = atom.args[0].args[0]
        if q.is_real is not True or not q.is_rational_function(r):
            continue
        at = sp.simplify(q.subs(r, point))
        if at.is_negative is True:
            transformed = transformed.xreplace(
                {atom: sp.sqrt(sp.log(-q) ** 2 + sp.pi**2)}
            )
            changed = True
    for atom in transformed.atoms(sp.atan2):
        y, q = atom.args
        at = sp.simplify(q.subs(r, point))
        if y == 0 and q.is_real is True and at.is_negative is True:
            transformed = transformed.xreplace({atom: sp.pi})
            changed = True
    # log(q**2)=2 log(|q|) on a real nonzero local germ.
    for atom in transformed.atoms(sp.log):
        q = atom.args[0]
        if q.is_Pow and q.exp == 2 and q.base.is_real is True:
            at = sp.simplify(q.base.subs(r, point))
            sign = _sign(at)
            if sign in (-1, 1):
                transformed = transformed.xreplace({atom: 2 * sp.log(sign * q.base)})
                changed = True
    if not changed:
        return None
    if any(
        a.func not in (sp.log, sp.Abs, sp.atan2) for a in transformed.atoms(sp.Function)
    ):
        return None
    transformed = sp.simplify(transformed)
    right = bounded_limit(transformed, r, point, allow_general=True)
    if right is None or right.has(
        r, sp.nan, sp.zoo, sp.AccumBounds, sp.Limit, sp.Subs, sp.Derivative
    ):
        return None
    if not (x.is_positive is True and point == 0) and domain is sp.S.true:
        left = bounded_limit(transformed, r, point, direction="-", allow_general=True)
        if left is None or left.has(
            r, sp.nan, sp.zoo, sp.AccumBounds, sp.Limit, sp.Subs, sp.Derivative
        ):
            return None
        if right != left and (
            right in (sp.oo, -sp.oo)
            or left in (sp.oo, -sp.oo)
            or sp.simplify(right - left).is_zero is False
        ):
            return (
                LimitStatus.DOES_NOT_EXIST,
                None,
                LimitEvidence(
                    "real_log_norm_side_conflict",
                    "the exact real norm formula gives incompatible side germs",
                ),
            )
        if right != left and sp.simplify(right - left) != 0:
            return None
    return _answer(
        right,
        "real_principal_log_norm",
        "on a negative real argument, |Log(q)|=sqrt(log(-q)**2+pi**2); locally constant atan2 branches are normalized before taking the germ",
    )


def constant_positive_power_certificate(expr, x, point, domain):
    if (
        not expr.is_Pow
        or expr.base.has(x)
        or expr.base.is_positive is not True
        or not expr.exp.has(x)
    ):
        return None
    if expr.base == 1:
        return _answer(
            sp.S.One, "constant_unit_power", "a nonzero unit base has value 1"
        )
    increasing = _sign(sp.log(expr.base))
    if increasing not in (-1, 1):
        return None
    sides = (
        ("+",)
        if point in (sp.oo, -sp.oo)
        or (x.is_positive is True and point == 0)
        or (domain is not sp.S.true and domain == (x > point))
        else ("+", "-")
    )
    values = []
    for side in sides:
        value = bounded_limit(expr.exp, x, point, direction=side, allow_general=True)
        if value in (sp.oo, -sp.oo):
            values.append(sp.oo if (value is sp.oo) == (increasing == 1) else sp.S.Zero)
        elif (
            value is not None
            and value.is_finite is True
            and not value.has(sp.AccumBounds, sp.Limit)
        ):
            values.append(expr.base**value)
        else:
            return None
    if len(values) == 2 and values[0] != values[1]:
        return (
            LimitStatus.DOES_NOT_EXIST,
            None,
            LimitEvidence(
                "positive_constant_power_side_conflict",
                "the monotone positive constant power maps the two exponent-side limits to incompatible values",
            ),
        )
    return _answer(
        values[0],
        "positive_constant_power_side_limit",
        "monotonicity of the positive constant power maps the resolved exponent germ",
    )


def fibonacci_tail_certificate(expr, x, point):
    if point is not sp.oo:
        return None
    f = sp.fibonacci(x)
    log_value = sp.lucas(x)
    phi = (1 + sp.sqrt(5)) / 2
    values = {log_value / f: sp.sqrt(5), sp.log(f) / x: sp.log(phi), f ** (1 / x): phi}
    if expr in values:
        return _answer(
            values[expr],
            "binet_dominant_positive_root",
            "Binet's positive root phi dominates the negative root of modulus 1/phi; the relative error decays exponentially",
        )
    return None


def principal_radical_infinity_certificate(expr, x, point):
    """Leading multiplicative polynomial radicals with their approached cut side."""
    if point not in (sp.oo, -sp.oo) or expr.free_symbols - {x} or expr.has(sp.Function):
        return None
    z = sp.Dummy("radical_tail", positive=True)
    expression = expr.subs(x, z if point is sp.oo else -z)
    if not expression.is_Mul or not any(
        a.is_Pow and a.exp.is_Rational and not a.exp.is_integer for a in expression.args
    ):
        return None
    coefficient = sp.S.One
    order = sp.S.Zero
    for factor in expression.args:
        power = factor.exp if factor.is_Pow else sp.S.One
        base = factor.base if factor.is_Pow else factor
        if power.is_Rational is not True or not base.is_polynomial(z):
            return None
        polynomial = sp.Poly(base, z)
        c = sp.expand_complex(polynomial.LC())
        if c == 0:
            return None
        degree = polynomial.degree()
        if power.is_integer is True or c.is_positive is True or c.is_real is False:
            coefficient *= c**power
        elif c.is_negative is True:
            imaginary = sp.im(sp.expand_complex(base))
            imag_poly = sp.Poly(imaginary, z)
            sign = _sign(imag_poly.LC()) if not imag_poly.is_zero else 0
            if sign is None:
                return None
            phase = sp.pi if sign in (0, 1) else -sp.pi
            coefficient *= (-c) ** power * sp.exp(sp.I * phase * power)
        else:
            return None
        order += degree * power
    if order == 0:
        return _answer(
            sp.simplify(coefficient),
            "principal_radical_polynomial_tail",
            "the polynomial leading coefficients and signed imaginary next germs select every principal-power cut boundary",
        )
    if order < 0:
        return _answer(
            sp.S.Zero,
            "principal_radical_polynomial_decay",
            "the nonzero principal-power leading product has negative total order",
        )
    return None


def hankel_integer_origin_certificate(expr, x, point):
    if point != 0:
        return None
    atoms = expr.atoms(sp.hankel1, sp.hankel2)
    if len(atoms) != 1:
        return None
    atom = next(iter(atoms))
    n, arg = atom.args
    if n.is_Integer is not True or n <= 0:
        return None
    a = sp.simplify(arg / x)
    if a.has(x) or a.is_positive is not True:
        return None
    multiplier = sp.cancel(expr / atom)
    coefficient, power = multiplier.as_coeff_exponent(x)
    if coefficient.has(x) or power.is_integer is not True:
        return None
    if power == n:
        phase = -sp.I if atom.func is sp.hankel1 else sp.I
        return _answer(
            coefficient * phase * 2**n * sp.factorial(n - 1) / (sp.pi * a**n),
            "integer_hankel_origin_leading_pole",
            "the integer Y_n leading pole is -Gamma(n)*(2/z)**n/pi; the J_n term is of higher order",
        )
    if power > n:
        return _answer(
            sp.S.Zero,
            "integer_hankel_origin_decay",
            "the multiplier order is higher than the integer Hankel pole order",
        )
    return None


def expint_integer_certificate(expr, x, point, domain):
    """E_n recurrence with a propagated principal-log series remainder."""
    atoms = expr.atoms(sp.expint)
    if not atoms:
        return None
    transformed = expr
    w = sp.Dummy("small_expint")
    for atom in atoms:
        n, arg = atom.args
        if n.is_Integer is not True or n < 1 or n > 20:
            return None
        symbol = sp.Dummy("expint_value")
        dependence = transformed.xreplace({atom: symbol})
        if (
            not dependence.is_polynomial(symbol)
            or sp.Poly(dependence, symbol).degree() > 1
        ):
            return None
        coefficient = sp.expand(dependence).coeff(symbol)
        av = bounded_limit(arg, x, point, allow_general=True)
        if av == 0:
            two_sided = (
                point not in (sp.oo, -sp.oo)
                and not (x.is_positive is True and point == 0)
                and domain is sp.S.true
            )
            if (
                two_sided
                and bounded_limit(arg, x, point, direction="-", allow_general=True) != 0
            ):
                return None
            remainder = bounded_limit(
                coefficient * arg ** (n + 2), x, point, allow_general=True
            )
            if remainder != 0:
                return None
            if (
                two_sided
                and bounded_limit(
                    coefficient * arg ** (n + 2),
                    x,
                    point,
                    direction="-",
                    allow_general=True,
                )
                != 0
            ):
                return None
            expansion = -sp.EulerGamma - sp.log(w) + w - w * w / 4
            for k in range(2, int(n) + 1):
                expansion = (sp.exp(-w) - w * expansion) / (k - 1)
            transformed = transformed.xreplace({atom: expansion.subs(w, arg)})
        elif av is sp.oo:
            coordinate = sp.Dummy("positive_coordinate", positive=True)
            if arg.subs(x, coordinate).is_positive is not True:
                return None
            cv = bounded_limit(coefficient, x, point, allow_general=True)
            if (
                cv is None
                or cv.is_finite is not True
                or cv.has(sp.AccumBounds, sp.Limit)
            ):
                return None
            transformed = transformed.xreplace({atom: sp.S.Zero})
        else:
            return None
    value = bounded_limit(sp.simplify(transformed), x, point, allow_general=True)
    if value is None or value.has(x, sp.nan, sp.zoo, sp.AccumBounds, sp.Limit):
        return None
    if (
        point not in (sp.oo, -sp.oo)
        and not (x.is_positive is True and point == 0)
        and domain is sp.S.true
    ):
        left = bounded_limit(
            sp.simplify(transformed), x, point, direction="-", allow_general=True
        )
        if left is None or sp.simplify(left - value) != 0:
            return None
    return _answer(
        value,
        "integer_expint_recurrence_germ",
        "the principal E1 logarithmic series and the exact integer En recurrence are used only after the multiplied series remainder vanishes",
    )


def primepi_expansion_certificate(expr, x, point):
    """Propagate a finite PNT expansion error through an affine expression."""
    if point is not sp.oo or not expr.has(sp.primepi(x)):
        return None
    y = sp.Dummy("prime_count")
    expression = sp.cancel(expr.xreplace({sp.primepi(x): y}))
    if not expression.is_polynomial(y) or sp.Poly(expression, y).degree() > 1:
        return None
    coefficient = sp.expand(expression).coeff(y)
    # DLMF 27.12.4: with six terms the remainder is O(x/log(x)**7).
    error = bounded_limit(
        coefficient * x / sp.log(x) ** 7, x, sp.oo, allow_general=True
    )
    if error != 0:
        return None
    approximation = sum(sp.factorial(k) * x / sp.log(x) ** (k + 1) for k in range(6))
    value = bounded_limit(
        sp.expand(expression.subs(y, approximation)), x, sp.oo, allow_general=True
    )
    if value is None or value.has(x, sp.nan, sp.zoo, sp.AccumBounds, sp.Limit):
        return None
    return _answer(
        value,
        "prime_number_theorem_remainder",
        "six terms of the prime-counting asymptotic expansion are used only after the multiplied remainder is proved to vanish",
    )


def lambert_positive_tail_certificate(expr, x, point):
    if point is not sp.oo or not expr.has(sp.LambertW(x)):
        return None
    z = sp.Dummy("positive_W", positive=True)
    outer = expr.xreplace({sp.LambertW(x): z})
    if outer.has(x) or any(
        a.func not in (sp.exp, sp.log, sp.sinh, sp.cosh, sp.tanh, sp.atan)
        for a in outer.atoms(sp.Function)
    ):
        return None
    value = bounded_limit(outer, z, sp.oo, allow_general=True)
    if value is None or value.has(z, sp.nan, sp.zoo, sp.AccumBounds, sp.Limit):
        return None
    return _answer(
        value,
        "lambertw_positive_infinity",
        "the principal real W(x) tends monotonically to infinity, so a resolved elementary outer tail composes with it",
    )


def gamma_positive_sum_certificate(expr, x, point, assumptions):
    if point is not sp.oo or not expr.is_Add or not expr.has(sp.gamma):
        return None
    for term in expr.args:
        if term.func is sp.gamma:
            argument = term.args[0]
            if (
                _sign(
                    argument.subs(x, sp.Dummy("positive_x", positive=True)), assumptions
                )
                != 1
            ):
                return None
            value = bounded_limit(argument, x, sp.oo, allow_general=True)
            if value not in (0, sp.oo):
                return None
        else:
            value = bounded_limit(term, x, sp.oo, allow_general=True)
            if value is not sp.oo and (
                value is None or value.is_nonnegative is not True
            ):
                return None
    return _answer(
        sp.oo,
        "positive_gamma_sum_divergence",
        "each term is nonnegative on the real tail and the positive gamma terms diverge at zero or infinity",
    )


def neighboring_sine_certificate(expr, x, point):
    if not expr.is_Add or len(expr.args) != 2:
        return None
    positive = [a for a in expr.args if a.func is sp.sin]
    negative = [-a for a in expr.args if (-a).func is sp.sin]
    if len(positive) != 1 or len(negative) != 1:
        return None
    a, b = positive[0].args[0], negative[0].args[0]
    t = (
        sp.Dummy("real_approach", positive=True)
        if point is sp.oo
        else sp.Dummy("real_approach", real=True)
    )
    if a.subs(x, t).is_real is not True or b.subs(x, t).is_real is not True:
        return None
    delta = bounded_limit(sp.simplify(a - b), x, point, allow_general=True)
    if point not in (sp.oo, -sp.oo) and not (x.is_positive is True and point == 0):
        if (
            bounded_limit(
                sp.simplify(a - b), x, point, direction="-", allow_general=True
            )
            != 0
        ):
            return None
    if delta == 0:
        return _answer(
            sp.S.Zero,
            "sine_uniform_lipschitz",
            "|sin(a)-sin(b)|<=|a-b| for real phases, whose difference tends to zero",
        )
    return None
