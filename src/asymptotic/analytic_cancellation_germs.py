"""Finite analytic jets with explicit cancellation and endpoint checks."""

from functools import lru_cache

import sympy as sp

from .compact_limit_germs import answer


def analytic_jet(expr, variable, order=8):
    """Return a bounded Taylor jet at zero, or decline an unverified germ.

    The accepted functions are analytic on a neighborhood of their checked
    centers. Truncated polynomial arithmetic avoids repeatedly expanding the
    original quotient or asking a general series engine to discover its scale.
    Every accepted jet has remainder O(variable**(order + 1)).
    """
    zero = (sp.S.Zero,) * (order + 1)

    def add(a, b):
        return tuple(sp.expand(x + y) for x, y in zip(a, b, strict=True))

    def scale(a, c):
        return tuple(sp.expand(v * c) for v in a)

    def multiply(a, b):
        return tuple(
            sp.expand(sum(a[j] * b[k - j] for j in range(k + 1)))
            for k in range(order + 1)
        )

    def compose(delta, coefficients):
        result, power = zero, (sp.S.One,) + zero[1:]
        for c in coefficients:
            result = add(result, scale(power, c))
            power = multiply(power, delta)
        return result

    @lru_cache(None)
    def jet(e):
        if not e.has(variable):
            return (e,) + zero[1:] if e.is_finite is True else None
        if e == variable:
            return (sp.S.Zero, sp.S.One) + zero[2:]
        if e.is_Add or e.is_Mul:
            result = zero if e.is_Add else (sp.S.One,) + zero[1:]
            for arg in e.args:
                value = jet(arg)
                if value is None:
                    return None
                result = add(result, value) if e.is_Add else multiply(result, value)
            return result
        if e.is_Pow and e.exp.is_Integer and abs(e.exp) <= 8:
            base = jet(e.base)
            if base is None:
                return None
            n = int(e.exp)
            if n < 0:
                if base[0].is_zero is not False:
                    return None
                delta = scale(base, 1 / base[0])
                delta = (sp.S.Zero,) + delta[1:]
                return scale(
                    compose(delta, [sp.binomial(n, k) for k in range(order + 1)]),
                    base[0] ** n,
                )
            result = (sp.S.One,) + zero[1:]
            for _ in range(n):
                result = multiply(result, base)
            return result
        if e.func not in (sp.gamma, sp.polygamma, sp.exp, sp.log, sp.sin, sp.cos):
            return None
        argument = e.args[-1]
        value = jet(argument)
        if value is None:
            return None
        center, delta = value[0], (sp.S.Zero,) + value[1:]
        if e.func in (sp.gamma, sp.polygamma):
            if center != 1:
                return None
            if e.func is sp.polygamma:
                m = e.args[0]
                if not m.is_Integer or not 0 <= m <= 3:
                    return None
                coefficients = [
                    -sp.EulerGamma
                    if m + k == 0
                    else (-1) ** (m + k + 1)
                    * sp.factorial(m + k)
                    * sp.zeta(m + k + 1)
                    / sp.factorial(k)
                    for k in range(order + 1)
                ]
                return compose(delta, coefficients)
            logs = [-sp.EulerGamma] + [
                (-1) ** k * sp.zeta(k) / k for k in range(2, order + 1)
            ]
            coefficients = [sp.S.One]
            for n in range(1, order + 1):
                coefficients.append(
                    sp.expand(
                        sum(
                            k * logs[k - 1] * coefficients[n - k]
                            for k in range(1, n + 1)
                        )
                        / n
                    )
                )
            return compose(delta, coefficients)
        if e.func is sp.log:
            if center.is_positive is not True:
                return None
            return compose(
                delta,
                [sp.log(center)]
                + [(-1) ** (k + 1) / (k * center**k) for k in range(1, order + 1)],
            )
        if e.func is sp.exp:
            return compose(
                delta, [sp.exp(center) / sp.factorial(k) for k in range(order + 1)]
            )
        if center != 0:
            return None
        coefficients = [
            (sp.sin(k * sp.pi / 2) if e.func is sp.sin else sp.cos(k * sp.pi / 2))
            / sp.factorial(k)
            for k in range(order + 1)
        ]
        return compose(delta, coefficients)

    if order < 1 or order > 8 or sp.count_ops(expr) > 220:
        return None
    if expr.free_symbols - {variable}:
        return None
    return jet(expr)


def _leading(jet):
    if jet is None:
        return None
    for k, coefficient in enumerate(jet):
        if coefficient == 0:
            continue
        if coefficient.is_zero is not False:
            return None
        return k, coefficient
    return None


def gamma_analytic_cancellation_certificate(expr, x, point):
    """Certify finite gamma cancellations before generic growth analysis."""
    if point != 0 or not expr.has(sp.gamma, sp.polygamma):
        return None
    if sp.count_ops(expr) > 220 or expr.free_symbols - {x}:
        return None
    numerator, denominator = sp.fraction(sp.together(expr))
    njet, djet = analytic_jet(numerator, x), analytic_jet(denominator, x)
    d = _leading(djet)
    if d is None:
        return None
    n = _leading(njet)
    if n is None:
        if njet is None or any(c != 0 for c in njet) or d[0] >= 8:
            return None
        value = sp.S.Zero
    elif n[0] > d[0]:
        value = sp.S.Zero
    elif n[0] == d[0]:
        value = sp.cancel(n[1] / d[1])
    else:
        return None
    return answer(
        value,
        "gamma_analytic_cancellation_jet",
        "Gamma and fixed-order polygamma are analytic at argument one. "
        "Their Taylor coefficients through order eight are combined by "
        "truncated polynomial arithmetic with O(x**9) remainder. The "
        "denominator has a certified nonzero leading coefficient, so it "
        "has no other sufficiently near zeros. Retained orders determine "
        "the quotient limit without discarding amplified remainders.",
    )


def acos_endpoint_ratio_certificate(expr, x, point):
    """Use the principal acos square-root germ on verified real endpoints."""
    if (
        x.is_real is False
        or point != 0
        or not expr.has(sp.acos)
        or sp.count_ops(expr) > 90
    ):
        return None
    numerator, denominator = sp.fraction(expr)
    top, bottom = numerator.atoms(sp.acos), denominator.atoms(sp.acos)
    if len(top) != 1 or len(bottom) != 1:
        return None
    a, b = next(iter(top)), next(iter(bottom))
    coefficient = sp.cancel(expr * b / a)
    if coefficient.has(x) or coefficient.is_finite is not True:
        return None
    leads = [_leading(analytic_jet(1 - atom.args[0], x)) for atom in (a, b)]
    if any(
        v is None or v[0] == 0 or v[0] % 2 or v[1].is_positive is not True
        for v in leads
    ):
        return None
    n, d = leads
    if n[0] < d[0]:
        return None
    value = sp.S.Zero if n[0] > d[0] else coefficient * sp.sqrt(n[1] / d[1])
    return answer(
        value,
        "acos_real_endpoint_ratio",
        "Both real analytic arguments approach one from below with "
        "positive even-order deficits. Principal acos(1-u)="
        "sqrt(2*u)*(1+O(u)); the square roots share their real side "
        "factor and the denominator is nonzero on the punctured germ.",
    )


def complex_logarithmic_real_pole_certificate(expr, x, point, domain, assumptions):
    """Retain the bounded principal phase that determines a real pole's sign."""
    from .local_nonexistence_families import _sides

    if expr.func is not sp.re or not expr.has(sp.log, sp.cos):
        return None
    if sp.count_ops(expr) > 60 or not _sides(x, sp.sympify(point), domain, assumptions):
        return None
    logs, cosines = expr.atoms(sp.log), expr.atoms(sp.cos)
    if len(logs) != 1 or len(cosines) != 1:
        return None
    logarithm, cosine = next(iter(logs)), next(iter(cosines))
    phase = cosine.args[0]
    rate = sp.diff(phase, x)
    if (
        phase.subs(x, point) != 0
        or rate.has(x)
        or rate.is_real is not True
        or rate.is_zero is not False
        or rate.is_finite is not True
    ):
        return None
    if logarithm.args[0] != (cosine - 1) * sp.exp(-sp.I * phase):
        return None
    coefficient = sp.cancel(expr.args[0] * (sp.exp(sp.I * phase) - 1) * logarithm)
    if coefficient.has(x) or coefficient.is_finite is not True:
        return None
    if coefficient.is_positive is True:
        value = -sp.oo
    elif coefficient.is_negative is True:
        value = sp.oo
    else:
        return None
    return answer(
        value,
        "principal_log_real_pole_phase",
        "On both real sides, u is a transverse local coordinate. "
        "Write exp(I*u)-1=a+I*b with a=-u**2/2+O(u**4), "
        "b=u+O(u**3). The principal logarithm of "
        "(cos(u)-1)*exp(-I*u) has real part 2*log(Abs(u))-log(2)+O(u**2) "
        "and imaginary part sign(u)*pi-u. The real reciprocal "
        "is (a*L-b*theta)/((a**2+b**2)*(L**2+theta**2)), "
        "whose dominant term is -pi/(4*Abs(u)*log(Abs(u))**2). "
        "Both original factors are nonzero on a sufficiently small "
        "punctured real neighborhood.",
    )


def logarithmic_power_sum_certificate(expr, x, point):
    """Read the common logarithmic scale of finite real-power sums."""
    if (
        x.is_real is False
        or point != 0
        or sp.count_ops(expr) > 70
        or not expr.has(sp.log(x))
    ):
        return None
    numerator, denominator = sp.fraction(expr)
    if denominator != sp.log(x) or numerator.func is not sp.log:
        return None

    def leading_power(value):
        terms = {}
        for term in sp.Add.make_args(sp.expand_mul(value)):
            power = term.as_powers_dict().get(x, sp.S.Zero)
            if (
                power.is_number is not True
                or power.is_real is not True
                or power.is_finite is not True
            ):
                return None
            coefficient = sp.cancel(term / x**power)
            if coefficient.has(x) or coefficient.is_finite is not True:
                return None
            terms[power] = terms.get(power, sp.S.Zero) + coefficient
        terms = {k: sp.expand(v) for k, v in terms.items() if sp.expand(v) != 0}
        if not terms:
            return None
        powers = sorted(terms)
        coefficient = terms[powers[0]]
        if coefficient.is_zero is not False:
            return None
        return powers[0]

    n, d = sp.fraction(sp.together(numerator.args[0]))
    a, b = leading_power(n), leading_power(d)
    if a is None or b is None:
        return None
    return answer(
        a - b,
        "finite_power_sum_log_scale",
        "A finite sum of fixed finite real powers has a nonzero "
        "lowest coefficient and a positive gap to every higher power. "
        "On either real side its modulus is a nonzero constant times "
        "Abs(x)**order*(1+o(1)); principal branch phases stay bounded. "
        "Both sums are eventually nonzero, so log(n/d)/log(x) tends "
        "to the difference of their orders.",
    )


def argument_step_germ_certificate(expr, x, point, domain, assumptions):
    """Resolve nested argument steps from attained real-side sign germs."""
    from .local_nonexistence_families import _sides

    if not expr.has(sp.Heaviside, sp.arg) or sp.count_ops(expr) > 55:
        return None
    sides = _sides(x, sp.sympify(point), domain, assumptions)
    if not sides:
        return None
    t = sp.Dummy("argument_step_side", positive=True)
    values = []
    for side in sides:
        local = expr.subs(x, point + side * t)
        for atom in sorted(local.atoms(sp.Heaviside), key=sp.count_ops):
            current = atom.xreplace({})
            if current.func is not sp.Heaviside or current.args[0].func is not sp.arg:
                return None
            inner = current.args[0].args[0]
            if inner.has(sp.Heaviside, sp.arg) or sp.count_ops(inner) > 40:
                return None
            inner = inner.replace(
                lambda node: node.func is sp.atanh and not node.has(t),
                lambda node: node.rewrite(sp.log),
            )
            imaginary = sp.expand_complex(inner).as_real_imag()[1]
            lead = _leading(analytic_jet(imaginary, t, 4))
            if lead is None:
                if imaginary != 0:
                    return None
                real = sp.expand_complex(inner).as_real_imag()[0]
                lead = _leading(analytic_jet(real, t, 4))
                if lead is None:
                    return None
                if lead[1].is_positive is True:
                    value = current.args[1]
                elif lead[1].is_negative is True:
                    value = sp.S.One
                else:
                    return None
            elif lead[1].is_positive is True:
                value = sp.S.One
            elif lead[1].is_negative is True:
                value = sp.S.Zero
            else:
                return None
            local = local.xreplace({atom: value})
            # Rebuild the list after each inner replacement, so an enclosing
            # step is analyzed with the attained inner value, not its old node.
            if local.has(sp.Heaviside):
                remainder = argument_step_germ_certificate(
                    local, t, sp.S.Zero, sp.S.true, sp.S.true
                )
                if remainder is None:
                    return None
                local = remainder[1]
                break
        if (
            local.has(t)
            or local.has(sp.Heaviside, sp.arg)
            or local.is_finite is not True
        ):
            return None
        values.append(local)
    if any(v != values[0] for v in values[1:]):
        return None
    return answer(
        values[0],
        "attained_argument_step_germ",
        "On each admitted real side a checked analytic real or "
        "imaginary component has a nonzero leading coefficient. "
        "Its eventual sign fixes the principal argument's half-plane "
        "and hence the step value. Nested steps are resolved from "
        "inside outward; the argument never vanishes on that germ.",
    )


def vanishing_elliptic_amplitude_certificate(expr, x, point):
    """Bound incomplete elliptic terms with vanishing amplitudes separately."""
    from ._symbolic_policy import bounded_limit

    atoms = expr.atoms(sp.elliptic_f, sp.elliptic_e, sp.elliptic_pi)
    if point is not sp.oo or not atoms or len(atoms) > 3 or sp.count_ops(expr) > 200:
        return None
    replacements = {a: sp.Dummy("elliptic_amplitude") for a in atoms}
    reduced = expr.xreplace(replacements)
    try:
        polynomial = sp.Poly(reduced, *replacements.values())
    except sp.PolynomialError:
        return None
    if polynomial.total_degree() > 1:
        return None
    for atom, marker in replacements.items():
        if (atom.func is sp.elliptic_pi and len(atom.args) != 3) or (
            atom.func in (sp.elliptic_f, sp.elliptic_e) and len(atom.args) != 2
        ):
            return None
        amplitude = atom.args[0] if len(atom.args) == 2 else atom.args[1]
        parameters = (
            atom.args[1:] if len(atom.args) == 2 else (atom.args[0], atom.args[2])
        )
        if any(p.has(x) or p.is_finite is not True for p in parameters):
            return None
        if bounded_limit(amplitude, x, point) != 0:
            return None
        coefficient = polynomial.coeff_monomial(marker)
        value = bounded_limit(coefficient, x, point)
        if value is None or value.is_finite is not True:
            return None
    remainder = polynomial.coeff_monomial(1)
    value = bounded_limit(remainder, x, point)
    if value is None or value.is_finite is not True:
        return None
    return answer(
        value,
        "vanishing_elliptic_amplitude",
        "For fixed finite parameters, incomplete F, E and Pi are "
        "holomorphic at amplitude zero with value zero: their "
        "integrands have nonzero denominators at that center. Each "
        "amplitude tends to zero and its multiplier has a separately "
        "certified finite limit, so every elliptic term vanishes. "
        "The remaining expression is evaluated without expanding "
        "the original coupled special-function quotient.",
    )


def inverse_sech_projective_pole_certificate(expr, x, point, domain, assumptions):
    """Prove norm divergence without expanding an oscillatory inverse-sech phase."""
    from .local_nonexistence_families import _sides
    from .local_tail_germs import rational_order

    if point != 0 or not expr.has(sp.asech, sp.exp) or sp.count_ops(expr) > 45:
        return None
    if not _sides(x, sp.S.Zero, domain, assumptions):
        return None
    exponentials, inverse = expr.atoms(sp.exp), expr.atoms(sp.asech)
    if len(exponentials) != 1 or len(inverse) != 1:
        return None
    exponential, inverse = next(iter(exponentials)), next(iter(inverse))
    if exponential.args[0] != -sp.sqrt(1 - inverse**2):
        return None
    rate = sp.cancel(inverse.args[0] / x)
    if (
        rate.has(x)
        or rate.is_real is not True
        or rate.is_finite is not True
        or rate.is_zero is not False
    ):
        return None
    coefficient = sp.cancel(expr / exponential)
    order = rational_order(coefficient, x)
    if order is None or order[0] >= 0:
        return None
    return answer(
        sp.zoo,
        "inverse_sech_bounded_modulus_pole",
        "On either sufficiently small real side, principal asech(a*x) "
        "has real part L=acosh(1/Abs(a*x))->infinity and imaginary "
        "part theta with Abs(theta)<=pi. If sqrt(1-(L+I*theta)**2) "
        "has real part u>=0, squaring gives "
        "u**2=(sqrt((L**2-theta**2-1)**2+4*L**2*theta**2) "
        "-(L**2-theta**2-1))/2. For large L this bounds u<=pi+1. "
        "The exponential modulus is therefore bounded below by "
        "exp(-pi-1). The original rational multiplier has a nonzero "
        "pole coefficient and diverging modulus, proving complex "
        "projective infinity. Its denominator is eventually nonzero.",
    )
