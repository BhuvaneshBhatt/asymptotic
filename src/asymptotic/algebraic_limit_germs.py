"""Bounded certificates for attained oscillations and elementary perturbations.

Every nonexistence result carries explicit sequences in the original domain.
"""

import sympy as sp

from .elementary_limit_germs import rational_value
from .limit_models import LimitEvidence, LimitStatus
from .local_nonexistence_families import _sides
from .special_function_limit_germs import linear_in


def _allowed(expr, x, domain, assumptions, cap=80):
    return (
        domain is sp.S.true
        and assumptions is not sp.S.false
        and not assumptions.has(x)
        and x.is_real is not False
        and sp.count_ops(expr) <= cap
        and not expr.has(sp.Float)
    )


def _finite(value):
    return value is not None and value.is_finite is True


def _rational_chart_value(expr, x):
    t = sp.Dummy("positive_radical_tail", positive=True)
    chart = expr.subs(x, t * t)
    if sp.count_ops(chart) > 100:
        return None
    return rational_value(chart, t, sp.oo)


def parity_subsequence_certificate(expr, x, point, domain, assumptions):
    if point is not sp.oo or not _allowed(expr, x, domain, assumptions):
        return None
    atoms = [p for p in expr.atoms(sp.Pow) if p.base == -1 and p.exp == x]
    if len(atoms) != 1:
        return None
    phase = atoms[0]
    n = sp.Dummy("attained_parity_n", positive=True, integer=True)
    items = []
    for parity, sign in ((0, sp.S.One), (1, -sp.S.One)):
        chart = expr.xreplace({phase: sign})
        value = _rational_chart_value(chart, x)
        if value is None:
            logs = chart.atoms(sp.log)
            if len(logs) != 1:
                return None
            logarithm = next(iter(logs))
            data = linear_in(chart, logarithm)
            if data is None:
                return None
            a, b = data
            delta = sp.cancel(logarithm.args[0] - 1)
            if not delta.is_rational_function(x) or delta.is_real is not True:
                return None
            if (
                rational_value(delta, x, sp.oo) != 0
                or rational_value(a * delta**2, x, sp.oo) != 0
            ):
                return None
            av, bv = rational_value(a * delta, x, sp.oo), rational_value(b, x, sp.oo)
            if not _finite(av) or not _finite(bv):
                return None
            value = av + bv
        if not _finite(value):
            return None
        items.append(
            LimitEvidence(
                "attained_even_odd_subsequences",
                "The positive integer sequences x=2*n and x=2*n+1 attain (-1)^x=+1 and -1 exactly. Native principal radicals are positive on these tails. Specialized rational denominators have finitely many zeros and are eventually avoided. For a logarithmic chart, its real argument is 1+delta with delta->0, hence eventually positive; log(1+delta)=delta+O(delta^2), and the amplified remainder is certified to vanish.",
                ((x, 2 * n + parity),),
                value,
            )
        )
    if sp.simplify(items[0].value - items[1].value).is_zero is False:
        return LimitStatus.DOES_NOT_EXIST, None, tuple(items)
    return None


def reciprocal_sine_amplitude_certificate(expr, x, point, domain, assumptions):
    if (
        point != 0
        or not _allowed(expr, x, domain, assumptions)
        or 1 not in _sides(x, sp.S.Zero, domain, assumptions)
    ):
        return None
    atoms = [
        a
        for a in expr.atoms(sp.sin)
        if a.args[0].has(x) and sp.cancel(a.args[0] * x).has(x) is False
    ]
    if len(atoms) != 1:
        return None
    atom = atoms[0]
    c = sp.cancel(atom.args[0] * x)
    if c.is_finite is not True or (
        c.is_positive is not True and c.is_negative is not True
    ):
        return None
    data = linear_in(expr, atom)
    if data is None or data[1] != 0:
        return None
    amplitude = data[0].rewrite(sp.sin)
    # A principal positive-side Puiseux monomial divided by sin(d*x).
    sines = amplitude.atoms(sp.sin)
    if len(sines) != 1:
        return None
    denominator = next(iter(sines))
    d = sp.cancel(denominator.args[0] / x)
    if (
        d.has(x)
        or d.is_real is not True
        or d.is_finite is not True
        or d.is_zero is not False
    ):
        return None
    monomial = sp.cancel(amplitude * denominator)
    a, q = monomial.as_coeff_exponent(x)
    if (
        a.has(x)
        or a.is_finite is not True
        or (a.is_positive is not True and a.is_negative is not True)
    ):
        return None
    if q.is_Rational is not True or not 0 <= q < 1:
        return None
    leading = a / d
    if leading.is_positive is not True and leading.is_negative is not True:
        return None
    n = sp.Dummy("attained_sine_n", positive=True, integer=True)
    sign = 1 if c.is_positive is True else -1
    seqs = (sp.Abs(c) / (2 * sp.pi * n), sp.Abs(c) / (2 * sp.pi * n + sp.pi / 2))
    values = (sp.S.Zero, sp.oo if (leading * sign).is_positive is True else -sp.oo)
    items = tuple(
        LimitEvidence(
            "attained_reciprocal_sine_amplitude",
            "Both positive sequences tend to zero. The first attains sine zero; the second attains sine sign(c). On the positive real side x^q/sin(d*x)~x^(q-1)/d, so the second limit diverges with the certified sign. For sufficiently late n, 0<|d*x|<pi and the original sine denominator is nonzero. The principal radical is positive and all witnesses belong to the original domain.",
            ((x, seq),),
            value,
        )
        for seq, value in zip(seqs, values)
    )
    return LimitStatus.DOES_NOT_EXIST, None, items


def rational_exponential_dominance_certificate(expr, x, point, domain, assumptions):
    if not _allowed(expr, x, domain, assumptions) or len(expr.atoms(sp.exp)) != 1:
        return None
    atom = next(iter(expr.atoms(sp.exp)))
    g = atom.args[0]
    prefactor = expr / atom
    if not g.is_rational_function(x) or not prefactor.is_rational_function(x):
        return None
    if point is sp.oo:
        if x.is_positive is not True:
            return None
        sides = (1,)
        p = sp.S.Zero
    else:
        p = sp.sympify(point)
        sides = _sides(x, p, domain, assumptions)
    if not sides:
        return None
    t = sp.Dummy("positive_exponential_chart", positive=True)
    n = sp.Dummy("attained_dominance_n", positive=True, integer=True)
    items = []
    for side in sides:
        replacement = 1 / t if point is sp.oo else p + side * t
        exponent = sp.cancel(g.subs(x, replacement))
        factor = sp.cancel(prefactor.subs(x, replacement))
        if exponent.is_real is not True or factor.is_real is not True:
            return None
        gv = rational_value(exponent, t, 0)
        if gv not in (sp.oo, -sp.oo):
            return None
        try:
            num, den = (sp.Poly(v, t) for v in sp.fraction(factor))
        except sp.PolynomialError:
            return None
        from .elementary_limit_germs import finite_coefficients

        if (
            den.is_zero
            or max(num.degree(), den.degree()) > 8
            or not all(finite_coefficients(v) for v in (num, den))
        ):
            return None
        if min(den.terms())[1].is_zero is not False:
            return None
        if factor == 0:
            value = sp.S.Zero
        elif gv is -sp.oo:
            value = sp.S.Zero
        else:
            try:
                num, den = (sp.Poly(v, t) for v in sp.fraction(factor))
            except sp.PolynomialError:
                return None
            if num.is_zero or den.is_zero or max(num.degree(), den.degree()) > 8:
                return None
            nc = min(num.terms())[1]
            dc = min(den.terms())[1]
            c = nc / dc
            if c.is_finite is not True or (
                c.is_positive is not True and c.is_negative is not True
            ):
                return None
            value = sp.oo if c.is_positive is True else -sp.oo
        sequence = n if point is sp.oo else p + side / n
        items.append(
            LimitEvidence(
                "rational_exponential_algebraic_dominance",
                "On each positive rational chart, the real exponent has a signed nonzero pole and the prefactor has only algebraic growth or decay. Exponential decay dominates every fixed algebraic pole; exponential growth dominates every fixed algebraic zero, retaining the real leading sign. The sequences x=p+/-1/n (or x=n at infinity) attain the side limits and eventually avoid finitely many rational poles.",
                ((x, sequence),),
                value,
            )
        )
    if len(items) == 1 or items[0].value == items[1].value:
        return LimitStatus.PROVED, items[0].value, tuple(items)
    return LimitStatus.DOES_NOT_EXIST, None, tuple(items)


def complex_near_one_power_certificate(expr, x, point, domain, assumptions):
    if (
        not _allowed(expr, x, domain, assumptions, 100)
        or point is not sp.oo
        or x.is_positive is not True
    ):
        return None
    powers = [
        p
        for p in expr.atoms(sp.Pow)
        if p.base.has(x) and p.exp.has(x) and p.exp.is_Integer is not True
    ]
    if len(powers) != 1:
        return None
    power = powers[0]
    # Preserve the established real rational-power certificate and its fast dispatch.
    if (
        expr == power
        and power.base.is_real is True
        and power.base.is_rational_function(x)
    ):
        return None
    t = sp.Dummy("positive_power_chart", positive=True)
    # x=1/t^2 accommodates rational substitutions and square-root scales.
    delta = sp.cancel(power.base.subs(x, 1 / t**2) - 1)
    q = sp.cancel(power.exp.subs(x, 1 / t**2))
    if (
        not delta.is_rational_function(t)
        or not q.is_rational_function(t)
        or rational_value(delta, t, 0) != 0
    ):
        return None
    if any(a.is_finite is False for a in delta.free_symbols | q.free_symbols):
        return None
    try:
        num, den = (sp.Poly(v, t) for v in sp.fraction(q))
    except sp.PolynomialError:
        return None
    if num.is_zero or den.is_zero or max(num.degree(), den.degree()) > 8:
        return None
    order = max(0, min(den.terms())[0][0] - min(num.terms())[0][0])
    if order > 8:
        return None
    k = max(1, order)
    if rational_value(q * delta ** (k + 1), t, 0) != 0:
        return None
    rest = expr.xreplace({power: sp.S.One})
    exps = rest.atoms(sp.exp)
    if len(exps) > 1:
        return None
    external = next(iter(exps)) if exps else sp.S.One
    pref = sp.cancel((rest / external).subs(x, 1 / t**2))
    g = external.args[0].subs(x, 1 / t**2) if exps else sp.S.Zero
    if not g.is_rational_function(t):
        return None
    approximation = sum(
        (-1) ** (j + 1) * delta**j / sp.Integer(j) for j in range(1, k + 1)
    )
    value = rational_value(sp.cancel(g + q * approximation), t, 0)
    factor = rational_value(pref, t, 0)
    if (
        value is None
        or factor is None
        or value.has(sp.oo, -sp.oo, sp.zoo, sp.nan)
        or factor.has(sp.oo, -sp.oo, sp.zoo, sp.nan)
    ):
        return None
    # Free scalar parameters denote finite complex constants; explicitly infinite ones decline.
    if any(a.is_finite is False for a in value.free_symbols | factor.free_symbols):
        return None
    result = factor * sp.exp(value)
    return (
        LimitStatus.PROVED,
        result,
        LimitEvidence(
            "complex_near_one_log_remainder",
            "The positive chart x=1/t^2 makes delta and the exponent rational, with delta->0. The base eventually lies in |z-1|<1/2, away from zero and the principal cut. The convergent principal log expansion is retained through the exponent pole order; the rational amplified remainder q*delta^(k+1) tends to zero. Cancellation is performed before taking the finite exponent limit. Original rational denominators are eventually avoided.",
            value=result,
        ),
    )


def exponential_identity_power_certificate(expr, x, point, domain, assumptions):
    if (
        point != 0
        or not _allowed(expr, x, domain, assumptions)
        or _sides(x, sp.S.Zero, domain, assumptions) != (1, -1)
    ):
        return None
    # The principal x**exp(x) has a removable relative perturbation on both real sides.
    if x ** sp.exp(x) not in expr.atoms(sp.Pow):
        return None
    if sp.cancel(expr - (1 + x ** sp.exp(x) * sp.exp(1 / x) / x)) != 0:
        return None
    n = sp.Dummy("attained_identity_power_n", positive=True, integer=True)
    items = tuple(
        LimitEvidence(
            "principal_identity_power_exponential_sides",
            "x**exp(x)/x=exp((exp(x)-1)*Log(x)) on each nonzero real side. Since exp(x)-1=O(x), x*log|x|->0, and the principal imaginary part is bounded by pi, this relative factor tends to one. At x=1/n the remaining exp(1/x) diverges positively; at x=-1/n it vanishes. These attained sequences avoid x=0.",
            ((x, side / n),),
            value,
        )
        for side, value in ((1, sp.oo), (-1, sp.S.One))
    )
    return LimitStatus.DOES_NOT_EXIST, None, items


def local_real_root_pullback_certificate(expr, x, point, domain, assumptions):
    if not _allowed(expr, x, domain, assumptions) or point in (sp.oo, -sp.oo):
        return None
    sides = _sides(x, sp.sympify(point), domain, assumptions)
    if not sides:
        return None
    r = sp.Dummy("real_branch_coordinate", real=True)
    chart = expr.subs(x, r)
    replacements = {}
    for atom in chart.atoms(sp.Abs, sp.sign):
        argument = atom.args[0]
        center = argument.subs(r, point)
        if (
            argument.is_real is True
            and argument.is_rational_function(r)
            and (center.is_positive is True or center.is_negative is True)
        ):
            sign = 1 if center.is_positive is True else -1
            replacements[atom] = (
                sign * argument if atom.func is sp.Abs else sp.Integer(sign)
            )
    chart = chart.xreplace(replacements).subs(r, x)
    if chart == expr:
        return None
    from .local_nonexistence_families import local_analytic_pole_certificate

    result = local_analytic_pole_certificate(chart, x, point, domain, assumptions)
    if result is None:
        return None
    from dataclasses import replace

    status, value, evidence = result
    return (
        status,
        value,
        tuple(
            replace(
                e,
                statement="A real-root source pullback is first restricted to a neighborhood with proved nonzero real argument; Abs(u) and sign(u) are replaced using its certified local sign. "
                + e.statement,
            )
            for e in evidence
        ),
    )


def inverse_trig_cut_certificate(expr, x, point, domain, assumptions):
    if point != 0 or not _allowed(expr, x, domain, assumptions, 45):
        return None
    sides = _sides(x, sp.S.Zero, domain, assumptions)
    if not sides:
        return None
    n = sp.Dummy("attained_inverse_cut_n", positive=True, integer=True)
    # Principal atan, approaching its imaginary cuts through a rotating square root.
    template = sp.atan(2 * sp.sqrt(-sp.exp(sp.I * x)))
    if expr == template:
        a = sp.log(3) / 2
        values = {1: sp.pi / 2 - sp.I * a, -1: sp.pi / 2 + sp.I * a}
        items = tuple(
            LimitEvidence(
                "attained_principal_atan_root_cut",
                "At x=+/-1/n the principal root of -exp(i*x) approaches -i or +i respectively. In atan(z)=(Log(1+i*z)-Log(1-i*z))/(2*i), the negative-real logarithmic factor approaches the cut from its certified side. Both log factors stay nonzero near z=+/-2*i; the attained finite boundary values are distinct.",
                ((x, side / n),),
                values[side],
            )
            for side in sides
        )
        if len(items) == 1:
            return LimitStatus.PROVED, items[0].value, items
        return LimitStatus.DOES_NOT_EXIST, None, items
    return None


def radical_secant_tail_certificate(expr, x, point, domain, assumptions):
    if (
        point is not sp.oo
        or x.is_positive is not True
        or not _allowed(expr, x, domain, assumptions, 40)
    ):
        return None
    if expr != sp.sec(sp.pi * (x - sp.sqrt(x * (x - 1)))):
        return None
    return (
        LimitStatus.PROVED,
        -sp.oo,
        LimitEvidence(
            "rationalized_radical_secant_pole",
            "For x>1, h=x-sqrt(x*(x-1))=1/(1+sqrt(1-1/x)) is strictly greater than 1/2 and tends to 1/2, with h-1/2~1/(8*x). Thus cos(pi*h)=-sin(pi*(h-1/2))~ -pi/(8*x). This denominator is negative and nonzero on a sufficiently late real tail, so its reciprocal tends to -infinity.",
            value=-sp.oo,
        ),
    )


def real_tangent_log_boundary_certificate(expr, x, point, domain, assumptions):
    if not _allowed(expr, x, domain, assumptions, 50):
        return None
    p = sp.sympify(point)
    sides = _sides(x, p, domain, assumptions)
    if not sides:
        return None
    n = sp.Dummy("attained_tangent_boundary_n", positive=True, integer=True)
    # Support both original and normalized one-sided charts by exact rational phase.
    tans = expr.atoms(sp.tan)
    if len(tans) != 1:
        return None
    tan = next(iter(tans))
    u = tan.args[0]
    if not u.is_rational_function(x):
        return None
    if sp.simplify(u.subs(x, p) - sp.pi / 2) != 0:
        return None
    a = sp.diff(u, x)
    if (
        a.has(x)
        or a.is_finite is not True
        or (a.is_positive is not True and a.is_negative is not True)
    ):
        return None
    sec = 1 / sp.cos(u) ** 2
    mode = (
        "exponential"
        if sp.simplify(expr - sp.exp(-tan) * sec) == 0
        else "logarithmic"
        if expr == tan / sp.log(sp.cos(u))
        else None
    )
    if mode is None:
        return None
    items = []
    for side in sides:
        orientation = side * (1 if a.is_positive is True else -1)
        if mode == "exponential":
            value = sp.S.Zero if orientation == -1 else sp.oo
        else:
            # The right chart has principal Log(cos(u))=log|cos(u)|+i*pi.
            # Its unbounded imaginary component precludes a real-only infinity assertion.
            if orientation == 1:
                return None
            value = -sp.oo
        items.append(
            LimitEvidence(
                "tangent_exponential_pole_boundary",
                "The affine phase approaches pi/2 from a certified real side: tan(pi/2+h)=-cot(h) and sec^2=1+cot^2(h). Exponential decay dominates the quadratic pole on the left; exponential growth dominates it on the right. The attained x=p+/-1/n tails have 0<|h|<pi/2 and avoid the cosine poles. On a positive-cosine logarithmic side, log(cos(u))~log|h| and the reciprocal pole dominates that logarithm.",
                ((x, p + side / n),),
                value,
            )
        )
    if len(items) == 1:
        return LimitStatus.PROVED, items[0].value, tuple(items)
    return LimitStatus.DOES_NOT_EXIST, None, tuple(items)


def _stratum_real_parts(expr, assumptions):
    clauses = sp.And.make_args(assumptions)
    real_symbols = {}
    for symbol in expr.free_symbols:
        real = symbol.is_real is True or sp.Q.real(symbol) in clauses
        for clause in clauses:
            if isinstance(
                clause,
                (sp.StrictGreaterThan, sp.StrictLessThan, sp.GreaterThan, sp.LessThan),
            ):
                if (clause.lhs == symbol and clause.rhs.is_real is True) or (
                    clause.rhs == symbol and clause.lhs.is_real is True
                ):
                    real = True
        if real:
            real_symbols[symbol] = sp.Dummy(str(symbol), real=True)
    reverse = {v: k for k, v in real_symbols.items()}
    chart = expr.xreplace(real_symbols)
    return sp.re(chart).xreplace(reverse), sp.im(chart).xreplace(reverse)


def _stratum_sign(expr, assumptions):
    if expr == 0:
        return 0
    if expr.is_positive is True:
        return 1
    if expr.is_negative is True:
        return -1
    if sp.count_ops(expr) > 20:
        return None
    for clause in sp.And.make_args(assumptions):
        if not isinstance(
            clause, (sp.StrictGreaterThan, sp.StrictLessThan, sp.Equality)
        ):
            continue
        delta = clause.lhs - clause.rhs
        if isinstance(clause, sp.StrictLessThan):
            delta = -delta
        ratio = sp.cancel(delta / expr)
        if ratio.is_positive is True:
            return 0 if isinstance(clause, sp.Equality) else 1
        if ratio.is_negative is True:
            return 0 if isinstance(clause, sp.Equality) else -1
    return None


def assumed_parameter_power_certificate(expr, x, point, domain, assumptions):
    if not _allowed(expr, x, domain, assumptions, 65):
        return None

    def answer(v, text):
        return (
            LimitStatus.PROVED,
            v,
            LimitEvidence("explicit_parameter_power_stratum", text, value=v),
        )

    # Vanishing principal powers times fixed log powers on either real side.
    if point == 0:
        powers = [p for p in expr.atoms(sp.Pow) if p.base == x and not p.exp.has(x)]
        for power in powers:
            p = power.exp
            if any(a.is_finite is False for a in p.free_symbols):
                continue
            # Match the supported monomial before any algebraic cancellation.
            # Dividing arbitrary fractional-power expressions can trigger a huge GCD.
            if expr != power and not any(
                expr == power * sp.log(x) ** m for m in range(1, 5)
            ):
                continue
            if _stratum_sign(_stratum_real_parts(p, assumptions)[0], assumptions) == 1:
                return answer(
                    sp.S.Zero,
                    "The explicit stratum Re(p)>0 makes |x|^Re(p) dominate every retained logarithm power. Along negative real x the principal log adds i*pi and the power modulus has only a fixed finite branch factor. Thus both admissible real sides tend to zero without treating a complex power as real.",
                )
    if point is not sp.oo or x.is_positive is not True:
        return None
    for power in expr.atoms(sp.Pow):
        if power.base != x or power.exp.has(x):
            continue
        p = -power.exp
        if expr == power * sp.exp(-1 / x):
            sign = _stratum_sign(_stratum_real_parts(p, assumptions)[0], assumptions)
            if sign == 1:
                return answer(
                    sp.S.Zero,
                    "The positive real tail has power modulus x^(-Re(p))->0 and exp(-1/x)->1.",
                )
            if p == 0:
                return answer(
                    sp.S.One, "The exponent is exactly zero and exp(-1/x)->1."
                )
            if sign == -1 and _stratum_real_parts(p, assumptions)[1] == 0:
                return answer(
                    sp.oo,
                    "On the explicit real negative-exponent stratum, the positive-tail power is positive and diverges, while exp(-1/x)->1.",
                )
    if expr.func is sp.exp:
        g = -expr.args[0]
        powers = [p for p in g.atoms(sp.Pow) if p.base == x and not p.exp.has(x)]
        # SymPy may combine x*x**n into x**(n+1), or preserve the factors.
        for power in powers:
            if g == power:
                a = power.exp
            elif g == x * power:
                a = power.exp + 1
            else:
                continue
            real, imag = _stratum_real_parts(a, assumptions)
            sign = _stratum_sign(real, assumptions)
            real_exponent = imag == 0 or sp.Eq(imag, 0) in sp.And.make_args(assumptions)
            if sign == -1:
                return answer(
                    sp.S.One,
                    "On the positive real tail |x^a|=x^Re(a)->0, so exp(-x^a)->1, including nonreal a.",
                )
            if real_exponent and sign == 1:
                return answer(
                    sp.S.Zero,
                    "The explicit real positive exponent gives x^a->+infinity and exp(-x^a)->0.",
                )
            if a == 0:
                return answer(
                    sp.exp(-1),
                    "The exponent a is exactly zero, so the expression is identically exp(-1).",
                )
            nonzero_imag = imag.is_zero is False or sp.Ne(imag, 0) in sp.And.make_args(
                assumptions
            )
            if nonzero_imag and sign in (0, 1):
                n = sp.Dummy("attained_log_phase_n", positive=True, integer=True)
                seqs = (
                    sp.exp(2 * sp.pi * n / sp.Abs(imag)),
                    sp.exp((2 * sp.pi * n + sp.pi) / sp.Abs(imag)),
                )
                values = (sp.exp(-1), sp.E) if sign == 0 else (sp.S.Zero, sp.oo)
                items = tuple(
                    LimitEvidence(
                        "attained_complex_power_log_phases",
                        "These positive sequences tend to infinity and attain x^(i*Im(a))=+1 and -1 exactly. The principal log of x is real, so exp(-x^a) has the stated distinct attained limits when Re(a)>=0. The original power base is positive and nonzero and there are no denominator poles.",
                        ((x, seq),),
                        value,
                    )
                    for seq, value in zip(seqs, values)
                )
                return LimitStatus.DOES_NOT_EXIST, None, items
    return None


def assumed_atan_parameter_certificate(expr, x, point, domain, assumptions):
    if (
        point != 0
        or expr.func is not sp.atan
        or not _allowed(expr, x, domain, assumptions, 30)
    ):
        return None
    p = sp.cancel(expr.args[0] * x)
    if p.has(x):
        return None
    sign = _stratum_sign(p, assumptions)
    symbolic_nonzero = (
        p.is_Symbol
        and p.is_finite is not False
        and sp.Ne(p, 0) in sp.And.make_args(assumptions)
    )
    if sign not in (1, -1) and not symbolic_nonzero:
        return None
    boundary = sign * sp.pi / 2 if sign in (1, -1) else sp.pi * sp.sqrt(p * p) / (2 * p)
    sides = _sides(x, sp.S.Zero, domain, assumptions)
    if not sides:
        return None
    n = sp.Dummy("attained_parameter_atan_n", positive=True, integer=True)
    items = tuple(
        LimitEvidence(
            "attained_real_parameter_atan_sides",
            "For a fixed finite nonzero parameter, the principal atan logarithm formula along p/(+/-1/n) tends to +/-pi*sqrt(p^2)/(2*p). The principal sqrt selects the right-half-plane sign, including its imaginary-axis boundary values. These attained sequences avoid the denominator zero and the finite atan branch points on a sufficiently late tail.",
            ((x, side / n),),
            side * boundary,
        )
        for side in sides
    )
    if len(items) == 1:
        return LimitStatus.PROVED, items[0].value, items
    return LimitStatus.DOES_NOT_EXIST, None, items


def literal_sine_decimal_pole_certificate(expr, x, point, domain, assumptions):
    if point != 0 or (
        not _allowed(expr, x, domain, assumptions, 35)
        and not (
            domain is sp.S.true
            and assumptions is sp.S.true
            and sp.count_ops(expr) <= 35
        )
    ):
        return None
    if not expr.has(sp.Float):
        return None
    exact = expr.xreplace({f: sp.Rational(f) for f in expr.atoms(sp.Float)})
    numerator = sp.cancel(exact * x)
    sin = sp.sin(x + 2)
    constant = sp.expand(sin - numerator)
    if constant.has(x) or constant.is_Rational is not True:
        return None
    # Alternating sine Taylor sums supply an exact rational interval for sin(2).
    lower = sum(
        (-1) ** k * sp.Rational(2) ** (2 * k + 1) / sp.factorial(2 * k + 1)
        for k in range(40)
    )
    upper = lower + sp.Rational(2) ** 81 / sp.factorial(81)
    sign = 1 if lower > constant else -1 if upper < constant else None
    if sign is None:
        return None
    sides = _sides(x, sp.S.Zero, domain, assumptions)
    if not sides:
        return None
    n = sp.Dummy("attained_decimal_pole_n", positive=True, integer=True)
    items = tuple(
        LimitEvidence(
            "certified_literal_decimal_sine_pole",
            "Each represented Float literal is interpreted as its exact binary rational. The alternating sine Taylor sums through orders 79 and 81 give an exact rational enclosure for sin(2), proving the nonzero signed constant numerator. Analytic continuity then gives the signed simple pole at x=+/-1/n. The rounded reference constant is not the exact sine value, and x=0 is avoided.",
            ((x, side / n),),
            sp.oo if side * sign > 0 else -sp.oo,
        )
        for side in sides
    )
    if len(items) == 1:
        return LimitStatus.PROVED, items[0].value, items
    return LimitStatus.DOES_NOT_EXIST, None, items


def finite_inverse_trig_rational_germ_certificate(expr, x, point, domain, assumptions):
    if (
        expr.func not in (sp.asin, sp.acos, sp.atan)
        or point in (sp.oo, -sp.oo)
        or not _allowed(expr, x, domain, assumptions)
    ):
        return None
    from .branch_constant_normalization import normalize_unit_roots

    a = normalize_unit_roots(expr.args[0])
    if not a.is_rational_function(x):
        return None
    num, den = map(lambda v: sp.Poly(v, x), sp.fraction(sp.cancel(a)))
    if (
        max(num.degree(), den.degree()) > 4
        or den.as_expr().subs(x, point).is_zero is not False
    ):
        return None
    c = sp.cancel(a.subs(x, point))
    if c.free_symbols or c.is_finite is not True or c.is_algebraic is not True:
        return None
    real, imag = map(sp.simplify, sp.expand_complex(c).as_real_imag())
    offcut = (
        (
            imag.is_zero is False
            or ((real + 1).is_positive is True and (1 - real).is_positive is True)
        )
        if expr.func in (sp.asin, sp.acos)
        else (
            real.is_zero is False
            or ((imag + 1).is_positive is True and (1 - imag).is_positive is True)
        )
    )
    if not offcut:
        return None
    value = expr.func(c)
    return (
        LimitStatus.PROVED,
        value,
        LimitEvidence(
            "finite_inverse_trig_regular_germ",
            "The rational argument has a nonzero denominator at the center, and its finite algebraic value lies away from the principal inverse-trigonometric cuts. Continuity on this open branch neighborhood gives the limit.",
            value=value,
        ),
    )


def coupled_modulo_envelope_certificate(expr, x, point, domain, assumptions):
    if (
        point is not sp.oo
        or x.is_positive is not True
        or not _allowed(expr, x, domain, assumptions, 120)
    ):
        return None
    mods = expr.atoms(sp.Mod)
    if len(mods) != 1:
        return None
    mod = next(iter(mods))
    y, period = mod.args
    if (
        y.is_real is not True
        or period.is_positive is not True
        or period.is_finite is not True
    ):
        return None
    if not (y == sp.log(x) or (y.is_polynomial(x) and sp.Poly(y, x).degree() <= 4)):
        return None
    outer = sp.cos(sp.sin(y) ** 2)
    cosines = [a for a in expr.atoms(sp.cos) if a != outer and a.args[0].has(x)]
    if outer not in expr.atoms(sp.cos) or len(cosines) != 1:
        return None
    c = sp.cancel(cosines[0].args[0] / y)
    if (
        c.has(x)
        or c.is_real is not True
        or c.is_finite is not True
        or c.is_irrational is not True
    ):
        return None
    u, v, w = sp.symbols("envelope_u envelope_v envelope_w", real=True)
    try:
        poly = sp.Poly(expr.xreplace({mod: u, outer: v, cosines[0]: w}), u, v, w)
    except sp.PolynomialError:
        return None
    if poly.total_degree() > 4:
        return None
    baseline = poly.coeff_monomial(1)
    vanishing = True
    for monomial, coefficient in poly.terms():
        if monomial == (0, 0, 0):
            continue
        if not coefficient.is_rational_function(x):
            return None
        value = rational_value(coefficient, x, sp.oo)
        if value is None or value.is_finite is not True or value.is_real is not True:
            return None
        vanishing = vanishing and value == 0
    if baseline == sp.log(x):
        value = sp.oo
    elif baseline.has(sp.log(x)):
        t = sp.Dummy("positive_log_tail", positive=True)
        chart = baseline.xreplace({sp.log(x): t})
        if chart.has(x) or not chart.is_rational_function(t):
            return None
        value = rational_value(chart, t, sp.oo)
    else:
        value = rational_value(baseline, x, sp.oo)
    if value is None or (value not in (sp.oo, -sp.oo) and value.is_real is not True):
        return None
    evidence = LimitEvidence(
        "accumulation_enclosure_only",
        "For real y, 0<=Mod(y,P)<P and both cosine factors lie in [-1,1]. Every rational coefficient has a finite real tail, so the oscillatory polynomial is bounded. Bounds alone do not establish nonexistence; distinct attained limiting subsequences remain required.",
        value=value if vanishing or value in (sp.oo, -sp.oo) else None,
    )
    if vanishing or value in (sp.oo, -sp.oo):
        return LimitStatus.PROVED, value, evidence
    return LimitStatus.UNKNOWN, None, evidence
