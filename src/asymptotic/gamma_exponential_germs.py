"""Reusable gamma germs and positive-tail envelopes.

Only controlled relative errors in products, or absolute bounds in sums, are
used. Bounds here never establish oscillatory nonexistence.
"""

import sympy as sp

from ._symbolic_policy import bounded_limit
from .compact_limit_germs import answer, gamma_stirling_product_certificate
from .limit_models import LimitEvidence, LimitStatus


def parameter_chart(expr, x, assumptions):
    """Exact bijective coordinates for fixed real parameters with lower bounds."""
    replacements = {}
    for clause in sp.And.make_args(sp.sympify(assumptions)):
        if not isinstance(clause, sp.StrictGreaterThan):
            continue
        a, b = clause.lhs, clause.rhs
        if (
            a.is_Symbol
            and a != x
            and not b.free_symbols
            and b.is_real is True
            and b.is_finite is True
        ):
            if a not in replacements:
                replacements[a] = b + sp.Dummy("positive_parameter", positive=True)
    reverse = {
        next(iter(v.free_symbols)): a - (v - next(iter(v.free_symbols)))
        for a, v in replacements.items()
    }
    return expr.xreplace(replacements), reverse


def clean_limit(expr, x, point):
    choices = sorted(expr.atoms(sp.Min, sp.Max), key=sp.count_ops, reverse=True)
    if choices:
        branches = [expr]
        for atom in choices:
            branches = [e.xreplace({atom: a}) for e in branches for a in atom.args]
            if len(branches) > 16:
                return None
        values = [clean_limit(e, x, point) for e in branches]
        return (
            values[0]
            if values and values[0] is not None and all(v == values[0] for v in values)
            else None
        )
    value = bounded_limit(expr, x, point, allow_general=True)
    if value is None or value.has(
        x, sp.nan, sp.zoo, sp.AccumBounds, sp.Limit, sp.Subs, sp.Derivative
    ):
        return None
    return value


def signed_gamma_germ_certificate(expr, x, point, assumptions=sp.S.true):
    """Resolve gamma poles and signs on a checked real parameter chart."""
    if not expr.has(sp.gamma, sp.factorial, sp.polygamma, sp.arg, sp.uppergamma):
        return None
    if sp.count_ops(expr) > 120:
        return None
    chart, reverse = parameter_chart(expr, x, assumptions)
    if point is sp.oo and chart.has(sp.polygamma):
        atoms = chart.atoms(sp.polygamma)
        if any(
            a.args[0] != 0
            or a.args[1].is_positive is not True
            or clean_limit(a.args[1], x, point) != sp.oo
            for a in atoms
        ):
            return None
        # Linear combinations with bounded coefficients preserve O(1/z) tails.
        markers = {a: sp.Dummy() for a in atoms}
        polynomial = chart.xreplace(markers)
        if not polynomial.is_polynomial(*markers.values()):
            return None
        p = sp.Poly(polynomial, *markers.values())
        if p.total_degree() > 1:
            return None
        for a, u in markers.items():
            c = p.coeff_monomial(u)
            if clean_limit(c / a.args[1], x, point) != 0:
                return None
        value = clean_limit(
            sp.expand_log(
                chart.xreplace({a: sp.log(a.args[1]) for a in atoms}), force=False
            ),
            x,
            point,
        )
        if value is not None:
            return answer(
                value.xreplace(reverse),
                "digamma_positive_tail_absolute_error",
                "psi(z)=log(z)+O(1/z) on the positive tail; each multiplied remainder tends to zero",
            )
    if point in (sp.oo, -sp.oo):
        return None
    if chart.func is sp.arg and chart.args[0].func is sp.gamma:
        z = chart.args[0].args[0]
        c = sp.simplify(z / (x - point))
        if c in (sp.I, -sp.I):
            right = -sp.pi / 2 if c == sp.I else sp.pi / 2
            if x.is_positive is True and point == 0:
                return answer(
                    right,
                    "gamma_imaginary_pole_side",
                    "Gamma(c*t)=1/(c*t)+O(1); its principal argument approaches the indicated side value",
                )
            k = sp.Dummy("k", positive=True, integer=True)
            witnesses = (
                LimitEvidence(
                    "gamma_pole_right_subsequence",
                    "k positive integer; x=p+1/k is defined eventually and Gamma(c/k)=(k/c)*(1+O(1/k))",
                    ((x, point + 1 / k),),
                    right,
                ),
                LimitEvidence(
                    "gamma_pole_left_subsequence",
                    "k positive integer; x=p-1/k is defined eventually and Gamma(-c/k)=(-k/c)*(1+O(1/k))",
                    ((x, point - 1 / k),),
                    -right,
                ),
            )
            return LimitStatus.DOES_NOT_EXIST, None, witnesses
    # Gamma/factorial ratios: exact recurrence, before evaluating either pole.
    rewritten = chart.replace(
        lambda a: a.func is sp.factorial, lambda a: sp.gamma(a.args[0] + 1)
    )
    numerator, denominator = sp.fraction(rewritten)
    if numerator.func is sp.gamma and denominator.func is sp.gamma:
        delta = sp.simplify(numerator.args[0] - denominator.args[0])
        if delta.is_Integer and abs(delta) <= 64:
            z = denominator.args[0]
            reduced = sp.rf(z, delta) if delta >= 0 else 1 / sp.rf(z + delta, -delta)
            value = clean_limit(reduced, x, point)
            if value is not None:
                return answer(
                    value,
                    "gamma_exact_integer_recurrence",
                    "Gamma(z+m)/Gamma(z) is the exact meromorphic rising-factorial germ, including cancellation at poles",
                )
    atoms = chart.atoms(sp.polygamma)
    if atoms:
        markers = {a: sp.Dummy() for a in atoms}
        linear = chart.xreplace(markers)
        if (
            not linear.is_polynomial(*markers.values())
            or sp.Poly(linear, *markers.values()).total_degree() > 1
        ):
            return None
        replacements = {}
        for a in atoms:
            m, z = a.args
            if not m.is_Integer or m < 0 or m > 8 or sp.simplify(z.subs(x, point)) != 0:
                return None
            replacements[a] = sp.polygamma(m, z + 1) + (-1) ** (m + 1) * sp.factorial(
                m
            ) / z ** (m + 1)
        germ = sp.expand(chart.xreplace(replacements))
        # Exact cancellation leaves analytic functions near 1. A remaining
        # pole can be assigned a sign only on an explicit positive side.
        analytic = germ.xreplace(
            {
                sp.polygamma(m, z + 1): sp.polygamma(m, 1)
                for a in atoms
                for m, z in [a.args]
            }
        )
        if x.is_positive is not True and any(
            t.has(x) and clean_limit(t, x, point) in (sp.oo, -sp.oo)
            for t in sp.Add.make_args(analytic)
        ):
            return None
        value = clean_limit(analytic, x, point)
        if value is not None and (
            value.is_finite is True or (x.is_positive is True and point == 0)
        ):
            # Substitution error is O(z); only unamplified linear occurrences.
            for a in atoms:
                coefficient = sp.expand(chart).coeff(a)
                if (
                    coefficient.has(x)
                    and clean_limit(coefficient * a.args[1], x, point) != 0
                ):
                    return None
            return answer(
                value,
                "polygamma_exact_pole_subtraction",
                "psi_m(z)=psi_m(z+1)+(-1)^(m+1)*m!/z^(m+1); the remaining germ is analytic at 1 with vanishing O(z) error",
            )
    # Scaled upper gamma near zero: recurrence plus a sectorial O(z) bound.
    atoms = chart.atoms(sp.uppergamma)
    if len(atoms) == 1:
        a = next(iter(atoms))
        order, z = a.args
        alpha = -order
        if (
            alpha.is_real is True
            and (alpha - 1).is_positive is True
            and clean_limit(z, x, point) == 0
        ):
            c = sp.simplify(z / (x - point))
            if c.is_imaginary is True and c.is_zero is False:
                scale = sp.cancel(chart / (alpha * z**alpha * a))
                if not scale.has(x) and scale.is_finite is True:
                    return answer(
                        scale.xreplace(reverse),
                        "upper_gamma_zero_scaled_sector",
                        "a*z**a*Gamma(-a,z)=exp(-z)-z**a*Gamma(1-a,z)=1+o(1) for fixed real a>1 on either imaginary ray, using the principal branch; the second term is O(z), with logarithmic corrections when required",
                    )
    return None


def positive_gamma_tail_certificate(expr, x, point, assumptions=sp.S.true):
    """Certify gamma domination and fixed-shift tails on the positive real axis."""
    if (
        point is not sp.oo
        or x.is_positive is not True
        or not expr.has(sp.gamma, sp.beta, sp.uppergamma)
    ):
        return None
    if sp.count_ops(expr) > 150:
        return None
    chart, reverse = parameter_chart(expr, x, assumptions)
    atoms = chart.atoms(sp.uppergamma)
    if len(atoms) == 1:
        atom = next(iter(atoms))
        n, z = atom.args
        if (
            not n.has(x)
            and not n.has(sp.oo, -sp.oo, sp.zoo, sp.nan)
            and z.is_positive is True
        ):
            ratio = sp.simplify(z / x)
            if not ratio.has(x) and ratio.is_positive is True:
                coefficient = sp.cancel(chart / atom)
                # Permit fixed complex polynomial powers: their moduli grow
                # polynomially. Exponential amplification is out.
                allowed = True
                for factor in sp.Mul.make_args(coefficient):
                    base, exponent = factor.as_base_exp()
                    if factor.has(x) and not (
                        (base == x and not exponent.has(x))
                        or (base == z and not exponent.has(x))
                    ):
                        allowed = False
                if allowed:
                    return answer(
                        sp.S.Zero,
                        "fixed_order_upper_gamma_tail",
                        "For each fixed finite complex order, Gamma(n,c*x)=exp(-c*x)*(c*x)**(n-1)*(1+O(1/x)), c>0; every fixed polynomial multiplier is dominated",
                    )
    # A uniform bounded-shift ratio; never replace two gamma terms separately.
    gammas = chart.atoms(sp.gamma)
    if len(gammas) == 2:
        for a in gammas:
            if a.args[0] != x:
                continue
            b = next(g for g in gammas if g != a)
            s = sp.expand(b.args[0] - x)
            if s.is_real is True and s.has(sp.sin, sp.cos):
                bounds = real_bounds(s, x)
                if bounds is not None and not any(v.has(x) for v in bounds):
                    ratio = b / (a * x**s)
                    scale = sp.cancel(chart / ratio)
                    if not scale.has(x) and scale.is_finite is True:
                        return answer(
                            scale.xreplace(reverse),
                            "gamma_uniform_bounded_shift_ratio",
                            "Gamma(x+s)/(Gamma(x)*x**s)=1+O(1/x), uniformly for s in the proved fixed real bounded interval, by integrating psi(x+t)=log(x)+O(1/x) over the bounded shift",
                        )
    # One large/small beta difference, normalized relative to its dominant
    # positive-pole term rather than subtracting separate infinite limits.
    from .univariate_perturbation_limits import linear_pair

    pair = linear_pair(chart, chart.atoms(sp.beta))
    if pair:
        a, b, c = pair
        for small, big, coefficient in ((a, b, c), (b, a, -c)):
            u, v = small.args
            s, t = big.args
            if (
                u != v
                or s != t
                or u.is_positive is not True
                or s.is_positive is not True
            ):
                continue
            if clean_limit(u, x, point) != 0 or clean_limit(s, x, point) != sp.oo:
                continue
            decay = gamma_stirling_product_certificate(
                sp.gamma(s) ** 2 / sp.gamma(2 * s), x, point
            )
            if not decay or decay[1] != 0:
                continue
            value = clean_limit(2 * coefficient / u, x, point)
            if value is not None and value.is_extended_real is True:
                return answer(
                    value.xreplace(reverse),
                    "beta_small_large_relative_dominance",
                    "B(u,u)=2/u*(1+O(u)) for u->0+; positive affine Stirling proves B(s,s)->0. Their difference is (2/u)*(1+o(1)), so the multiplying coefficient retains a relative, not subtractive, error",
                )
    # Products are evaluated with relative O(1/x) or O(z) remainders. Sums
    # are accepted only if every term resolves individually and no infinities
    # cancel, so leading-order replacement cannot invent cancellation terms.
    values = []
    for term in sp.Add.make_args(sp.expand_mul(chart)):
        local = term
        for factor in sp.Mul.make_args(term):
            base, exponent = factor.as_base_exp()
            if factor.has(sp.gamma, sp.beta):
                if (
                    base.func not in (sp.gamma, sp.beta)
                    or exponent.has(x)
                    or exponent.is_real is not True
                    or exponent.is_finite is not True
                ):
                    return None
        replacements = {}
        for b in local.atoms(sp.beta):
            u, v = b.args
            if (
                u.is_positive is True
                and v.is_positive is True
                and clean_limit(u, x, point) == clean_limit(v, x, point) == 0
            ):
                replacements[b] = (u + v) / (u * v)
            else:
                replacements[b] = sp.gamma(u) * sp.gamma(v) / sp.gamma(u + v)
        local = local.xreplace(replacements)
        small = {}
        for g in local.atoms(sp.gamma):
            z = g.args[0]
            if z.is_positive is True and clean_limit(z, x, point) == 0:
                small[g] = 1 / z
        local = local.xreplace(small)
        if not local.has(sp.gamma):
            value = clean_limit(local, x, point)
        else:
            # Split a product into its gamma-Stirling part and a continuous
            # elementary factor. Exact gamma-ratio constants are retained.
            factors = sp.Mul.make_args(local)

            def core(f):
                if f.has(sp.gamma):
                    return True
                base, exponent = f.as_base_exp()
                return (base == x and exponent.is_real is True) or (
                    not f.has(x) and f.is_positive is True
                )

            gamma_factors = [f for f in factors if core(f)]
            plain = sp.Mul(*(f for f in factors if not core(f)))
            result = gamma_stirling_product_certificate(
                sp.Mul(*gamma_factors), x, point
            )
            value = None
            if result and result[1].is_finite is True and result[1].is_zero is False:
                other = clean_limit(plain, x, point)
                if other is not None:
                    value = other * result[1]
            if value is None:
                result = gamma_stirling_product_certificate(local, x, point)
                if result:
                    value = result[1]
        if value is None or value.has(sp.nan, sp.zoo, sp.AccumBounds):
            return None
        values.append(value)
    total = sp.Add(*values)
    if total.has(sp.nan, sp.zoo):
        return None
    return answer(
        total.xreplace(reverse),
        "gamma_positive_tail_products",
        "Gamma(z)=z**(-1)*(1+O(z)) at 0+, beta(u,v)=(u+v)/(u*v)*(1+O(u+v)) at positive zero; positive affine gamma products use controlled Stirling remainders. Only non-cancelling individually resolved terms are combined",
    )


def real_bounds(node, x):
    """Outward bounds on the real x>1 tail, retaining all dependencies safely."""
    if node == x:
        return x, x
    if not node.has(x):
        return (node, node) if node.is_real is True and node.is_finite is True else None
    if node.func in (sp.sin, sp.cos) and node.args[0].is_real is True:
        return -sp.S.One, sp.S.One
    if node.func is sp.exp:
        b = real_bounds(node.args[0], x)
        return None if b is None else (sp.exp(b[0]), sp.exp(b[1]))
    if node.func is sp.atan:
        b = real_bounds(node.args[0], x)
        return None if b is None else (sp.atan(b[0]), sp.atan(b[1]))
    if (
        node.is_Pow
        and node.base == sp.log(x)
        and not node.exp.has(x)
        and node.exp.is_real is True
        and node.exp.is_finite is True
    ):
        # log(x)>0 on this helper's x>1 tail, even if global symbol
        # assumptions cannot decide its sign for small positive x.
        return node, node
    if (
        node.is_Pow
        and not node.base.has(x)
        and node.base.is_positive is True
        and node.base.is_finite is True
    ):
        b = real_bounds(node.exp, x)
        if b is None:
            return None
        if (node.base - 1).is_positive is True:
            return node.base ** b[0], node.base ** b[1]
        if (node.base - 1).is_negative is True:
            return node.base ** b[1], node.base ** b[0]
        return None
    if node.func is sp.log:
        b = real_bounds(node.args[0], x)
        if b and b[0].is_positive is True:
            return sp.log(b[0]), sp.log(b[1])
        return None
    if node.is_Add:
        b = [real_bounds(a, x) for a in node.args]
        return (
            None
            if any(v is None for v in b)
            else (sum(v[0] for v in b), sum(v[1] for v in b))
        )
    if node.is_Mul:
        lower = upper = sp.S.One
        for a in node.args:
            b = real_bounds(a, x)
            if b is None:
                return None
            candidates = [sp.expand_mul(u * v) for u in (lower, upper) for v in b]
            lower, upper = sp.Min(*candidates), sp.Max(*candidates)
        return lower, upper
    if node.is_Pow and node.base == x:
        b = real_bounds(node.exp, x)
        if b:
            return x ** b[0], x ** b[1]
    if node.is_Pow and not node.exp.has(x) and node.exp.is_real is True:
        b = real_bounds(node.base, x)
        if b and b[0].is_positive is True:
            if node.exp.is_positive is True:
                return b[0] ** node.exp, b[1] ** node.exp
            if node.exp.is_negative is True:
                return b[1] ** node.exp, b[0] ** node.exp
            if node.exp == 0:
                return sp.S.One, sp.S.One
    return None


def exponential_envelope_certificate(expr, x, point, assumptions=sp.S.true):
    """Control an oscillatory exponential by an independently bounded amplitude."""
    if point is not sp.oo or x.is_positive is not True:
        return None
    if not expr.has(sp.sin, sp.cos, sp.exp) and not any(
        p.exp.has(x) for p in expr.atoms(sp.Pow)
    ):
        return None
    if sp.count_ops(expr) > 100:
        return None
    chart, reverse = parameter_chart(expr, x, assumptions)
    # exp(g)-1=g*(1+O(g)); do not replace individual exponentials in sums.
    for factor in sp.Mul.make_args(chart):
        if factor.is_Add and len(factor.args) == 2 and -1 in factor.args:
            atom = next(a for a in factor.args if a != -1)
            if atom.func is sp.exp and atom.args[0].is_real is True:
                g = atom.args[0]
                if clean_limit(g, x, point) == 0:
                    leading = chart.xreplace({factor: g})
                    value = clean_limit(leading, x, point)
                    if value is not None and value.is_extended_real is True:
                        return answer(
                            value.xreplace(reverse),
                            "expm1_relative_small_germ",
                            "For real g->0, exp(g)-1=g*(1+O(g)); the relative error is retained after multiplication",
                        )
    if chart.has(sp.sin, sp.cos):
        bounds = real_bounds(chart, x)
        if bounds:
            lo, hi = (clean_limit(a, x, point) for a in bounds)
            if lo == hi and lo is not None:
                return answer(
                    lo.xreplace(reverse),
                    "real_oscillatory_outward_squeeze",
                    "Outward real bounds on x>1 have the same extended limit; no attainment or nonexistence is inferred",
                )
            if lo is sp.oo:
                return answer(
                    sp.oo,
                    "real_oscillatory_growing_lower_bound",
                    "The outward lower bound tends to positive infinity on the real positive tail",
                )
            if hi is -sp.oo:
                return answer(
                    -sp.oo,
                    "real_oscillatory_decreasing_upper_bound",
                    "The outward upper bound tends to negative infinity on the real positive tail",
                )
    # Positive real power/exponential germs with exact parameter charts.
    if reverse and all(a.func in (sp.exp, sp.log) for a in chart.atoms(sp.Function)):
        if chart.is_real is True:
            value = clean_limit(chart, x, point)
            if value is not None:
                return answer(
                    value.xreplace(reverse),
                    "positive_tail_elementary_parameter_chart",
                    "Strict parameter inequalities are represented by exact positive coordinates; real exp/log/power germs retain their signs and branches",
                )
    # Positive dominant exponential with a complex subdominant phase. The
    # conclusion includes both modulus divergence and direction tending to 1.
    terms = sp.Add.make_args(chart)
    data = []
    for term in terms:
        powers = [p for p in term.atoms(sp.Pow) if p.exp.has(x) and not p.base.has(x)]
        if len(powers) != 1:
            break
        p = powers[0]
        base = p.base
        if base.is_real is not True or base.is_zero is not False:
            break
        try:
            poly = sp.Poly(p.exp, x)
        except sp.PolynomialError:
            break
        if poly.degree() != 1:
            break
        a, b = poly.nth(1), poly.nth(0)
        c = sp.cancel(term / p)
        if (
            a.is_real is not True
            or b.is_real is not True
            or c.has(x)
            or c.is_finite is not True
        ):
            break
        data.append((a * sp.log(sp.Abs(base)), p, c))
    if len(data) == len(terms) and len(data) > 1:
        for rate, p, c in data:
            if (
                p.base.is_positive is not True
                or c.is_positive is not True
                or rate.is_positive is not True
            ):
                continue
            if all(q == p or (rate - r).is_positive is True for r, q, d in data):
                return answer(
                    sp.oo,
                    "positive_direction_exponential_dominance",
                    "The positive real dominant exponential diverges; every other principal real-base power has exponentially vanishing relative modulus. Thus |f|->infinity and f/|f|->1, giving positive directed infinity without assuming the expression real-valued",
                )
    # Complex unit phases may be dropped only in a finite uniform limit.
    phases = {}
    for p in chart.atoms(sp.Pow):
        if (
            p.base.is_negative is True
            and not p.base.has(x)
            and p.exp.has(x)
            and p.exp.is_real is True
        ):
            u = sp.Dummy("unit_phase", nonzero=True)
            phases[p] = (-p.base) ** p.exp * u
    if not phases:
        return None
    transformed = sp.cancel(chart.xreplace(phases))
    markers = tuple(v.args[-1] if v.is_Mul else v for v in phases.values())
    # Extract markers explicitly, independent of canonical factor order.
    markers = tuple(
        sorted(
            set().union(*(v.atoms(sp.Dummy) for v in phases.values()))
            - set(chart.atoms(sp.Dummy)),
            key=sp.default_sort_key,
        )
    )
    n, d = sp.fraction(transformed)
    if not n.is_polynomial(*markers) or not d.is_polynomial(*markers):
        return None
    pn, pd = sp.Poly(n, *markers), sp.Poly(d, *markers)
    # Candidate scales include both numerator and denominator coefficients;
    # coefficient-wise convergence is uniform on the compact unit torus.
    for scale in pd.coeffs():
        if scale == 0:
            continue
        ns = {power: clean_limit(c / scale, x, point) for power, c in pn.terms()}
        ds = {power: clean_limit(c / scale, x, point) for power, c in pd.terms()}
        if any(
            v is None or v.is_finite is not True for v in (*ns.values(), *ds.values())
        ):
            continue
        np = sp.Add(
            *(c * sp.Mul(*(u**k for u, k in zip(markers, p))) for p, c in ns.items())
        )
        dp = sp.Add(
            *(c * sp.Mul(*(u**k for u, k in zip(markers, p))) for p, c in ds.items())
        )
        terms = sp.Poly(dp, *markers).terms()
        if len(terms) != 1 or terms[0][1].is_zero is not False:
            continue
        result = sp.cancel(np / dp)
        if not result.has(*markers) and result.is_finite is True:
            return answer(
                result.xreplace(reverse),
                "unit_phase_uniform_rational_limit",
                "Principal negative-base powers have unit-modulus phase for real exponents. Scaled polynomial coefficients converge uniformly on the unit torus, and the limiting denominator is a nonzero monomial, uniformly bounded away from zero",
            )
    return None
