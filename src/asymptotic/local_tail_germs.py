"""Bounded analytic germs, attained pole witnesses, and gamma-tail bounds."""

from functools import lru_cache

import sympy as sp

from .compact_limit_germs import answer
from .limit_models import LimitEvidence, LimitStatus


@lru_cache(maxsize=256)
def _operation_count(expr):
    return sp.count_ops(expr)


@lru_cache(maxsize=256)
def _bounded_integer_powers(expr):
    return not any(
        power.exp.is_Integer and abs(power.exp) > 16 for power in expr.atoms(sp.Pow)
    )


def budget(expr, n=120):
    """Check structural work bounds shared by the local certificates.

    SymPy expression structure is immutable. Bounded caches reuse its cost
    across providers and thresholds without caching assumption-dependent proofs.
    """
    return _operation_count(expr) <= n and _bounded_integer_powers(expr)


def fixed_finite(e):
    # Plain free parameters denote fixed finite complex numbers; this check
    # still rejects unguarded parameter denominators such as 1/(v-1).
    if any(s.is_finite is False for s in e.free_symbols):
        return False
    chart = e.xreplace(
        {
            s: sp.Dummy("finite_parameter", complex=True, finite=True)
            for s in e.free_symbols
        }
    )
    return chart.is_finite is True


def rational_order(e, x):
    n, d = sp.fraction(sp.cancel(e))
    try:
        pn, pd = sp.Poly(n, x), sp.Poly(d, x)
    except sp.PolynomialError:
        return None
    if pn.is_zero or pd.is_zero or max(pn.degree(), pd.degree()) > 8:
        return None
    if any(
        c.is_real is not True or c.is_finite is not True
        for p in (pn, pd)
        for c in p.all_coeffs()
    ):
        return None
    kn = min(k[0] for k, c in pn.terms())
    kd = min(k[0] for k, c in pd.terms())
    c = pn.nth(kn) / pd.nth(kd)
    if c.is_zero is not False:
        return None
    return kn - kd, c


def analytic_origin(e, x):
    for f in e.atoms(sp.Function):
        if f.func in (sp.sin, sp.cos):
            try:
                p = sp.Poly(f.args[0], x)
            except sp.PolynomialError:
                return False
            if p.degree() > 4 or any(
                c.is_finite is not True or c.is_real is not True for c in p.all_coeffs()
            ):
                return False
        elif f.func is sp.log:
            a = f.args[0]
            if not a.is_polynomial(x) or a.subs(x, 0).is_positive is not True:
                return False
        else:
            return False
    return all(p.exp.is_Integer for p in e.atoms(sp.Pow))


def ci_trig_cancellation_certificate(expr, x, point):
    if point != 0 or not expr.has(sp.Ci) or not budget(expr):
        return None
    atoms = expr.atoms(sp.Ci)
    if len(atoms) > 3:
        return None
    rewrite = {
        a: 2 * sp.cos(a.args[0] / 2) ** 2 - 1
        for a in expr.atoms(sp.cos)
        if sp.cos(a.args[0] / 2) in expr.atoms(sp.cos)
    }
    original_den = sp.denom(sp.together(expr))
    if not budget(original_den, 60) or not analytic_origin(original_den, x):
        return None
    # Nonzero analytic leading term proves original denominator avoidance.
    try:
        dp = sp.Poly(sp.series(original_den, x, 0, 9).removeO(), x)
    except (ValueError, NotImplementedError, sp.PolynomialError):
        return None
    if dp.is_zero or not any(c.is_zero is False for c in dp.all_coeffs()):
        return None
    reduced = sp.cancel(expr.xreplace(rewrite))
    errors = []
    replacements = {}
    for atom in atoms:
        a = sp.cancel(atom.args[0] / x)
        if a.has(x) or a.is_positive is not True or a.is_finite is not True:
            return None
        error = sp.Dummy("bounded_Ci_remainder", real=True)
        errors.append(error)
        replacements[atom] = sp.EulerGamma + sp.log(x) + sp.log(a) + error * x * x
    germ = sp.cancel(reduced.xreplace(replacements))
    if germ.has(sp.log(x)) or not budget(germ, 80) or not analytic_origin(germ, x):
        return None
    try:
        ep = sp.Poly(germ, *errors)
        if ep.total_degree() > 1:
            return None
        # Each unknown O(x**2) coefficient must vanish independently.
        for error in errors:
            c = ep.coeff_monomial(error)
            if sp.series(c, x, 0, 1).removeO() != 0:
                return None
        value = sp.series(germ.subs(dict.fromkeys(errors, 0)), x, 0, 1).removeO()
    except (ValueError, NotImplementedError, sp.PolynomialError):
        return None
    if value.has(x) or value.is_finite is not True:
        return None
    return answer(
        value,
        "ci_exact_trig_cancellation",
        "Exact double-angle identities and rational cancellation are applied on the original punctured domain. The original analytic denominator has a nonzero Taylor coefficient of order at most eight, so it has no other nearby zeros. DLMF 6.6.6 gives Ci(a*x)=EulerGamma+log(x)+log(a)+O(x**2) for fixed a>0, including the principal negative-real boundary. Every multiplied remainder vanishes separately; the remaining analytic germ has the recorded limit on both real sides.",
    )


def secant_odd_pi_pole_certificate(expr, x, point):
    if point != 0 or not expr.has(sp.sec) or not budget(expr, 70):
        return None
    atoms = expr.atoms(sp.sec)
    if len(atoms) != 1:
        return None
    atom = next(iter(atoms))
    phase = atom.args[0]
    replacements = {}
    for a in phase.atoms(sp.Abs):
        v = a.args[0]
        if not v.is_polynomial(x) or v.is_real is False:
            return None
        at = v.subs(x, 0)
        if at.is_positive is True:
            replacements[a] = v
        elif at.is_negative is True:
            replacements[a] = -v
        else:
            return None
    phase = sp.expand(phase.xreplace(replacements))
    try:
        p = sp.Poly(phase, x)
    except sp.PolynomialError:
        return None
    if p.degree() > 4 or any(
        c.is_real is not True or c.is_finite is not True for c in p.all_coeffs()
    ):
        return None
    odd = sp.cancel(p.nth(0) / sp.pi)
    if odd.is_Integer is not True or (
        odd.is_odd is not True and odd.is_even is not True
    ):
        return None
    delta = phase - p.nth(0)
    data = rational_order(delta, x)
    if data is None:
        return None
    q, d = data
    if q <= 0:
        return None
    # SymPy evaluates sec(pi+h) to -sec(h). Account for both exact
    # presentations of the same odd-pi pole without restoring held heads.
    coefficient = sp.cancel(expr * (1 - (-1) ** odd * atom))
    if coefficient.has(sp.sec) or not budget(coefficient, 40):
        return None
    data = rational_order(coefficient, x)
    if data is None:
        return None
    m, c = data
    power = m - 2 * q
    leading = -2 * c / d**2
    if power > 0:
        return answer(
            sp.S.Zero,
            "secant_odd_pi_local_germ",
            "cos(odd*pi+delta)=-1+delta**2/2+O(delta**4); the original quotient has positive vanishing order and all original denominators are eventually nonzero.",
        )
    if power == 0:
        return answer(
            leading,
            "secant_odd_pi_local_germ",
            "The exact rational prefactor and cos(odd*pi+delta)=-1+delta**2/2+O(delta**4) give the recorded finite limit. Nonzero polynomial leading terms prove eventual original-domain membership.",
        )
    if leading.is_positive is True:
        right = sp.oo
    elif leading.is_negative is True:
        right = -sp.oo
    else:
        return None
    left = right if power % 2 == 0 else -right
    statement = f"Local phase delta={delta} has leading term ({d})*x**{q}; after exact secant pi-shift normalization the original expression is ({leading})*x**{power}*(1+O(x)). For sufficiently small nonzero real x, delta is nonzero and |delta|<pi/2, so cos(phase) is nonzero and the original factor 1-(-1)**({odd})*sec(phase) is nonzero. The rational prefactor denominator also has a nonzero polynomial leading term. Thus these attained sequences eventually avoid every original pole and lie in the punctured domain."
    if x.is_positive is True:
        return answer(right, "secant_odd_pi_local_germ", statement)
    if x.is_negative is True:
        return answer(left, "secant_odd_pi_local_germ", statement)
    if left == right:
        return answer(right, "secant_odd_pi_local_germ", statement)
    n = sp.Dummy("attained_pole_sequence_n", integer=True, positive=True)
    return (
        LimitStatus.DOES_NOT_EXIST,
        None,
        (
            LimitEvidence(
                "attained_secant_right_subsequence", statement, ((x, 1 / n),), right
            ),
            LimitEvidence(
                "attained_secant_left_subsequence", statement, ((x, -1 / n),), left
            ),
        ),
    )


def upper_gamma_fixed_argument_tail_certificate(expr, x, point):
    if point is not sp.oo or not expr.has(sp.uppergamma) or not budget(expr, 70):
        return None
    atoms = expr.atoms(sp.uppergamma)
    if len(atoms) != 1:
        return None
    atom = next(iter(atoms))
    a, w = atom.args
    if w.has(x) or not fixed_finite(w):
        return None
    try:
        p = sp.Poly(a, x)
    except sp.PolynomialError:
        return None
    if (
        p.degree() != 1
        or p.LC().is_positive is not True
        or any(c.is_real is not True or c.is_finite is not True for c in p.all_coeffs())
    ):
        return None
    gamma_atoms = expr.atoms(sp.gamma)
    if len(gamma_atoms) != 1:
        return None
    gamma = next(iter(gamma_atoms))
    shift = sp.expand(gamma.args[0] - a)
    if shift.is_Integer is not True or shift < 0 or shift > 4:
        return None
    marker = sp.Dummy("upper_gamma_ratio")
    reduced = sp.cancel(
        expr.xreplace(
            {atom: marker * sp.gamma(a), gamma: sp.rf(a, shift) * sp.gamma(a)}
        )
    )
    if reduced.has(sp.gamma):
        return None
    if sp.diff(reduced, marker, 2) != 0 or reduced.subs(marker, 0) != 0:
        return None
    coefficient = sp.diff(reduced, marker)
    if coefficient.has(x):
        n, d = sp.fraction(coefficient)
        try:
            pn, pd = sp.Poly(n, x), sp.Poly(d, x)
        except sp.PolynomialError:
            return None
        if max(pn.degree(), pd.degree()) > 4 or pn.degree() > pd.degree():
            return None
        if pd.LC().is_zero is not False:
            return None
        if any(not fixed_finite(c) for p in (pn, pd) for c in p.all_coeffs()):
            return None
        value = 0 if pn.degree() < pd.degree() else pn.LC() / pd.LC()
    else:
        value = coefficient
    if not fixed_finite(sp.sympify(value)):
        return None
    return answer(
        value,
        "upper_gamma_fixed_argument_superfactorial_tail",
        "For fixed finite complex w and real a->+infinity, the straight-segment lower-gamma integral gives |lowergamma(a,w)|<=exp(abs(w))*abs(w)**a/a on its principal branch. Dividing by Gamma(a) and Stirling proves uppergamma(a,w)/Gamma(a)->1 with superfactorial error. Exact integer shifts of the denominator are retained; the remaining rational coefficient has a finite tail. Both gamma arguments are eventually positive and rational denominators eventually nonzero.",
    )


def reciprocal_gamma_exponential_domination_certificate(expr, x, point):
    if point is not sp.oo or not expr.has(sp.gamma) or not budget(expr, 60):
        return None
    atoms = expr.atoms(sp.gamma)
    if len(atoms) != 2:
        return None
    offsets = []
    for a in atoms:
        b = sp.expand(a.args[0] - x)
        if b.has(x) or not fixed_finite(b):
            return None
        offsets.append(b)
    rest = expr
    for a in atoms:
        rest = rest * a
    rest = sp.cancel(rest)
    if rest.has(sp.gamma):
        return None
    if rest == 1:
        pass
    elif rest.is_Pow and not rest.base.has(x):
        b, e = rest.args
        if not fixed_finite(b):
            return None
        try:
            p = sp.Poly(e, x)
        except sp.PolynomialError:
            return None
        if p.degree() > 1 or any(
            c.is_real is not True or c.is_finite is not True for c in p.all_coeffs()
        ):
            return None
        eventually_positive = (p.degree() == 1 and p.LC().is_positive is True) or (
            p.degree() == 0 and p.nth(0).is_positive is True
        )
        if b.is_zero is not False and not eventually_positive:
            return None
    else:
        return None
    return answer(
        sp.S.Zero,
        "reciprocal_gamma_factorial_domination",
        "For fixed finite complex shifts b,c, Stirling on x+b and x+c with x real positive gives log|Gamma(x+b)*Gamma(x+c)|=2*x*log(x)-2*x+O(log(x)). A fixed finite base to a real affine exponent grows at most exponentially; zero base with an eventually positive exponent is identically zero. Thus the quotient tends to zero. Gamma arguments stay in a right half-plane and have no tail poles. The O(log(x)) shift correction cannot compete with the factorial scale.",
    )


def radical_trig_tail_certificate(expr, x, point):
    if (
        point is not sp.oo
        or not expr.has(sp.cos, sp.sec)
        or not budget(expr, 70)
        or expr.free_symbols - {x}
    ):
        return None
    if any(f.func not in (sp.cos, sp.sec) for f in expr.atoms(sp.Function)):
        return None
    t = sp.Dummy("positive_reciprocal_tail", positive=True)
    germ = expr.subs(x, 1 / t)
    atoms = germ.atoms(sp.cos, sp.sec)
    if len(atoms) > 2:
        return None
    replacements = {}
    for atom in atoms:
        phase = atom.args[0]
        # Extract only positive powers of the positive real tail coordinate.
        # The polynomial numerator is proved positive near zero. This avoids
        # splitting principal square roots on an unproved complex chart.
        roots = {}
        for power in phase.atoms(sp.Pow):
            if power.exp != sp.S.Half or power.base.is_polynomial(t):
                continue
            num, den = sp.fraction(sp.cancel(power.base))
            c, j = den.as_coeff_exponent(t)
            if (
                not num.is_polynomial(t)
                or num.subs(t, 0).is_positive is not True
                or c.is_positive is not True
                or j.is_Integer is not True
                or abs(j) > 4
            ):
                return None
            pn = sp.Poly(num, t)
            if pn.degree() > 4 or any(
                a.is_real is not True or a.is_finite is not True
                for a in pn.all_coeffs()
            ):
                return None
            roots[power] = sp.sqrt(num / c) / t ** (j / 2)
        phase = sp.cancel(phase.xreplace(roots))
        for power in phase.atoms(sp.Pow):
            if power.exp.is_Integer:
                continue
            if power.exp != sp.S.Half:
                return None
            # Only positive analytic polynomial radicands after the exact
            # positive-coordinate substitution; no principal-power splitting.
            if (
                not power.base.is_polynomial(t)
                or power.base.subs(t, 0).is_positive is not True
            ):
                return None
            pn = sp.Poly(power.base, t)
            if pn.degree() > 4 or any(
                a.is_real is not True or a.is_finite is not True
                for a in pn.all_coeffs()
            ):
                return None
        if phase.has(sp.Function):
            return None
        try:
            p = sp.Poly(sp.series(phase, t, 0, 4).removeO(), t)
        except (ValueError, NotImplementedError, sp.PolynomialError):
            return None
        if p.degree() > 3 or any(
            c.is_real is not True or c.is_finite is not True for c in p.all_coeffs()
        ):
            return None
        cos_germ = sp.series(sp.cos(p.as_expr()), t, 0, 4).removeO()
        data = rational_order(cos_germ, t)
        if data is None or data[0] not in (0, 1, 2):
            return None
        replacements[atom] = cos_germ if atom.func is sp.cos else 1 / cos_germ
    reduced = sp.cancel(germ.xreplace(replacements))
    # Restrict to an exact monomial cosine ratio with a regular rational
    # prefactor; fourth-order phase errors cannot be amplified by other poles.
    marker = {a: sp.Dummy("trig_marker", nonzero=True) for a in atoms}
    algebra = germ.xreplace(
        {a: z if a.func is sp.cos else 1 / z for a, z in marker.items()}
    )
    coefficient = algebra
    for a, z in marker.items():
        coefficient = coefficient / (z if a.func is sp.cos else 1 / z)
    coefficient = sp.cancel(coefficient)
    if any(
        coefficient.has(z) for z in marker.values()
    ) or not coefficient.is_rational_function(t):
        return None
    data = rational_order(coefficient, t)
    if data is None or data[0] < 0:
        return None
    data = rational_order(reduced, t)
    if data is None or data[0] < 0:
        return None
    value = sp.S.Zero if data[0] > 0 else data[1]
    return answer(
        value,
        "radical_trig_reciprocal_tail_germ",
        "The exact positive-tail coordinate t=1/x makes the square-root radicands analytic and positive near zero. Each finite trigonometric phase is expanded with O(t**4) remainder. Every cosine denominator has a nonzero leading coefficient of order at most two, so its relative error vanishes and its original zeros are eventually avoided. A regular rational prefactor prevents error amplification; the leading quotient gives the recorded limit.",
    )
