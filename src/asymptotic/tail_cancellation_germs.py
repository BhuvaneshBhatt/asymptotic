"""Bounded, reusable certificates with retained remainder or attained witnesses."""

import sympy as sp

from ._symbolic_policy import bounded_limit
from .compact_limit_germs import answer
from .limit_models import LimitEvidence, LimitStatus


def within_budget(expr, ops):
    return sp.count_ops(expr) <= ops and not any(
        p.exp.is_Integer and abs(p.exp) > 16 for p in expr.atoms(sp.Pow)
    )


def clean(expr, x, point, *, direction="+"):
    value = bounded_limit(expr, x, point, direction=direction, allow_general=True)
    if value is None or value.has(
        x, sp.nan, sp.zoo, sp.AccumBounds, sp.Limit, sp.Subs, sp.Derivative
    ):
        return None
    # A substituted unresolved function at infinity is not a limit value.
    # Only canonical signed infinities may survive a real-tail certificate.
    if value not in (sp.oo, -sp.oo) and value.has(sp.oo, -sp.oo):
        return None
    if value.free_symbols - (expr.free_symbols - {x}):
        return None
    return value


def _fixed_finite_constant(expr):
    if expr.is_finite is True:
        return True
    if expr.is_finite is False:
        return False
    if expr.is_Symbol:
        return True
    if expr.is_Add or expr.is_Mul:
        return all(_fixed_finite_constant(a) for a in expr.args)
    if expr.is_Pow and expr.base.is_positive is True and expr.base.is_number is True:
        return _fixed_finite_constant(expr.exp)
    return False


def _finite_dirichlet_projection_value(expr, x):
    """Group exact numeric exponential scales before taking a tail limit."""
    groups = {}
    for term in sp.Add.make_args(sp.expand_mul(sp.expand_power_exp(expr))):
        base = sp.S.One
        coefficient = term
        for power in term.atoms(sp.Pow):
            if not power.exp.has(x):
                continue
            if power.base.is_Rational is not True or power.base <= 0:
                return None
            rate = sp.expand(power.exp).coeff(x)
            offset = sp.expand(power.exp - rate * x)
            if rate.is_Rational is not True or offset.has(x):
                return None
            base *= power.base**rate
            coefficient = coefficient.xreplace({power: power.base**offset})
        groups[base] = groups.get(base, 0) + coefficient
    result = sp.S.Zero
    for base, coefficient in groups.items():
        coefficient = sp.simplify(coefficient)
        if coefficient == 0:
            continue
        if coefficient.has(x):
            return None
        if not _fixed_finite_constant(coefficient):
            return None
        if base < 1:
            continue
        if base == 1:
            result += coefficient
            continue
        return None
    return sp.simplify(result)


def perturbation_budget(expr, ops):
    # Accumulating periodic poles cannot be disposed of by generic growth
    # limits. These theorem routes provide no pole-avoidance hypotheses.
    return within_budget(expr, ops) and not expr.has(sp.tan, sp.cot, sp.sec, sp.csc)


def finite_clean(expr, x, point, domain):
    directions = (
        ("+",)
        if (x.is_positive is True and point == 0) or domain == sp.Gt(x, point)
        else ("-",)
        if x.is_negative is True and point == 0
        else ("+", "-")
    )
    values = [clean(expr, x, point, direction=d) for d in directions]
    if any(v is None for v in values):
        return None
    if len(values) == 1 or values[0] == values[1]:
        return values[0]
    if sp.simplify(values[0] - values[1]) == 0:
        return values[0]
    return None


def dirichlet_cancellation_certificate(expr, x, point):
    """Finite Dirichlet projection, with an explicit absolute tail bound.

    For real s=x+b, q>0 fixed, the remainder beyond q+M is at most
    (q+M+1)**(-s)*(1+(q+M+1)/(s-1)). Its multiplied bound must vanish.
    Unlike replacing each zeta by its first term, this retains cancellations.
    """
    atoms = expr.atoms(sp.zeta)
    if (
        point is not sp.oo
        or x.is_positive is not True
        or not atoms
        or len(atoms) > 3
        or not perturbation_budget(expr, 100)
    ):
        return None
    # This route accepts linear projections only. Reject nonlinear powers
    # before constructing six-term expansions, not after polynomial expansion.
    if any(p.base.has(sp.zeta) and p.exp != 1 for p in expr.atoms(sp.Pow)):
        return None
    replacements = {}
    errors = []
    for atom in sorted(atoms, key=sp.default_sort_key):
        s = atom.args[0]
        q = atom.args[1] if len(atom.args) == 2 else sp.S.One
        offset = sp.expand(s - x)
        if offset.has(x) or offset.is_finite is False:
            return None
        if offset.is_finite is not True:
            parameters = tuple(offset.free_symbols)
            if (
                not parameters
                or any(p.is_finite is False for p in parameters)
                or not offset.is_polynomial(*parameters)
            ):
                return None
        if q.has(x) or q.is_positive is not True or q.is_finite is not True:
            return None
        error = sp.Dummy("dirichlet_error", real=True)
        # Fixed depth bounds work, not a search over arbitrarily many terms.
        replacements[atom] = sum((q + j) ** (-s) for j in range(6)) + error
        real_s = x + sp.re(offset)
        errors.append((error, (q + 6) ** (-real_s) * (1 + (q + 6) / (real_s - 1))))
    transformed = sp.expand_mul(expr.xreplace(replacements))
    symbols = [e for e, b in errors]
    try:
        poly = sp.Poly(transformed, *symbols)
    except sp.PolynomialError:
        return None
    if poly.total_degree() > 1:
        return None
    for error, bound in errors:
        coefficient = transformed.coeff(error)
        if (
            coefficient.has(*symbols)
            or clean(sp.Abs(coefficient) * bound, x, point) != 0
        ):
            return None
    leading = transformed.xreplace(dict.fromkeys(symbols, sp.S.Zero))
    value = _finite_dirichlet_projection_value(leading, x)
    if value is None:
        value = clean(leading, x, point)
    if value is None:
        return None
    return answer(
        value,
        "zeta_finite_dirichlet_projection",
        "Six exact Dirichlet terms retain all selected cancellations, including fixed finite complex shifts. The integral-test absolute bound uses Re(s); every amplified tail tends to zero.",
    )


def zeta_finite_pole_certificate(expr, x, point, assumptions, domain=sp.S.true):
    if (
        point in (sp.oo, -sp.oo)
        or not expr.has(sp.zeta)
        or not perturbation_budget(expr, 100)
    ):
        return None
    replacements = {}
    errors = []
    clauses = sp.And.make_args(assumptions)
    for atom in expr.atoms(sp.zeta):
        s = atom.args[0]
        q = atom.args[1] if len(atom.args) == 2 else sp.S.One
        delta = sp.cancel(s - 1)
        if (
            not delta.is_rational_function(x)
            or delta == 0
            or delta.subs(x, point) != 0
            or q.has(x)
        ):
            return None
        if not (q.is_positive is True or sp.Gt(sp.re(q), 0) in clauses):
            return None
        error = sp.Dummy("zeta_analytic_error")
        replacements[atom] = 1 / delta - sp.polygamma(0, q) + error
        errors.append((error, delta))
    transformed = sp.expand_mul(expr.xreplace(replacements))
    symbols = [e for e, d in errors]
    try:
        poly = sp.Poly(transformed, *symbols)
    except sp.PolynomialError:
        return None
    if poly.total_degree() > 1:
        return None
    for error, delta in errors:
        coefficient = transformed.coeff(error)
        if (
            coefficient.has(*symbols)
            or finite_clean(coefficient * delta, x, point, domain) != 0
        ):
            return None
    value = finite_clean(
        transformed.xreplace(dict.fromkeys(symbols, sp.S.Zero)), x, point, domain
    )
    if value is None:
        return None
    return answer(
        value,
        "zeta_finite_laurent_remainder",
        "For fixed Re(q)>0, zeta(1+delta,q)=1/delta-psi(q)+O(delta); every coefficient multiplying its analytic remainder times delta tends to zero. Unknown q pole/branch strata are declined.",
    )


def harmonic_tail_certificate(expr, x, point):
    atoms = expr.atoms(sp.harmonic)
    if (
        point is not sp.oo
        or x.is_positive is not True
        or len(atoms) != 1
        or not perturbation_budget(expr, 60)
    ):
        return None
    atom = next(iter(atoms))
    n = atom.args[0]
    s = atom.args[1] if len(atom.args) > 1 else sp.S.One
    if (
        n != x
        or s.is_number is not True
        or s.is_real is not True
        or not (1 - s).is_positive
    ):
        return None
    coefficient = sp.cancel(expr / atom)
    fixed_power = (
        coefficient.is_Pow and coefficient.base == x and coefficient.exp.is_Rational
    )
    if coefficient.has(sp.harmonic) or not (
        coefficient.is_rational_function(x) or fixed_power
    ):
        return None
    value = clean(coefficient * x ** (1 - s) / (1 - s), x, point)
    if value is None or value.is_extended_real is not True:
        return None
    return answer(
        value,
        "harmonic_integral_test_tail",
        "For fixed real order s<1, Euler summation/integral comparison gives H(x,s)=x**(1-s)/(1-s)*(1+o(1)) on the positive real analytic continuation. A rational multiplier or fixed real monomial preserves this relative error.",
    )


def small_log_power_certificate(expr, x, point):
    """Near-one powers with small quadratic log error, before expansion."""
    if (
        point is not sp.oo
        or x.is_positive is not True
        or not expr.is_Pow
        or not expr.exp.has(x)
        or not perturbation_budget(expr, 70)
    ):
        return None
    delta = sp.cancel(expr.base - 1)
    power = expr.exp
    if (
        delta.is_real is not True
        or power.is_real is not True
        or clean(delta, x, point) != 0
    ):
        return None
    if clean(sp.Abs(power) * delta**2, x, point) != 0:
        return None
    logarithm = clean(power * delta, x, point)
    if logarithm is None or logarithm.is_extended_real is not True:
        return None
    value = (
        sp.S.Zero
        if logarithm is -sp.oo
        else sp.oo
        if logarithm is sp.oo
        else sp.exp(logarithm)
    )
    return answer(
        value,
        "near_one_power_log_remainder",
        "delta->0 is real, so 1+delta>0 eventually. log(1+delta)=delta+O(delta**2); |power|*delta**2->0 controls the exponentiation error without expanding the power.",
    )


def gamma_ratio_scale_certificate(expr, x, point):
    if (
        point is not sp.oo
        or x.is_positive is not True
        or not perturbation_budget(expr, 80)
    ):
        return None
    atoms = expr.atoms(sp.gamma)
    if len(atoms) != 2:
        return None
    for denominator in atoms:
        numerator = next(a for a in atoms if a != denominator)
        g = denominator.args[0]
        delta = sp.expand(numerator.args[0] - g)
        if (
            delta.has(x)
            or delta.is_real is not True
            or delta.is_finite is not True
            or g.is_real is not True
        ):
            continue
        if clean(g, x, point) is not sp.oo:
            continue
        coefficient = sp.cancel(expr * denominator / numerator)
        if coefficient.has(sp.gamma):
            continue
        # A product only: no additive residual/cancellation is replaced.
        value = clean(coefficient * g**delta, x, point)
        if value is None or value.is_extended_real is not True:
            continue
        return answer(
            value,
            "gamma_fixed_shift_relative_scale",
            "For real g->+infinity and fixed finite real delta, Gamma(g+delta)/Gamma(g)=g**delta*(1+O(1/g)). The exact multiplying coefficient is retained; no subtraction of approximated gamma terms is permitted.",
        )
    return None


def monomial_phase_subsequences(expr, x, point, domain):
    """Invert one attained phase exactly; do not infer DNE from bounds."""
    if (
        not within_budget(expr, 80)
        or expr.free_symbols - {x}
        or domain is not sp.S.true
    ):
        return None
    trig = (sp.sin, sp.cos, sp.tan, sp.cot, sp.sec, sp.csc)
    atoms = expr.atoms(*trig)
    if not atoms:
        return None
    if any(
        a.func in (sp.tan, sp.cot, sp.sec, sp.csc) and a.args[0].has(*trig)
        for a in atoms
    ):
        return None
    if any(
        a.func not in trig + (sp.exp, sp.log, sp.Abs, sp.atan)
        for a in expr.atoms(sp.Function)
    ):
        return None
    primitive = [a for a in atoms if a.has(x) and not a.args[0].has(*trig)]
    if not primitive:
        return None
    phase = primitive[0].args[0]
    if any(a.args[0] != phase for a in primitive):
        return None
    if point is sp.oo:
        a, r = phase.as_coeff_exponent(x)
        if a.has(x) or a.is_positive is not True or r.is_Rational is not True or r <= 0:
            return None

        def inverse(y):
            return (y / a) ** (1 / r)
    elif point not in (sp.oo, -sp.oo):
        if x.is_nonpositive is True:
            return None
        u = sp.Dummy("phase_local", positive=True)
        a, r = phase.subs(x, point + u).as_coeff_exponent(u)
        if a.has(u) or a.is_positive is not True or r.is_Rational is not True or r >= 0:
            return None

        def inverse(y):
            return point + (y / a) ** (1 / r)
    else:
        return None
    n = sp.Dummy("subsequence_n", integer=True, positive=True)
    witnesses = []
    for offset in (sp.pi / 2, 3 * sp.pi / 2, sp.pi / 4, 5 * sp.pi / 4):
        sequence = inverse(2 * sp.pi * n + offset)
        # Structural inversion on positive bases and rational powers is exact.
        if (
            sp.simplify(
                sp.powdenest(phase.subs(x, sequence), force=False)
                - (2 * sp.pi * n + offset)
            )
            != 0
        ):
            continue
        along = sp.simplify(expr.subs(x, sequence))
        if along.has(sp.nan, sp.zoo, sp.oo, -sp.oo) or any(
            a.has(n) for a in along.atoms(*trig)
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
        value = clean(along, n, sp.oo)
        if value is None:
            continue
        evidence = LimitEvidence(
            "attained_monomial_phase_subsequence",
            "The positive real inverse phase approaches the target and has exactly the recorded phase; trig poles are avoided and the substituted expression has a certified limit.",
            ((x, sequence),),
            value,
        )
        for other in witnesses:
            if value != other.value and (
                value in (sp.oo, -sp.oo)
                or other.value in (sp.oo, -sp.oo)
                or sp.simplify(value - other.value).is_zero is False
            ):
                return LimitStatus.DOES_NOT_EXIST, None, (other, evidence)
        witnesses.append(evidence)
    return None


def tangent_pole_subsequences(expr, x, point, domain):
    if (
        point is not sp.oo
        or domain is not sp.S.true
        or expr.free_symbols - {x}
        or not within_budget(expr, 60)
    ):
        return None
    atoms = expr.atoms(sp.tan)
    if len(atoms) != 1:
        return None
    atom = next(iter(atoms))
    phase = atom.args[0]
    a = sp.diff(phase, x)
    b = sp.expand(phase - a * x)
    if a.is_Rational is not True or a <= 0 or b.has(x) or b.is_real is not True:
        return None
    marker = sp.Dummy("tangent")
    try:
        poly = sp.Poly(sp.expand_mul(expr.xreplace({atom: marker})), marker)
    except sp.PolynomialError:
        return None
    if poly.degree() != 1:
        return None
    coefficient, rest = poly.nth(1), poly.nth(0)
    if not coefficient.is_rational_function(x) or not rest.is_rational_function(x):
        return None
    numerator, denominator = sp.fraction(sp.cancel(coefficient))
    q = sp.degree(denominator, x) - sp.degree(numerator, x)
    if q not in (1, 2, 3, 4):
        return None
    scale = clean(coefficient * x**q, x, point)
    base = clean(rest, x, point)
    if (
        scale is None
        or scale.is_zero is not False
        or base is None
        or base.is_finite is not True
    ):
        return None
    n = sp.Dummy("pole_avoidance_n", positive=True, integer=True)
    zero = (sp.pi * n - b) / a
    near = (sp.pi * n + sp.pi / 2 - b - n ** (-q)) / a
    changed = base + scale * (a / sp.pi) ** q
    return (
        LimitStatus.DOES_NOT_EXIST,
        None,
        (
            LimitEvidence(
                "attained_tangent_zero_subsequence",
                "Arithmetic phase pi*n gives tan=0. Rational denominators have only finitely many zeros; the sequence is eventually in the original domain.",
                ((x, zero),),
                base,
            ),
            LimitEvidence(
                "attained_tangent_near_pole_subsequence",
                "Phase pi*n+pi/2-n**(-q) avoids every tangent pole for n>=1. tan=cot(n**(-q))=n**q*(1+O(n**(-2*q))); rational coefficient asymptotics give a different attained limit.",
                ((x, near),),
                changed,
            ),
        ),
    )


def upper_gamma_cut_certificate(expr, x, point, domain):
    """Signed cut values from the exact gamma monodromy relation.

    Restricted to fixed 0<a<1 rational orders: the jump's imaginary part is
    nonzero by positivity of integral_0^r t**(a-1)*exp(t) dt, not numerics.
    """
    if (
        point in (sp.oo, -sp.oo)
        or expr.func is not sp.uppergamma
        or not within_budget(expr, 40)
    ):
        return None
    a, z = expr.args
    if a.is_Rational is not True or not 0 < a < 1:
        return None
    if not z.is_polynomial(x) or sp.degree(z, x) > 4:
        return None
    boundary = sp.simplify(z.subs(x, point))
    if boundary.is_negative is not True:
        return None
    if domain is not sp.S.true and domain != sp.Gt(x, point):
        return None
    t = sp.Dummy("cut_approach", positive=True)
    sides = (
        (1,)
        if (x.is_positive is True and point == 0) or domain == sp.Gt(x, point)
        else (-1,)
        if x.is_negative is True and point == 0
        else (1, -1)
    )
    evidence = []
    for side in sides:
        imaginary = sp.expand(sp.im(z.subs(x, point + side * t)))
        if not imaginary.is_polynomial(t):
            return None
        poly = sp.Poly(imaginary, t)
        if poly.is_zero:
            return None
        sign = None
        for k in range(poly.degree() + 1):
            c = poly.nth(k)
            if c == 0:
                continue
            sign = 1 if c.is_positive is True else -1 if c.is_negative is True else None
            break
        if sign is None:
            return None
        upper = sp.uppergamma(a, boundary)
        value = (
            upper
            if sign == 1
            else sp.exp(-2 * sp.pi * sp.I * a) * upper
            + (1 - sp.exp(-2 * sp.pi * sp.I * a)) * sp.gamma(a)
        )
        evidence.append(
            LimitEvidence(
                "upper_gamma_signed_cut_germ",
                "The first nonzero imaginary Taylor coefficient selects the cut side. The lower value follows Gamma(a,z*exp(-2*pi*i))=exp(-2*pi*i*a)*Gamma(a,z)+(1-exp(-2*pi*i*a))*Gamma(a).",
                ((x, point + side * t),),
                value,
            )
        )
    if len(evidence) == 2 and evidence[0].value != evidence[1].value:
        jump = LimitEvidence(
            "upper_gamma_nonzero_cut_jump",
            "For 0<a<1 and r>0 the imaginary jump has magnitude 2*sin(pi*a)*integral_0^r t**(a-1)*exp(t) dt>0. Thus both attained one-sided boundary values differ.",
        )
        return LimitStatus.DOES_NOT_EXIST, None, tuple(evidence) + (jump,)
    return LimitStatus.PROVED, evidence[0].value, tuple(evidence)


def quadratic_radical_perturbation_certificate(expr, x, point):
    """Uniform bounded perturbations of a positive quadratic radical tail."""
    if point is not sp.oo or x.is_positive is not True or not within_budget(expr, 70):
        return None
    roots = [p for p in expr.atoms(sp.Pow) if p.exp == sp.Rational(1, 2) and p.has(x)]
    if len(roots) != 1:
        return None
    root = roots[0]
    marker = sp.Dummy("quadratic_root")
    try:
        poly = sp.Poly(sp.expand_mul(expr.xreplace({root: marker})), marker)
    except sp.PolynomialError:
        return None
    if poly.degree() != 1:
        return None
    coefficient, rest = poly.nth(1), poly.nth(0)
    if (
        coefficient.has(x)
        or coefficient.is_real is not True
        or coefficient.is_finite is not True
    ):
        return None
    radicand = sp.expand_mul(root.base)
    u, v = radicand.coeff(x, 2), radicand.coeff(x, 1)
    if (
        u.has(x)
        or v.has(x)
        or u.is_positive is not True
        or v.is_real is not True
        or v.is_finite is not True
    ):
        return None
    remainder = sp.expand_mul(radicand - u * x * x - v * x)
    from .gamma_exponential_germs import real_bounds

    bounds = real_bounds(remainder, x)
    if bounds is None or any(b.has(x) or b.is_finite is not True for b in bounds):
        return None
    constant = sp.simplify(rest + coefficient * sp.sqrt(u) * x)
    if constant.has(x) or constant.is_finite is not True:
        return None
    value = constant + coefficient * v / (2 * sp.sqrt(u))
    return answer(
        value,
        "quadratic_radical_uniform_remainder",
        "For u>0, fixed real v and a proved bounded real remainder r(x), sqrt(u*x**2+v*x+r)=sqrt(u)*x+v/(2*sqrt(u))+O(1/x), uniformly on the positive tail. The radicand is eventually positive and the fixed finite multiplier preserves the vanishing error; no oscillatory attainment is inferred.",
    )
