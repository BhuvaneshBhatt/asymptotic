"""Real positive-tail perturbations with relative errors preserved through cancellation."""

import sympy as sp

from ._symbolic_policy import bounded_limit
from .compact_limit_germs import answer


def linear_pair(expr, atoms):
    if len(atoms) != 2:
        return None
    a, b = sorted(atoms, key=sp.default_sort_key)
    u, v = sp.Dummy(), sp.Dummy()
    replaced = sp.expand_mul(expr.xreplace({a: u, b: v}))
    try:
        p = sp.Poly(replaced, u, v)
    except sp.PolynomialError:
        return None
    if p.total_degree() != 1 or p.nth(0, 0) != 0:
        return None
    ca, cb = p.nth(1, 0), p.nth(0, 1)
    if sp.cancel(ca + cb) != 0 or ca == 0:
        return None
    return a, b, ca


def zeta_tail_certificate(expr, x, point):
    if (
        point is not sp.oo
        or x.is_positive is not True
        or not expr.has(sp.zeta)
        or sp.count_ops(expr) > 120
    ):
        return None
    pair = linear_pair(expr, expr.atoms(sp.zeta))
    if pair:
        a, b, c = pair
        if len(a.args) == len(b.args) == 1:
            for base, shift, coefficient in ((b, a, c), (a, b, -c)):
                offset = sp.expand(base.args[0] - x)
                if (
                    offset.has(x)
                    or offset.is_real is not True
                    or offset.is_finite is not True
                ):
                    continue
                delta = sp.expand(shift.args[0] - base.args[0])
                if (
                    delta.is_real is not True
                    or delta == 0
                    or bounded_limit(delta, x, sp.oo, allow_general=True) != 0
                ):
                    continue
                leading = -coefficient * sp.log(2) * delta * 2 ** (-x - offset)
                value = bounded_limit(leading, x, sp.oo, allow_general=True)
                if (
                    value is not None
                    and value.is_extended_real is True
                    and not value.has(sp.AccumBounds, sp.Limit)
                ):
                    return answer(
                        value,
                        "zeta_small_shift_dirichlet_remainder",
                        "the k=2 Dirichlet term of zeta(s+delta)-zeta(s) is -log(2)*delta*2**(-s)*(1+o(1)); k>=3 derivatives are O(3**(-s)), uniformly for real delta->0. The error is relative to the difference, not to each zeta separately",
                    )

    def replace(node):
        if node.is_Add and len(node.args) == 2 and -1 in node.args:
            other = next(a for a in node.args if a != -1)
            if other.func is sp.zeta and len(other.args) == 1:
                offset = sp.expand(other.args[0] - x)
                if (
                    not offset.has(x)
                    and offset.is_real is True
                    and offset.is_finite is True
                ):
                    return 2 ** (-x - offset)
        if node.is_Mul:
            vals = [replace(a) for a in node.args]
            if any(v is None for v in vals):
                return None
            return sp.Mul(*vals)
        if node.is_Pow and node.exp.is_Integer:
            b = replace(node.base)
            return None if b is None else b**node.exp
        if not node.has(sp.zeta):
            return node
        return None

    leading = replace(expr)
    if leading is None or leading.has(sp.zeta):
        return None
    value = bounded_limit(leading, x, sp.oo, allow_general=True)
    if (
        value is None
        or value.is_extended_real is not True
        or value.has(sp.AccumBounds, sp.Limit)
    ):
        return None
    return answer(
        value,
        "zeta_dirichlet_leading_product",
        "for fixed real b, zeta(x+b)-1=2**(-x-b)*(1+O((2/3)**x)); multiplication and integer powers preserve relative errors without subtracting leading terms",
    )


def gamma_small_shift_certificate(expr, x, point):
    if (
        point is not sp.oo
        or x.is_positive is not True
        or not expr.has(sp.gamma)
        or sp.count_ops(expr) > 120
    ):
        return None
    pair = linear_pair(expr, expr.atoms(sp.gamma))
    if not pair:
        return None
    a, b, c = pair
    if a == sp.gamma(x):
        shift = b
        coefficient = -c
    elif b == sp.gamma(x):
        shift = a
        coefficient = c
    else:
        return None
    delta = sp.expand(shift.args[0] - x)
    if (
        delta.is_real is not True
        or bounded_limit(delta * sp.log(x), x, sp.oo, allow_general=True) != 0
    ):
        return None
    amplitude = sp.factor_terms(coefficient * delta)
    sign = (
        sp.S.One
        if amplitude.is_positive is True
        else -sp.S.One
        if amplitude.is_negative is True
        else None
    )
    if sign is None:
        return None
    loggamma = (x - sp.Rational(1, 2)) * sp.log(x) - x + sp.log(2 * sp.pi) / 2
    logarithm = (
        sp.expand_log(sp.log(sign * amplitude), force=False)
        + loggamma
        + sp.log(sp.log(x))
    )
    if logarithm.has(sp.gamma):
        return None
    value = bounded_limit(logarithm, x, sp.oo, allow_general=True)
    if value is sp.oo:
        result = sign * sp.oo
    elif value is -sp.oo:
        result = sp.S.Zero
    elif value is not None and value.is_real is True and value.is_finite is True:
        result = sign * sp.exp(value)
    else:
        return None
    return answer(
        result,
        "gamma_small_shift_relative_remainder",
        "for real delta with delta*log(x)->0, Gamma(x+delta)-Gamma(x)=delta*Gamma(x)*log(x)*(1+o(1)); Taylor's integral remainder and psi(x)~log(x) give a relative error. Positive-tail Stirling determines the signed amplitude in logarithmic coordinates",
    )


def gamma_exponential_difference_certificate(expr, x, point):
    if (
        point is not sp.oo
        or x.is_positive is not True
        or expr.is_Add is not True
        or len(expr.args) != 2
    ):
        return None
    baseline = sp.exp(sp.gamma(x))
    if -baseline not in expr.args:
        return None
    raised = next(a for a in expr.args if a != -baseline)
    if raised.func is not sp.exp:
        return None
    argument = raised.args[0]
    atoms = argument.atoms(sp.gamma)
    if len(atoms) != 1:
        return None
    atom = next(iter(atoms))
    scale = sp.cancel(argument / atom)
    if scale.func is not sp.exp:
        return None
    c = sp.cancel(scale.args[0] * x)
    if c.has(x) or c.is_number is not True or c.is_positive is not True:
        return None
    delta = sp.expand(atom.args[0] - x)
    if (
        delta.is_real is not True
        or bounded_limit(x * delta * sp.log(x), x, sp.oo, allow_general=True) != 0
    ):
        return None
    return answer(
        sp.oo,
        "gamma_exponential_difference_gap",
        "exp(c/x)*Gamma(x+delta)-Gamma(x)~c*Gamma(x)/x->+infinity when c>0 and x*delta*log(x)->0; factoring the outer exponential difference preserves this positive diverging gap",
    )


def zeta_pole_projection_certificate(expr, x, point):
    if (
        point is not sp.oo
        or x.is_positive is not True
        or not expr.has(sp.zeta)
        or sp.count_ops(expr) > 100
    ):
        return None
    replacements = {}
    constants = {}
    for projection in expr.atoms(sp.re, sp.im):
        atom = projection.args[0]
        if atom.func is not sp.zeta or len(atom.args) != 1:
            continue
        c = sp.cancel((atom.args[0] - 1) * x)
        if (
            c.has(x)
            or c.is_number is not True
            or c.is_finite is not True
            or c.is_zero is not False
        ):
            return None
        terms = projection.func(1 / c) * x + (
            sp.EulerGamma if projection.func is sp.re else 0
        )
        for k in range(1, 5):
            marker = sp.Dummy("real_stieltjes_constant", real=True)
            constants[marker] = sp.stieltjes(k)
            terms += (
                (-1) ** k * marker * projection.func(c**k) / (sp.factorial(k) * x**k)
            )
        replacements[projection] = terms
    if not replacements:
        return None
    transformed = sp.expand(expr.xreplace(replacements))
    if transformed.has(sp.zeta):
        return None
    # Bound amplification of the omitted analytic O(x^-5) remainder.
    for projection in replacements:
        marker = sp.Dummy()
        p = expr.xreplace({projection: marker})
        try:
            poly = sp.Poly(p, marker)
        except sp.PolynomialError:
            return None
        if poly.degree() > 1:
            return None
        coefficient = poly.nth(1)
        if coefficient.has(sp.zeta):
            return None
        try:
            n, d = map(lambda a: sp.Poly(a, x), sp.fraction(sp.cancel(coefficient)))
        except sp.PolynomialError:
            return None
        if n.degree() - d.degree() > 3:
            return None
    value = bounded_limit(transformed, x, sp.oo, allow_general=True)
    if value is None or value.has(x, sp.AccumBounds, sp.Limit, sp.nan, sp.zoo):
        return None
    return answer(
        value.xreplace(constants),
        "zeta_pole_stieltjes_projection",
        "the Laurent expansion zeta(1+z)=1/z+EulerGamma+sum((-1)**k*gamma_k*z**k/k!) has real Stieltjes constants; fixed complex z=c/x permits real/imaginary projection. Linear coefficients grow at most cubically, so the multiplied analytic O(x^-5) remainder vanishes",
    )


def nested_gamma_growth_certificate(expr, x, point):
    if point is not sp.oo or x.is_positive is not True:
        return None
    nested = sp.log(sp.gamma(sp.gamma(x)))
    if nested in expr.atoms(sp.log):
        marker = sp.Dummy()
        p = expr.xreplace({nested: marker})
        try:
            poly = sp.Poly(p, marker)
        except sp.PolynomialError:
            return None
        if poly.degree() != 1 or poly.nth(0) != 0:
            return None
        coefficient = poly.nth(1)
        if coefficient.has(sp.gamma):
            return None
        sign = (
            1
            if coefficient.is_positive is True
            else -1
            if coefficient.is_negative is True
            else None
        )
        if sign is None:
            return None
        amplitude_log = (
            sp.expand_log(sp.log(sign * coefficient), force=False)
            + (x - sp.Rational(1, 2)) * sp.log(x)
            - x
            + sp.log(2 * sp.pi) / 2
            + sp.log(x)
            + sp.log(sp.log(x))
        )
        value = bounded_limit(amplitude_log, x, sp.oo, allow_general=True)
        if value is sp.oo:
            return answer(
                sign * sp.oo,
                "nested_gamma_log_stirling_growth",
                "log(Gamma(Gamma(x)))~Gamma(x)*log(Gamma(x)) and log(Gamma(x))~x*log(x); logarithmic coordinates decide the positive amplitude with relative errors tending to zero",
            )
        if value is -sp.oo:
            return answer(
                sp.S.Zero,
                "nested_gamma_log_stirling_growth",
                "Stirling relative errors and the logarithmic amplitude give decay",
            )
    psi = sp.polygamma(0, x)
    atoms = expr.atoms(sp.gamma)
    near = [a for a in atoms if a.args[0].func is sp.exp]
    if len(near) != 1 or psi not in expr.atoms(sp.polygamma):
        return None
    atom = near[0]
    c = sp.cancel(atom.args[0].args[0] * sp.gamma(x))
    if c.has(x) or c.is_real is not True or c.is_finite is not True:
        return None
    u, v = sp.Dummy(), sp.Dummy()
    try:
        p = sp.Poly(sp.expand_mul(expr.xreplace({atom: u, psi: v})), u, v)
    except sp.PolynomialError:
        return None
    if p.total_degree() != 1:
        return None
    coefficient = p.nth(1, 0)
    if (
        coefficient == 0
        or sp.cancel(p.nth(0, 1) + coefficient) != 0
        or sp.cancel(p.nth(0, 0) + coefficient) != 0
    ):
        return None
    value = bounded_limit(-coefficient * sp.log(x), x, sp.oo, allow_general=True)
    if value not in (sp.oo, -sp.oo, sp.S.Zero):
        return None
    return answer(
        value,
        "gamma_near_one_digamma_relative_growth",
        "Gamma(exp(c/Gamma(x))) tends to 1 for fixed finite real c; psi(x)~log(x). Therefore Gamma(exp(c/Gamma(x)))-psi(x)-1 ~ -log(x), with relative remainder tending to zero",
    )


def iterated_digamma_exponential_certificate(expr, x, point):
    if point is not sp.oo or x.is_positive is not True:
        return None
    t = x
    for _ in range(3):
        t = sp.polygamma(0, t)
    tower = t
    for _ in range(3):
        tower = sp.exp(tower)
    if not expr.has(tower):
        return None
    coefficient = sp.cancel(expr / tower)
    if coefficient.has(sp.polygamma, sp.exp) or not coefficient.is_rational_function(x):
        return None
    try:
        n, d = map(lambda a: sp.Poly(a, x), sp.fraction(coefficient))
    except sp.PolynomialError:
        return None
    if n.degree() - d.degree() > -1:
        return None
    return answer(
        sp.S.Zero,
        "iterated_digamma_exponential_bound",
        "for large positive t, psi(t)<log(t) and exp(psi(t))<=t-1/4. All iterated psi arguments eventually diverge positively. Propagating this strict gap gives the triple exponential at most x**exp(-1/4), so a rational coefficient of degree at most -1 forces decay",
    )


def regularized_hypergeometric_decay_certificate(expr, x, point):
    from .reference_normalization import RegularizedHypergeometric0F1

    if point is not sp.oo or x.is_positive is not True or sp.count_ops(expr) > 100:
        return None
    atoms = expr.atoms(RegularizedHypergeometric0F1)
    if len(atoms) != 1:
        return None
    atom = next(iter(atoms))
    b, z = atom.args
    if (
        z != x
        or b.has(x)
        or b.is_finite is False
        or not (b.is_Symbol or (b.is_number and b.is_finite is True))
    ):
        return None
    marker = sp.Dummy()
    try:
        p = sp.Poly(expr.xreplace({atom: marker}), marker)
    except sp.PolynomialError:
        return None
    if p.degree() != 1 or p.nth(0) != 0:
        return None
    exponential = 0
    algebraic = sp.S.One
    for factor in sp.Mul.make_args(p.nth(1)):
        if factor.func is sp.exp:
            exponential += factor.args[0]
        else:
            algebraic *= factor
    if not algebraic.is_rational_function(x):
        return None
    try:
        q = sp.Poly(exponential, x)
    except sp.PolynomialError:
        return None
    if q.degree() < 1 or q.LC().is_negative is not True:
        return None
    return answer(
        sp.S.Zero,
        "regularized_0f1_subexponential_envelope",
        "for every fixed finite complex b, regularized 0F1(b;x) has at most polynomial times exp(C*sqrt(x)) growth, including nonpositive-integer b by its entire-parameter series. A negative-leading polynomial exponential and rational prefactor dominate this bound",
    )


def elementary_minmax_tail_certificate(expr, x, point):
    if point is not sp.oo or not expr.has(sp.Min, sp.Max) or sp.count_ops(expr) > 80:
        return None
    original = expr
    for atom in sorted(expr.atoms(sp.Min, sp.Max), key=sp.count_ops):
        winner = None
        for candidate in atom.args:
            for other in atom.args:
                if other == candidate:
                    continue
                if candidate.is_positive is True and other.is_positive is True:
                    gap = sp.log(candidate) - sp.log(other)
                    gap = sp.expand_log(gap, force=False)
                elif candidate.is_real is True and other.is_real is True:
                    gap = candidate - other
                else:
                    break
                value = bounded_limit(gap, x, sp.oo, allow_general=True)
                if value is None:
                    break
                desired = (
                    (value is -sp.oo or value.is_negative is True)
                    if atom.func is sp.Min
                    else (value is sp.oo or value.is_positive is True)
                )
                if not desired:
                    break
            else:
                winner = candidate
                break
        if winner is None:
            return None
        expr = expr.xreplace({atom: winner})
    if expr == original:
        return None
    value = bounded_limit(expr, x, sp.oo, allow_general=True)
    if value is None or value.has(x, sp.AccumBounds, sp.Limit, sp.nan, sp.zoo):
        return None
    return answer(
        value,
        "elementary_minmax_eventual_branch",
        "strict signed limits of pairwise real differences or positive-branch logarithmic ratios prove eventual Min/Max branch selection before target substitution; indeterminate comparisons are declined",
    )


def shifted_gamma_ratio_cancellation_certificate(expr, x, point):
    if point is not sp.oo or x.is_positive is not True or sp.count_ops(expr) > 100:
        return None
    atoms = expr.atoms(sp.gamma)
    if len(atoms) != 2:
        return None
    numerator, denominator = sorted(atoms, key=sp.default_sort_key)
    a, b = [sp.expand(g.args[0] - x) for g in (numerator, denominator)]
    if not all(
        c.is_number is True and c.is_real is True and c.is_finite is True
        for c in (a, b)
    ):
        return None
    q = sum(
        (-1) ** (k + 1)
        * (sp.bernoulli(k + 1, a) - sp.bernoulli(k + 1, b))
        / (k * (k + 1) * x**k)
        for k in range(1, 5)
    )
    remainder = sp.Dummy("relative_gamma_ratio_error")
    replaced = sp.cancel(
        expr.xreplace({numerator: denominator * x ** (a - b) * (sp.exp(q) + remainder)})
    )
    if replaced.has(sp.gamma):
        return None
    try:
        p = sp.Poly(replaced, remainder)
    except sp.PolynomialError:
        return None
    if p.degree() > 1:
        return None
    coefficient = p.nth(1)
    if not coefficient.is_rational_function(x):
        return None
    try:
        n, d = map(lambda t: sp.Poly(t, x), sp.fraction(sp.cancel(coefficient)))
    except sp.PolynomialError:
        return None
    if n.degree() - d.degree() > 3:
        return None
    value = bounded_limit(p.nth(0), x, sp.oo, allow_general=True)
    if value is None or value.has(x, sp.AccumBounds, sp.Limit, sp.nan, sp.zoo):
        return None
    return answer(
        value,
        "shifted_gamma_ratio_cancellation_remainder",
        "the shifted log-gamma ratio through x**-4 has O(x**-5) remainder. Exponentiation retains a relative O(x**-5) error; the expression is linear in that error with rational coefficient of degree at most 3, so cancellation and amplification preserve the vanishing remainder",
    )
