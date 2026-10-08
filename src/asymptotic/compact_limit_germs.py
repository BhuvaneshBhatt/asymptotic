"""Limits that preserve compact powers and normalize Lambert-W scales exactly."""

import sympy as sp

from ._symbolic_policy import bounded_limit
from .limit_models import LimitEvidence, LimitStatus


def answer(value, method, statement):
    return LimitStatus.PROVED, value, LimitEvidence(method, statement, value=value)


def compact_rational_certificate(expr, x, point):
    """Use the expression tree, never expand an integer power into a polynomial."""

    def tail(node):
        if node == x:
            return 1, sp.S.One
        if not node.has(x):
            if (
                node.is_finite is True
                and node.is_real is True
                and node.is_zero is False
            ):
                return 0, node
            if node == 0:
                return None
            return None
        if node.is_Mul:
            data = [tail(a) for a in node.args]
            if any(v is None for v in data):
                return None
            return sum(d for d, c in data), sp.Mul(*(c for d, c in data))
        if node.is_Pow and node.exp.is_Integer:
            data = tail(node.base)
            if data is None:
                return None
            d, c = data
            # Avoid constructing astronomical integers, even in a coefficient.
            coefficient = (
                sp.Pow(c, node.exp, evaluate=False)
                if abs(node.exp) > 128 and c not in (-1, 1)
                else c**node.exp
            )
            return d * node.exp, coefficient
        if node.is_Add:
            data = [tail(a) for a in node.args]
            if any(v is None for v in data):
                return None
            d = max(v[0] for v in data)
            leading = [v[1] for v in data if v[0] == d]
            if len(leading) == 1:
                c = leading[0]
            else:
                if any(
                    abs(p.exp) > 128
                    for c in leading
                    for p in c.atoms(sp.Pow)
                    if p.exp.is_Integer
                ):
                    return None
                c = sp.simplify(sum(leading))
            # Cancellation requires the next order, so decline.
            return (d, c) if c.is_zero is False else None
        return None

    if point is sp.oo:
        data = tail(expr)
        if data is None:
            return None
        degree, coefficient = data
        if degree == 0:
            value = coefficient
        elif degree < 0:
            value = sp.S.Zero
        elif coefficient.is_positive is True:
            value = sp.oo
        elif coefficient.is_negative is True:
            value = -sp.oo
        else:
            return None
        return answer(
            value,
            "compact_rational_leading_order",
            "integer-power expression-tree degrees and non-cancelling leading coefficients give the rational tail without polynomial expansion",
        )
    if point not in (sp.oo, -sp.oo) and any(
        p.exp.is_Integer and abs(p.exp) > 128 for p in expr.atoms(sp.Pow)
    ):

        def local(node):
            if node == x:
                return point
            if not node.has(x):
                return node if node.is_finite is True else None
            if node.is_Add or node.is_Mul:
                values = [local(a) for a in node.args]
                if any(v is None for v in values):
                    return None
                return node.func(*values)
            if node.is_Pow and node.exp.is_Integer:
                base = local(node.base)
                if base is None or (node.exp < 0 and base.is_zero is not False):
                    return None
                if abs(node.exp) > 128 and base not in (-1, 0, 1):
                    return None
                return base**node.exp
            return None

        value = local(expr)
        if value is not None and value.is_finite is True:
            return answer(
                value,
                "compact_integer_power_continuity",
                "finite continuous integer-power composition is substituted before expansion; every denominator stays nonzero",
            )
    return None


def lambert_small_scale_certificate(expr, x, point):
    if point is not sp.oo:
        return None
    atoms = expr.atoms(sp.LambertW)
    if len(atoms) != 1:
        return None
    atom = next(iter(atoms))
    if len(atom.args) > 1 and atom.args[1] != 0:
        return None
    g = atom.args[0]
    if g.has(sp.LambertW) or g.is_real is not True:
        return None
    if bounded_limit(g, x, sp.oo, allow_general=True) != sp.oo:
        return None
    t = sp.Dummy("positive_small_W_scale", positive=True)
    transformed = sp.cancel(expr.xreplace({atom: g * t * t}))
    if transformed.has(x):
        transformed = sp.simplify(transformed)
    if transformed.has(x) or transformed.has(sp.LambertW):
        return None
    if any(
        a.func not in (sp.exp, sp.cos, sp.sin, sp.tan, sp.sinh, sp.cosh, sp.log)
        for a in transformed.atoms(sp.Function)
    ):
        return None
    value = bounded_limit(transformed, t, 0, allow_general=True)
    if value is None or value.has(t, sp.Limit, sp.nan, sp.zoo, sp.AccumBounds):
        return None
    return answer(
        value,
        "lambert_exact_small_scale",
        "on the positive tail W(g)/g=exp(-W(g)) tends to zero; exact substitution t=sqrt(W(g)/g) removes the large variable before the local limit",
    )


def positive_lambert_composition_certificate(expr, x, point):
    if point is not sp.oo or not expr.has(sp.LambertW):
        return None

    def value(node):
        if node == x:
            return sp.oo
        if not node.has(x):
            return node if node.is_positive is True and node.is_finite is True else None
        if node.func in (sp.log, sp.LambertW):
            if node.func is sp.LambertW and len(node.args) > 1 and node.args[1] != 0:
                return None
            inner = value(node.args[0])
            if inner is sp.oo:
                return sp.oo
            return None
        if node.is_Pow and node.exp.is_positive is True and node.exp.is_number:
            inner = value(node.base)
            return sp.oo if inner is sp.oo else None
        if node.is_Add or node.is_Mul:
            values = [value(a) for a in node.args]
            if any(v is None for v in values):
                return None
            if sp.oo in values:
                return sp.oo
            return node.func(*values)
        return None

    if value(expr) is sp.oo:
        return answer(
            sp.oo,
            "positive_lambert_growth_composition",
            "positive factors and monotone principal W/log compositions on eventually positive divergent real arguments certify divergence",
        )
    return None


def gamma_stirling_product_certificate(expr, x, point):
    """Four shifted Stirling terms; all multiplied remainders tend to zero."""
    if point is not sp.oo or not expr.has(sp.gamma) or expr.is_Add:
        return None

    def moderate(exponent):
        if exponent.is_real is not True or not exponent.is_rational_function(x):
            return False
        try:
            n, d = map(lambda q: sp.Poly(q, x), sp.fraction(sp.cancel(exponent)))
            return n.degree() - d.degree() <= 1
        except sp.PolynomialError:
            return False

    def affine(base):
        try:
            p = sp.Poly(base, x)
        except sp.PolynomialError:
            return None
        if p.degree() != 1:
            return None
        a, b = p.nth(1), p.nth(0)
        return (
            (a, b)
            if a.is_positive is True
            and a.is_finite is True
            and b.is_real is True
            and b.is_finite is True
            else None
        )

    logarithm = 0
    phase = False
    for factor in sp.Mul.make_args(expr):
        base, exponent = factor.as_base_exp()
        if base == -1 and exponent.is_real is True:
            phase = True
            continue
        if base.func is sp.gamma:
            data = affine(base.args[0])
            if data is None or not moderate(exponent):
                return None
            a, b = data
            expansion = (
                a * x * (sp.log(x) + sp.log(a) - 1)
                + (b - sp.Rational(1, 2)) * (sp.log(x) + sp.log(a))
                + sp.log(2 * sp.pi) / 2
            )
            for k in range(1, 5):
                expansion += (
                    (-1) ** (k + 1)
                    * sp.bernoulli(k + 1, b)
                    / (k * (k + 1) * (a * x) ** k)
                )
            logarithm += exponent * expansion
        elif factor.func is sp.exp:
            if factor.args[0].is_real is not True or factor.args[0].has(sp.gamma):
                return None
            logarithm += factor.args[0]
        elif base.is_positive is True and not base.has(x):
            if not moderate(exponent):
                return None
            logarithm += exponent * sp.log(base)
        else:
            data = affine(base)
            if data is None or not moderate(exponent):
                return None
            a, b = data
            expansion = (
                sp.log(x)
                + sp.log(a)
                + sum((-1) ** (k + 1) * (b / (a * x)) ** k / k for k in range(1, 5))
            )
            logarithm += exponent * expansion
    approximation = sp.expand(logarithm)
    value = bounded_limit(approximation, x, sp.oo, allow_general=True)
    if value is None or value.has(
        x, sp.AccumBounds, sp.Limit, sp.nan, sp.zoo, sp.Subs, sp.Derivative
    ):
        return None
    if value is -sp.oo:
        result = sp.S.Zero
    elif phase:
        return None
    elif value is sp.oo:
        result = sp.oo
    elif value is not None and value.is_real is True and value.is_finite is True:
        result = sp.exp(value)
    else:
        return None
    return answer(
        result,
        "shifted_stirling_product_remainder",
        "positive affine gamma arguments use four shifted Stirling terms; real rational exponents grow at most linearly, so all multiplied O(x**-5) remainders vanish. Unit-modulus phases are discarded only when the amplitude tends to zero",
    )


def real_monotone_composition_certificate(expr, x, point):
    """Compose genuine real extended limits, declining every indeterminate form."""
    if not expr.has(sp.gamma, sp.exp, sp.log):
        return None
    if x.is_positive is not True or point not in (0, sp.oo):
        return None

    def value(node):
        if node == x:
            return sp.S.Zero if point == 0 else sp.oo
        if not node.has(x):
            return node if node.is_real is True and node.is_finite is True else None
        if node.is_Add or node.is_Mul:
            values = [value(a) for a in node.args]
            if any(v is None for v in values):
                return None
            result = node.func(*values)
            return None if result.has(sp.nan, sp.zoo, sp.AccumBounds) else result
        if node.is_Pow and node.exp.is_number and node.exp.is_real is True:
            base = value(node.base)
            if base is None:
                return None
            if base == 0 and node.exp < 0:
                return sp.oo if node.base.is_positive is True else None
            if (
                base.is_negative is True or base is -sp.oo
            ) and node.exp.is_Integer is not True:
                return None
            result = base**node.exp
            return (
                None
                if result.has(sp.nan, sp.zoo) or result.is_extended_real is not True
                else result
            )
        if node.func is sp.exp and node.args[0].is_real is True:
            inner = value(node.args[0])
            return sp.exp(inner) if inner is not None else None
        if node.func is sp.log:
            inner = value(node.args[0])
            if inner is sp.oo:
                return sp.oo
            if inner == 0 and node.args[0].is_positive is True:
                return -sp.oo
            if inner is not None and inner.is_positive is True:
                return sp.log(inner)
        if (
            node.func is sp.gamma
            and node.args[0].is_real is True
            and value(node.args[0]) is sp.oo
        ):
            return sp.oo
        return None

    result = value(expr)
    if (
        result is None
        or result.has(x, sp.nan, sp.zoo, sp.AccumBounds)
        or result.is_extended_real is not True
    ):
        return None
    return answer(
        result,
        "real_monotone_extended_composition",
        "real exp/log and eventually positive divergent gamma arguments compose through resolved extended limits; zero-times-infinity and opposite-infinity sums are declined",
    )


def bounded_oscillatory_envelope_certificate(expr, x, point):
    if point is not sp.oo or not expr.has(sp.sin, sp.cos, sp.cosh):
        return None
    if sp.count_ops(expr) > 200:
        return None
    result = 0
    for term in sp.Add.make_args(sp.expand_mul(expr)):
        atoms = term.atoms(sp.sin, sp.cos, sp.cosh)
        if not atoms:
            value = bounded_limit(term, x, point, allow_general=True)
            if (
                value is None
                or value.is_finite is not True
                or value.has(sp.AccumBounds, sp.Limit, sp.nan, sp.zoo)
            ):
                return None
            result += value
            continue
        replacements = {}
        for atom in atoms:
            argument = atom.args[0]
            if atom.func in (sp.sin, sp.cos):
                if argument.is_real is not True:
                    return None
            else:
                realpart = sp.simplify(sp.re(argument))
                if realpart.has(x) or realpart.is_finite is not True:
                    return None
            marker = sp.Dummy("bounded_oscillator")
            replacements[atom] = marker
        transformed = term.xreplace(replacements)
        markers = tuple(replacements.values())
        if not transformed.is_polynomial(*markers):
            return None
        for powers, coefficient in sp.Poly(transformed, *markers).terms():
            value = bounded_limit(coefficient, x, point, allow_general=True)
            if any(powers):
                if value != 0:
                    return None
            elif value is None or value.is_finite is not True:
                return None
            else:
                result += value
    return answer(
        result,
        "bounded_oscillatory_envelope",
        "real sin/cos and cosh with constant real part are uniformly bounded; every nonconstant oscillator coefficient tends to zero",
    )


def integer_floor_side_certificate(expr, x, point):
    if point != 0 or x.is_positive is not True:
        return None
    atoms = expr.atoms(sp.floor, sp.ceiling)
    if not atoms:
        return None
    replacements = {}
    for atom in atoms:
        try:
            p = sp.Poly(atom.args[0], x)
        except sp.PolynomialError:
            return None
        if p.degree() != 1:
            return None
        m, c = p.nth(0), p.nth(1)
        if m.is_integer is not True:
            return None
        if c.is_positive is True:
            value = m if atom.func is sp.floor else m + 1
        elif c.is_negative is True:
            value = m - 1 if atom.func is sp.floor else m
        else:
            return None
        replacements[atom] = value
    value = expr.xreplace(replacements)
    if value.has(x) or value.is_finite is not True:
        return None
    return answer(
        value,
        "floor_affine_positive_side",
        "floor(m+c*u) is m for c>0 and m-1 for c<0 when the positive local parameter is sufficiently small",
    )


def gamma_reciprocal_shift_certificate(expr, x, point):
    """Resolve tiny gamma shifts using a derivative and explicit remainder bound."""
    if point is not sp.oo or not expr.has(sp.gamma) or sp.count_ops(expr) > 200:
        return None
    base = sp.gamma(x)
    if base not in expr.atoms(sp.gamma):
        return None
    atoms = expr.atoms(sp.gamma) - {base}
    if not atoms:
        return None
    psi = (
        sp.log(x) - 1 / (2 * x) - 1 / (12 * x * x) + 1 / (120 * x**4) - 1 / (252 * x**6)
    )
    replacements = {}
    markers = []
    for atom in atoms:
        c = sp.cancel((atom.args[0] - x) * base)
        if (
            c.has(x)
            or c.is_number is not True
            or c.is_real is not True
            or c.is_finite is not True
        ):
            return None
        marker = sp.Dummy("gamma_shift_remainder")
        replacements[atom] = base + c * psi + marker
        markers.append(marker)
    transformed = sp.expand(expr.xreplace(replacements))

    def moderate_coefficient(coefficient):
        for term in sp.Add.make_args(sp.expand(coefficient)):
            power = 0
            for factor in sp.Mul.make_args(term):
                b, e = factor.as_base_exp()
                if factor.is_number and factor.is_finite is True:
                    continue
                if e.is_Integer is not True:
                    return False
                if b == x:
                    power += e
                elif b != sp.log(x):
                    return False
            if power > 5:
                return False
        return True

    for marker in markers:
        coefficient = transformed.coeff(marker)
        if coefficient.has(*markers) or not moderate_coefficient(coefficient):
            return None
    remainder_free = sp.expand(
        transformed - sum(transformed.coeff(r) * r for r in markers)
    )
    if remainder_free.has(*markers, sp.gamma):
        return None
    value = bounded_limit(remainder_free, x, point, allow_general=True)
    if value is None or value.has(x, sp.AccumBounds, sp.Limit, sp.nan, sp.zoo):
        return None
    return answer(
        value,
        "gamma_reciprocal_shift_remainder",
        "Taylor expansion gives Gamma(x+c/Gamma(x))-Gamma(x)=c*psi(x)+O(log(x)**2/Gamma(x)); the digamma remainder is O(x**-8). Every coefficient grows at most x**5 times a fixed log power, so both multiplied remainders vanish",
    )


def minmax_interval_dominance_certificate(expr, x, point):
    if point is not sp.oo or expr.func not in (sp.Min, sp.Max):
        return None

    def interval(node):
        if not node.has(x) and node.is_real is True and node.is_finite is True:
            return node, node
        if node.func in (sp.sin, sp.cos) and node.args[0].is_real is True:
            if node.func is sp.cos:
                inner = interval(node.args[0])
                if (
                    inner
                    and (inner[0] >= -1) is sp.S.true
                    and (inner[1] <= 1) is sp.S.true
                ):
                    square = sp.Max(inner[0] ** 2, inner[1] ** 2)
                    return 1 - square / 2, sp.S.One
            return -sp.S.One, sp.S.One
        if (
            node.func is sp.Mod
            and node.args[0].is_real is True
            and not node.args[1].has(x)
            and node.args[1].is_positive is True
        ):
            return sp.S.Zero, node.args[1]
        if node.is_Add:
            values = [interval(a) for a in node.args]
            if any(v is None for v in values):
                return None
            return sum(a for a, b in values), sum(b for a, b in values)
        if node.is_Mul:
            lo = hi = sp.S.One
            for factor in node.args:
                bounds = interval(factor)
                if bounds is None:
                    return None
                products = [
                    lo * bounds[0],
                    lo * bounds[1],
                    hi * bounds[0],
                    hi * bounds[1],
                ]
                lo, hi = sp.Min(*products), sp.Max(*products)
            return lo, hi
        if node.is_Pow and node.exp == 2:
            bounds = interval(node.base)
            if bounds is None:
                return None
            lo, hi = bounds
            if (lo <= 0) is sp.S.true and (hi >= 0) is sp.S.true:
                return sp.S.Zero, sp.Max(lo * lo, hi * hi)
            return sp.Min(lo * lo, hi * hi), sp.Max(lo * lo, hi * hi)
        return None

    for candidate in expr.args:
        if not candidate.is_rational_function(x):
            continue
        value = bounded_limit(candidate, x, point, allow_general=True)
        if value is None or value.is_real is not True or value.is_finite is not True:
            continue
        for other in expr.args:
            if other == candidate:
                continue
            bounds = interval(other)
            if bounds is None:
                break
            gap = bounds[0] - value if expr.func is sp.Min else value - bounds[1]
            if gap.is_positive is not True:
                break
        else:
            return answer(
                value,
                "minmax_strict_interval_dominance",
                "a rational branch has a finite limit strictly below every other Min lower bound (or above every Max upper bound), so it is eventually the selected branch. Bounds are used for dominance only, never to assert attained clusters",
            )
    return None
