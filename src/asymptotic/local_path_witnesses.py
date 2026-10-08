"""Bounded attained subsequences for local meromorphic nonexistence proofs."""

from itertools import product

import sympy as sp

from ._limit_composition import _exact_equal
from ._multivariate_analytic import _certified_real_analytic
from ._polynomial_bounds import bounded_degree
from .limit_models import LimitEvidence
from .local_expansion import adaptive_path_limit, local_series


def _nonzero_germ(expression, variable):
    """Prove eventual nonvanishing from an analytic leading term and remainder."""
    if expression.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
        return False
    if not _certified_real_analytic(expression, (variable,)):
        return False
    # Intrinsic assumptions can prove nonvanishing on the positive path
    # without a Taylor jet, including monomials beyond the jet depth.
    if expression.is_zero is False:
        return True
    expansion = local_series(expression, variable, depth=8)
    if expansion is None:
        return False
    try:
        polynomial = sp.Poly(expansion.prefix, variable)
    except sp.PolynomialError:
        return False
    if polynomial.is_zero:
        return False
    exponent, coefficient = min(polynomial.terms(), key=lambda item: item[0][0])
    if coefficient.is_zero is not False:
        return False
    if expansion.order is None:
        return expression.is_polynomial(variable) is True
    # The common series provider supplies O(t**8); it must be smaller than
    # the nonzero leading monomial to exclude zeros on a punctured interval.
    return (
        sp.Order(expansion.order, variable).contains(variable ** exponent[0]) is False
    )


def local_path_conflict(expr, variables, target, domain, assumptions):
    """Return two attained, pole-avoiding subsequences with distinct limits.

    Only unrestricted real coordinates at the origin and a small elementary
    expression tree are eligible. Analyticity and nonzero denominator germs
    prove that t=1/j stays in the expression domain for all sufficiently large
    integers j. Failure of the bounded search provides no existence conclusion.
    """
    variables = tuple(variables)
    if (
        not isinstance(expr, sp.Expr)
        or len(variables) not in (2, 3)
        or tuple(target) != (0,) * len(variables)
        or domain is not sp.S.true
        or assumptions is not sp.S.true
        or expr.free_symbols - set(variables)
        or expr.has(sp.Float)
        or sp.count_ops(expr) > 40
        or any(
            v.assumptions0 != sp.Symbol(v.name, real=True).assumptions0
            for v in variables
        )
    ):
        return None
    allowed = {
        sp.sin,
        sp.cos,
        sp.tan,
        sp.sinh,
        sp.cosh,
        sp.tanh,
        sp.exp,
        sp.atan,
        sp.Abs,
    }
    if not expr.has(sp.Function) or any(
        atom.func not in allowed for atom in expr.atoms(sp.Function)
    ):
        return None
    # Symmetric germs are left to the uniform analytic certificates; probing
    # their repeated equal path limits can be costly under high cancellation.
    if expr.xreplace({v: -v for v in variables}) == expr:
        return None
    t = sp.Dummy("local_path_parameter", positive=True)
    paths = []
    for i in range(len(variables)):
        for sign in (1, -1):
            paths.append(
                tuple(sign * t if k == i else sp.S.Zero for k in range(len(variables)))
            )
    paths.extend([tuple(t for _ in variables), tuple(-t for _ in variables)])
    if len(variables) == 2:
        paths.extend([(t, t**2), (t**2, t)])
    inverse_index = sp.Dummy("attained_path_index", positive=True, integer=True)
    witnessed = []
    denominators = [
        power.base for power in expr.atoms(sp.Pow) if power.exp.is_negative is True
    ]
    for path in paths:
        substitutions = dict(zip(variables, path, strict=True))
        guards = [base.subs(substitutions, simultaneous=True) for base in denominators]
        if any(not _nonzero_germ(guard, t) for guard in guards):
            continue
        along = expr.subs(substitutions, simultaneous=True)
        if along.has(sp.nan, sp.zoo):
            continue
        numerator, denominator = sp.fraction(sp.together(along))
        if not _certified_real_analytic(numerator, (t,)) or not _nonzero_germ(
            denominator, t
        ):
            continue
        if along == 0:
            value = sp.S.Zero
        else:
            expanded = adaptive_path_limit(along, t, depths=(2,))
            if expanded is None:
                continue
            value = expanded[0]
        if value.is_extended_real is not True:
            continue
        sequence = tuple(
            (v, p.subs(t, 1 / inverse_index))
            for v, p in zip(variables, path, strict=True)
        )
        evidence = LimitEvidence(
            "attained_local_path",
            "The exact t=1/j subsequence approaches the origin. Nonzero analytic "
            "leading denominator terms and smaller remainders exclude poles for "
            "all sufficiently large j; a vanishing expansion remainder certifies its limit.",
            sequence,
            value,
        )
        for previous in witnessed:
            if _exact_equal(previous.value, value) is False:
                return previous, evidence
        witnessed.append(evidence)
    return None


def reciprocal_phase_conflict(expr, variables, target, domain, assumptions):
    """Lift two attained reciprocal-power sine/cosine phases along a real axis.

    The axis restriction must be affine in one oscillator, with a continuous
    finite offset and a continuous nonzero amplitude. Analytic denominator
    germs separately exclude poles before the explicit phase sequences are used.
    """
    from ._limit_composition import _regular_composition

    target = tuple(map(sp.sympify, target))

    if (
        not isinstance(expr, sp.Expr)
        or len(variables) not in (2, 3)
        or len(target) != len(variables)
        or any(
            p.is_number is not True
            or p.is_real is not True
            or p.is_finite is not True
            or p.has(sp.Float)
            for p in target
        )
        or domain is not sp.S.true
        or assumptions is not sp.S.true
        or expr.free_symbols - set(variables)
        or expr.has(sp.Float)
        or sp.count_ops(expr) > 40
        or any(
            v.assumptions0 != sp.Symbol(v.name, real=True).assumptions0
            for v in variables
        )
        or not any(
            power.exp.is_negative is True
            for atom in expr.atoms(sp.sin, sp.cos)
            for power in atom.args[0].atoms(sp.Pow)
        )
    ):
        return None
    t = sp.Dummy("phase_axis_parameter", positive=True)
    q = sp.Dummy("oscillator_value", real=True)
    index = sp.Dummy("attained_phase_index", positive=True, integer=True)
    denominators = [
        power.base for power in expr.atoms(sp.Pow) if power.exp.is_negative is True
    ]
    denominators.extend(atom.args[0] for atom in expr.atoms(sp.log))
    denominators.extend(sp.cos(atom.args[0]) for atom in expr.atoms(sp.tan))
    for axis in range(len(variables)):
        path = tuple(p + (t if i == axis else sp.S.Zero) for i, p in enumerate(target))
        substitutions = dict(zip(variables, path, strict=True))
        if any(
            not _nonzero_germ(base.subs(substitutions, simultaneous=True), t)
            for base in denominators
        ):
            continue
        along = expr.subs(substitutions, simultaneous=True)
        oscillators = list(along.atoms(sp.sin, sp.cos))
        for atom in oscillators:
            coefficient, exponent = atom.args[0].as_coeff_exponent(t)
            if (
                coefficient.is_Rational is not True
                or coefficient == 0
                or exponent not in (-1, -2, -3, -4)
            ):
                continue
            skeleton = along.xreplace({atom: q})
            if bounded_degree(skeleton, (q,), 1) is None:
                continue
            try:
                polynomial = sp.Poly(skeleton, q)
            except sp.PolynomialError:
                continue
            if polynomial.degree() != 1:
                continue
            amplitude = _regular_composition(polynomial.coeff_monomial(q), (t,), (0,))
            offset = _regular_composition(polynomial.coeff_monomial(1), (t,), (0,))
            if (
                amplitude is None
                or offset is None
                or amplitude.is_zero is not False
                or amplitude.is_real is not True
                or offset.is_real is not True
            ):
                continue
            if atom.func is sp.sin:
                phases = (
                    sp.pi / 2 + 2 * sp.pi * index,
                    3 * sp.pi / 2 + 2 * sp.pi * index,
                )
                values = (sp.sign(coefficient), -sp.sign(coefficient))
            else:
                phases = (2 * sp.pi * index, sp.pi + 2 * sp.pi * index)
                values = (sp.S.One, -sp.S.One)
            evidence = []
            for phase, value in zip(phases, values, strict=True):
                radius = (sp.Abs(coefficient) / phase) ** (-1 / exponent)
                sequence = tuple(
                    (v, p.subs(t, radius)) for v, p in zip(variables, path, strict=True)
                )
                evidence.append(
                    LimitEvidence(
                        "attained_reciprocal_phase",
                        "The positive radius about the exact finite target, (abs(c)/theta_j)**(1/p), approaches zero "
                        "and attains the prescribed sine/cosine phase exactly. Nonzero analytic "
                        "denominator germs avoid poles eventually; continuous amplitude and "
                        "offset give distinct subsequential limits.",
                        sequence,
                        amplitude * value + offset,
                    )
                )
            return tuple(evidence)
    return None


def parameter_monomial_conflict(expr, variables, target, domain, assumptions):
    """Construct attained monomial paths for finite real parameter families.

    A parameter-independent monomial denominator never vanishes on signed
    nonzero monomial paths. Exact weighted polynomial terms determine each
    path limit. Divergence is accepted only when the leading coefficient has
    a certified sign for every parameter value allowed by its assumptions.
    """
    if (
        not isinstance(expr, sp.Expr)
        or len(variables) != 2
        or tuple(target) != (0, 0)
        or domain is not sp.S.true
        or assumptions is not sp.S.true
        or expr.has(sp.Float)
        or sp.count_ops(expr) > 30
        or any(
            v.assumptions0 != sp.Symbol(v.name, real=True).assumptions0
            for v in variables
        )
    ):
        return None
    parameters = expr.free_symbols - set(variables)
    if not parameters or any(
        p.is_real is not True or p.is_finite is not True for p in parameters
    ):
        return None
    numerator, denominator = sp.fraction(expr)
    if (
        bounded_degree(numerator, variables, 12) is None
        or bounded_degree(denominator, variables, 12) is None
    ):
        return None
    try:
        polynomial = sp.Poly(numerator, *variables, domain=sp.EX)
        divisor = sp.Poly(denominator, *variables, domain=sp.QQ)
    except (sp.PolynomialError, sp.polys.polyerrors.CoercionFailed):
        return None
    if len(divisor.terms()) != 1 or divisor.is_zero:
        return None
    if any(
        c.is_real is not True or c.is_finite is not True for c in polynomial.coeffs()
    ):
        return None
    divisor_powers, divisor_coefficient = divisor.terms()[0]
    weights = [(1, 1)]
    weights.extend(pair for k in range(2, 7) for pair in ((1, k), (k, 1)))
    index = sp.Dummy("attained_parameter_index", positive=True, integer=True)
    witnessed = []
    for weight in weights:
        divisor_degree = sum(w * n for w, n in zip(weight, divisor_powers, strict=True))
        for signs in product((-1, 1), repeat=2):
            divisor_sign = divisor_coefficient * sp.prod(
                s**n for s, n in zip(signs, divisor_powers, strict=True)
            )
            groups = {}
            for powers, coefficient in polynomial.terms():
                degree = sum(w * n for w, n in zip(weight, powers, strict=True))
                coefficient *= (
                    sp.prod(s**n for s, n in zip(signs, powers, strict=True))
                    / divisor_sign
                )
                groups[degree] = groups.get(degree, sp.S.Zero) + coefficient
            leading = next(
                (
                    (degree, coefficient)
                    for degree, coefficient in sorted(groups.items())
                    if coefficient.is_zero is not True
                ),
                None,
            )
            if leading is None:
                value = sp.S.Zero
            else:
                degree, coefficient = leading
                order = degree - divisor_degree
                if order > 0:
                    value = sp.S.Zero
                elif order == 0:
                    value = coefficient
                elif coefficient.is_positive is True:
                    value = sp.oo
                elif coefficient.is_negative is True:
                    value = -sp.oo
                else:
                    continue
            sequence = tuple(
                (v, s / index**w)
                for v, s, w in zip(variables, signs, weight, strict=True)
            )
            evidence = LimitEvidence(
                "attained_monomial_parameter_path",
                "The signed monomial sequence approaches the origin and all coordinates "
                "are nonzero for every positive integer j. The exact monomial denominator "
                "therefore avoids poles independently of the finite real parameters. "
                "Weighted polynomial terms certify the limit; divergent leading terms "
                "have a parameter-uniform sign.",
                sequence,
                value,
            )
            for previous in witnessed:
                if _exact_equal(previous.value, value) is False:
                    return previous, evidence
            witnessed.append(evidence)
    return None


def _polynomial_pole_paths(numerator, denominator, variables, parameter):
    """Build a ray and a polynomial curve next to a rational linear pole.

    A degree-at-least-two phase divided by a linear form vanishes on a regular
    ray. If the phase restricts nontrivially to the pole plane, a transverse
    displacement of order degree+1 makes the quotient diverge on a defined
    curve. A bounded rational grid searches that plane; no hit stays unresolved.
    """
    if (
        bounded_degree(numerator, variables, 4) is None
        or bounded_degree(denominator, variables, 1) is None
    ):
        return None
    try:
        phase = sp.Poly(numerator, *variables, domain=sp.QQ)
        divisor = sp.Poly(denominator, *variables, domain=sp.QQ)
    except (sp.PolynomialError, sp.polys.polyerrors.CoercionFailed):
        return None
    if (
        phase.is_zero
        or not 2 <= phase.total_degree() <= 4
        or min(sum(p) for p, _ in phase.terms()) < 2
        or len(phase.terms()) > 12
        or divisor.total_degree() != 1
        or divisor.coeff_monomial(1) != 0
    ):
        return None
    coefficients = tuple(divisor.coeff_monomial(v) for v in variables)
    pivot = next(i for i, c in enumerate(coefficients) if c != 0)
    transverse = tuple(
        1 / coefficients[i] if i == pivot else sp.S.Zero for i in range(len(variables))
    )
    exponent = phase.total_degree() + 1
    for coordinates in product((0, 1, 2), repeat=len(variables) - 1):
        direction = list(map(sp.sympify, coordinates))
        direction.insert(pivot, sp.S.Zero)
        direction[pivot] = (
            -sum(c * v for c, v in zip(coefficients, direction, strict=True))
            / coefficients[pivot]
        )
        restricted = sp.Poly(
            phase.as_expr().xreplace(
                dict(zip(variables, (v * parameter for v in direction), strict=True))
            ),
            parameter,
            domain=sp.QQ,
        )
        if restricted.is_zero:
            continue
        regular = tuple(w * parameter for w in transverse)
        curve = tuple(
            v * parameter + w * parameter**exponent
            for v, w in zip(direction, transverse, strict=True)
        )
        along = sp.Poly(
            phase.as_expr().xreplace(dict(zip(variables, curve, strict=True))),
            parameter,
            domain=sp.QQ,
        )
        (order,), coefficient = min(along.terms(), key=lambda row: row[0][0])
        if not 0 < order < exponent:
            continue
        return regular, curve, coefficient, order, exponent
    return None


def polynomial_pole_conflict(expr, variables, target, domain, assumptions):
    """Prove nonexistence with attained approaches to a linear pole plane.

    Analytic first-order functions of a polynomial phase over a linear
    denominator and real exponentials of such quotients share the same exact
    geometry. Original denominators and tangent poles are checked on both
    approaches. This is a conflict certificate; a failed search proves nothing.
    """
    from ._limit_composition import _regular_composition

    if (
        not isinstance(expr, sp.Expr)
        or len(variables) not in (2, 3)
        or tuple(target) != (0,) * len(variables)
        or domain is not sp.S.true
        or assumptions is not sp.S.true
        or expr.free_symbols - set(variables)
        or expr.has(sp.Float)
        or sp.count_ops(expr) > 40
        or any(
            v.assumptions0 != sp.Symbol(v.name, real=True).assumptions0
            for v in variables
        )
        or any(
            a.func
            not in (sp.sin, sp.cos, sp.tan, sp.sinh, sp.cosh, sp.exp, sp.atan, sp.Abs)
            for a in expr.atoms(sp.Function)
        )
    ):
        return None
    parameter = sp.Dummy("pole_curve_parameter", positive=True)
    numerator, denominator = sp.fraction(expr)
    coefficient, head = numerator.as_coeff_Mul()
    mode = None
    if (
        coefficient.is_Rational
        and coefficient != 0
        and head.func in (sp.sin, sp.tan, sp.sinh, sp.atan)
    ):
        geometry = _polynomial_pole_paths(
            head.args[0], denominator, variables, parameter
        )
        # sin, tan, sinh and atan all satisfy f(u) = u + O(u**3) at zero.
        if geometry is None:
            return None
        mode = "first_order"
        offset = sp.S.Zero
    else:
        for term in sp.Add.make_args(expr):
            coefficient, head = term.as_coeff_Mul()
            if (
                coefficient.is_Rational is not True
                or coefficient == 0
                or head.func is not sp.exp
            ):
                continue
            phase, divisor = sp.fraction(head.args[0])
            geometry = _polynomial_pole_paths(phase, divisor, variables, parameter)
            if geometry is None:
                continue
            offset = _regular_composition(expr - term, variables, target)
            if (
                offset is not None
                and offset.is_real is True
                and offset.is_finite is True
            ):
                mode = "exponential"
                break
    if mode is None:
        return None
    regular, curve, leading, order, pole_order = geometry
    if mode == "first_order":
        first = sp.S.Zero
        second = sp.sign(coefficient * leading) * sp.oo
    else:
        first = offset + coefficient
        second = offset + (coefficient * sp.oo if leading > 0 else sp.S.Zero)
    if _exact_equal(first, second) is not False:
        return None
    guards = [a.base for a in expr.atoms(sp.Pow) if a.exp.is_negative is True]
    guards.extend(sp.cos(a.args[0]) for a in expr.atoms(sp.tan))
    guards.extend(a.args[0] for a in expr.atoms(sp.log))
    index = sp.Dummy("attained_pole_index", positive=True, integer=True)
    evidence = []
    for path, value in ((regular, first), (curve, second)):
        chart = dict(zip(variables, path, strict=True))
        if any(not _nonzero_germ(g.xreplace(chart), parameter) for g in guards):
            return None
        evidence.append(
            LimitEvidence(
                "attained_polynomial_pole",
                f"The regular ray has a phase of order at least two and a linear denominator. "
                f"The pole curve has denominator t**{pole_order} and phase leading term {leading}*t**{order}. "
                "Analytic first-order germs or real exponential endpoint limits give the distinct values. "
                "The exact t=1/j sequences stay in the original domain: nonzero analytic denominator germs "
                "and tangent cosines avoid poles eventually.",
                tuple(
                    (v, p.xreplace({parameter: 1 / index}))
                    for v, p in zip(variables, path, strict=True)
                ),
                value,
            )
        )
    return tuple(evidence)
