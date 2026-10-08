"""Analytic, divided-difference, Taylor, and growth-scale certificates."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import sympy as sp
from sympy.core.function import AppliedUndef
from sympy.core.relational import Relational

from ._multivariate_germ import (
    ParityCertificate,
    _coercive_polynomial_model,
    _denominator_zero_directions,
    _leading_homogeneous,
    _obvious_definite,
    _shift,
    _shift_for,
    _weighted_leading,
    _weighted_poly_order,
)
from ._symbolic_policy import bounded_ask, bounded_limit


def _constant_wrt(expr, variables):
    return not (sp.sympify(expr).free_symbols & set(variables))


def _analytic_unary_derivative(func, center):
    """Return a certified finite derivative for a supported real-analytic function."""
    q = sp.Dummy("_q", real=True)
    if func in (sp.sin, sp.cos, sp.sinh, sp.cosh, sp.exp, sp.asinh, sp.atanh):
        pass
    elif func is sp.tan:
        if sp.simplify(sp.cos(center)).is_zero is not False:
            return None
    elif func is sp.log:
        if sp.sympify(center).is_positive is not True:
            return None
    else:
        return None
    try:
        value = sp.simplify(sp.diff(func(q), q).subs(q, center))
    except (TypeError, ValueError, NotImplementedError):
        return None
    if value.has(sp.nan, sp.zoo) or value.is_finite is False:
        return None
    return value


def _function_difference_signatures(expr, variables, target):
    """Certified first/nonzero-jet normal forms for an analytic difference.

    Each returned pair ``(delta, scale, proof)`` means that the expression is
    ``scale*delta*(1+o(1))`` relative to the punctured set ``delta != 0``.
    The first-derivative case is the analytic divided-difference theorem.
    For cos/cosh at zero we additionally use their analytic dependence on
    the squared argument, which makes the second-jet reduction uniform even
    when the two inner arguments coalesce to unusually high order.
    """
    out = []
    terms = sp.Add.make_args(sp.expand(expr))
    if len(terms) != 2:
        return out

    variants = []
    odd = (sp.sin, sp.tan, sp.sinh)
    for term in terms:
        coeff, body = term.as_independent(*variables, as_Add=False)
        if not _constant_wrt(coeff, variables) or not getattr(
            body, "is_Function", False
        ):
            return out
        local = [(sp.sympify(coeff), body.func, body.args[0])]
        if body.func in odd:
            local.append((-sp.sympify(coeff), body.func, -body.args[0]))
        variants.append(local)

    point = dict(zip(variables, target, strict=True))
    for c1, f1, a1 in variants[0]:
        for c2, f2, a2 in variants[1]:
            if f1 is not f2 or sp.simplify(c1 + c2) != 0:
                continue
            center1 = sp.simplify(a1.subs(point))
            center2 = sp.simplify(a2.subs(point))
            if sp.simplify(center1 - center2) != 0:
                continue
            derivative = _analytic_unary_derivative(f1, center1)
            if derivative is not None and derivative != 0:
                out.append(
                    (
                        sp.expand(a1 - a2),
                        sp.simplify(c1 * derivative),
                        "analytic first divided-difference remainder",
                    )
                )
            if f1 in (sp.cos, sp.cosh) and center1 == 0:
                second = sp.diff(f1(sp.Symbol("_q")), sp.Symbol("_q"), 2).subs(
                    sp.Symbol("_q"), 0
                )
                if second != 0:
                    out.append(
                        (
                            sp.expand(a1**2 - a2**2),
                            sp.simplify(c1 * second / 2),
                            (
                                "even analytic function is analytic in the squared "
                                "argument"
                            ),
                        )
                    )
    return out


def generalized_divided_difference_certificate(
    expr, variables, target, *, _context=None
):
    """Certify divided differences with nonlinear/multivariate inner arguments."""
    expr = sp.sympify(expr)
    numerator, denominator = sp.fraction(sp.cancel(expr))
    numerator_forms = [(numerator, sp.S.One, "exact numerator")]
    denominator_forms = [(denominator, sp.S.One, "exact denominator")]
    numerator_forms.extend(
        _function_difference_signatures(numerator, variables, target)
    )
    denominator_forms.extend(
        _function_difference_signatures(denominator, variables, target)
    )

    for delta_n, scale_n, proof_n in numerator_forms:
        for delta_d, scale_d, proof_d in denominator_forms:
            if delta_n == numerator and delta_d == denominator:
                continue
            try:
                ratio = sp.cancel(delta_n / delta_d)
            except (TypeError, ValueError, ZeroDivisionError, sp.PolynomialError):
                continue
            if not _constant_wrt(ratio, variables):
                continue
            value = sp.simplify(scale_n * ratio / scale_d)
            if value.has(sp.nan, sp.zoo):
                continue
            return ParityCertificate(
                "generalized_divided_difference",
                True,
                value,
                "generalized analytic divided differences have a common certified "
                "inner increment",
                (delta_n, delta_d, proof_n, proof_d),
            )
    return ParityCertificate("generalized_divided_difference", False)


def _certified_real_analytic(expr, variables):
    """Conservative structural real-analyticity certificate near the origin."""
    expr = sp.sympify(expr)
    if not expr.has(*variables):
        return True
    if expr.is_Symbol:
        return expr in variables
    if expr.is_Add or expr.is_Mul:
        return all(_certified_real_analytic(arg, variables) for arg in expr.args)
    if expr.is_Pow:
        base, exponent = expr.as_base_exp()
        if not _certified_real_analytic(base, variables) or not exponent.is_number:
            return False
        base0 = sp.simplify(base.subs(dict.fromkeys(variables, 0)))
        if exponent.is_integer is True:
            return exponent.is_nonnegative is True or base0.is_zero is False
        if exponent.is_Rational is True:
            return base0.is_positive is True
        return False
    if getattr(expr, "is_Function", False):
        if not all(_certified_real_analytic(arg, variables) for arg in expr.args):
            return False
        if len(expr.args) != 1:
            return False
        arg0 = sp.simplify(expr.args[0].subs(dict.fromkeys(variables, 0)))
        if expr.func in (sp.sin, sp.cos, sp.sinh, sp.cosh, sp.exp, sp.asinh):
            return True
        if expr.func in (sp.atan, sp.tanh):
            return arg0.is_real is True and arg0.is_finite is True
        if expr.func is sp.tan:
            return sp.simplify(sp.cos(arg0)).is_zero is False
        if expr.func is sp.log:
            return arg0.is_positive is True
    return False


def _tensor_taylor_trunc(expr, variables, order):
    out = sp.sympify(expr)
    for variable in variables:
        out = sp.series(out, variable, 0, order + 1).removeO()
    return sp.expand(out)


def analytic_equivalence_rewrite_certificate(
    expr, variables, target, *, order=12, _context=None
):
    """Try bounded Taylor orders until a uniform remainder proves the limit.

    Each attempted order independently checks coercivity and the remainder
    inequality. A successful low-order jet avoids constructing unused terms.
    """
    if type(order) is not int or order < 1:
        raise ValueError("Taylor order must be a positive integer")
    for degree in sorted({min(degree, order) for degree in (2, 4, 8, 12, order)}):
        certificate = _analytic_equivalence_at_order(
            expr, variables, target, order=degree, _context=_context
        )
        if certificate.certified:
            return certificate
    return ParityCertificate("analytic_equivalence_rewrite", False)


def _analytic_equivalence_at_order(expr, variables, target, *, order, _context=None):
    """Replace analytic germs by first relevant Taylor jets with a remainder proof.

    The certificate uses a tensor-product Taylor polynomial.  Analyticity gives a
    uniform remainder in which every omitted term has exponent at least
    ``order + 1`` in some coordinate.  Against a positive weight vector ``w``
    this is ``O(r**((order+1)*min(w)))``.  Hence a coercive polynomial
    denominator model of lower weighted degree is stable under the omitted
    analytic remainder, and candidate subtraction is certified in the same way.
    """
    u, f = _shift_for(expr, variables, target, _context)
    numerator, denominator = sp.fraction(f)
    if numerator.is_polynomial(*u) and denominator.is_polynomial(*u):
        return ParityCertificate("analytic_equivalence_rewrite", False)
    if not (
        _certified_real_analytic(numerator, u)
        and _certified_real_analytic(denominator, u)
    ):
        return ParityCertificate("analytic_equivalence_rewrite", False)
    try:
        tn = _tensor_taylor_trunc(numerator, u, order)
        td = _tensor_taylor_trunc(denominator, u, order)
    except (TypeError, ValueError, NotImplementedError, sp.PoleError):
        return ParityCertificate("analytic_equivalence_rewrite", False)
    if not (tn.is_polynomial(*u) and td.is_polynomial(*u)):
        return ParityCertificate("analytic_equivalence_rewrite", False)

    model = _coercive_polynomial_model(td, u)
    if model is None:
        return ParityCertificate("analytic_equivalence_rewrite", False)
    weights, degree, sign, principal, provider = model
    remainder_order = (order + 1) * min(weights)
    if remainder_order <= degree:
        return ParityCertificate("analytic_equivalence_rewrite", False)

    numerator_order, numerator_lead = _weighted_leading(tn, u, weights)
    if numerator_order is sp.oo or numerator_order > degree:
        return ParityCertificate(
            "analytic_equivalence_rewrite",
            True,
            sp.S.Zero,
            "analytic numerator jet and certified uniform remainder are higher "
            "order than a coercive denominator jet",
            (tn, td, weights, degree, remainder_order, provider),
        )
    if numerator_order != degree:
        return ParityCertificate("analytic_equivalence_rewrite", False)
    denominator_order, denominator_lead = _weighted_leading(td, u, weights)
    if denominator_order != degree:
        return ParityCertificate("analytic_equivalence_rewrite", False)
    try:
        candidate = sp.cancel(numerator_lead / denominator_lead)
    except (TypeError, ValueError, ZeroDivisionError, sp.PolynomialError):
        return ParityCertificate("analytic_equivalence_rewrite", False)
    if not _constant_wrt(candidate, u):
        return ParityCertificate("analytic_equivalence_rewrite", False)
    residual = sp.expand(tn - candidate * td)
    residual_order = _weighted_poly_order(residual, u, weights)
    if residual_order is not sp.oo and residual_order <= degree:
        return ParityCertificate("analytic_equivalence_rewrite", False)
    return ParityCertificate(
        "analytic_equivalence_rewrite",
        True,
        sp.simplify(candidate),
        "candidate-centered analytic jet replacement has a certified higher-order "
        "multivariate remainder",
        (
            tn,
            td,
            weights,
            degree,
            remainder_order,
            residual_order,
            sign,
            principal,
            provider,
        ),
    )


def _polynomial_path_valuation(poly, variables, path, t):
    substituted = sp.expand(
        sp.sympify(poly).subs(
            dict(zip(variables, path, strict=True)), simultaneous=True
        )
    )
    try:
        p = sp.Poly(substituted, t)
    except (sp.PolynomialError, TypeError, ValueError):
        return None
    if p.is_zero:
        return sp.oo, sp.S.Zero
    exponent = min(monom[0] for monom, coeff in p.terms() if coeff != 0)
    coefficient = sp.simplify(
        sp.Add(*(coeff for monom, coeff in p.terms() if monom[0] == exponent))
    )
    return exponent, coefficient


def _analytic_taylor_path_limit(tn, td, variables, path, order, t):
    """Certified path limit when tensor-Taylor data determine leading terms."""
    nonzero_path_orders = []
    for component in path:
        data = _polynomial_path_valuation(component, (t,), (t,), t)
        if data is None:
            return None
        valuation, _ = data
        if valuation is not sp.oo:
            nonzero_path_orders.append(valuation)
    if not nonzero_path_orders:
        return None
    remainder_order = (order + 1) * min(nonzero_path_orders)
    numerator_data = _polynomial_path_valuation(tn, variables, path, t)
    denominator_data = _polynomial_path_valuation(td, variables, path, t)
    if numerator_data is None or denominator_data is None:
        return None
    on, cn = numerator_data
    od, cd = denominator_data
    if od is sp.oo or od >= remainder_order:
        return None
    numerator_order = remainder_order if on is sp.oo else min(on, remainder_order)
    if numerator_order > od:
        return sp.S.Zero
    if on is sp.oo or on >= remainder_order:
        return None
    if on == od:
        value = sp.simplify(cn / cd)
        return None if value.has(sp.nan, sp.zoo) else value
    if on < od:
        leading = sp.simplify(cn / cd)
        if leading.is_positive is True:
            return sp.oo
        if leading.is_negative is True:
            return -sp.oo
    return None


def analytic_jet_path_conflict_certificate(
    expr, variables, target, *, order=6, _context=None
):
    """Prove analytic DNE from two Taylor-certified accumulating paths."""
    u, f = _shift_for(expr, variables, target, _context)
    numerator, denominator = sp.fraction(f)
    if numerator.is_polynomial(*u) and denominator.is_polynomial(*u):
        return ParityCertificate("analytic_jet_path_dne", False)
    if not (
        _certified_real_analytic(numerator, u)
        and _certified_real_analytic(denominator, u)
    ):
        return ParityCertificate("analytic_jet_path_dne", False)
    try:
        tn = _tensor_taylor_trunc(numerator, u, order)
        td = _tensor_taylor_trunc(denominator, u, order)
    except (TypeError, ValueError, NotImplementedError, sp.PoleError):
        return ParityCertificate("analytic_jet_path_dne", False)
    if not (tn.is_polynomial(*u) and td.is_polynomial(*u)):
        return ParityCertificate("analytic_jet_path_dne", False)

    t = sp.Dummy("_path_t", positive=True)
    paths = []
    model = _coercive_polynomial_model(td, u)
    if model is not None and len(u) <= 3:
        weights, degree, _, _, _ = model
        if (order + 1) * min(weights) <= degree:
            return ParityCertificate("analytic_jet_path_dne", False)
        for direction in product((-1, 0, 1), repeat=len(u)):
            if not any(direction):
                continue
            paths.append(
                tuple(
                    sp.Integer(coefficient) * t ** int(weight)
                    for coefficient, weight in zip(direction, weights, strict=True)
                )
            )
    elif len(u) <= 3:
        degree, leading_denominator = _leading_homogeneous(td, u)
        if degree in (1, 2):
            simple = [
                tuple(sp.S.One if i == j else sp.S.Zero for i in range(len(u)))
                for j in range(len(u))
            ]
            paths.extend(tuple(c * t for c in direction) for direction in simple)
            for direction in _denominator_zero_directions(leading_denominator, u):
                for perturb_index in range(len(u)):
                    for exponent in range(2, 7):
                        paths.append(
                            tuple(
                                coefficient * t
                                + (t**exponent if i == perturb_index else 0)
                                for i, coefficient in enumerate(direction)
                            )
                        )

    witnesses = []
    for path in paths:
        value = _analytic_taylor_path_limit(tn, td, u, path, order, t)
        if value is None:
            continue
        witnesses.append((path, value))
    for i, (path_a, value_a) in enumerate(witnesses):
        for path_b, value_b in witnesses[i + 1 :]:
            same = value_a == value_b
            if not same and value_a.is_finite is True and value_b.is_finite is True:
                same = sp.simplify(value_a - value_b) == 0
            if same:
                continue
            translated_a = tuple(
                sp.simplify(center + component)
                for center, component in zip(target, path_a, strict=True)
            )
            translated_b = tuple(
                sp.simplify(center + component)
                for center, component in zip(target, path_b, strict=True)
            )
            return ParityCertificate(
                "analytic_jet_path_dne",
                True,
                sp.nan,
                "two accumulating paths have distinct limits certified by uniform "
                "analytic Taylor remainders",
                (translated_a, value_a, translated_b, value_b, order),
            )
    return ParityCertificate("analytic_jet_path_dne", False)


def _sqrt_difference_data(expr, variables):
    """Return ``(scale, A, B)`` for ``scale*(sqrt(A)-sqrt(B))``."""
    terms = sp.Add.make_args(sp.expand(expr))
    if len(terms) != 2:
        return None
    parsed = []
    for term in terms:
        coeff, body = term.as_independent(*variables, as_Add=False)
        if not _constant_wrt(coeff, variables):
            return None
        base, exponent = body.as_base_exp()
        if exponent != sp.Rational(1, 2):
            return None
        parsed.append((sp.sympify(coeff), base))
    (c1, a), (c2, b) = parsed
    if sp.simplify(c1 + c2) != 0 or c1 == 0:
        return None
    return c1, a, b


def _continuous_radical_value(expr, variables, target):
    """Certify relative continuity and collect principal-real radical domains."""
    point = dict(zip(variables, target, strict=True))
    constraints = []

    def visit(node):
        node = sp.sympify(node)
        if not node.has(*variables):
            return True
        if node.is_Symbol:
            return node in variables
        if node.is_Add or node.is_Mul:
            return all(visit(arg) for arg in node.args)
        if node.is_Pow:
            base, exponent = node.as_base_exp()
            if not visit(base) or exponent.is_Rational is not True:
                return False
            base0 = sp.simplify(base.subs(point))
            if exponent < 0 and base0.is_zero is not False:
                return False
            if exponent.is_integer is not True:
                if int(exponent.q) % 2 == 0:
                    # Principal real q-th roots are continuous on base >= 0.
                    if base0.is_nonnegative is not True:
                        return False
                    constraints.append(sp.Ge(base, 0))
                elif base0.is_positive is not True:
                    # Stay away from the principal-power branch point/cut for
                    # unsupported odd-denominator noninteger powers.
                    return False
            return True
        if getattr(node, "is_Function", False) and len(node.args) == 1:
            if not visit(node.args[0]):
                return False
            arg0 = sp.simplify(node.args[0].subs(point))
            if node.func in (sp.sin, sp.cos, sp.sinh, sp.cosh, sp.exp, sp.Abs):
                return True
            if node.func is sp.tan:
                return sp.simplify(sp.cos(arg0)).is_zero is False
            if node.func is sp.log:
                return arg0.is_positive is True
        return False

    if not visit(expr):
        return None
    try:
        value = sp.simplify(expr.subs(point))
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if value.has(sp.nan, sp.zoo) or value.is_finite is False:
        return None
    return value, tuple(constraints)


def _rational_small_t_sign(expr, t):
    """Exact eventual sign of a univariate rational germ for ``t -> 0+``."""
    expr = sp.cancel(sp.sympify(expr))
    numerator, denominator = sp.fraction(expr)
    try:
        npoly = sp.Poly(sp.expand(numerator), t)
        dpoly = sp.Poly(sp.expand(denominator), t)
    except (sp.PolynomialError, TypeError, ValueError):
        return None
    if dpoly.is_zero:
        return None
    if npoly.is_zero:
        return 0

    def leading(poly):
        exponent = min(monom[0] for monom, coeff in poly.terms() if coeff != 0)
        coeff = sp.Add(*(c for monom, c in poly.terms() if monom[0] == exponent))
        return sp.simplify(coeff)

    coefficient = sp.simplify(leading(npoly) / leading(dpoly))
    if coefficient.is_positive is True:
        return 1
    if coefficient.is_negative is True:
        return -1
    if coefficient.is_zero is True:
        return 0
    return None


def _eventually_satisfies(formula, variables, target, direction):
    t = sp.Dummy("_domain_t", positive=True)
    substitution = {
        variable: sp.sympify(center) + sp.sympify(step) * t
        for variable, center, step in zip(variables, target, direction, strict=True)
    }

    def check(condition):
        condition = sp.sympify(condition)
        if condition in (sp.S.true, True):
            return True
        if condition in (sp.S.false, False):
            return False
        if isinstance(condition, sp.And):
            return all(check(arg) for arg in condition.args)
        if isinstance(condition, sp.Or):
            return any(check(arg) for arg in condition.args)
        if isinstance(condition, Relational):
            delta = sp.cancel((condition.lhs - condition.rhs).subs(substitution))
            sign = _rational_small_t_sign(delta, t)
            if sign is None:
                return False
            if isinstance(condition, sp.StrictGreaterThan):
                return sign > 0
            if isinstance(condition, sp.GreaterThan):
                return sign >= 0
            if isinstance(condition, sp.StrictLessThan):
                return sign < 0
            if isinstance(condition, sp.LessThan):
                return sign <= 0
            if isinstance(condition, sp.Equality):
                return sign == 0
            if isinstance(condition, sp.Unequality):
                return sign != 0
        return False

    return check(formula)


def _accumulating_domain_direction(domain, constraints, variables, target):
    """Find an exact linear approach ray inside a simple relative domain."""
    if len(variables) > 4:
        return None
    formula = sp.And(sp.sympify(domain), *constraints)
    for direction in product((-2, -1, 0, 1, 2), repeat=len(variables)):
        if not any(direction):
            continue
        if _eventually_satisfies(formula, variables, target, direction):
            return direction
    return None


def radical_rationalization_certificate(
    expr, variables, target, *, domain=True, _context=None
):
    """Rationalize a principal-square-root difference with domain tracking."""
    expr = sp.sympify(expr)
    numerator, denominator = sp.fraction(expr)
    numerator_root = _sqrt_difference_data(numerator, variables)
    denominator_root = _sqrt_difference_data(denominator, variables)
    if numerator_root is None and denominator_root is None:
        return ParityCertificate("radical_rationalization", False)

    transformed_numerator = numerator
    transformed_denominator = denominator
    exact_constraints = []
    if numerator_root is not None:
        scale, a, b = numerator_root
        exact_constraints.extend((sp.Ge(a, 0), sp.Ge(b, 0)))
        transformed_numerator = scale * (a - b) / (sp.sqrt(a) + sp.sqrt(b))
    if denominator_root is not None:
        scale, a, b = denominator_root
        # On the principal real branch sqrt(A)=sqrt(B) iff A=B, so this
        # polynomial inequality exactly tracks the deleted denominator variety.
        exact_constraints.extend((sp.Ge(a, 0), sp.Ge(b, 0), sp.Ne(a - b, 0)))
        transformed_denominator = scale * (a - b) / (sp.sqrt(a) + sp.sqrt(b))
    else:
        point = dict(zip(variables, target, strict=True))
        denominator_at_target = sp.simplify(denominator.subs(point))
        if denominator_at_target.is_zero is not False:
            exact_constraints.append(sp.Ne(denominator, 0))

    transformed = sp.cancel(transformed_numerator / transformed_denominator)
    continuity = _continuous_radical_value(transformed, variables, target)
    if continuity is None:
        return ParityCertificate("radical_rationalization", False)
    value, transformed_constraints = continuity
    constraints = tuple(exact_constraints) + tuple(transformed_constraints)
    direction = _accumulating_domain_direction(domain, constraints, variables, target)
    if direction is None:
        return ParityCertificate("radical_rationalization", False)
    return ParityCertificate(
        "radical_rationalization",
        True,
        value,
        "exact conjugate rationalization plus relative-domain continuity "
        "certifies the limit",
        (sp.factor(transformed), constraints, direction),
    )


def oscillatory_phase_sequence_certificate(expr, variables, target, *, _context=None):
    """Prove DNE for reciprocal max-radius sine/cosine by two exact sequences."""
    u, f = _shift_for(expr, variables, target, _context)
    if f.func not in (sp.sin, sp.cos) or len(f.args) != 1:
        return ParityCertificate("oscillatory_phase_sequences_dne", False)
    phase = sp.cancel(f.args[0])
    numerator, denominator = sp.fraction(phase)
    if sp.simplify(numerator) not in (1, -1):
        return ParityCertificate("oscillatory_phase_sequences_dne", False)
    if denominator.func is not sp.Max:
        return ParityCertificate("oscillatory_phase_sequences_dne", False)
    expected = {sp.Abs(variable) for variable in u}
    if set(denominator.args) != expected:
        return ParityCertificate("oscillatory_phase_sequences_dne", False)

    n = sp.Symbol("n", integer=True, positive=True)
    phase_sign = sp.simplify(numerator)
    if f.func is sp.sin:
        theta_a = sp.pi / 2 + 2 * sp.pi * n
        theta_b = 3 * sp.pi / 2 + 2 * sp.pi * n
        value_a, value_b = phase_sign, -phase_sign
    else:
        theta_a = 2 * sp.pi * n
        theta_b = sp.pi + 2 * sp.pi * n
        value_a, value_b = sp.S.One, -sp.S.One
    radius_a = sp.simplify(1 / theta_a)
    radius_b = sp.simplify(1 / theta_b)
    seq_a_shifted = (radius_a,) + (sp.S.Zero,) * (len(u) - 1)
    seq_b_shifted = (radius_b,) + (sp.S.Zero,) * (len(u) - 1)
    seq_a = tuple(
        sp.simplify(center + delta)
        for center, delta in zip(target, seq_a_shifted, strict=True)
    )
    seq_b = tuple(
        sp.simplify(center + delta)
        for center, delta in zip(target, seq_b_shifted, strict=True)
    )
    return ParityCertificate(
        "oscillatory_phase_sequences_dne",
        True,
        sp.nan,
        "two explicit accumulating reciprocal-phase sequences give distinct "
        "exact oscillatory values",
        (seq_a, value_a, seq_b, value_b, n),
    )


def analytic_meromorphic_taylor_reduction(expr, variables, target, *, order=6):
    """Reduce analytic/meromorphic germs to finite multivariate Taylor data."""
    u, f = _shift(expr, variables, target)
    num, den = sp.fraction(f)

    def trunc(g):
        out = sp.sympify(g)
        for x in u:
            try:
                out = sp.series(out, x, 0, order + 1).removeO()
            except (TypeError, ValueError, NotImplementedError):
                return None
        return sp.expand(out)

    tn, td = trunc(num), trunc(den)
    if tn is None or td is None:
        return ParityCertificate("analytic_taylor", False)
    # Exact only when the original numerator and denominator are polynomial in
    # the limit variables.  A transcendental blacklist is unsound: functions
    # such as tan/sinh/special functions could otherwise be mistaken for an
    # exact finite Taylor polynomial merely because they were not enumerated.
    exact = bool(num.is_polynomial(*u) and den.is_polynomial(*u))
    return ParityCertificate(
        "analytic_taylor",
        exact,
        sp.cancel(tn / td) if exact else None,
        "finite Taylor germ reduction"
        if exact
        else "Taylor data computed; remainder domination unresolved",
        (tn, td, order),
    )


def real_projection_simplification_certificate(
    expr, variables, target, *, _context=None
):
    """Simplify Re/Im/conjugate on the real simultaneous-limit variables."""
    from .limits import LimitStatus, limit

    u, f = _shift_for(expr, variables, target, _context)
    replacements = {}
    for atom in sp.preorder_traversal(f):
        if atom.func is sp.re and atom.args[0] in u:
            replacements[atom] = atom.args[0]
        elif atom.func is sp.im and atom.args[0] in u:
            replacements[atom] = sp.S.Zero
        elif atom.func is sp.conjugate and atom.args[0] in u:
            replacements[atom] = atom.args[0]
    if not replacements:
        return ParityCertificate("real_projection", False)
    reduced = sp.simplify(f.xreplace(replacements))
    if reduced == f:
        return ParityCertificate("real_projection", False)
    result = limit(reduced, u, (0,) * len(u), return_result=True)
    if result.status is LimitStatus.PROVED:
        return ParityCertificate(
            "real_projection",
            True,
            result.value,
            "real limit variables make Re(x)=x, Im(x)=0, and conjugate(x)=x",
            (reduced,),
        )
    if result.status is LimitStatus.DOES_NOT_EXIST:
        return ParityCertificate(
            "real_projection",
            True,
            None,
            "real projection reduction preserves two conflicting approaches",
            (reduced,),
        )
    return ParityCertificate("real_projection", False, data=(reduced,))


def radial_norm_canonicalization_certificate(expr, variables, target, *, _context=None):
    """Cheap Euclidean-radius certificates before general semialgebraic machinery."""
    u, f = _shift_for(expr, variables, target, _context)
    r2 = sp.Add(*(v**2 for v in u))
    rho = sp.Dummy("_rho", positive=True)

    # Exact radial expressions become a one-sided univariate limit.
    radial = f
    for atom in sorted(f.atoms(sp.Pow), key=sp.default_sort_key, reverse=True):
        base, exponent = atom.as_base_exp()
        if sp.expand(base - r2) == 0:
            radial = radial.xreplace({atom: rho ** (2 * exponent)})
    radial = radial.xreplace({r2: rho**2})
    if radial != f and not radial.has(*u):
        try:
            value = bounded_limit(radial, rho, 0, direction="+", allow_general=True)
        except (TypeError, ValueError, NotImplementedError, sp.PoleError):
            value = sp.nan
        if value not in (sp.nan, sp.zoo):
            return ParityCertificate(
                "radial_norm",
                True,
                value,
                "Euclidean norm/radius canonicalization reduces to a positive radial limit",
                (radial,),
            )

    # Nonzero continuous numerator over a positive Euclidean-radius power has
    # a signed infinite limit, independent of angle.
    num, den = sp.fraction(sp.cancel(f))
    n0 = sp.simplify(num.subs(dict.fromkeys(u, 0)))
    for k in range(1, 9):
        root = r2 ** sp.Rational(k, 2)
        q = sp.simplify(den / root)
        if not q.has(*u) and q.is_positive is True and n0.is_nonzero is True:
            if n0.is_positive is True:
                return ParityCertificate(
                    "radial_norm",
                    True,
                    sp.oo,
                    "positive nonzero numerator over a vanishing Euclidean radius",
                    (n0, k),
                )
            if n0.is_negative is True:
                return ParityCertificate(
                    "radial_norm",
                    True,
                    -sp.oo,
                    "negative nonzero numerator over a vanishing Euclidean radius",
                    (n0, k),
                )

    # Polynomial / rho**k: a numerator of total order > k is uniformly o(1),
    # since every coordinate is bounded by rho.
    for k in range(1, 9):
        root = r2 ** sp.Rational(k, 2)
        q = sp.simplify(den / root)
        if q.has(*u) or q == 0:
            continue
        try:
            poly = sp.Poly(num, *u)
        except sp.PolynomialError:
            continue
        orders = [sum(mon) for mon, coeff in poly.terms() if coeff != 0]
        if orders and min(orders) > k:
            return ParityCertificate(
                "radial_norm",
                True,
                sp.S.Zero,
                "polynomial numerator order dominates the Euclidean-radius denominator",
                (min(orders), k),
            )
    return ParityCertificate("radial_norm", False)


def logarithmic_germ_order_certificate(expr, variables, target, *, _context=None):
    """Certify common log radial germs using r^a |log r| -> 0 and log1p jets."""
    u, f = _shift_for(expr, variables, target, _context)
    r2 = sp.Add(*(v**2 for v in u))
    rho = sp.Dummy("_rho", positive=True)

    # Canonical radial logarithms, including translated Log10 = log()/log(10).
    repl = f
    for logatom in list(f.atoms(sp.log)):
        arg = sp.expand(logatom.args[0])
        if sp.expand(arg - r2) == 0:
            repl = repl.xreplace({logatom: 2 * sp.log(rho)})
        elif arg.func is sp.Pow and sp.expand(arg.base - r2) == 0:
            repl = repl.xreplace({logatom: 2 * arg.exp * sp.log(rho)})
    # Replace a radial polynomial prefactor exactly when possible.
    repl = repl.xreplace({r2: rho**2})
    if repl != f and not repl.has(*u):
        try:
            value = bounded_limit(repl, rho, 0, direction="+", allow_general=True)
        except (TypeError, ValueError, NotImplementedError, sp.PoleError):
            value = sp.nan
        if value not in (sp.nan, sp.zoo):
            return ParityCertificate(
                "logarithmic_germ",
                True,
                value,
                "radial logarithmic germ reduced using r^a*log(r) order",
                (repl,),
            )

    # p(u)*log(r2): any polynomial p with positive vanishing order is o(1).
    logs = [a for a in f.atoms(sp.log) if sp.expand(a.args[0] - r2) == 0]
    if len(logs) == 1:
        logatom = logs[0]
        q = sp.simplify(f / logatom)
        try:
            poly = sp.Poly(q, *u)
        except sp.PolynomialError:
            poly = None
        if poly is not None:
            orders = [sum(mon) for mon, coeff in poly.terms() if coeff != 0]
            if orders and min(orders) > 0:
                return ParityCertificate(
                    "logarithmic_germ",
                    True,
                    sp.S.Zero,
                    "positive polynomial order dominates a radial logarithmic singularity",
                    (min(orders),),
                )
    return ParityCertificate("logarithmic_germ", False)


@dataclass(frozen=True)
class ComplexSingularGerm:
    """Leading principal-branch germ with explicit real-axis boundary data.

    ``positive_real`` and ``negative_real`` are leading real projections on the
    two boundary rays.  ``log_coefficient`` records the coefficient of the
    principal logarithm in the singular expansion; it is metadata rather than
    an invitation to erase branch information.
    """

    function: object
    order: int
    positive_real: sp.Expr
    negative_real: sp.Expr
    log_coefficient: sp.Expr
    description: str


_HANKEL1_PRINCIPAL_GERMS = {
    0: ComplexSingularGerm(
        sp.hankel1,
        0,
        sp.S.One,
        -sp.S.One,
        2 * sp.I / sp.pi,
        "H1_0(z)=1+(2 i/pi)(log(z/2)+gamma)+O(z^2 log z), with the principal-log jump",
    ),
    1: ComplexSingularGerm(
        sp.hankel1,
        -1,
        sp.S.Zero,
        sp.S.Zero,
        sp.I / sp.pi,
        "H1_1(z)=-2 i/(pi z)+z/2+(i z/pi)(log(z/2)+gamma-1/2)+O(z^3 log z)",
    ),
}


def complex_hankel_projection_certificate(expr, variables, target, *, _context=None):
    """Certify branch-sensitive DNE for real projections of principal Hankel germs.

    The proof compares the two real-axis boundary germs of the principal branch.
    It never replaces a Hankel function by a branch-blind Taylor series.
    """
    u, f = _shift_for(expr, variables, target, _context)
    re_nodes = tuple(a for a in sp.preorder_traversal(f) if a.func is sp.re)
    hankels = tuple(f.atoms(sp.hankel1))
    if not hankels or not re_nodes:
        return ParityCertificate("complex_hankel_projection", False)
    args = {a.args[1] for a in hankels if len(a.args) == 2}
    if len(args) != 1:
        return ParityCertificate("complex_hankel_projection", False)
    x = next(iter(args))
    if x not in u or any(a.args[0] not in _HANKEL1_PRINCIPAL_GERMS for a in hankels):
        return ParityCertificate("complex_hankel_projection", False)

    # Every Hankel occurrence must be owned by a real projection whose remaining
    # coefficient is real on the witness axis.  This excludes hidden complex
    # multipliers for which Im(H) contributes to Re(c H).
    owned = set()
    decomposed = []
    for node in re_nodes:
        hs = tuple(node.args[0].atoms(sp.hankel1))
        if not hs:
            continue
        if len(hs) != 1:
            return ParityCertificate("complex_hankel_projection", False)
        h = hs[0]
        coeff = sp.simplify(node.args[0] / h)
        decomposed.append((node, h, coeff))
        owned.add(h)
    if owned != set(hankels):
        return ParityCertificate("complex_hankel_projection", False)

    t = sp.Dummy("_hankel_t", positive=True)
    zeros = {v: sp.S.Zero for v in u if v != x}
    values = []
    witnesses = []
    for side in (sp.S.One, -sp.S.One):
        repl = {}
        for node, h, coeff in decomposed:
            axis_coeff = sp.simplify(coeff.subs(zeros).subs(x, side * t))
            if axis_coeff.is_real is not True:
                return ParityCertificate("complex_hankel_projection", False)
            n = int(h.args[0])
            # Principal boundary values for t>0:
            # Re H0^(1)(+t)=+J0(t), Re H0^(1)(-t)=-J0(t);
            # Re H1^(1)(±t)=J1(t).  Retaining the first nonzero J1 term
            # matters when its real coefficient has a reciprocal zero.
            real_germ = side if n == 0 else t / 2
            repl[node] = axis_coeff * real_germ
        projected = sp.simplify(f.xreplace(repl).subs(zeros).subs(x, side * t))
        if projected.has(sp.hankel1):
            return ParityCertificate("complex_hankel_projection", False)
        try:
            value = bounded_limit(projected, t, 0, direction="+", allow_general=True)
        except (TypeError, ValueError, NotImplementedError, sp.PoleError):
            return ParityCertificate("complex_hankel_projection", False)
        if value in (sp.nan, sp.zoo) or value.has(t):
            return ParityCertificate("complex_hankel_projection", False)
        values.append(value)
        witnesses.append((side, projected))

    if (
        all(v not in (sp.oo, -sp.oo) for v in values)
        and sp.simplify(values[0] - values[1]) != 0
    ):
        return ParityCertificate(
            "complex_hankel_projection_dne",
            True,
            None,
            "principal Hankel logarithmic branch has distinct real-axis boundary cluster values",
            tuple(values) + tuple(witnesses),
        )
    return ParityCertificate("complex_hankel_projection", False, data=tuple(values))


@dataclass(frozen=True)
class SpecialFunctionGermContract:
    function: object
    arity: int
    parameter_condition: object
    singularity: str
    leading_power: sp.Expr
    leading_coefficient: sp.Expr
    logarithmic_power: sp.Expr
    remainder_order: sp.Expr
    branch_structure: str
    needs_zero_inner_limit: bool = True


_SPECIAL_GERM_REGISTRY = {
    sp.fresnelc: SpecialFunctionGermContract(
        sp.fresnelc, 1, sp.S.true, "regular", 1, sp.S.One, 0, 5, "entire"
    ),
    sp.fresnels: SpecialFunctionGermContract(
        sp.fresnels, 1, sp.S.true, "regular", 3, sp.pi / 6, 0, 7, "entire"
    ),
    sp.erf: SpecialFunctionGermContract(
        sp.erf, 1, sp.S.true, "regular", 1, 2 / sp.sqrt(sp.pi), 0, 3, "entire"
    ),
    sp.erfi: SpecialFunctionGermContract(
        sp.erfi, 1, sp.S.true, "regular", 1, 2 / sp.sqrt(sp.pi), 0, 3, "entire"
    ),
}


def special_function_germ_contract(atom, assumptions=sp.S.true):
    """Return a uniform certified local-germ contract for a supported atom."""
    atom = sp.sympify(atom)
    assumptions = sp.sympify(assumptions)
    fixed = _SPECIAL_GERM_REGISTRY.get(atom.func)
    if fixed is not None:
        return fixed
    if atom.func in (sp.besselj, sp.bessely) and len(atom.args) == 2:
        nu, _ = atom.args
        # Parameter-dependent order is admitted only on a proved cell.  J_nu
        # has leading (z/2)^nu/Gamma(nu+1) for nu>-1 away from Gamma poles.
        if (
            bounded_ask(sp.Q.real(nu), assumptions) is True
            and bounded_ask(sp.Q.positive(nu + 1), assumptions) is True
        ):
            if atom.func is sp.besselj:
                return SpecialFunctionGermContract(
                    sp.besselj,
                    2,
                    sp.Gt(nu, -1),
                    "regular_or_algebraic_branch",
                    nu,
                    1 / (2**nu * sp.gamma(nu + 1)),
                    0,
                    nu + 2,
                    "principal power for noninteger order",
                )
            # Y_nu needs separate integer/noninteger branch cells; do not merge
            # them because logarithmic terms appear at integer order.
            if (
                bounded_ask(sp.Q.integer(nu), assumptions) is False
                and bounded_ask(sp.Q.positive(nu), assumptions) is True
            ):
                return SpecialFunctionGermContract(
                    sp.bessely,
                    2,
                    sp.And(sp.Gt(nu, 0), sp.Not(sp.Contains(nu, sp.S.Integers))),
                    "algebraic_pole",
                    -nu,
                    -sp.gamma(nu) * 2**nu / sp.pi,
                    0,
                    sp.Min(nu, 2 - nu),
                    "principal power",
                )
    return None


def special_function_local_germ_certificate(expr, variables, target, *, _context=None):
    """Registry-driven certified local germs for supported special functions.

    Only entries with a simple real analytic leading monomial are rewritten here.
    Bessel/Hankel singular germs are recognized explicitly but left to dedicated
    order/branch rules rather than using an unsafe principal-branch jet.
    """
    from .limits import LimitStatus, limit

    u, f = _shift_for(expr, variables, target, _context)

    # Integer-order Bessel-Y singular germ: Y_1(z) = -2/(pi*z) + O(z*log(z)).
    # Replace only the removable product z*Y_1(z), whose remainder tends to zero.
    for atom in f.atoms(sp.bessely):
        if len(atom.args) != 2 or atom.args[0] != 1:
            continue
        z = atom.args[1]
        if sp.simplify(z.subs(dict.fromkeys(u, 0))) != 0:
            continue
        try:
            multiplier = sp.cancel(f / (z * atom))
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        multiplier_limit = limit(multiplier, u, (0,) * len(u), return_result=True)
        if (
            multiplier_limit.status is not LimitStatus.PROVED
            or multiplier_limit.value in (sp.oo, -sp.oo)
        ):
            continue
        reduced = sp.simplify((-2 / sp.pi) * multiplier)
        result = limit(reduced, u, (0,) * len(u), return_result=True)
        if result.status is LimitStatus.PROVED:
            return ParityCertificate(
                "special_function_germ",
                True,
                result.value,
                "BesselY(1,z) singular germ z*Y_1(z)=-2/pi+O(z^2 log z)",
                (atom, reduced),
            )

    # Uniform parameter-dependent Bessel-J contract.  It participates only
    # when the order cell itself proves the admissibility condition.
    for atom in f.atoms(sp.besselj):
        entry = special_function_germ_contract(atom)
        if entry is None:
            continue
        arg = atom.args[-1]
        if sp.simplify(arg.subs(dict.fromkeys(u, 0))) != 0:
            continue
        leading = sp.simplify(entry.leading_coefficient * arg**entry.leading_power)
        reduced = f.xreplace({atom: leading})
        result = limit(reduced, u, (0,) * len(u), return_result=True)
        if result.status is LimitStatus.PROVED and result.value == 0:
            return ParityCertificate(
                "special_function_germ",
                True,
                sp.S.Zero,
                "uniform special-function germ contract with certified parameter cell and higher-order remainder",
                (atom, entry.parameter_condition, entry.remainder_order),
            )

    for atom in sp.preorder_traversal(f):
        entry = special_function_germ_contract(atom)
        if entry is None or len(atom.args) != 1:
            continue
        arg = atom.args[0]
        if sp.simplify(arg.subs(dict.fromkeys(u, 0))) != 0:
            continue
        leading = sp.simplify(
            entry.leading_coefficient
            * arg**entry.leading_power
            * sp.log(arg) ** entry.logarithmic_power
        )
        proof = f"{entry.function}: leading valuation {entry.leading_power}, remainder order {entry.remainder_order}, branch={entry.branch_structure}"
        # Safe quotient/product replacement is delegated to the existing analytic
        # engine; this fast path handles expressions where the leading replacement
        # itself settles the limit and the omitted order is strictly higher.
        reduced = f.xreplace({atom: leading})
        if reduced == f:
            continue
        result = limit(reduced, u, (0,) * len(u), return_result=True)
        if result.status is LimitStatus.PROVED and result.value == 0:
            return ParityCertificate(
                "special_function_germ",
                True,
                sp.S.Zero,
                "registered special-function leading germ with higher-order remainder",
                (atom, leading, proof),
            )
    return ParityCertificate("special_function_germ", False)


def growth_scale_limit(expr, variables, target, *, _context=None):
    """Certify common polynomial-times-log/exp decay scales without CAD."""
    u, f = _shift_for(expr, variables, target, _context)
    r2 = sp.Add(*(x**2 for x in u))
    # Factor cancellation-expanded radial polynomials before substitution.
    try:
        f = sp.factor(f)
    except (sp.PolynomialError, TypeError, ValueError):
        pass
    rho = sp.Dummy("_rho", positive=True)
    repl = f.subs(r2, rho**2)
    if repl.has(*u):
        # SymPy may distribute the radial quadratic before substitution.
        for radial in (r2 * sp.log(r2), sp.log(r2), sp.exp(-1 / r2)):
            try:
                q = sp.cancel(f / radial)
            except (TypeError, ValueError, ZeroDivisionError):
                continue
            if not q.has(*u):
                repl = q * radial.subs(r2, rho**2)
                break
    if repl != f and not repl.has(*u):
        try:
            v = bounded_limit(repl, rho, 0, direction="+", allow_general=True)
        except (TypeError, ValueError, NotImplementedError):
            return ParityCertificate("growth_scale", False)
        if v not in (sp.nan, sp.zoo):
            return ParityCertificate(
                "growth_scale", True, v, "radial polynomial/log/exp growth scale"
            )
    return ParityCertificate("growth_scale", False)


def divided_difference_certificate(expr, variables, target, *, _context=None):
    """Exact common unary divided-difference limits at a common target."""
    if len(variables) != 2 or target[0] != target[1]:
        return ParityCertificate("divided_difference", False)
    x, y = variables
    a = target[0]
    num, den = sp.fraction(sp.cancel(expr))
    # Identify F(x)-F(y) up to an overall constant.
    fx = sp.simplify(num.subs(y, a))
    # Direct x-y denominator: mean-value/divided-difference theorem.
    q = sp.cancel(den / (x - y))
    if not q.has(x, y):
        F = sp.simplify(fx - fx.subs(x, a))
        candidate = sp.simplify(sp.diff(F, x).subs(x, a) / q)
        if sp.simplify(num - (F - F.xreplace({x: y}))) == 0:
            return ParityCertificate(
                "divided_difference",
                True,
                candidate,
                "analytic first divided difference tends to the derivative",
            )
    # x^2-y^2 with an even analytic numerator F(x)-F(y).  The quotient
    # is the divided difference of G(t)=F(sqrt(t)); G'(0)=F''(0)/2.
    q = sp.cancel(den / (x**2 - y**2))
    if not q.has(x, y):
        F = sp.simplify(fx - fx.subs(x, a))
        if (
            a == 0
            and sp.simplify(F.subs(x, -x) - F) == 0
            and sp.simplify(num - (F - F.xreplace({x: y}))) == 0
        ):
            candidate = sp.simplify(sp.diff(F, x, 2).subs(x, 0) / (2 * q))
            return ParityCertificate(
                "divided_difference",
                True,
                candidate,
                "even analytic divided difference in squared coordinates",
            )
    return ParityCertificate("divided_difference", False)


def _analytic_ratio_eligible(expr, variables, target, _context=None):
    """Cheaply reject germs that cannot be analytic quotients at the target."""
    u, f = _shift_for(expr, variables, target, _context)
    if f.has(
        AppliedUndef, sp.Abs, sp.sign, sp.floor, sp.ceiling, sp.Heaviside, sp.Piecewise
    ):
        return False
    zero = dict.fromkeys(u, 0)
    for power in f.atoms(sp.Pow):
        if power.exp.is_integer is True:
            continue
        try:
            center = sp.simplify(power.base.subs(zero))
        except (TypeError, ValueError, NotImplementedError):
            return False
        if center == 0 or center.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
            return False
    for atom in f.atoms(sp.Function):
        if atom.func is sp.log:
            continue
        try:
            center = (
                sp.simplify(atom.args[0].subs(dict.fromkeys(u, 0)))
                if atom.args
                else sp.S.Zero
            )
        except (TypeError, ValueError, NotImplementedError):
            return False
        if center.has(sp.nan, sp.zoo, sp.oo, -sp.oo) or center.is_finite is False:
            return False
    for atom in f.atoms(sp.log):
        try:
            center = sp.simplify(atom.args[0].subs(zero))
        except (TypeError, ValueError, NotImplementedError):
            return False
        if center.is_positive is not True:
            return False
    return True


def analytic_leading_ratio_certificate(
    expr, variables, target, *, order=7, _context=None
):
    """Certify analytic quotient limits from proportional leading Taylor forms."""
    if not _analytic_ratio_eligible(expr, variables, target, _context):
        return ParityCertificate("analytic_leading_ratio", False)
    u, f = _shift_for(expr, variables, target, _context)
    if f.has(AppliedUndef):
        return ParityCertificate("analytic_leading_ratio", False)
    num, den = sp.fraction(f)

    def trunc(g):
        out = g
        for x in u:
            out = sp.series(out, x, 0, order + 1).removeO()
        return sp.expand(out)

    try:
        tn, td = trunc(num), trunc(den)
    except (TypeError, ValueError, NotImplementedError, sp.PoleError):
        return ParityCertificate("analytic_leading_ratio", False)
    dn, hn = _leading_homogeneous(tn, u)
    dd, hd = _leading_homogeneous(td, u)
    if dn is None or dd is None or dd is sp.oo:
        return ParityCertificate("analytic_leading_ratio", False)
    sd = _obvious_definite(hd, u)
    if sd is not None and dn is not sp.oo and dn > dd:
        return ParityCertificate(
            "analytic_leading_ratio",
            True,
            sp.S.Zero,
            "higher analytic numerator order over a definite leading denominator",
        )
    if sd is not None and dn is not sp.oo and dn < dd:
        sn = _obvious_definite(hn, u)
        if sn is not None:
            return ParityCertificate(
                "analytic_leading_ratio",
                True,
                sp.oo if sn * sd > 0 else -sp.oo,
                "lower analytic numerator order and certified signs imply a pole",
            )
    if dn != dd:
        return ParityCertificate("analytic_leading_ratio", False)
    # Determine a constant quotient of leading forms.
    try:
        ratio = sp.cancel(hn / hd)
    except (TypeError, ValueError, ZeroDivisionError):
        return ParityCertificate("analytic_leading_ratio", False)
    if ratio.free_symbols & set(u):
        return ParityCertificate("analytic_leading_ratio", False)
    if sd is None:
        return ParityCertificate("analytic_leading_ratio", False)
    return ParityCertificate(
        "analytic_leading_ratio",
        True,
        ratio,
        "proportional leading analytic Taylor forms with definite denominator",
        (hn, hd),
    )


def certified_analytic_taylor_reduction(
    expr, variables, target, *, order=6, domain=sp.S.true
):
    """Taylor reduction with an actual uniform multivariate remainder certificate."""
    try:
        from .multivariate_expansion import multivariate_expansion

        expansion = multivariate_expansion(
            expr, variables, target, order=order, domain=domain
        )
    except (TypeError, ValueError, NotImplementedError, RuntimeError, ArithmeticError):
        return ParityCertificate("analytic_taylor_remainder", False)
    if (
        getattr(expansion, "certified", False)
        and expansion.remainder_certificate is not None
    ):
        return ParityCertificate(
            "analytic_taylor_remainder",
            True,
            expansion.truncated_expression(),
            "finite multivariate Taylor model with certified uniform radial remainder",
            (expansion, expansion.remainder_certificate),
        )
    return ParityCertificate("analytic_taylor_remainder", False, data=(expansion,))


def removable_transcendental_normalization_certificate(
    expr, variables, target, *, domain=True, _context=None
):
    """Normalize standard removable scalar germs before multivariate analysis.

    The transformation is used only after the inner argument is independently
    certified to tend to zero.  The residual factor is then sent back through
    the ordinary multivariate limit engine; no path agreement is promoted to a
    proof.
    """
    from .limits import LimitStatus, limit
    from .stratification import AsymptoticStratification

    expr = sp.sympify(expr)
    candidates = []
    for atom in sp.preorder_traversal(expr):
        if atom.func in (sp.sin, sp.tan, sp.sinh) and atom.args:
            h = atom.args[0]
            candidates.append((atom / h, h, sp.S.One, atom.func.__name__))
        elif atom.func is sp.exp and atom.args:
            h = atom.args[0]
            candidates.append(((atom - 1) / h, h, sp.S.One, "expm1"))
        elif atom.func is sp.log and atom.args:
            arg = atom.args[0]
            h = sp.expand(arg - 1)
            candidates.append((atom / h, h, sp.S.One, "log1p"))
        elif atom.func is sp.cos and atom.args:
            h = atom.args[0]
            candidates.append(((1 - atom) / h**2, h, sp.Rational(1, 2), "cosm1"))
    for removable, h, removable_limit, name in candidates:
        if not h.has(*variables):
            continue
        try:
            residual = sp.cancel(expr / removable)
        except (TypeError, ValueError, ZeroDivisionError, sp.PolynomialError):
            continue
        # The normalization must make structural progress.  In particular,
        # dividing an inverse-sinc factor by sinc would square its singular
        # denominator and can recurse forever.
        trans_heads = (sp.sin, sp.cos, sp.tan, sp.sinh, sp.exp, sp.log)

        def complexity(value, heads=trans_heads):
            return (
                sum(len(value.atoms(head)) for head in heads),
                int(sp.count_ops(value, visual=False)),
            )

        before = complexity(expr)
        options = []
        if residual != expr and complexity(residual) < before:
            options.append((complexity(residual), residual, removable_limit))
        try:
            inverse_residual = sp.cancel(expr * removable)
        except (TypeError, ValueError, ZeroDivisionError, sp.PolynomialError):
            inverse_residual = expr
        if inverse_residual != expr and complexity(inverse_residual) < before:
            options.append(
                (
                    complexity(inverse_residual),
                    inverse_residual,
                    sp.simplify(1 / removable_limit),
                )
            )
        if not options:
            continue
        _, residual, removable_limit = min(options, key=lambda item: item[0])
        inner = limit(h, variables, target, domain=domain, return_result=True)
        if (
            isinstance(inner, AsymptoticStratification)
            or inner.status is not LimitStatus.PROVED
            or inner.value != 0
        ):
            continue
        rest = limit(residual, variables, target, domain=domain, return_result=True)
        if (
            isinstance(rest, AsymptoticStratification)
            or rest.status is not LimitStatus.PROVED
            or rest.value is None
        ):
            continue
        value = sp.simplify(removable_limit * rest.value)
        if value.has(sp.nan, sp.zoo):
            continue
        return ParityCertificate(
            "removable_transcendental_normalization",
            True,
            value,
            f"certified {name}-like removable germ times a certified residual limit",
            (h, residual),
        )
    return ParityCertificate("removable_transcendental_normalization", False)


def parameter_weighted_order_certificate(expr, variables, target, *, _context=None):
    """Return conditional zero information for radial powers with free exponents.

    This fast path is conservative: it never turns a symbolic
    exponent comparison into an unconditional limit.  The condition is stored
    in certificate data for the parameter-stratification layer and ordinary
    value-mode therefore remains UNKNOWN.
    """
    u, f = _shift_for(expr, variables, target, _context)
    num, den = sp.fraction(f)
    r2 = sp.Add(*(v**2 for v in u))
    if not den.is_Pow or sp.expand(den.base - r2) != 0:
        return ParityCertificate("parameter_weighted_order", False)
    p = den.exp
    params = p.free_symbols - set(u)
    if not params:
        return ParityCertificate("parameter_weighted_order", False)
    try:
        poly = sp.Poly(num, *u)
    except sp.PolynomialError:
        return ParityCertificate("parameter_weighted_order", False)
    # |u^alpha| <= r^|alpha|, so numerator order m and r^(2p) denominator
    # give a sufficient zero condition m-2p>0.
    orders = [sum(mon) for mon, c in poly.terms() if c != 0]
    if not orders:
        return ParityCertificate(
            "parameter_weighted_order", True, sp.S.Zero, "identically zero numerator"
        )
    m = min(orders)
    condition = sp.StrictLessThan(p, sp.Rational(m, 2))
    return ParityCertificate(
        "parameter_weighted_order", False, data=(sp.S.Zero, condition, m, p)
    )


def generalized_log_product_order_certificate(
    expr, variables, target, *, _context=None
):
    """Certify products of positive-order germs with logarithmic singularities."""
    u, f = _shift_for(expr, variables, target, _context)
    logs = [a for a in f.atoms(sp.log) if a.has(*u)]
    if len(logs) != 1:
        return ParityCertificate("log_product_order", False)
    logatom = logs[0]
    try:
        q = sp.cancel(f / logatom)
        poly = sp.Poly(q, *u)
        orders = [sp.Rational(sum(mon)) for mon, c in poly.terms() if c != 0]
    except (sp.PolynomialError, TypeError, ValueError):
        # Cheap algebraic monomial order, including sqrt(Abs(y)).
        order = sp.S.Zero
        ok = True
        for factor in sp.Mul.make_args(q):
            if factor.is_number:
                continue
            base, exp = factor.as_base_exp()
            if base in u and exp.is_positive is True:
                order += exp
                continue
            if base.func is sp.Abs and base.args[0] in u and exp.is_positive is True:
                order += exp
                continue
            try:
                pp = sp.Poly(factor, *u)
                os = [sum(mon) for mon, c in pp.terms() if c != 0]
                if os and min(os) > 0:
                    order += min(os)
                    continue
            except sp.PolynomialError:
                pass
            ok = False
            break
        orders = [order] if ok else []
    if not orders or min(orders) <= 0:
        return ParityCertificate("log_product_order", False)
    arg = logatom.args[0]
    # Products/absolute values of coordinate germs and positive sums of squares
    # have at most logarithmic growth.  Polynomial decay of any positive order
    # dominates every fixed logarithmic power.
    allowed = arg.has(sp.Abs) or arg.is_Mul or arg.is_Pow
    try:
        ap = sp.Poly(arg, *u)
        allowed = allowed or (
            ap.total_degree() > 0 and sp.simplify(arg.subs(dict.fromkeys(u, 0))) == 0
        )
    except sp.PolynomialError:
        pass
    if not allowed:
        return ParityCertificate("log_product_order", False)
    return ParityCertificate(
        "log_product_order",
        True,
        sp.S.Zero,
        "positive polynomial order dominates logarithmic growth of a vanishing algebraic germ",
        (min(orders), arg),
    )


def angular_cluster_path_certificate(expr, variables, target, *, _context=None):
    """Cheap exact angular cluster/DNE test on a small certified ray atlas."""
    from ._symbolic_policy import bounded_limit

    u, f = _shift_for(expr, variables, target, _context)
    t = sp.Dummy("_angular_t", positive=True)
    rays = []
    n = len(u)
    for i in range(n):
        d = [sp.S.Zero] * n
        d[i] = sp.S.One
        rays.append(tuple(d))
        d = [sp.S.Zero] * n
        d[i] = -sp.S.One
        rays.append(tuple(d))
    if n >= 2:
        rays.extend([tuple(sp.S.One for _ in u), tuple(-sp.S.One for _ in u)])
    values = []
    for d in rays:
        sub = {v: t * a for v, a in zip(u, d, strict=True)}
        try:
            g = sp.simplify(f.subs(sub))
            val = bounded_limit(g, t, 0, direction="+", allow_general=True)
        except (TypeError, ValueError, NotImplementedError, RecursionError):
            continue
        if (
            val is None
            or isinstance(val, (sp.Set, sp.AccumBounds))
            or val.has(sp.nan, sp.zoo, sp.oo, -sp.oo)
        ):
            continue
        if all(sp.simplify(val - v) != 0 for v, _ in values):
            values.append((val, d))
        if len(values) >= 2:
            return ParityCertificate(
                "angular_cluster_dne",
                True,
                None,
                "two exact accumulating rays have distinct cluster values",
                tuple(values[:2]),
            )
    return ParityCertificate("angular_cluster", False)


def inverse_branch_limit_certificate(expr, variables, target, *, _context=None):
    """One-sided real branch certificates for inverse functions at endpoints."""
    u, f = _shift_for(expr, variables, target, _context)
    if f.func is not sp.atanh or len(f.args) != 1:
        return ParityCertificate("inverse_branch", False)
    z = sp.factor(f.args[0])
    # atanh(z)->+oo as z->1 from inside (-1,1).  Prove the common rational
    # 2a/(1+a^2+b^2) form by the exact identity 1-z=((a-1)^2+b^2)/den.
    delta = sp.factor(1 - z)
    num, den = sp.fraction(delta)
    if sp.simplify(delta.subs(dict.fromkeys(u, 0))) != 0:
        return ParityCertificate("inverse_branch", False)
    try:
        sp.Poly(num, *u)
    except sp.PolynomialError:
        return ParityCertificate("inverse_branch", False)
    if den.subs(dict.fromkeys(u, 0)).is_positive is not True:
        return ParityCertificate("inverse_branch", False)
    # Sum-of-squares / nonnegative quadratic numerator is enough here.
    h = sp.hessian(num, u)
    if num.subs(dict.fromkeys(u, 0)) == 0 and all(
        sp.simplify(x) == 0
        for x in [sp.diff(num, v).subs(dict.fromkeys(u, 0)) for v in u]
    ):
        eig = [sp.simplify(e) for e in h.eigenvals()]
        if (
            eig
            and all(e.is_nonnegative is True for e in eig)
            and any(e.is_positive is True for e in eig)
        ):
            return ParityCertificate(
                "inverse_branch",
                True,
                sp.oo,
                "atanh argument approaches 1 from its real interior branch",
                (delta,),
            )
    return ParityCertificate("inverse_branch", False)


def singular_polylog_special_germ_certificate(
    expr, variables, target, *, _context=None
):
    """Dominant singular Bessel-Y products with certified polylog remainder."""
    u, f = _shift_for(expr, variables, target, _context)
    # Normalize every removable z*Y1(z) factor to its finite leading constant.
    reduced = f
    changed = False
    for atom in list(f.atoms(sp.bessely)):
        if atom.args[0] != 1:
            continue
        z = atom.args[1]
        # x*Y1(x) -> -2/pi, with O(x^2 log|x|) remainder.
        pat = z * atom
        if reduced.has(pat):
            reduced = sp.expand(reduced).subs(pat, -2 / sp.pi)
            changed = True
    if not changed:
        # Expanded multiplication may hide the product; algebraically divide each term.
        terms = []
        for term in sp.Add.make_args(sp.expand(f)):
            done = False
            for atom in term.atoms(sp.bessely):
                if atom.args[0] == 1:
                    z = atom.args[1]
                    q = sp.cancel(term / (z * atom))
                    if not q.has(atom):
                        terms.append(sp.simplify(-2 * q / sp.pi))
                        done = True
                        changed = True
                        break
            if not done:
                terms.append(term)
        reduced = sp.Add(*terms)
    if not changed:
        return ParityCertificate("singular_polylog_germ", False)
    from .limits import LimitStatus, limit

    r = limit(reduced, u, (0,) * len(u), return_result=True)
    if r.status is LimitStatus.PROVED:
        return ParityCertificate(
            "singular_polylog_germ",
            True,
            r.value,
            "Y1 singular germ with O(z^2 log|z|) remainder",
            (reduced,),
        )
    # A negative even reciprocal pole plus a finite continuous remainder is -oo.
    for v in u:
        pole = -1 / v**2
        rem = sp.simplify(reduced - pole)
        try:
            rv = rem.subs(dict.fromkeys(u, 0), simultaneous=True)
        except (TypeError, ValueError, NotImplementedError):
            continue
        if not rv.has(sp.nan, sp.zoo, sp.oo, -sp.oo) and not rv.has(*u):
            return ParityCertificate(
                "singular_polylog_germ",
                True,
                -sp.oo,
                "negative even reciprocal pole dominates finite Bessel polylog remainder",
                (reduced,),
            )
    return ParityCertificate("singular_polylog_germ", False, data=(reduced,))


def exponential_small_power_certificate(expr, variables, target, *, _context=None):
    """Certify ``(1+h)**(1/k) -> 1`` when ``h/k -> 0`` uniformly.

    The implemented form covers products of real coordinates over sums of their
    absolute values.  The proof uses ``|xy|/(|x|+|y|) <= min(|x|,|y|)`` and
    ``log(1+h)=h+O(h**2)``.
    """
    u, f = _shift_for(expr, variables, target, _context)
    if not f.is_Pow:
        return ParityCertificate("exponential_small_power", False)
    base, exponent = f.as_base_exp()
    h = sp.expand(base - 1)
    if not h.has(*u):
        return ParityCertificate("exponential_small_power", False)
    den = sp.simplify(1 / exponent)
    abs_sum = sp.Add(*(sp.Abs(v) for v in u))
    if sp.simplify(den - abs_sum) != 0:
        return ParityCertificate("exponential_small_power", False)
    factors = sp.Mul.make_args(h)
    variable_factors = [a for a in factors if a in u]
    constant = sp.simplify(h / sp.Mul(*variable_factors)) if variable_factors else h
    if len(variable_factors) < 2 or constant.has(*u):
        return ParityCertificate("exponential_small_power", False)
    return ParityCertificate(
        "exponential_small_power",
        True,
        sp.S.One,
        "logarithm of the power is bounded by a vanishing coordinate product over an absolute-value radius",
        (h, den),
    )


def variable_power_branch_certificate(expr, variables, target, *, _context=None):
    """Detect principal-power branch conflict at a vanishing signed base."""
    u, f = _shift_for(expr, variables, target, _context)
    if not f.is_Pow:
        return ParityCertificate("variable_power_branch", False)
    base, exponent = f.as_base_exp()
    zero = dict.fromkeys(u, 0)
    if sp.simplify(base.subs(zero)) != 0:
        return ParityCertificate("variable_power_branch", False)
    try:
        e0 = sp.simplify(exponent.subs(zero))
    except (TypeError, ValueError):
        return ParityCertificate("variable_power_branch", False)
    if e0.is_real is not True or e0.is_negative is not True or e0.is_integer is True:
        return ParityCertificate("variable_power_branch", False)
    # A nonzero linear part gives approaches with opposite signs of the base.
    grad = [sp.simplify(sp.diff(base, v).subs(zero)) for v in u]
    if not any(g.is_nonzero is True for g in grad):
        return ParityCertificate("variable_power_branch", False)
    return ParityCertificate(
        "variable_power_branch_dne",
        True,
        None,
        "opposite signs of a vanishing real base approach different principal-power infinity directions",
        (base, e0, tuple(grad)),
    )


def complex_sech_phase_certificate(expr, variables, target, *, _context=None):
    """Certify DNE for a simple complex pole inside periodic ``sech``."""
    u, f = _shift_for(expr, variables, target, _context)
    powers = [
        a for a in sp.preorder_traversal(f) if a.is_Pow and a.base.func is sp.sech
    ]
    atoms = list(f.atoms(sp.sech))
    if len(atoms) != 1 or (f != atoms[0] and not powers):
        return ParityCertificate("complex_sech_phase", False)
    z = atoms[0].args[0]
    # On some coordinate axis require z = a + b*I/t exactly.  Two phase
    # subsequences differing by pi/2 then give distinct finite sech^2 values.
    t = sp.Dummy("_sech_t", positive=True)
    for v in u:
        axis = sp.simplify(z.subs({w: (t if w == v else 0) for w in u}))
        a = sp.simplify(
            bounded_limit(sp.re(axis), t, 0, direction="+", allow_general=True)
        )
        b = sp.simplify(
            bounded_limit(t * sp.im(axis), t, 0, direction="+", allow_general=True)
        )
        if a.is_finite is not True or b.is_real is not True or b.is_nonzero is not True:
            continue
        return ParityCertificate(
            "complex_sech_phase_dne",
            True,
            None,
            "a complex reciprocal phase inside sech has two accumulating periodic phase subsequences",
            (v, a, b),
        )
    return ParityCertificate("complex_sech_phase", False)


def discontinuity_branch_certificate(expr, variables, target, *, _context=None):
    """Certified discontinuity witnesses for selected real branch functions."""
    u, f = _shift_for(expr, variables, target, _context)
    # FractionalPart of a real rational pole has distinct accumulating phase
    # subsequences.  Restricting to a coordinate axis gives a rigorous witness.
    if f.func is sp.frac and len(f.args) == 1:
        arg = f.args[0]
        t = sp.Dummy("_frac_t", positive=True)
        for v in u:
            axis = sp.cancel(arg.subs({w: (t if w == v else 0) for w in u}))
            try:
                pole = bounded_limit(
                    sp.log(sp.Abs(axis)) / sp.log(t),
                    t,
                    0,
                    direction="+",
                    allow_general=True,
                )
            except (TypeError, ValueError, NotImplementedError, sp.PoleError):
                continue
            if pole.is_number and pole.is_negative is True:
                return ParityCertificate(
                    "fractional_part_dne",
                    True,
                    None,
                    "a real rational-pole axis has fractional-part subsequences with phases 0 and 1/2",
                    (v, axis, sp.S.Zero, sp.Rational(1, 2)),
                )
    return ParityCertificate("discontinuity_branch", False)
