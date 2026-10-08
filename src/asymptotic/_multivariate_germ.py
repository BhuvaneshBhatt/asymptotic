"""Polynomial/order/sign certificates for multivariate germs."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import lcm

import sympy as sp

from ._symbolic_policy import bounded_limit, bounded_solve_one


@dataclass(frozen=True)
class ParityCertificate:
    method: str
    certified: bool
    value: sp.Expr | None = None
    statement: str = ""
    data: tuple = ()


def _shift(expr, variables, target):
    u = sp.symbols(f"_u0:{len(variables)}", real=True)
    return u, sp.cancel(
        sp.sympify(expr).subs(
            dict(
                zip(
                    variables,
                    (a + x for a, x in zip(target, u, strict=True)),
                    strict=True,
                )
            )
        )
    )


@dataclass(frozen=True)
class GermAnalysis:
    """Lazy per-evaluation symbolic normal forms shared by fast certificates."""

    expression: sp.Expr
    variables: tuple[sp.Symbol, ...]
    target: tuple[sp.Expr, ...]
    _cache: dict = field(
        default_factory=dict, init=False, repr=False, compare=False, hash=False
    )

    @classmethod
    def create(cls, expression, variables, target):
        return cls(
            sp.sympify(expression), tuple(variables), tuple(map(sp.sympify, target))
        )

    def _get(self, key, factory):
        cache = self._cache
        if key not in cache:
            cache[key] = factory()
        return cache[key]

    @property
    def shifted(self):
        return self._get(
            "shifted", lambda: _shift(self.expression, self.variables, self.target)
        )

    @property
    def fraction(self):
        def build():
            u, shifted = self.shifted
            num, den = sp.fraction(sp.cancel(shifted))
            return u, sp.expand(num), sp.expand(den)

        return self._get("fraction", build)


def _shift_for(expr, variables, target, context=None):
    if context is not None:
        return context.shifted
    return _shift(expr, variables, target)


def _poly_order(expr, variables):
    try:
        p = sp.Poly(sp.expand(expr), *variables)
    except (sp.PolynomialError, TypeError, ValueError):
        return None
    if p.is_zero:
        return sp.oo
    return min(sum(m) for m, c in p.terms() if c != 0)


def isolated_zero_certificate(poly, variables, target):
    """Cheap exact isolated-zero certificate for sign-definite polynomial germs."""
    u, f = _shift(poly, variables, target)
    try:
        p = sp.Poly(sp.expand(f), *u)
    except (sp.PolynomialError, TypeError, ValueError):
        return ParityCertificate("isolated_zero", False)
    if p.TC() != 0:
        return ParityCertificate("isolated_zero", True, sp.S.false, "nonzero at target")
    terms = p.terms()
    # Sum of positive even pure powers is positive away from the origin.
    if terms and all(
        c.is_positive is True and sum(m) % 2 == 0 and sum(v != 0 for v in m) == 1
        for m, c in terms
    ):
        covered = {i for m, _ in terms for i, e in enumerate(m) if e}
        if covered == set(range(len(u))):
            return ParityCertificate(
                "isolated_zero",
                True,
                sp.S.true,
                "positive even-power form has an isolated zero",
            )
    return ParityCertificate("isolated_zero", False)


def _cheap_lojasiewicz_exponent_bound(expr, variables, target):
    """Certify a local radial Łojasiewicz lower bound for simple polynomial germs.

    Returns exponent ``m`` when ``|f(x)| >= c ||x-a||**m`` is certified locally.
    The current cheap path covers sign-definite sums of even pure powers.
    """
    u, f = _shift(expr, variables, target)
    try:
        p = sp.Poly(sp.expand(f), *u)
    except (sp.PolynomialError, TypeError, ValueError):
        return ParityCertificate("lojasiewicz", False)
    pure = []
    for monom, coeff in p.terms():
        if coeff.is_positive is not True or sum(e != 0 for e in monom) != 1:
            return ParityCertificate("lojasiewicz", False)
        degree = sum(monom)
        if degree % 2:
            return ParityCertificate("lojasiewicz", False)
        pure.append((monom, degree))
    covered = {i for m, _ in pure for i, e in enumerate(m) if e}
    if covered != set(range(len(u))):
        return ParityCertificate("lojasiewicz", False)
    exponent = max(d for _, d in pure)
    return ParityCertificate(
        "lojasiewicz",
        True,
        sp.Integer(exponent),
        f"positive pure-power germ dominates a radial power of order {exponent}",
    )


def sertoz_rational_limit(expr, variables, target, *, _context=None):
    """Cheap exact rational criterion based on homogeneous leading orders."""
    u, f = _shift_for(expr, variables, target, _context)
    num, den = map(sp.expand, sp.fraction(sp.cancel(f)))
    on, od = _poly_order(num, u), _poly_order(den, u)
    if on is None or od is None or od is sp.oo:
        return ParityCertificate("sertoz_rational", False)
    if on is sp.oo:
        return ParityCertificate(
            "sertoz_rational", True, sp.S.Zero, "zero numerator germ"
        )
    iso = isolated_zero_certificate(den, u, (0,) * len(u))
    _loj_fast = _cheap_lojasiewicz_exponent_bound
    loj = _loj_fast(den, u, (0,) * len(u))
    if (
        iso.certified
        and iso.value is sp.S.true
        and loj.certified
        and on > int(loj.value)
    ):
        return ParityCertificate(
            "sertoz_rational",
            True,
            sp.S.Zero,
            "numerator order exceeds a certified denominator Łojasiewicz exponent",
        )
    return ParityCertificate("sertoz_rational", False, data=(on, od))


def _leading_homogeneous(poly, variables):
    try:
        p = sp.Poly(sp.expand(poly), *variables)
    except (sp.PolynomialError, TypeError, ValueError):
        return None, None
    if p.is_zero:
        return sp.oo, sp.S.Zero
    d = min(sum(m) for m, c in p.terms() if c != 0)
    return d, sp.Add(
        *(
            c * sp.prod(x**e for x, e in zip(variables, m, strict=True))
            for m, c in p.terms()
            if sum(m) == d
        )
    )


def _weighted_poly_order(poly, variables, weights):
    """Return the exact positive-weight valuation of a polynomial."""
    from .blowup_geometry import Valuation

    try:
        return Valuation(tuple(variables), tuple(weights)).polynomial(poly)
    except (TypeError, ValueError):
        return None


def _weighted_leading(poly, variables, weights):
    """Return ``(valuation, initial form)`` from the common valuation engine."""
    from .blowup_geometry import Valuation

    try:
        valuation = Valuation(tuple(variables), tuple(weights))
    except (TypeError, ValueError):
        return None, None
    degree = valuation.polynomial(poly)
    if degree is None:
        return None, None
    if degree is sp.oo:
        return sp.oo, sp.S.Zero
    return degree, valuation.initial_form(poly)


def _quadratic_definite_sign(poly, variables):
    """Exact Sylvester certificate for a homogeneous real quadratic form."""
    try:
        p = sp.Poly(sp.expand(poly), *variables)
    except (sp.PolynomialError, TypeError, ValueError):
        return None
    if p.is_zero or p.total_degree() != 2:
        return None
    if any(sum(monom) != 2 for monom, coeff in p.terms() if coeff != 0):
        return None
    matrix = sp.hessian(p.as_expr(), variables) / 2
    positive = True
    negative = True
    for size in range(1, len(variables) + 1):
        minor = sp.simplify(matrix[:size, :size].det())
        if minor.is_positive is not True:
            positive = False
        expected_negative = -1 if size % 2 else 1
        signed = sp.simplify(expected_negative * minor)
        if signed.is_positive is not True:
            negative = False
    if positive:
        return 1
    if negative:
        return -1
    return None


def _obvious_definite(poly, variables):
    """Return +1/-1 for a conservative exact definite homogeneous form."""
    try:
        p = sp.Poly(sp.expand(poly), *variables)
    except (sp.PolynomialError, TypeError, ValueError):
        return None
    if p.total_degree() == 0:
        c = p.as_expr()
        return 1 if c.is_positive is True else -1 if c.is_negative is True else None
    quadratic = _quadratic_definite_sign(p.as_expr(), variables)
    if quadratic is not None:
        return quadratic
    signs = []
    pure_covered = set()
    for monom, coeff in p.terms():
        # Same-sign sums of even monomials are semidefinite.  Strict
        # punctured-neighborhood definiteness additionally requires a positive
        # pure even power for every variable; products such as x**2*y**2 alone
        # vanish on coordinate axes and are not definite.
        if any(e % 2 for e in monom) or coeff == 0:
            return None
        sg = (
            1
            if coeff.is_positive is True
            else -1
            if coeff.is_negative is True
            else None
        )
        if sg is None:
            return None
        signs.append(sg)
        if sum(e != 0 for e in monom) == 1:
            pure_covered.update(i for i, e in enumerate(monom) if e)
    if pure_covered != set(range(len(variables))):
        return None
    if signs and all(x == 1 for x in signs):
        return 1
    if signs and all(x == -1 for x in signs):
        return -1
    return None


def _coercive_polynomial_model(poly, variables):
    """Certify a weighted punctured-neighborhood sign/coercivity model.

    The return value is ``(weights, degree, sign, principal_form, provider)``.
    Two cheap exact families are recognized:

    * a definite quadratic initial form (ordinary radial weights), and
    * a positive/negative pure-even-power core covering every variable, with
      every sign-indefinite perturbation of strictly higher weighted order.

    In the second family, same-sign even cross monomials may occur at any
    weighted order because they only strengthen the lower bound.  The proof is
    the standard compact weighted-blow-up argument: the pure-power core is
    strictly positive on the angular sphere, while all unsafe perturbations are
    uniformly ``o(r**degree)``.
    """
    expr = sp.expand(poly)
    try:
        p = sp.Poly(expr, *variables)
    except (sp.PolynomialError, TypeError, ValueError):
        return None
    if p.is_zero:
        return None

    ordinary_degree, ordinary_lead = _leading_homogeneous(expr, variables)
    if ordinary_degree == 2:
        sign = _quadratic_definite_sign(ordinary_lead, variables)
        if sign is not None:
            return (
                (1,) * len(variables),
                2,
                sign,
                ordinary_lead,
                "definite_quadratic_initial_form",
            )

    for sign in (1, -1):
        oriented = sp.Poly(sp.expand(sign * expr), *variables)
        pure = {}
        for monom, coeff in oriented.terms():
            nonzero = [i for i, exponent in enumerate(monom) if exponent]
            if len(nonzero) != 1 or coeff.is_positive is not True:
                continue
            i = nonzero[0]
            exponent = int(monom[i])
            if exponent % 2:
                continue
            previous = pure.get(i)
            if previous is None or exponent < previous[0]:
                pure[i] = (exponent, coeff, monom)
        if set(pure) != set(range(len(variables))):
            continue
        exponents = tuple(pure[i][0] for i in range(len(variables)))
        degree = 1
        for exponent in exponents:
            degree = lcm(degree, exponent)
        weights = tuple(degree // exponent for exponent in exponents)
        core_monomials = {pure[i][2] for i in range(len(variables))}
        unsafe = False
        for monom, coeff in oriented.terms():
            if monom in core_monomials:
                continue
            same_sign_nonnegative = coeff.is_nonnegative is True and all(
                e % 2 == 0 for e in monom
            )
            if same_sign_nonnegative:
                continue
            weighted_degree = sum(
                int(e) * int(w) for e, w in zip(monom, weights, strict=True)
            )
            if weighted_degree <= degree:
                unsafe = True
                break
        if unsafe:
            continue
        principal = sp.Add(
            *(pure[i][1] * variables[i] ** pure[i][0] for i in range(len(variables)))
        )
        return (
            weights,
            degree,
            sign,
            sp.expand(sign * principal),
            "weighted_pure_even_core",
        )
    return None


def _structural_semidefinite_sign(poly, variables):
    """Return a fixed weak sign for simple polynomial squares/even powers."""
    expr = sp.factor(sp.sympify(poly))
    coeff, rest = expr.as_coeff_Mul()
    coeff_sign = (
        1 if coeff.is_positive is True else -1 if coeff.is_negative is True else None
    )
    if coeff_sign is None:
        return None
    if rest.is_Pow and rest.exp.is_integer is True and rest.exp.is_even is True:
        return coeff_sign
    if _obvious_nonnegative(sp.expand(rest), variables) is True:
        return coeff_sign
    return None


def _rational_path_limit(expr, variables, path):
    """Exact ``t -> 0+`` limit on a polynomial/rational path."""
    t = sp.Dummy("_t", positive=True)
    path_symbols = set().union(
        *(sp.sympify(component).free_symbols for component in path)
    )
    if len(path_symbols) > 1:
        return None
    source_t = next(iter(path_symbols), None)
    normalized_path = tuple(
        (
            sp.sympify(component).subs(source_t, t)
            if source_t is not None
            else sp.sympify(component)
        )
        for component in path
    )
    substituted = sp.cancel(
        sp.sympify(expr).subs(
            dict(zip(variables, normalized_path, strict=True)),
            simultaneous=True,
        )
    )
    numerator, denominator = sp.fraction(substituted)
    try:
        npoly = sp.Poly(sp.expand(numerator), t, extension=True)
        dpoly = sp.Poly(sp.expand(denominator), t, extension=True)
    except (sp.PolynomialError, TypeError, ValueError):
        return None
    if dpoly.is_zero:
        return None

    def valuation_and_coefficient(poly):
        terms = poly.terms()
        exponent = min(monom[0] for monom, coeff in terms if coeff != 0)
        coefficient = sp.Add(*(coeff for monom, coeff in terms if monom[0] == exponent))
        return exponent, sp.simplify(coefficient)

    od, cd = valuation_and_coefficient(dpoly)
    if cd.is_finite is not True or cd.is_zero is not False:
        return None
    if npoly.is_zero:
        return sp.S.Zero
    if any(c.is_finite is not True for c in npoly.all_coeffs()):
        return None
    on, cn = valuation_and_coefficient(npoly)
    if on > od:
        return sp.S.Zero
    leading = sp.simplify(cn / cd)
    if on == od:
        return leading
    if leading.is_positive is True:
        return sp.oo
    if leading.is_negative is True:
        return -sp.oo
    return None


def _denominator_zero_directions(leading, variables):
    """Find exact low-degree real directions on a leading denominator variety."""
    degree = sp.Poly(sp.expand(leading), *variables).total_degree()
    directions = []
    n = len(variables)
    if degree == 1:
        coeffs = [sp.expand(leading).coeff(v) for v in variables]
        for i in range(n):
            for j in range(i + 1, n):
                if coeffs[i] == 0 and coeffs[j] == 0:
                    continue
                vector = [sp.S.Zero] * n
                vector[i] = coeffs[j]
                vector[j] = -coeffs[i]
                if any(value != 0 for value in vector):
                    directions.append(tuple(vector))
                    break
            if directions:
                break
    elif degree == 2:
        matrix = sp.hessian(leading, variables) / 2
        for vector in matrix.nullspace():
            direction = tuple(sp.simplify(value) for value in vector)
            if any(value != 0 for value in direction):
                directions.append(direction)
        if n == 2:
            x, y = variables
            for fixed, solve_for in ((y, x), (x, y)):
                polynomial = sp.expand(leading.subs(fixed, 1))
                try:
                    roots = (
                        bounded_solve_one(
                            sp.Eq(polynomial, 0), solve_for, allow_general=True
                        )
                        or ()
                    )
                except (NotImplementedError, TypeError, ValueError):
                    roots = []
                for root in roots:
                    if root.is_real is False:
                        continue
                    direction = (root, sp.S.One) if solve_for == x else (sp.S.One, root)
                    value = sp.simplify(
                        leading.subs(dict(zip(variables, direction, strict=True)))
                    )
                    if value == 0:
                        directions.append(direction)
    unique = []
    for direction in directions:
        if direction not in unique:
            unique.append(direction)
    return tuple(unique)


def _denominator_variety_dne_certificate(expr, variables, target, shifted=None):
    """Prove rational DNE by two exact polynomial paths, one near a pole variety."""
    if len(variables) > 3:
        return ParityCertificate("candidate_rational", False)
    u, f = shifted if shifted is not None else _shift(expr, variables, target)
    numerator, denominator = map(sp.expand, sp.fraction(sp.cancel(f)))
    if not (numerator.is_polynomial(*u) and denominator.is_polynomial(*u)):
        return ParityCertificate("candidate_rational", False)
    od, leading_denominator = _leading_homogeneous(denominator, u)
    if od not in (1, 2):
        return ParityCertificate("candidate_rational", False)
    directions = _denominator_zero_directions(leading_denominator, u)
    if not directions:
        return ParityCertificate("candidate_rational", False)

    path_t = sp.Symbol("_path_t", positive=True)
    witnesses = []
    simple_directions = [
        tuple(sp.S.One if i == j else sp.S.Zero for i in range(len(u)))
        for j in range(len(u))
    ]
    simple_directions.append((sp.S.One,) * len(u))
    for direction in simple_directions:
        value = _rational_path_limit(
            f, u, tuple(component * path_t for component in direction)
        )
        if value is not None and not value.has(sp.nan, sp.zoo):
            witnesses.append(
                (tuple(component * path_t for component in direction), value)
            )

    for direction in directions:
        direct_path = tuple(component * path_t for component in direction)
        value = _rational_path_limit(f, u, direct_path)
        if value is not None and not value.has(sp.nan, sp.zoo):
            witnesses.append((direct_path, value))
        for perturb_index in range(len(u)):
            for exponent in range(2, 7):
                perturbed = tuple(
                    component * path_t + (path_t**exponent if i == perturb_index else 0)
                    for i, component in enumerate(direction)
                )
                value = _rational_path_limit(f, u, perturbed)
                if value is not None and not value.has(sp.nan, sp.zoo):
                    witnesses.append((perturbed, value))

    for i, (path_a, value_a) in enumerate(witnesses):
        for path_b, value_b in witnesses[i + 1 :]:
            from ._limit_composition import _exact_equal

            if _exact_equal(value_a, value_b) is not False:
                continue
            denominator_degree = sp.Poly(denominator, *u).total_degree()
            if denominator_degree > 1:
                method = "exceptional_curve:rational_divisor_contact_dne"
            else:
                if len(u) == 2 and sp.Poly(leading_denominator, *u).degree() == 1:
                    cx = sp.diff(leading_denominator, u[0])
                    chart = "x_over_y" if cx != 0 else "y_over_x"
                else:
                    chart = "projective"
                method = f"exceptional_direction:{chart}_dne"
            return ParityCertificate(
                method,
                True,
                sp.nan,
                "two exact accumulating polynomial paths have different "
                "rational limits",
                (path_a, value_a, path_b, value_b, leading_denominator),
            )
    return ParityCertificate("candidate_rational", False)


def candidate_rational_limit_certificate(expr, variables, target, *, _context=None):
    """Candidate-centered exact certification for rational multivariate germs.

    Rather than treating only zero and pole candidates, the denominator is first
    given an exact local coercivity/sign model.  A finite candidate ``L`` is then
    certified by proving ``numerator - L*denominator`` has strictly larger
    weighted order.  If the denominator is not coercive, low-degree denominator
    varieties are searched for exact conflicting polynomial paths.  A separate
    semidefinite-denominator argument handles signed poles even when the zero set
    of the denominator is positive-dimensional.
    """
    u, f = _shift_for(expr, variables, target, _context)
    numerator, denominator = map(sp.expand, sp.fraction(sp.cancel(f)))
    if not (numerator.is_polynomial(*u) and denominator.is_polynomial(*u)):
        return ParityCertificate("candidate_rational", False)
    if denominator == 0:
        return ParityCertificate("candidate_rational", False)
    if numerator == 0:
        return ParityCertificate(
            "candidate_rational",
            True,
            sp.S.Zero,
            "identically zero rational numerator",
        )

    model = _coercive_polynomial_model(denominator, u)
    if model is not None:
        weights, degree, denominator_sign, principal, provider = model
        numerator_order, numerator_lead = _weighted_leading(numerator, u, weights)
        if numerator_order is sp.oo:
            return ParityCertificate(
                "candidate_rational", True, sp.S.Zero, "zero numerator germ"
            )
        if numerator_order > degree:
            return ParityCertificate(
                "candidate_rational",
                True,
                sp.S.Zero,
                "weighted numerator order exceeds a certified coercive "
                "denominator order",
                (weights, degree, denominator_sign, principal, provider),
            )
        if numerator_order == degree:
            denominator_order, denominator_lead = _weighted_leading(
                denominator, u, weights
            )
            if denominator_order != degree:
                denominator_lead = None
            try:
                candidate = (
                    None
                    if denominator_lead is None
                    else sp.cancel(numerator_lead / denominator_lead)
                )
            except (TypeError, ValueError, ZeroDivisionError):
                candidate = None
            if candidate is not None and not (candidate.free_symbols & set(u)):
                residual = sp.expand(numerator - candidate * denominator)
                residual_order = _weighted_poly_order(residual, u, weights)
                if residual_order is sp.oo or residual_order > degree:
                    return ParityCertificate(
                        "candidate_rational",
                        True,
                        sp.simplify(candidate),
                        "candidate residual has strictly higher weighted order "
                        "over a certified coercive denominator",
                        (
                            weights,
                            degree,
                            denominator_sign,
                            principal,
                            provider,
                            residual_order,
                        ),
                    )

    # A denominator may vanish on a local variety and still force a signed pole:
    # a fixed-sign numerator of lower radial order divided by a fixed weak-sign
    # denominator of higher order has magnitude bounded below by a negative
    # radial power on every point where the quotient is defined.
    numerator_model = _coercive_polynomial_model(numerator, u)
    denominator_semisign = _structural_semidefinite_sign(denominator, u)
    od = _poly_order(denominator, u)
    if numerator_model is not None:
        numerator_weights, numerator_degree = numerator_model[:2]
        numerator_lower_order = max(
            numerator_degree // weight for weight in numerator_weights
        )
    else:
        numerator_lower_order = None
    if (
        numerator_model is not None
        and denominator_semisign is not None
        and od not in (None, sp.oo)
        and numerator_lower_order < od
    ):
        numerator_sign = numerator_model[2]
        value = sp.oo if numerator_sign * denominator_semisign > 0 else -sp.oo
        return ParityCertificate(
            "candidate_rational",
            True,
            value,
            "a coercive numerator lower bound and a fixed-sign higher-order "
            "denominator certify a signed pole off the denominator variety",
            (
                numerator_lower_order,
                od,
                numerator_model,
                denominator_semisign,
            ),
        )

    # If a coercive denominator model exists but candidate subtraction did not
    # settle the quotient, preserve the established weighted/path machinery.
    # The new denominator-variety search is intended for genuinely noncoercive
    # leading varieties; running it here would merely steal cheap
    # monomial-curve evidence without extending the solved set.
    if model is not None:
        return ParityCertificate("candidate_rational", False)

    return _denominator_variety_dne_certificate(f, u, (0,) * len(u), shifted=(u, f))


def _obvious_nonnegative(poly, variables):
    try:
        p = sp.Poly(sp.expand(poly), *variables)
    except sp.PolynomialError:
        return None
    for monom, coeff in p.terms():
        if any(e % 2 for e in monom) or coeff.is_nonnegative is not True:
            return None
    return True


def _obvious_positive_punctured(poly, variables):
    try:
        p = sp.Poly(sp.expand(poly), *variables)
    except sp.PolynomialError:
        return None
    covered = set()
    for monom, coeff in p.terms():
        if any(e % 2 for e in monom) or coeff.is_nonnegative is not True:
            return None
        if coeff.is_positive is True and sum(e != 0 for e in monom) == 1:
            covered.update(i for i, e in enumerate(monom) if e)
    return covered == set(range(len(variables)))


def rational_power_order_certificate(expr, variables, target, *, _context=None):
    """Certify zero for polynomial-over-positive-rational-power germs by order.

    If the denominator is ``base**q`` with positive rational ``q`` and the
    leading homogeneous form of ``base`` is definite, then ``base**q`` is
    comparable to a radial power.  A strictly higher numerator order therefore
    proves a zero limit without introducing algebraic graph auxiliaries/CAD.
    """
    u, f = _shift_for(expr, variables, target, _context)
    num, den = sp.fraction(f)
    if not num.is_polynomial(*u) or not isinstance(den, sp.Pow):
        return ParityCertificate("rational_power_order", False)
    base, exponent = den.as_base_exp()
    if not (exponent.is_Rational and exponent > 0 and base.is_polynomial(*u)):
        return ParityCertificate("rational_power_order", False)
    dn, hn = _leading_homogeneous(sp.expand(num), u)
    db, hb = _leading_homogeneous(sp.expand(base), u)
    if dn in (None, sp.oo) or db in (None, sp.oo):
        return ParityCertificate("rational_power_order", False)
    if _obvious_definite(hb, u) != 1:
        return ParityCertificate(
            "rational_power_order", False, data=(dn, db, exponent, hb)
        )
    denominator_order = sp.Rational(db) * exponent
    if sp.Rational(dn) > denominator_order:
        return ParityCertificate(
            "rational_power_order",
            True,
            sp.S.Zero,
            "numerator order exceeds a certified positive rational-power denominator order",
            (dn, db, exponent, hn, hb),
        )
    return ParityCertificate("rational_power_order", False, data=(dn, db, exponent, hb))


def _nonnegative_rational_axis_minimum(expr, variables, target):
    """Return 0 when nonnegativity and an exact zero approach certify the local infimum."""
    u, f = _shift(expr, variables, target)
    num, den = map(sp.expand, sp.fraction(sp.cancel(f)))
    if not (num.is_polynomial(*u) and den.is_polynomial(*u)):
        return None
    # This accepts only syntactically nonnegative polynomial sums;
    # harder sign questions belong to the semialgebraic optimizer.
    if (
        _obvious_nonnegative(num, u) is not True
        or _obvious_positive_punctured(den, u) is not True
    ):
        return None
    for axis in u:
        subs = {z: sp.S.Zero for z in u if z != axis}
        restricted = sp.cancel(f.subs(subs))
        try:
            value = bounded_limit(
                restricted, axis, 0, direction="+-", allow_general=True
            )
        except (TypeError, ValueError, NotImplementedError):
            continue
        if value == 0:
            return sp.S.Zero
    return None


def local_sign_certificate(expr, variables, target):
    """Cheap exact local sign from the first nonzero homogeneous polynomial."""
    u, f = _shift(expr, variables, target)
    if not f.is_polynomial(*u):
        return ParityCertificate("local_sign", False)
    degree, lead = _leading_homogeneous(f, u)
    sign = _obvious_definite(lead, u) if degree is not sp.oo else None
    if sign is not None:
        return ParityCertificate(
            "local_sign",
            True,
            sp.Integer(sign),
            "leading homogeneous form has certified strict punctured-neighborhood sign",
            (degree, lead),
        )
    return ParityCertificate("local_sign", False, data=(degree, lead))


def _axis_signs(poly, variables):
    vals = set()
    for x in variables:
        for a in (sp.S.One, -sp.S.One):
            sub = dict.fromkeys(variables, sp.S.Zero)
            sub[x] = a
            v = sp.simplify(poly.subs(sub))
            if v.is_positive is True:
                vals.add(1)
            if v.is_negative is True:
                vals.add(-1)
    return vals


def reciprocal_pole_certificate(expr, variables, target, *, _context=None):
    """Certify signed infinite rational limits or sign-changing pole DNE cheaply."""
    u, f = _shift_for(expr, variables, target, _context)
    num, den = sp.fraction(sp.cancel(f))
    if not (num.is_polynomial(*u) and den.is_polynomial(*u)):
        return ParityCertificate("reciprocal_pole", False)
    on, ln = _leading_homogeneous(num, u)
    od, ld = _leading_homogeneous(den, u)
    if on in (None, sp.oo) or od in (None, sp.oo) or on >= od:
        return ParityCertificate("reciprocal_pole", False)
    sn, sd = _obvious_definite(ln, u), _obvious_definite(ld, u)
    if sn is not None and sd is not None:
        value = sp.oo if sn * sd > 0 else -sp.oo
        return ParityCertificate(
            "reciprocal_pole",
            True,
            value,
            "certified signs and lower numerator order imply a signed pole",
            (on, od, ln, ld),
        )
    # Sign variation in either leading form yields opposite signed pole paths
    # when the other form has a fixed nonzero sign.
    nvar, dvar = _axis_signs(ln, u), _axis_signs(ld, u)
    if (sd is not None and nvar == {1, -1}) or (sn is not None and dvar == {1, -1}):
        return ParityCertificate(
            "reciprocal_pole_dne",
            True,
            sp.nan,
            "opposite signed pole paths certify nonexistence",
            (on, od, ln, ld),
        )
    # Anisotropic positive germs can have no useful total-degree leading sign.
    # If the reciprocal is independently certified to tend to zero and both
    # polynomials are positive on the punctured neighborhood, the original
    # quotient has a positive infinite limit.
    if (
        _obvious_positive_punctured(num, u) is True
        and _obvious_positive_punctured(den, u) is True
    ):
        inverse = sertoz_rational_limit(sp.cancel(den / num), u, (0,) * len(u))
        if inverse.certified and inverse.value == 0:
            return ParityCertificate(
                "reciprocal_pole",
                True,
                sp.oo,
                "positive anisotropic reciprocal is certified to vanish",
                (on, od, inverse),
            )
    return ParityCertificate("reciprocal_pole", False)
