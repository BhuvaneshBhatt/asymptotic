from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache

import sympy as sp

from ._linear_ode_operator import linear_operator_coefficients
from ._symbolic_errors import SYMBOLIC_ERRORS
from ._symbolic_policy import bounded_limit, bounded_simplify
from ._symbolic_primitives import certification_primitive
from .context import AsymptoticContext
from .function_properties import PropertyDecision, PropertyKnowledge, PropertyProvenance
from .instrumentation import record_symbolic_event
from .remainder import AsymptoticRemainder, RemainderKind, RemainderProvenance


@dataclass(frozen=True)
class RemainderTheoremCertificate:
    """Auditable proof record for an operation-specific remainder theorem."""

    theorem: str
    hypotheses: tuple[PropertyDecision, ...]
    conclusion: AsymptoticRemainder
    note: str | None = None

    def __post_init__(self) -> None:
        if self.conclusion.kind is RemainderKind.UNKNOWN:
            record_symbolic_event("unknown_remainders")

    @property
    def certified(self) -> bool:
        return all(h.verdict is True for h in self.hypotheses) and self.conclusion.is_certified

    def replay(self) -> bool | None:
        """Replay theorem evidence without strengthening unresolved hypotheses."""

        if any(h.verdict is False for h in self.hypotheses):
            return False
        if any(h.verdict is not True for h in self.hypotheses):
            return None
        if not self.conclusion.is_certified:
            return False
        checked = self.conclusion.check()
        return True if checked is None else checked


@dataclass(frozen=True)
class GreenMode:
    """One homogeneous mode used by a certified scalar Green inverse."""

    characteristic_root: sp.Expr
    multiplicity_index: int
    expression: sp.Expr
    dichotomy_side: str
    relation_to_particular: str | None = None


@dataclass(frozen=True)
class ExponentialDichotomyCertificate:
    """Exact hyperbolic splitting for a constant-coefficient scalar operator.

    ``stable_modes`` decay at the requested end and ``unstable_modes`` grow
    there.  Center modes make the certificate unresolved; they are never
    silently assigned to either side.
    """

    variable: sp.Symbol
    point: sp.Expr
    characteristic_poly: sp.Expr
    stable_modes: tuple[GreenMode, ...]
    unstable_modes: tuple[GreenMode, ...]
    center_roots: tuple[sp.Expr, ...]

    @property
    def certified(self) -> bool:
        return not self.center_roots

    def replay(self) -> bool:
        lam = sp.Symbol("__lambda")
        modes = (*self.stable_modes, *self.unstable_modes)
        for mode in modes:
            if sp.simplify(self.characteristic_poly.subs(lam, mode.characteristic_root)) != 0:
                return False
            side = _root_half_plane(mode.characteristic_root, self.point)
            expected = "stable" if side == -1 else "unstable" if side == 1 else "center"
            if mode.dichotomy_side != expected:
                return False
        return not self.center_roots


@dataclass(frozen=True)
class GreenOperatorCertificate:
    """Replayable Green/right-inverse construction.

    Constant-coefficient certificates may be exact.  For an asymptotically
    constant operator, ``dichotomy`` and ``particular`` belong to the limiting
    operator while ``defect`` is replayed in the full operator.
    """

    order: int
    coefficients: tuple[sp.Expr, ...]
    forcing: sp.Expr
    particular: sp.Expr | None
    defect: sp.Expr | None
    dichotomy: ExponentialDichotomyCertificate | None
    selected_end_condition: str
    limiting_coefficients: tuple[sp.Expr, ...] | None = None
    coeff_perturbations: tuple[sp.Expr, ...] | None = None
    perturbation_limits: tuple[sp.Expr | None, ...] | None = None

    @property
    def asymptotically_constant(self) -> bool:
        """Whether the dichotomy belongs to a constant limiting operator."""

        return self.limiting_coefficients is not None

    @property
    def exact_right_inverse(self) -> bool:
        return self.particular is not None and self.defect == 0

    def replay(self, variable: sp.Symbol | None = None) -> bool | None:
        if self.particular is None:
            return None
        x = variable or (self.dichotomy.variable if self.dichotomy is not None else None)
        if x is None:
            return self.defect == 0
        defect = sp.simplify(
            sum(
                self.coefficients[k] * sp.diff(self.particular, x, k) for k in range(self.order + 1)
            )
            + self.forcing
        )
        if defect != 0:
            return False
        return self.dichotomy is None or self.dichotomy.replay()

    def replay_asymptotic(self, variable: sp.Symbol | None = None) -> bool | None:
        """Replay an asymptotically-constant Green certificate.

        For a genuinely variable operator this verifies coefficient convergence,
        the limiting dichotomy, and ``L(q)+R=o(R)``.  Constant-coefficient
        certificates delegate to :meth:`replay`.
        """

        if not self.asymptotically_constant:
            return self.replay(variable)
        if self.particular is None or self.coeff_perturbations is None:
            return None
        x = variable or (self.dichotomy.variable if self.dichotomy is not None else None)
        if x is None:
            return None
        point = self.dichotomy.point if self.dichotomy is not None else sp.oo
        for perturbation in self.coeff_perturbations:
            limit = bounded_limit(perturbation, x, point)
            if limit != 0:
                return False if limit is not None else None
        if self.dichotomy is None or not self.dichotomy.replay():
            return False
        if self.defect is None or self.forcing == 0:
            return None
        defect_ratio = bounded_limit(sp.Abs(self.defect / self.forcing), x, point)
        if defect_ratio == 0:
            return True
        return False if defect_ratio is not None else None


def _decision(name: str, verdict: bool | None, *, reason: str, source: str) -> PropertyDecision:
    return PropertyDecision(
        sp.Symbol(name),
        verdict,
        sp.S.true,
        PropertyKnowledge.SUFFICIENT,
        (PropertyProvenance(source, note=reason),),
        (reason,),
    )


def _classify_exact_error(
    exact_error: sp.Expr,
    candidate_scale: sp.Expr,
    variable: sp.Symbol,
    point: sp.Expr,
    *,
    source: str,
) -> AsymptoticRemainder:
    exact_error = sp.simplify(exact_error)
    candidate_scale = sp.simplify(candidate_scale)
    if exact_error == 0:
        return AsymptoticRemainder.exact_zero(variable, point, source=source)
    if candidate_scale == 0:
        return AsymptoticRemainder.unknown(
            variable, point, exact_expression=exact_error, source=source
        )
    ctx = AsymptoticContext(variable, point=point)
    try:
        lim = ctx.limit(sp.Abs(sp.simplify(exact_error / candidate_scale)))
    except SYMBOLIC_ERRORS:
        lim = None
    if lim == 0:
        return AsymptoticRemainder.little_o(
            candidate_scale, variable, point, exact_expression=exact_error, source=source
        )
    if lim not in (None, sp.oo, -sp.oo, sp.zoo) and getattr(lim, "is_finite", None) is True:
        return AsymptoticRemainder.big_o(
            candidate_scale, variable, point, exact_expression=exact_error, source=source
        )
    return AsymptoticRemainder.unknown(variable, point, exact_expression=exact_error, source=source)


def _safe_sum_remainders(
    remainders: Iterable[AsymptoticRemainder],
    *,
    source: str,
) -> AsymptoticRemainder:
    """Combine finitely many certified remainders without requiring scale comparability."""

    items = tuple(remainders)
    if not items:
        raise ValueError("at least one remainder is required")
    result = items[0]
    for item in items[1:]:
        combined = result.add(item)
        if combined.is_certified:
            result = combined
            continue
        if result.kind.name == "UNKNOWN" or item.kind.name == "UNKNOWN":
            result = combined
            continue
        scales = [sp.Abs(sp.sympify(r.scale)) for r in (result, item) if r.scale is not None]
        if not scales:
            result = combined
            continue
        envelope = bounded_simplify(sum(scales, sp.S.Zero))
        exact = (
            bounded_simplify(result.exact_expression + item.exact_expression)
            if result.exact_expression is not None and item.exact_expression is not None
            else None
        )
        kind = (
            RemainderKind.LITTLE_O
            if result.kind is item.kind is RemainderKind.LITTLE_O
            else RemainderKind.BIG_O
        )
        result = AsymptoticRemainder(
            result.variable,
            result.point,
            kind,
            envelope,
            exact,
            result.provenance + item.provenance + (RemainderProvenance(source),),
        )
    return result


def certify_scaling_remainder(
    factor: sp.Expr,
    remainder: AsymptoticRemainder,
) -> RemainderTheoremCertificate:
    """Certify multiplication of an approximation error by an exact factor."""

    factor = sp.sympify(factor)
    conclusion = remainder.scale_by(factor)
    hypothesis = _decision(
        "exact_scaling_factor",
        True,
        reason="the scaling factor is part of the exact finite prefix algebra",
        source="exact scaling remainder theorem",
    )
    return RemainderTheoremCertificate("exact scaling remainder theorem", (hypothesis,), conclusion)


def certify_antiderivative_remainder(
    remainder: AsymptoticRemainder,
) -> RemainderTheoremCertificate:
    """Certify a primitive error when a bounded exact primitive is available.

    An abstract O/o bound does not in general survive indefinite integration.
    This theorem upgrades only cases where the exact error is stored and both
    it and the declared scale admit bounded elementary primitives; the claimed
    output bound is then checked directly.
    """

    variable, point = remainder.variable, remainder.point
    if remainder.is_exact:
        conclusion = AsymptoticRemainder.exact_zero(
            variable, point, source="primitive of exact-zero remainder"
        )
        hypothesis = _decision(
            "antiderivative_exact_zero",
            True,
            reason="the source error is identically zero",
            source="antiderivative remainder theorem",
        )
        return RemainderTheoremCertificate(
            "exact antiderivative remainder theorem", (hypothesis,), conclusion
        )
    if remainder.exact_expression is None or remainder.scale is None:
        conclusion = AsymptoticRemainder.unknown(
            variable, point, source="antiderivative requires an exact stored error and scale"
        )
        hypothesis = _decision(
            "antiderivative_exact_error_available",
            None,
            reason="abstract O/o data alone does not control an indefinite primitive",
            source="antiderivative remainder theorem",
        )
        return RemainderTheoremCertificate(
            "antiderivative remainder theorem", (hypothesis,), conclusion
        )
    exact_primitive = certification_primitive(remainder.exact_expression, variable)
    scale_primitive = certification_primitive(sp.sympify(remainder.scale), variable)
    if exact_primitive is None or scale_primitive is None or scale_primitive == 0:
        conclusion = AsymptoticRemainder.unknown(
            variable,
            point,
            source="bounded primitive oracle could not establish an integrated scale",
        )
        hypothesis = _decision(
            "antiderivative_primitives_available",
            None,
            reason="exact bounded primitives were not available for both error and scale",
            source="antiderivative remainder theorem",
        )
        return RemainderTheoremCertificate(
            "antiderivative remainder theorem", (hypothesis,), conclusion
        )
    conclusion = _classify_exact_error(
        exact_primitive,
        scale_primitive,
        variable,
        point,
        source="directly replayed antiderivative error bound",
    )
    hypothesis = _decision(
        "antiderivative_direct_bound",
        conclusion.is_certified,
        reason="integrated exact error was compared directly with an integrated candidate scale",
        source="antiderivative remainder theorem",
    )
    return RemainderTheoremCertificate(
        "antiderivative remainder theorem", (hypothesis,), conclusion
    )


def certify_finite_sum_remainder(
    remainders: Iterable[AsymptoticRemainder],
) -> RemainderTheoremCertificate:
    """Certify the error in a finite sum of asymptotic approximations.

    Comparable scales retain the sharpest existing bound.  When exact growth
    comparison is unavailable, the theorem falls back to the safe envelope
    ``sum(abs(scale_i))`` instead of discarding otherwise valid O/o data.
    """

    items = tuple(remainders)
    if not items:
        raise ValueError("at least one remainder is required")
    conclusion = _safe_sum_remainders(items, source="finite remainder sum theorem")
    hypothesis = _decision(
        "finite_sum_compatible",
        conclusion.is_certified,
        reason="all summands use compatible asymptotic variables/points and finite-sum bounds were combined",
        source="finite sum remainder theorem",
    )
    return RemainderTheoremCertificate("finite sum remainder theorem", (hypothesis,), conclusion)


def certify_product_remainder(
    left_prefix: sp.Expr,
    right_prefix: sp.Expr,
    left_remainder: AsymptoticRemainder,
    right_remainder: AsymptoticRemainder,
) -> RemainderTheoremCertificate:
    """Certify ``(a+R_a)(b+R_b)-ab`` from the exact product identity.

    The error is exactly ``b R_a + a R_b + R_a R_b``.  This theorem therefore
    preserves O/o information even when neither finite prefix is asymptotically
    constant.
    """

    left_remainder._check_compatible(right_remainder)
    a = sp.sympify(left_prefix)
    b = sp.sympify(right_prefix)
    pieces = (
        left_remainder.scale_by(b),
        right_remainder.scale_by(a),
        left_remainder.product(right_remainder),
    )
    conclusion = _safe_sum_remainders(pieces, source="finite product remainder theorem")
    if left_remainder.exact_expression is not None and right_remainder.exact_expression is not None:
        ra = left_remainder.exact_expression
        rb = right_remainder.exact_expression
        exact = bounded_simplify(b * ra + a * rb + ra * rb)
        conclusion = AsymptoticRemainder(
            conclusion.variable,
            conclusion.point,
            conclusion.kind,
            conclusion.scale,
            exact,
            conclusion.provenance,
        )
    hypothesis = _decision(
        "product_error_identity",
        conclusion.is_certified,
        reason="used exact identity b*R_a + a*R_b + R_a*R_b",
        source="product remainder theorem",
    )
    return RemainderTheoremCertificate(
        "binary product remainder theorem", (hypothesis,), conclusion
    )


def certify_finite_product_remainder(
    prefixes: Iterable[sp.Expr],
    remainders: Iterable[AsymptoticRemainder],
) -> RemainderTheoremCertificate:
    """Certify a finite product by repeated exact binary product identities."""

    prefix_items = tuple(map(sp.sympify, prefixes))
    remainder_items = tuple(remainders)
    if not prefix_items or len(prefix_items) != len(remainder_items):
        raise ValueError("prefixes and remainders must be nonempty and have equal length")
    prefix = prefix_items[0]
    remainder = remainder_items[0]
    hypotheses = []
    for next_prefix, next_remainder in zip(prefix_items[1:], remainder_items[1:]):
        cert = certify_product_remainder(prefix, next_prefix, remainder, next_remainder)
        hypotheses.extend(cert.hypotheses)
        remainder = cert.conclusion
        prefix = bounded_simplify(prefix * next_prefix)
    return RemainderTheoremCertificate(
        "finite product remainder theorem",
        tuple(hypotheses),
        remainder,
    )


def certify_reciprocal_remainder(
    prefix: sp.Expr,
    remainder: AsymptoticRemainder,
) -> RemainderTheoremCertificate:
    """Certify a reciprocal under eventual nonvanishing and relative smallness.

    If ``A = a + R`` with ``a`` eventually nonzero and ``R/a -> 0``, then
    ``1/A - 1/a`` has the same O/o kind as ``R`` with scale ``scale(R)/a**2``.
    The theorem deliberately requires the *declared scale* to be small relative
    to ``a`` so an abstract big-O statement is sufficient by itself.
    """

    variable, point = remainder.variable, remainder.point
    a = sp.sympify(prefix)
    ctx = AsymptoticContext(variable, point=point)
    sign = ctx.eventual_sign(a)
    nonzero = sign in (-1, 1)
    if remainder.is_exact:
        if nonzero:
            conclusion = AsymptoticRemainder.exact_zero(
                variable, point, source="reciprocal of exact eventually-nonzero approximation"
            )
        else:
            conclusion = AsymptoticRemainder.unknown(
                variable,
                point,
                source="exact reciprocal prefix is not certified eventually nonzero",
            )
        hypotheses = (
            _decision(
                "reciprocal_input_exact",
                True,
                reason="input remainder is identically zero",
                source="reciprocal remainder theorem",
            ),
            _decision(
                "reciprocal_prefix_nonzero",
                nonzero if sign is not None else None,
                reason=f"eventual sign of reciprocal prefix: {sign}",
                source="reciprocal remainder theorem",
            ),
        )
        return RemainderTheoremCertificate("exact reciprocal", hypotheses, conclusion)
    if remainder.scale is None or a == 0:
        conclusion = AsymptoticRemainder.unknown(
            variable, point, source="reciprocal prefix/scale unavailable"
        )
        h = _decision(
            "reciprocal_nondegenerate",
            False if a == 0 else None,
            reason="a nonzero prefix and remainder scale are required",
            source="reciprocal remainder theorem",
        )
        return RemainderTheoremCertificate("reciprocal stability theorem", (h,), conclusion)

    relative_limit = ctx.limit(sp.Abs(sp.sympify(remainder.scale) / a))
    relative_small = relative_limit == 0
    propagated_scale = bounded_simplify(sp.sympify(remainder.scale) / a**2)
    exact = None
    if remainder.exact_expression is not None:
        err = remainder.exact_expression
        exact = bounded_simplify(1 / (a + err) - 1 / a)
    if nonzero and relative_small:
        conclusion = AsymptoticRemainder(
            variable,
            point,
            remainder.kind,
            propagated_scale,
            exact,
            remainder.provenance + (RemainderProvenance("reciprocal stability theorem"),),
        )
    else:
        conclusion = AsymptoticRemainder.unknown(
            variable,
            point,
            exact_expression=exact,
            source="reciprocal nonvanishing or relative-smallness hypothesis unresolved",
        )
    hypotheses = (
        _decision(
            "reciprocal_prefix_nonzero",
            nonzero if sign is not None else None,
            reason=f"eventual sign of reciprocal prefix: {sign}",
            source="reciprocal remainder theorem",
        ),
        _decision(
            "reciprocal_relative_error_small",
            relative_small if relative_limit is not None else None,
            reason="checked remainder_scale/prefix -> 0",
            source="reciprocal remainder theorem",
        ),
    )
    return RemainderTheoremCertificate("reciprocal stability theorem", hypotheses, conclusion)


def certify_quotient_remainder(
    numerator_prefix: sp.Expr,
    denominator_prefix: sp.Expr,
    numerator_remainder: AsymptoticRemainder,
    denominator_remainder: AsymptoticRemainder,
) -> RemainderTheoremCertificate:
    """Certify a quotient by reciprocal stability followed by product propagation."""

    reciprocal = certify_reciprocal_remainder(denominator_prefix, denominator_remainder)
    if not reciprocal.conclusion.is_certified:
        return RemainderTheoremCertificate(
            "quotient remainder theorem",
            reciprocal.hypotheses,
            AsymptoticRemainder.unknown(
                numerator_remainder.variable,
                numerator_remainder.point,
                source="denominator reciprocal could not be certified",
            ),
        )
    product = certify_product_remainder(
        numerator_prefix,
        1 / sp.sympify(denominator_prefix),
        numerator_remainder,
        reciprocal.conclusion,
    )
    return RemainderTheoremCertificate(
        "quotient remainder theorem",
        reciprocal.hypotheses + product.hypotheses,
        product.conclusion,
    )


def certify_algebraic_substitution_remainder(
    outer: sp.Expr,
    argument: sp.Symbol,
    prefix: sp.Expr,
    input_remainder: AsymptoticRemainder,
    *,
    output_variable: sp.Symbol,
    point: sp.Expr,
) -> RemainderTheoremCertificate:
    """Certify polynomial/rational substitution using exact algebraic identities.

    Polynomial substitution expands the finite Taylor identity exactly in the
    perturbation.  Rational substitution applies that theorem to numerator and
    denominator and then invokes the certified quotient theorem, so poles are
    handled through an explicit eventual-nonvanishing hypothesis.
    """

    outer = sp.sympify(outer)
    try:
        rational = bool(outer.is_rational_function(argument))
    except SYMBOLIC_ERRORS:
        rational = False
    if not rational:
        conclusion = AsymptoticRemainder.unknown(
            output_variable,
            point,
            source="outer expression is not rational in the substitution argument",
        )
        h = _decision(
            "algebraic_outer_rational",
            False,
            reason="algebraic substitution theorem covers rational functions",
            source="algebraic substitution remainder theorem",
        )
        return RemainderTheoremCertificate("algebraic substitution theorem", (h,), conclusion)

    numerator, denominator = sp.fraction(sp.cancel(outer))
    delta = sp.Dummy("_delta")

    def polynomial_part(poly_expr: sp.Expr) -> RemainderTheoremCertificate:
        base = bounded_simplify(poly_expr.subs(argument, prefix))
        difference = sp.Poly(
            sp.expand(poly_expr.subs(argument, prefix + delta) - base),
            delta,
        )
        pieces = []
        for (degree,), coefficient in difference.terms():
            if degree <= 0:
                continue
            power = input_remainder
            for _ in range(1, degree):
                power = power.product(input_remainder)
            pieces.append(power.scale_by(coefficient))
        if not pieces:
            conclusion = AsymptoticRemainder.exact_zero(
                output_variable, point, source="constant algebraic substitution"
            )
        else:
            conclusion = _safe_sum_remainders(
                pieces, source="polynomial substitution finite Taylor identity"
            )
        h = _decision(
            "polynomial_taylor_identity",
            conclusion.is_certified,
            reason="expanded the exact finite polynomial perturbation identity",
            source="algebraic substitution remainder theorem",
        )
        return RemainderTheoremCertificate("polynomial substitution theorem", (h,), conclusion)

    try:
        sp.Poly(numerator, argument)
        sp.Poly(denominator, argument)
    except sp.PolynomialError:
        conclusion = AsymptoticRemainder.unknown(
            output_variable,
            point,
            source="rational numerator/denominator are not polynomial in argument",
        )
        h = _decision(
            "algebraic_polynomial_parts",
            False,
            reason="could not represent rational function as polynomial numerator/denominator",
            source="algebraic substitution remainder theorem",
        )
        return RemainderTheoremCertificate("algebraic substitution theorem", (h,), conclusion)

    numerator_cert = polynomial_part(numerator)
    if denominator == 1:
        return RemainderTheoremCertificate(
            "polynomial substitution theorem",
            numerator_cert.hypotheses,
            numerator_cert.conclusion,
        )
    denominator_cert = polynomial_part(denominator)
    quotient = certify_quotient_remainder(
        numerator.subs(argument, prefix),
        denominator.subs(argument, prefix),
        numerator_cert.conclusion,
        denominator_cert.conclusion,
    )
    return RemainderTheoremCertificate(
        "rational substitution theorem",
        numerator_cert.hypotheses + denominator_cert.hypotheses + quotient.hypotheses,
        quotient.conclusion,
    )


def certify_differentiation_remainder(
    remainder: AsymptoticRemainder,
    order: int = 1,
) -> RemainderTheoremCertificate:
    """Certify differentiation only when its regularity hypothesis is provable.

    General O/o differentiation is invalid.  The theorem is therefore applied
    automatically only when the exact remainder is known, in which case the
    derivative bound is replayed directly against the differentiated scale.
    """

    if order < 0:
        raise ValueError("order must be nonnegative")
    if remainder.is_exact:
        conclusion = AsymptoticRemainder.exact_zero(
            remainder.variable, remainder.point, source="differentiation of exact remainder"
        )
        hypothesis = _decision(
            "exact_remainder_differentiable",
            True,
            reason="the exact remainder is identically zero",
            source="differentiation remainder theorem",
        )
        return RemainderTheoremCertificate("exact differentiation", (hypothesis,), conclusion)

    if remainder.exact_expression is None:
        conclusion = AsymptoticRemainder.unknown(
            remainder.variable,
            remainder.point,
            source="no proof of derivative control for an abstract O/o remainder",
        )
        hypothesis = _decision(
            "derivative_control",
            None,
            reason="O/o alone does not control derivatives",
            source="differentiation remainder theorem",
        )
        return RemainderTheoremCertificate(
            "differentiation under derivative control", (hypothesis,), conclusion
        )

    exact_d = sp.diff(remainder.exact_expression, remainder.variable, order)
    scale_d = (
        sp.diff(sp.sympify(remainder.scale), remainder.variable, order)
        if remainder.scale is not None
        else exact_d
    )
    conclusion = _classify_exact_error(
        exact_d,
        scale_d,
        remainder.variable,
        remainder.point,
        source="direct replay of exact differentiated remainder",
    )
    hypothesis = _decision(
        "derivative_control",
        conclusion.is_certified,
        reason="the exact differentiated error was compared with the differentiated scale",
        source="differentiation remainder theorem",
    )
    return RemainderTheoremCertificate(
        "differentiation by exact-error replay", (hypothesis,), conclusion
    )


def certify_unary_composition_remainder(
    outer: sp.Expr,
    argument: sp.Symbol,
    prefix: sp.Expr,
    input_remainder: AsymptoticRemainder,
    *,
    output_variable: sp.Symbol,
    point: sp.Expr,
    taylor_order: int | None = None,
    perturbation_scale: sp.Expr | None = None,
) -> RemainderTheoremCertificate:
    """Propagate a remainder through a smooth local composition.

    This is a finite-order Taylor stability theorem.  It finds the first
    nonzero derivative ``F^(k)(P)`` (rather than assuming ``k=1``) and proves
    the next Taylor term is smaller by checking
    ``F^(k+1)(P)*m/F^(k)(P) -> 0``.  Consequently stationary points such as
    ``cos(P+R)`` at ``P=0`` can retain a certified quadratic remainder scale.

    Rational outer functions are delegated to the stronger exact algebraic
    substitution theorem.  Exact represented input errors are replayed directly.
    """

    outer = sp.sympify(outer)
    if input_remainder.is_exact:
        conclusion = AsymptoticRemainder.exact_zero(
            output_variable, point, source="composition of exact input"
        )
        h = _decision(
            "composition_input_exact",
            True,
            reason="input remainder is zero",
            source="composition remainder theorem",
        )
        return RemainderTheoremCertificate("exact composition", (h,), conclusion)

    try:
        is_rational = bool(outer.is_rational_function(argument))
    except SYMBOLIC_ERRORS:
        is_rational = False
    if is_rational:
        algebraic = certify_algebraic_substitution_remainder(
            outer,
            argument,
            prefix,
            input_remainder,
            output_variable=output_variable,
            point=point,
        )
        if algebraic.conclusion.is_certified:
            return algebraic

    scale = perturbation_scale if perturbation_scale is not None else input_remainder.scale
    if scale is None:
        conclusion = AsymptoticRemainder.unknown(
            output_variable, point, source="input perturbation scale unavailable"
        )
        h = _decision(
            "composition_taylor_scale",
            None,
            reason="no input remainder scale",
            source="composition remainder theorem",
        )
        return RemainderTheoremCertificate("local composition stability", (h,), conclusion)

    max_order = max(1, int(taylor_order or 6))
    first_order: int | None = None
    first_derivative: sp.Expr | None = None
    for order in range(1, max_order + 1):
        derivative = bounded_simplify(sp.diff(outer, argument, order).subs(argument, prefix))
        if derivative == 0 or derivative.is_zero is True:
            continue
        first_order = order
        first_derivative = derivative
        break

    if first_order is None or first_derivative is None:
        conclusion = AsymptoticRemainder.unknown(
            output_variable,
            point,
            source="no nonzero Taylor derivative found within the finite composition order",
        )
        h = _decision(
            "composition_nonzero_taylor_coefficient",
            None,
            reason=f"no nonzero derivative found through order {max_order}",
            source="composition remainder theorem",
        )
        return RemainderTheoremCertificate("finite-order composition stability", (h,), conclusion)

    coefficient = bounded_simplify(first_derivative / sp.factorial(first_order))
    propagated_scale = bounded_simplify(sp.Abs(coefficient) * sp.sympify(scale) ** first_order)
    ctx = AsymptoticContext(output_variable, point=point)
    next_derivative = bounded_simplify(
        sp.diff(outer, argument, first_order + 1).subs(argument, prefix)
    )
    if next_derivative == 0 or next_derivative.is_zero is True:
        stable: bool | None = True
        stable_limit = sp.S.Zero
    else:
        stable_limit = ctx.limit(
            sp.Abs(bounded_simplify(next_derivative * sp.sympify(scale) / first_derivative))
        )
        stable = stable_limit == 0

    exact_error = None
    if input_remainder.exact_expression is not None:
        exact_error = bounded_simplify(
            outer.subs(argument, prefix + input_remainder.exact_expression)
            - outer.subs(argument, prefix)
        )
        conclusion = _classify_exact_error(
            exact_error,
            propagated_scale,
            output_variable,
            point,
            source="exact replay of composed remainder",
        )
        if conclusion.is_certified:
            stable = True
    elif stable is True:
        conclusion = AsymptoticRemainder(
            output_variable,
            point,
            input_remainder.kind,
            propagated_scale,
            None,
            input_remainder.provenance
            + (RemainderProvenance(f"order-{first_order} local composition stability theorem"),),
        )
    else:
        conclusion = AsymptoticRemainder.unknown(
            output_variable,
            point,
            source="composition Taylor stability unresolved",
        )

    hypotheses = (
        _decision(
            "composition_first_nonzero_taylor_order",
            True,
            reason=f"first nonzero Taylor derivative found at order {first_order}",
            source="composition remainder theorem",
        ),
        _decision(
            "composition_next_term_stable",
            stable,
            reason=f"checked F^({first_order + 1})(P)*scale/F^({first_order})(P) -> 0; limit={stable_limit}",
            source="composition remainder theorem",
        ),
    )
    return RemainderTheoremCertificate(
        "finite-order local composition theorem", hypotheses, conclusion
    )


def certify_inverse_remainder(
    function: sp.Expr,
    variable: sp.Symbol,
    inverse_variable: sp.Symbol,
    inverse_prefix: sp.Expr,
    *,
    source_point: sp.Expr = sp.oo,
    target_point: sp.Expr = sp.oo,
) -> RemainderTheoremCertificate:
    """Mean-value/Newton remainder theorem for an asymptotic inverse prefix.

    Let r=f(g)-y and q=r/f'(g).  If f'(g) is eventually nonzero and
    f''(g)q/f'(g)->0, then the true inverse differs from g by O(q).  If the
    residual is exactly zero, the inverse prefix is exact.
    """

    # Avoid turning certification itself into an uncontrolled symbolic
    # composition problem.  A theorem is only replayed when the residual can
    # reasonably be formed in the current finite expression model.
    if (sp.count_ops(function) + sp.count_ops(inverse_prefix) > 80) or (
        function.has(sp.exp) and inverse_prefix.has(sp.log)
    ):
        conclusion = AsymptoticRemainder.unknown(
            inverse_variable,
            target_point,
            source="inverse residual replay exceeds the conservative complexity boundary",
        )
        h = _decision(
            "inverse_residual_replay",
            None,
            reason="residual composition was not certified within the finite complexity policy",
            source="inverse remainder theorem",
        )
        return RemainderTheoremCertificate("inverse mean-value/Newton theorem", (h,), conclusion)

    residual = sp.simplify(function.subs(variable, inverse_prefix) - inverse_variable)
    if residual == 0:
        conclusion = AsymptoticRemainder.exact_zero(
            inverse_variable, target_point, source="exact inverse residual"
        )
        h = _decision(
            "inverse_residual_zero",
            True,
            reason="f(g)-y is identically zero",
            source="inverse remainder theorem",
        )
        return RemainderTheoremCertificate("exact inverse", (h,), conclusion)

    fp = sp.simplify(sp.diff(function, variable).subs(variable, inverse_prefix))
    if fp == 0:
        conclusion = AsymptoticRemainder.unknown(
            inverse_variable,
            target_point,
            exact_expression=None,
            source="inverse derivative vanishes",
        )
        h = _decision(
            "inverse_nondegenerate", False, reason="f'(g)=0", source="inverse remainder theorem"
        )
        return RemainderTheoremCertificate("inverse mean-value theorem", (h,), conclusion)

    ctx = AsymptoticContext(inverse_variable, point=target_point)
    sign = ctx.eventual_sign(fp)
    q = sp.simplify(residual / fp)
    stable: bool | None = None
    try:
        fpp = sp.simplify(sp.diff(function, variable, 2).subs(variable, inverse_prefix))
        lim = ctx.limit(sp.Abs(sp.simplify(fpp * q / fp)))
        stable = lim == 0
    except SYMBOLIC_ERRORS:
        stable = None
    nondeg = sign in (-1, 1)
    if nondeg and stable is True:
        conclusion = AsymptoticRemainder.big_o(
            q,
            inverse_variable,
            target_point,
            source="inverse mean-value/Newton remainder theorem",
        )
    else:
        conclusion = AsymptoticRemainder.unknown(
            inverse_variable,
            target_point,
            source="inverse nondegeneracy or derivative stability unresolved",
        )
    hypotheses = (
        _decision(
            "inverse_derivative_nonzero",
            nondeg if sign is not None else None,
            reason=f"eventual sign of f'(g): {sign}",
            source="inverse remainder theorem",
        ),
        _decision(
            "inverse_derivative_stable",
            stable,
            reason="checked f''(g)*(residual/f'(g))/f'(g) -> 0",
            source="inverse remainder theorem",
        ),
    )
    return RemainderTheoremCertificate("inverse mean-value/Newton theorem", hypotheses, conclusion)


def certify_nonlinear_lifting_remainder(
    residual: sp.Expr,
    linearized_coefficient: sp.Expr,
    variable: sp.Symbol,
    point: sp.Expr,
) -> RemainderTheoremCertificate:
    """Simple-root implicit/Newton theorem for a lifted nonlinear branch.

    This theorem is intentionally restricted to a certified scalar local
    correction operator: when the leading Fréchet coefficient is eventually
    nonzero, a residual R gives a next correction O(R/L).  More general
    differential inverse-operator estimates remain UNKNOWN.
    """

    residual = sp.simplify(residual)
    if residual == 0:
        conclusion = AsymptoticRemainder.exact_zero(
            variable, point, source="zero nonlinear residual"
        )
        h = _decision(
            "nonlinear_residual_zero",
            True,
            reason="residual vanishes exactly",
            source="nonlinear lifting remainder theorem",
        )
        return RemainderTheoremCertificate("exact nonlinear branch", (h,), conclusion)
    coeff = sp.simplify(linearized_coefficient)
    ctx = AsymptoticContext(variable, point=point)
    sign = None if coeff == 0 else ctx.eventual_sign(coeff)
    nondeg = sign in (-1, 1)
    if nondeg:
        conclusion = AsymptoticRemainder.big_o(
            sp.simplify(residual / coeff),
            variable,
            point,
            source="simple-root nonlinear residual theorem",
        )
    else:
        conclusion = AsymptoticRemainder.unknown(
            variable, point, source="linearized correction coefficient not certified invertible"
        )
    h = _decision(
        "nonlinear_linearization_nondegenerate",
        nondeg if sign is not None else None,
        reason=f"eventual sign of leading linearized coefficient: {sign}",
        source="nonlinear lifting remainder theorem",
    )
    return RemainderTheoremCertificate("simple-root nonlinear lifting", (h,), conclusion)


def _linear_operator_coefficients(
    linearized_operator: sp.Expr,
    delta: sp.Expr,
    variable: sp.Symbol,
) -> tuple[tuple[sp.Expr, ...], int] | None:
    return linear_operator_coefficients(linearized_operator, delta, variable)


@lru_cache(maxsize=256)
def _characteristic_root_data(
    coefficients: tuple[sp.Expr, ...],
) -> tuple[sp.Expr, tuple[tuple[sp.Expr, int], ...]]:
    """Cached exact characteristic polynomial and root multiplicities."""
    lam = sp.Symbol("__lambda")
    charpoly = sp.expand(sum(coefficients[k] * lam**k for k in range(len(coefficients))))
    roots = sp.roots(charpoly, lam)
    ordered = tuple(
        sorted(
            ((bounded_simplify(root), int(mult)) for root, mult in roots.items()),
            key=lambda item: sp.default_sort_key(item[0]),
        )
    )
    return charpoly, ordered


def clear_characteristic_poly_cache() -> None:
    _characteristic_root_data.cache_clear()


def characteristic_poly_cache_info():
    return _characteristic_root_data.cache_info()


def _root_half_plane(root: sp.Expr, point: sp.Expr) -> int | None:
    """Return -1 for decay, +1 for growth, 0 for center at the selected end."""
    real = sp.simplify(sp.re(root))
    sign: int | None
    if real.is_positive is True:
        sign = 1
    elif real.is_negative is True:
        sign = -1
    elif real.is_zero is True:
        sign = 0
    elif real.is_number:
        try:
            val = complex(sp.N(real, 30))
            sign = 1 if val.real > 0 else -1 if val.real < 0 else 0
        except SYMBOLIC_ERRORS:
            sign = None
    else:
        sign = None
    if point is -sp.oo and sign is not None:
        sign = -sign
    return sign


def _constant_coefficient_green_candidate(
    residual: sp.Expr,
    coeffs: tuple[sp.Expr, ...],
    variable: sp.Symbol,
    point: sp.Expr,
) -> tuple[GreenOperatorCertificate, tuple[PropertyDecision, ...]]:
    """Construct and replay a Green inverse for a hyperbolic constant operator.

    The scalar operator is factored through its characteristic roots. Each
    first-order factor is inverted with a verified elementary primitive, and
    the resulting particular solution is substituted back into the original
    operator before the certificate is accepted.
    """

    order = len(coeffs) - 1
    lead = sp.simplify(coeffs[-1])
    charpoly, root_data = _characteristic_root_data(coeffs)
    hypotheses = []
    if lead == 0:
        hypotheses.append(
            _decision(
                "green_leading_coefficient_nonzero",
                False,
                reason="highest differential coefficient is zero",
                source="Green/exponential-dichotomy theorem",
            )
        )
        return GreenOperatorCertificate(order, coeffs, residual, None, None, None, "none"), tuple(
            hypotheses
        )
    if any(c.has(variable) for c in coeffs):
        hypotheses.append(
            _decision(
                "green_constant_coefficients",
                None,
                reason="current rigorous Green construction requires exact constant coefficients",
                source="Green/exponential-dichotomy theorem",
            )
        )
        return GreenOperatorCertificate(order, coeffs, residual, None, None, None, "none"), tuple(
            hypotheses
        )
    if sum(multiplicity for _root, multiplicity in root_data) != order:
        hypotheses.append(
            _decision(
                "green_characteristic_roots_complete",
                None,
                reason="characteristic polynomial did not split into certified symbolic roots",
                source="Green/exponential-dichotomy theorem",
            )
        )
        return GreenOperatorCertificate(order, coeffs, residual, None, None, None, "none"), tuple(
            hypotheses
        )

    stable = []
    unstable = []
    center = []
    ordered_roots = []
    for root, multiplicity in root_data:
        side = _root_half_plane(root, point)
        if side in (None, 0):
            center.append(root)
        for j in range(multiplicity):
            mode = sp.simplify(variable**j * sp.exp(root * variable))
            gm = GreenMode(
                root, j, mode, "stable" if side == -1 else "unstable" if side == 1 else "center"
            )
            if side == -1:
                stable.append(gm)
            elif side == 1:
                unstable.append(gm)
            ordered_roots.append(root)
    dichotomy = ExponentialDichotomyCertificate(
        variable, point, charpoly, tuple(stable), tuple(unstable), tuple(center)
    )
    hypotheses.append(
        _decision(
            "green_exponential_dichotomy",
            dichotomy.certified,
            reason="all characteristic roots have certified nonzero real part"
            if dichotomy.certified
            else f"center/unresolved roots: {center}",
            source="Green/exponential-dichotomy theorem",
        )
    )

    # P(D)=a_n prod(D-r_j).  Compose exact first-order right inverses.  The
    # zero integration constants are the canonical Green selection.  This is
    # an algebraic identity after differentiation; no numerical ODE solve is
    # used in the certificate.
    q = sp.simplify(-residual / lead)
    for root in reversed(ordered_roots):
        primitive = certification_primitive(sp.exp(-root * variable) * q, variable)
        if primitive is None:
            q = None
            break
        q = sp.simplify(sp.exp(root * variable) * primitive)
    defect = None
    if q is not None:
        defect = sp.simplify(
            sum(coeffs[k] * sp.diff(q, variable, k) for k in range(order + 1)) + residual
        )
    hypotheses.append(
        _decision(
            "green_exact_right_inverse",
            defect == 0 if defect is not None else None,
            reason="verified L(G[-R])+R=0 exactly"
            if defect == 0
            else "closed-form Green particular solution was unavailable or did not replay exactly",
            source="Green/exponential-dichotomy theorem",
        )
    )
    cert = GreenOperatorCertificate(
        order,
        coeffs,
        residual,
        q,
        defect,
        dichotomy,
        "zero constants in all first-order Green factors",
    )
    return cert, tuple(hypotheses)


def _asymptotically_constant_green_candidate(
    residual: sp.Expr,
    coeffs: tuple[sp.Expr, ...],
    variable: sp.Symbol,
    point: sp.Expr,
) -> tuple[GreenOperatorCertificate, tuple[PropertyDecision, ...]]:
    """Construct a limiting Green inverse for ``L=L0+E`` on an infinite end.

    The operator is first normalized to monic form.  Each normalized
    coefficient must converge to a finite constant and the limiting monic
    operator must possess a hyperbolic exponential dichotomy.  Roughness of
    exponential dichotomies then preserves a dichotomy on a sufficiently far
    tail because the companion-matrix perturbation tends uniformly to zero.

    The returned particular solution is the *limiting* Green inverse applied
    to the normalized forcing.  Its defect is replayed in the full variable-
    coefficient operator; the caller requires that defect to be ``o(R)``.
    """

    order = len(coeffs) - 1
    hypotheses = []
    if point not in (sp.oo, -sp.oo):
        hypotheses.append(
            _decision(
                "green_infinite_end",
                False,
                reason="asymptotically-constant dichotomy roughness is certified only at +/-infinity",
                source="asymptotically-constant Green theorem",
            )
        )
        return GreenOperatorCertificate(order, coeffs, residual, None, None, None, "none"), tuple(
            hypotheses
        )

    ctx = AsymptoticContext(variable, point=point)
    lead = bounded_simplify(coeffs[-1])
    lead_sign = None if lead == 0 else ctx.eventual_sign(lead)
    lead_ok = lead_sign in (-1, 1)
    hypotheses.append(
        _decision(
            "green_leading_coefficient_eventually_nonzero",
            lead_ok if lead_sign is not None else None,
            reason=f"eventual sign of highest differential coefficient: {lead_sign}",
            source="asymptotically-constant Green theorem",
        )
    )
    if not lead_ok:
        return GreenOperatorCertificate(order, coeffs, residual, None, None, None, "none"), tuple(
            hypotheses
        )

    normalized = tuple(
        sp.S.One if k == order else bounded_simplify(coeffs[k] / lead) for k in range(order + 1)
    )
    limits = []
    perturbations = []
    perturbation_limits = []
    convergence = []
    for coefficient in normalized:
        limit = bounded_limit(
            coefficient,
            variable,
            point,
            direction=ctx.direction,
            allow_general=True,
        )
        finite_constant = (
            limit is not None
            and variable not in sp.sympify(limit).free_symbols
            and limit not in (sp.oo, -sp.oo, sp.zoo, sp.nan)
            and getattr(limit, "is_finite", None) is not False
        )
        if not finite_constant:
            limits.append(sp.nan)
            perturbations.append(sp.nan)
            perturbation_limits.append(None)
            convergence.append(None)
            continue
        limit = bounded_simplify(sp.sympify(limit))
        perturbation = bounded_simplify(coefficient - limit)
        plimit = bounded_limit(
            perturbation,
            variable,
            point,
            direction=ctx.direction,
            allow_general=True,
        )
        limits.append(limit)
        perturbations.append(perturbation)
        perturbation_limits.append(plimit)
        convergence.append(plimit == 0)

    coeffs_converge = all(value is True for value in convergence)
    hypotheses.append(
        _decision(
            "green_normalized_coefficients_converge",
            True if coeffs_converge else None,
            reason=(
                "all normalized coefficients converge to finite constants"
                if coeffs_converge
                else "at least one normalized coefficient limit or perturbation limit is unresolved"
            ),
            source="asymptotically-constant Green theorem",
        )
    )
    if not coeffs_converge:
        cert = GreenOperatorCertificate(
            order,
            coeffs,
            residual,
            None,
            None,
            None,
            "none",
            tuple(limits),
            tuple(perturbations),
            tuple(perturbation_limits),
        )
        return cert, tuple(hypotheses)

    limiting = tuple(limits)
    normalized_residual = bounded_simplify(residual / lead)
    limiting_green, limiting_hypotheses = _constant_coefficient_green_candidate(
        normalized_residual, limiting, variable, point
    )
    # The constant-coefficient premise in the nested certificate describes L0,
    # so retain its root/right-inverse evidence while adding the roughness
    # hypotheses above for the full operator.
    for hypothesis in limiting_hypotheses:
        predicate = str(hypothesis.predicate)
        suffix = predicate.removeprefix("green_")
        hypotheses.append(
            _decision(
                f"green_limiting_{suffix}",
                hypothesis.verdict,
                reason=hypothesis.reasons[0] if hypothesis.reasons else predicate,
                source="asymptotically-constant Green theorem",
            )
        )
    dichotomy_ok = limiting_green.dichotomy is not None and limiting_green.dichotomy.certified
    hypotheses.append(
        _decision(
            "green_dichotomy_roughness",
            True if (coeffs_converge and dichotomy_ok) else None,
            reason=(
                "the hyperbolic limiting companion system has a tail exponential dichotomy under E(x)->0"
                if (coeffs_converge and dichotomy_ok)
                else "a hyperbolic limiting dichotomy was not certified"
            ),
            source="asymptotically-constant Green theorem",
        )
    )

    q = limiting_green.particular
    defect = None
    if q is not None:
        defect = bounded_simplify(
            sum(coeffs[k] * sp.diff(q, variable, k) for k in range(order + 1)) + residual
        )
    cert = GreenOperatorCertificate(
        order,
        coeffs,
        residual,
        q,
        defect,
        limiting_green.dichotomy,
        "limiting Green selection continued by exponential-dichotomy roughness",
        limiting,
        tuple(perturbations),
        tuple(perturbation_limits),
    )
    return cert, tuple(hypotheses)


def _green_mode_control(
    green: GreenOperatorCertificate,
    q: sp.Expr,
    variable: sp.Symbol,
    point: sp.Expr,
) -> tuple[bool | None, str]:
    """Check endpoint homogeneous-mode control, including rough perturbations."""

    if green.dichotomy is None or not green.dichotomy.certified:
        return None, "limiting exponential dichotomy unavailable"
    ctx = AsymptoticContext(variable, point=point)
    q_limit = bounded_limit(sp.Abs(q), variable, point, direction=ctx.direction)
    q_small = q_limit == 0

    if not green.asymptotically_constant:
        statuses = []
        notes = []
        for mode in (*green.dichotomy.stable_modes, *green.dichotomy.unstable_modes):
            ratio = bounded_limit(
                sp.Abs(mode.expression / q), variable, point, direction=ctx.direction
            )
            mode_abs = bounded_limit(
                sp.Abs(mode.expression), variable, point, direction=ctx.direction
            )
            if ratio == 0:
                statuses.append(True)
                notes.append(f"{mode.expression}=o(q)")
            elif q_small and mode_abs is sp.oo:
                statuses.append(True)
                notes.append(
                    f"{mode.expression} grows and is excluded by the asymptotically-small-tail condition"
                )
            else:
                statuses.append(None)
                notes.append(f"mode {mode.expression} is not certified negligible/excluded")
        return (all(value is True for value in statuses) if statuses else True), "; ".join(notes)

    # For L=L0+E, individual perturbed modes need not be asymptotic to the
    # exact L0 exponentials when E=o(1).  A strict exponential-rate gap is the
    # robust condition: dichotomy roughness can shrink the L0 rates slightly
    # while preserving this separation on a sufficiently far tail.
    tail = variable if point is sp.oo else -variable
    q_rate = bounded_limit(
        sp.log(sp.Abs(q)) / tail,
        variable,
        point,
        direction=ctx.direction,
    )
    if q_rate is None or getattr(q_rate, "is_real", None) is False:
        return None, "exponential rate of the limiting Green particular is unresolved"
    statuses = []
    notes = []
    for mode in (*green.dichotomy.stable_modes, *green.dichotomy.unstable_modes):
        root_rate = bounded_simplify(
            sp.re(mode.characteristic_root) * (1 if point is sp.oo else -1)
        )
        if mode.dichotomy_side == "stable":
            gap = bounded_simplify(q_rate - root_rate)
            ok = gap.is_positive is True
            statuses.append(True if ok else None)
            notes.append(
                f"strict stable rate gap q_rate-root_rate={gap}"
                if ok
                else f"stable rate gap unresolved/nonpositive: {gap}"
            )
        elif mode.dichotomy_side == "unstable" and q_small:
            statuses.append(True)
            notes.append("unstable mode excluded by the asymptotically-small-tail condition")
        else:
            statuses.append(None)
            notes.append(
                "unstable-mode exclusion requires an asymptotically small selected correction"
            )
    return (all(value is True for value in statuses) if statuses else True), "; ".join(notes)


def certify_green_inverse_operator_remainder(
    residual: sp.Expr,
    linearized_operator: sp.Expr,
    correction_function: sp.FunctionClass | sp.Function,
    variable: sp.Symbol,
    point: sp.Expr,
) -> tuple[RemainderTheoremCertificate, GreenOperatorCertificate | None]:
    """Certify a higher-order scalar Green inverse using a hyperbolic dichotomy.

    Exact constant-coefficient operators use a replayed Green right inverse.
    Variable coefficients are additionally supported at ``+/-oo`` when monic
    normalized coefficients converge to a finite hyperbolic constant operator
    ``L0``.  Exponential-dichotomy roughness then supplies a tail dichotomy;
    the limiting Green particular is accepted only when its defect in the full
    operator is ``o(R)`` and a strict exponential-rate gap controls the selected
    small-tail homogeneous modes.  Unproved hypotheses remain ``UNKNOWN``.
    """
    residual = sp.simplify(residual)
    if residual == 0:
        conclusion = AsymptoticRemainder.exact_zero(variable, point, source="zero Fréchet residual")
        h = _decision(
            "green_residual_zero",
            True,
            reason="residual is zero",
            source="Green/exponential-dichotomy theorem",
        )
        return RemainderTheoremCertificate("exact Green inverse", (h,), conclusion), None
    delta = (
        correction_function(variable)
        if isinstance(correction_function, sp.FunctionClass)
        else correction_function
    )
    parsed = _linear_operator_coefficients(linearized_operator, delta, variable)
    if parsed is None:
        h = _decision(
            "green_scalar_linear_operator",
            False,
            reason="Fréchet expression is not scalar linear",
            source="Green/exponential-dichotomy theorem",
        )
        return RemainderTheoremCertificate(
            "Green inverse operator",
            (h,),
            AsymptoticRemainder.unknown(
                variable, point, source="nonlinear or unparsed Fréchet operator"
            ),
        ), None
    coeffs, order = parsed
    if any(coefficient.has(variable) for coefficient in coeffs):
        green, hyps = _asymptotically_constant_green_candidate(residual, coeffs, variable, point)
    else:
        green, hyps = _constant_coefficient_green_candidate(residual, coeffs, variable, point)
    if order < 2:
        # Let the dedicated first-order theorem supply its sharper result.
        h = _decision(
            "green_higher_order",
            None,
            reason="operator order is below two",
            source="Green/exponential-dichotomy theorem",
        )
        return RemainderTheoremCertificate(
            "Green inverse operator",
            (h,),
            AsymptoticRemainder.unknown(
                variable, point, source="use first-order Fréchet inverse theorem"
            ),
        ), green
    q = green.particular
    mode_control: bool | None = None
    mode_note = "particular unavailable"
    defect_small: bool | None = None
    if q is not None:
        if green.asymptotically_constant:
            normalized_residual = bounded_simplify(residual / coeffs[-1])
            normalized_defect = (
                bounded_simplify(green.defect / coeffs[-1]) if green.defect is not None else None
            )
            if normalized_defect is not None and normalized_residual != 0:
                defect_ratio = bounded_limit(
                    sp.Abs(normalized_defect / normalized_residual),
                    variable,
                    point,
                )
                defect_small = defect_ratio == 0
        else:
            defect_small = green.defect == 0
        mode_control, mode_note = _green_mode_control(green, q, variable, point)
    mode_h = _decision(
        "green_homogeneous_modes_controlled",
        mode_control,
        reason=mode_note,
        source="Green/exponential-dichotomy theorem",
    )
    defect_h = _decision(
        "green_full_operator_defect_small",
        defect_small,
        reason=(
            "the full variable-coefficient defect is o(R)"
            if defect_small is True and green.asymptotically_constant
            else "the Green right-inverse identity replays exactly"
            if defect_small is True
            else "the full-operator Green defect is not certified negligible"
        ),
        source="asymptotically-constant Green theorem"
        if green.asymptotically_constant
        else "Green/exponential-dichotomy theorem",
    )
    all_hyps = hyps + (defect_h, mode_h)
    if all(h.verdict is True for h in all_hyps) and q is not None:
        conclusion = AsymptoticRemainder.big_o(
            q, variable, point, source="certified higher-order Green/exponential-dichotomy estimate"
        )
    else:
        conclusion = AsymptoticRemainder.unknown(
            variable, point, source="higher-order Green/dichotomy hypotheses unresolved"
        )
    return RemainderTheoremCertificate(
        (
            "asymptotically-constant higher-order Fréchet Green/exponential-dichotomy theorem"
            if green.asymptotically_constant
            else "higher-order Fréchet Green/exponential-dichotomy theorem"
        ),
        all_hyps,
        conclusion,
        note=(
            f"Green particular: {q}; characteristic polynomial: "
            f"{green.dichotomy.characteristic_poly if green.dichotomy else None}; "
            f"limiting coefficients: {green.limiting_coefficients}"
        ),
    ), green


def certify_frechet_inverse_operator_remainder(
    residual: sp.Expr,
    linearized_operator: sp.Expr,
    correction_function: sp.FunctionClass | sp.Function,
    variable: sp.Symbol,
    point: sp.Expr,
) -> RemainderTheoremCertificate:
    """Certify a scalar Fréchet inverse, first order or higher order.

    First order uses the integrating-factor theorem.  Orders >=2 are delegated
    to constant- or asymptotically-constant Green/exponential-dichotomy
    certification when its hypotheses are provable; otherwise the result
    remains UNKNOWN.
    """
    residual = sp.simplify(residual)
    if residual == 0:
        h = _decision(
            "frechet_residual_zero",
            True,
            reason="residual is zero",
            source="Fréchet inverse-operator theorem",
        )
        return RemainderTheoremCertificate(
            "exact Fréchet inverse",
            (h,),
            AsymptoticRemainder.exact_zero(variable, point, source="zero Fréchet residual"),
        )
    delta = (
        correction_function(variable)
        if isinstance(correction_function, sp.FunctionClass)
        else correction_function
    )
    parsed = _linear_operator_coefficients(linearized_operator, delta, variable)
    if parsed is None:
        h = _decision(
            "frechet_linear",
            False,
            reason="operator is not scalar linear",
            source="Fréchet inverse-operator theorem",
        )
        return RemainderTheoremCertificate(
            "Fréchet inverse operator",
            (h,),
            AsymptoticRemainder.unknown(variable, point, source="nonlinear correction operator"),
        )
    coeffs, order = parsed
    if order >= 2:
        theorem, _green = certify_green_inverse_operator_remainder(
            residual, linearized_operator, correction_function, variable, point
        )
        return theorem
    if order != 1:
        h = _decision(
            "frechet_inverse_available",
            None,
            reason="zero-order operator should use the simple-root theorem",
            source="Fréchet inverse-operator theorem",
        )
        return RemainderTheoremCertificate(
            "Fréchet inverse operator",
            (h,),
            AsymptoticRemainder.unknown(
                variable, point, source="not a differential inverse problem"
            ),
        )
    a0, a1 = coeffs
    ctx = AsymptoticContext(variable, point=point)
    sign = ctx.eventual_sign(a1) if a1 != 0 else None
    nondeg = sign in (-1, 1)
    q: sp.Expr | None = None
    homogeneous: sp.Expr | None = None
    try:
        p = sp.simplify(a0 / a1)
        mu_exp = certification_primitive(p, variable)
        if mu_exp is not None:
            mu = sp.exp(mu_exp)
            homogeneous = sp.simplify(1 / mu)
            primitive = certification_primitive(sp.simplify(mu * residual / a1), variable)
            if primitive is not None:
                q = sp.simplify(-primitive / mu)
        # A leading algebraic inverse is often enough for an asymptotic
        # estimate even when the exact variation-of-constants primitive is a
        # special function (for example exp(x)/x**2 -> Ei).  Verify the
        # candidate by substitution instead of asking the integrator for a
        # closed form.
        if q is None and a0 != 0:
            leading_candidate = sp.simplify(-residual / a0)
            if leading_candidate != 0:
                q = leading_candidate
    except SYMBOLIC_ERRORS:
        q = None
    inverse_ok: bool | None = None
    hom_small: bool | None = None
    if q is not None and q != 0:
        defect = sp.simplify(a1 * sp.diff(q, variable) + a0 * q + residual)
        try:
            defect_lim = ctx.limit(sp.Abs(defect / residual))
            inverse_ok = defect_lim == 0
        except SYMBOLIC_ERRORS:
            inverse_ok = None
        if homogeneous is not None:
            try:
                hom_small = ctx.limit(sp.Abs(homogeneous / q)) == 0
            except SYMBOLIC_ERRORS:
                hom_small = None
    if nondeg and inverse_ok is True and hom_small is True and q is not None:
        conclusion = AsymptoticRemainder.big_o(
            q, variable, point, source="certified first-order Fréchet inverse-operator estimate"
        )
    else:
        conclusion = AsymptoticRemainder.unknown(
            variable, point, source="Fréchet inverse hypotheses unresolved"
        )
    hypotheses = (
        _decision(
            "frechet_leading_coefficient_nonzero",
            nondeg if sign is not None else None,
            reason=f"eventual sign of a1: {sign}",
            source="Fréchet inverse-operator theorem",
        ),
        _decision(
            "frechet_inverse_defect_small",
            inverse_ok,
            reason="checked L(q)+R=o(R)",
            source="Fréchet inverse-operator theorem",
        ),
        _decision(
            "frechet_homogeneous_mode_excluded",
            hom_small,
            reason="checked homogeneous mode=o(q) at the selected end",
            source="Fréchet inverse-operator theorem",
        ),
    )
    return RemainderTheoremCertificate(
        "first-order Fréchet inverse operator",
        hypotheses,
        conclusion,
        note=f"candidate correction: {q}",
    )
