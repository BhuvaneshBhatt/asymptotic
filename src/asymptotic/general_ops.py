"""Scale-aware composition, inversion, and integration for finite transseries.

The algorithms follow Shackell's multiseries strategy: composition is reduced
through analytic/meromorphic local expansions; inverse functions are reduced by
leading-scale normalization and the Ecalle ``(I+K)^-1`` iteration from
Chapter 7; integration chooses the leading integral scale and then performs
recursive integration by parts when no exact primitive remains in the current
field.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from ._power_simplify import analytic_powsimp
from ._symbolic_errors import SYMBOLIC_ERRORS
from ._symbolic_policy import bounded_primitive, bounded_solve_one
from .context import AsymptoticContext
from .function_properties import (
    PropertyDecision,
    PropertyKnowledge,
    PropertyProvenance,
    analytic_at_decision,
    decide,
    require_decision,
)
from .remainder import AsymptoticRemainder
from .remainder_theorems import (
    certify_antiderivative_remainder,
    certify_finite_sum_remainder,
    certify_inverse_remainder,
)
from .transseries import (
    TransseriesExpansion,
    transseries_from_expression,
)


def _trim(series: TransseriesExpansion, terms: int) -> TransseriesExpansion:
    """Trim without silently discarding the omitted-tail semantics."""

    return series.normalized().prefix(max(0, int(terms)))


def _constant_series(value: sp.Expr, inner: TransseriesExpansion) -> TransseriesExpansion:
    return TransseriesExpansion.from_terms(
        inner.variable, inner.point, (), center=sp.sympify(value), complete=True
    )


def compose_transseries(
    outer: sp.Expr | TransseriesExpansion,
    inner: object,
    *,
    argument: sp.Symbol | None = None,
    terms: int = 6,
    assumptions: sp.Expr | bool = sp.S.true,
    allow_unknown_properties: bool = False,
) -> TransseriesExpansion:
    """Compose a finite LE expression/transseries with an asymptotic argument.

    Protocol-compatible representations are converted to their finite
    transseries view. Arithmetic, exp, log and constant powers are evaluated recursively in the
    native transseries algebra. Other functions are admitted only through a
    certified finite-center analytic expansion.
    """

    if terms < 1:
        raise ValueError("terms must be positive")
    if not isinstance(inner, TransseriesExpansion):
        from .algebra import asymptotic_element

        inner = asymptotic_element(inner).to_transseries(terms)
    if isinstance(outer, TransseriesExpansion):
        z = outer.variable
        expr = outer.truncate()
    else:
        expr = sp.sympify(outer)
        z = argument
        if z is None:
            symbols = tuple(sorted(expr.free_symbols - {inner.variable}, key=sp.default_sort_key))
            if len(symbols) != 1:
                raise ValueError("composition requires an explicit argument symbol")
            z = symbols[0]
    if z == inner.variable:
        # This is allowed, but the outer expression must be interpreted before
        # substituting the inner series.
        z = sp.Dummy("composition_argument")
        expr = expr.xreplace({inner.variable: z})

    def evaluate(node: sp.Expr) -> TransseriesExpansion:
        """Evaluate one algebraic operation through native methods before transseries fallback."""
        if node == z:
            return inner
        if z not in node.free_symbols:
            return _constant_series(node, inner)
        if node.is_Add:
            result = _constant_series(0, inner)
            for arg in node.args:
                result = _trim(result + evaluate(arg), terms)
            return result
        if node.is_Mul:
            result = _constant_series(1, inner)
            for arg in node.args:
                result = _trim(result * evaluate(arg), terms)
            return result
        if node.is_Pow:
            base, exponent = node.as_base_exp()
            if z in exponent.free_symbols:
                # a(x)^b(x) = exp(b log a), provided both pieces are supported.
                return _trim(
                    (evaluate(exponent) * evaluate(base).log(terms=terms)).exp(terms=terms), terms
                )
            return _trim(evaluate(base).constant_power(exponent, terms=terms), terms)
        if node.func is sp.exp:
            return _trim(evaluate(node.args[0]).exp(terms=terms), terms)
        if node.func is sp.log:
            return _trim(evaluate(node.args[0]).log(terms=terms), terms)

        # General meromorphic/analytic composition at a finite limiting center.
        arg_series = evaluate(node.args[0]) if len(node.args) == 1 else None
        if arg_series is None:
            raise NotImplementedError(f"unsupported multivariate composition node {node.func}")
        if arg_series.center in (sp.oo, -sp.oo, sp.zoo):
            raise NotImplementedError(f"{node.func} requires a finite analytic composition center")
        decision = analytic_at_decision(node.func(z), z, arg_series.center, assumptions=assumptions)
        require_decision(
            decision,
            operation=f"composition with {node.func}",
            allow_unknown=allow_unknown_properties,
        )
        result = _trim(
            arg_series._compose_analytic_taylor(node.func(z), argument=z, terms=terms), terms
        )
        metadata = dict(result.metadata)
        metadata.setdefault("property_decisions", []).append(decision)
        return TransseriesExpansion.from_terms(
            result.variable,
            result.point,
            result.terms,
            center=result.center,
            complete=result.complete,
            metadata=metadata,
            remainder=result.remainder,
        )

    result = _trim(evaluate(expr), terms)
    metadata = dict(result.metadata)
    metadata.setdefault("operation_provenance", []).append(
        PropertyProvenance("asymptotic.compose_transseries", reference="Shackell §5.3")
    )
    return TransseriesExpansion.from_terms(
        result.variable,
        result.point,
        result.terms,
        center=result.center,
        complete=result.complete,
        metadata=metadata,
        remainder=result.remainder,
    )


@dataclass(frozen=True)
class LogExpInverseResult:
    """Finite multiseries approximation to a functional inverse."""

    expression: sp.Expr
    variable: sp.Symbol
    inverse_variable: sp.Symbol
    point: sp.Expr
    seed: sp.Expr
    iterations: int
    series: TransseriesExpansion
    transformed_by_log: int = 0

    def truncate(self, n: int | None = None) -> sp.Expr:
        return self.series.truncate(n)


def _dominant_additive_term(expr: sp.Expr, x: sp.Symbol, point: sp.Expr) -> sp.Expr:
    series = transseries_from_expression(expr, x, point=point)
    leading = series.normalized().leading_term
    if leading is None:
        if series.center != 0:
            return series.center
        raise ValueError("zero expression has no invertible leading term")
    return sp.simplify(leading.expression)


def _solve_leading_inverse(f0: sp.Expr, x: sp.Symbol, y: sp.Symbol) -> sp.Expr | None:
    sols = bounded_solve_one(sp.Eq(f0, y), x, allow_general=True) or ()
    if sols:
        # Prefer a real/principal expression with the smallest operation count.
        return min((sp.simplify(s) for s in sols), key=sp.count_ops)
    if f0 == x:
        return y
    return None


def inverse_logexp(
    expr: sp.Expr | TransseriesExpansion,
    variable: sp.Symbol,
    inverse_variable: sp.Symbol | None = None,
    *,
    point: sp.Expr = sp.oo,
    terms: int = 6,
    iterations: int | None = None,
    assumptions: sp.Expr | bool = sp.S.true,
    allow_unknown_properties: bool = False,
    _log_depth: int = 0,
) -> LogExpInverseResult:
    """Invert a finite-height exp-log asymptotic function at infinity.

    This implements the Chapter-7 normalization/iteration pattern. When the
    dominant multiplicative scale is exponential, ``log(f)`` is inverted first
    and the result is composed with ``log(y)``. Otherwise a leading inverse
    ``y0`` is found and Ecalle's operator

        K(h) = h(x + g(y0(x))) - h(x)

    is iterated, returning ``sum (-1)^i K^i y0``.
    """

    if point not in (sp.oo, -sp.oo):
        raise NotImplementedError("general log-exp inverse targets directed infinity")
    if terms < 1:
        raise ValueError("terms must be positive")
    y = inverse_variable or sp.Symbol("y", positive=True)
    f = expr.truncate() if isinstance(expr, TransseriesExpansion) else sp.sympify(expr)
    ctx = AsymptoticContext(variable, point=point)
    lim = ctx.limit(f)
    if lim not in (sp.oo, -sp.oo):
        raise NotImplementedError("log-exp inverse requires f to tend to directed infinity")

    derivative = sp.diff(f, variable)
    sign = ctx.eventual_sign(derivative)
    monotone_predicate = sp.Symbol(f"eventually_monotone({sp.sstr(f)})")
    monotone_decision = PropertyDecision(
        monotone_predicate,
        sign in (-1, 1) if sign is not None else None,
        sp.sympify(assumptions),
        PropertyKnowledge.SUFFICIENT,
        (
            PropertyProvenance(
                "asymptotic.inverse_logexp",
                reference="Shackell Theorem 22",
                note="eventual nonzero derivative",
            ),
        ),
        (f"eventual derivative sign: {sign}",),
    )
    require_decision(
        monotone_decision, operation="functional inversion", allow_unknown=allow_unknown_properties
    )

    # Shackell stage 1: if the largest scale occurs as an exponential factor,
    # lower the height by taking logs and restore the target with log(y).
    if _log_depth < 8:
        try:
            leading = _dominant_additive_term(f, variable, point)
        except ValueError:
            leading = f
        if leading.has(sp.exp):
            reduced = inverse_logexp(
                sp.expand_log(sp.log(f), force=False),
                variable,
                y,
                point=point,
                terms=terms,
                iterations=iterations,
                assumptions=assumptions,
                allow_unknown_properties=allow_unknown_properties,
                _log_depth=_log_depth + 1,
            )
            mapped = sp.expand(reduced.truncate().xreplace({y: sp.log(y)}))
            series = _trim(
                transseries_from_expression(mapped, y, point=sp.oo, complete=False), terms
            )
            md = dict(series.metadata)
            md.setdefault("property_decisions", []).append(monotone_decision)
            md.setdefault("operation_provenance", []).append(
                PropertyProvenance(
                    "asymptotic.inverse_logexp", reference="Shackell §7.2 / Ecalle iteration"
                )
            )
            certificate = certify_inverse_remainder(
                f, variable, y, series.truncate(), source_point=point, target_point=sp.oo
            )
            md.setdefault("remainder_certificates", []).append(certificate)
            remainder = (
                certificate.conclusion if certificate.conclusion.is_certified else series.remainder
            )
            series = TransseriesExpansion.from_terms(
                series.variable,
                series.point,
                series.terms,
                center=series.center,
                complete=remainder.is_exact,
                metadata=md,
                remainder=remainder,
            )
            return LogExpInverseResult(
                f,
                variable,
                y,
                point,
                reduced.seed,
                reduced.iterations,
                series,
                reduced.transformed_by_log + 1,
            )

    f0 = _dominant_additive_term(f, variable, point)
    y0 = _solve_leading_inverse(f0, variable, y)
    if y0 is None:
        raise NotImplementedError(f"could not invert leading asymptotic term {f0}")

    g = sp.simplify(f - f0)
    # Since f0(y0(x))=x, Shackell's G is simply g(y0(x)).
    G = sp.simplify(g.xreplace({variable: y0}))
    count = int(iterations if iterations is not None else max(terms, 2))
    phi = y0
    total = y0
    for i in range(1, count):
        shifted = sp.sympify(phi).xreplace({y: y + G})
        next_phi = sp.expand(shifted - phi)
        if sp.simplify(next_phi) == 0:
            break
        total = sp.expand(total + (-1) ** i * next_phi)
        phi = next_phi
        try:
            ts = transseries_from_expression(total, y, point=sp.oo, complete=False)
            total = _trim(ts, terms).truncate()
        except SYMBOLIC_ERRORS:
            total = sp.expand(total)

    # A final asymptotic expansion lets SymPy simplify expressions such as
    # log(y+log(y)) while the native parser retains the resulting LE terms.
    try:
        total = sp.series(total, y, sp.oo, max(terms + 2, 5)).removeO()
    except SYMBOLIC_ERRORS:
        total = sp.expand(total)
    series = _trim(transseries_from_expression(total, y, point=sp.oo, complete=False), terms)
    md = dict(series.metadata)
    md.setdefault("property_decisions", []).append(monotone_decision)
    md.setdefault("operation_provenance", []).append(
        PropertyProvenance(
            "asymptotic.inverse_logexp", reference="Shackell §7.2 / Ecalle iteration"
        )
    )
    certificate = certify_inverse_remainder(
        f, variable, y, series.truncate(), source_point=point, target_point=sp.oo
    )
    md.setdefault("remainder_certificates", []).append(certificate)
    remainder = certificate.conclusion if certificate.conclusion.is_certified else series.remainder
    series = TransseriesExpansion.from_terms(
        series.variable,
        series.point,
        series.terms,
        center=series.center,
        complete=remainder.is_exact,
        metadata=md,
        remainder=remainder,
    )
    return LogExpInverseResult(f, variable, y, point, y0, count, series, 0)


def _ibp_exponential_term(term: sp.Expr, x: sp.Symbol, terms: int) -> sp.Expr | None:
    """Formal integral of ``a*exp(Q)`` by repeated integration by parts."""

    q = sp.S.Zero
    amplitude = sp.S.One
    found = False
    for factor in sp.Mul.make_args(analytic_powsimp(term)):
        if factor.func is sp.exp:
            q += factor.args[0]
            found = True
        else:
            amplitude *= factor
    if not found:
        return None
    qp = sp.simplify(sp.diff(q, x))
    if qp == 0:
        return None
    piece = sp.simplify(amplitude / qp)
    total = sp.S.Zero
    for _ in range(max(1, terms)):
        total += piece
        piece = sp.simplify(-sp.diff(piece, x) / qp)
    return analytic_powsimp(sp.exp(q) * total)


def _power_log_integral(
    term: sp.Expr,
    x: sp.Symbol,
    terms: int,
    assumptions: sp.Expr | bool = sp.S.true,
    allow_unknown: bool = False,
) -> sp.Expr | None:
    """Asymptotic primitive for ``c*x**a*log(x)**b`` at +infinity."""

    expr = analytic_powsimp(sp.expand_log(term, force=False))
    pd = expr.as_powers_dict()
    a = sp.sympify(pd.get(x, 0))
    b = sp.sympify(pd.get(sp.log(x), 0))
    base = x**a * sp.log(x) ** b
    c = sp.simplify(expr / base)
    if x in c.free_symbols:
        return None

    power_resonance = decide(
        sp.Eq(a + 1, 0),
        assumptions,
        provenance=(
            PropertyProvenance(
                "asymptotic.asymptotic_integrate", reference="Shackell Theorem 14 case 2/3"
            ),
        ),
    )
    if power_resonance.verdict is True:
        log_resonance = decide(
            sp.Eq(b + 1, 0),
            assumptions,
            provenance=(
                PropertyProvenance(
                    "asymptotic.asymptotic_integrate", reference="Shackell Theorem 14 case 3"
                ),
            ),
        )
        if log_resonance.verdict is True:
            return sp.simplify(c * sp.log(sp.log(x)))
        if log_resonance.verdict is None:
            require_decision(
                PropertyDecision(
                    sp.Ne(b + 1, 0),
                    None,
                    sp.sympify(assumptions),
                    PropertyKnowledge.EXACT,
                    log_resonance.provenance,
                ),
                operation="logarithmic integration resonance",
                allow_unknown=allow_unknown,
            )
        return sp.simplify(c * sp.log(x) ** (b + 1) / (b + 1))
    if power_resonance.verdict is None:
        require_decision(
            PropertyDecision(
                sp.Ne(a + 1, 0),
                None,
                sp.sympify(assumptions),
                PropertyKnowledge.EXACT,
                power_resonance.provenance,
            ),
            operation="power integration resonance",
            allow_unknown=allow_unknown,
        )

    total = sp.S.Zero
    coeff = sp.S.One
    current_b = b
    denom = a + 1
    for _ in range(max(1, terms)):
        total += coeff * x ** (a + 1) * sp.log(x) ** current_b / denom
        coeff = sp.simplify(-coeff * current_b / denom)
        current_b = sp.simplify(current_b - 1)
    return analytic_powsimp(c * total)


def asymptotic_integrate(
    obj: object,
    variable: sp.Symbol | None = None,
    *,
    point: sp.Expr = sp.oo,
    constant: sp.Expr = 0,
    terms: int = 6,
    assumptions: sp.Expr | bool = sp.S.true,
    allow_unknown_properties: bool = False,
) -> TransseriesExpansion:
    """Compute a scale-aware finite asymptotic primitive.

    Any common-protocol representation may be supplied directly. Exact primitives are preferred. Otherwise exponential scales use repeated
    integration by parts, and power/log scales use the Hardy/Shackell leading
    integral forms with recursive lower-log corrections.
    """

    if terms < 1:
        raise ValueError("terms must be positive")
    if isinstance(obj, TransseriesExpansion):
        x = obj.variable
        point = obj.point
        source = obj.normalized()
    elif variable is None and not isinstance(obj, sp.Expr):
        from .algebra import asymptotic_element

        adapted = asymptotic_element(obj)
        x = adapted.variable
        point = adapted.point
        source = adapted.to_transseries(terms).normalized()
    else:
        if variable is None:
            raise ValueError("variable is required when integrating an expression")
        x = variable
        source = transseries_from_expression(sp.sympify(obj), x, point=point)

    primitive = sp.sympify(constant)
    integration_remainder = AsymptoticRemainder.exact_zero(
        x, point, source="exactly integrated finite prefix"
    )
    if source.center != 0:
        primitive += source.center * x
    for item in source.terms:
        term = item.expression
        # Use the explicit power/log rule first at +infinity. Besides exposing
        # the scale transition x^-1 -> log(x), this avoids branch-dependent
        # antiderivative normal forms such as log(-log(x)).
        approx = (
            _power_log_integral(term, x, terms, assumptions, allow_unknown_properties)
            if point is sp.oo
            else None
        )
        if approx is not None:
            primitive += approx
            more = (
                _power_log_integral(term, x, terms + 1, assumptions, allow_unknown_properties)
                if point is sp.oo
                else None
            )
            if more is not None:
                next_piece = analytic_powsimp(sp.expand(more - approx))
                if next_piece != 0:
                    integration_remainder = integration_remainder.add(
                        AsymptoticRemainder.big_o(
                            next_piece,
                            x,
                            point,
                            source="next integration-by-parts/logarithmic term",
                        )
                    )
            continue
        exact = bounded_primitive(term, x, allow_general=True, risch=True)
        if exact is not None:
            primitive += exact
            continue
        approx = _ibp_exponential_term(term, x, terms)
        if approx is None:
            raise NotImplementedError(f"no certified asymptotic integration rule for {term}")
        primitive += approx
        more = _ibp_exponential_term(term, x, terms + 1)
        if more is not None:
            next_piece = analytic_powsimp(sp.expand(more - approx))
            if next_piece != 0:
                integration_remainder = integration_remainder.add(
                    AsymptoticRemainder.big_o(
                        next_piece, x, point, source="next exponential integration-by-parts term"
                    )
                )

    source_remainder = source.remainder
    if source_remainder is None:
        raise RuntimeError("source transseries is missing its remainder")
    source_certificate = certify_antiderivative_remainder(source_remainder)
    combined_certificate = certify_finite_sum_remainder(
        (integration_remainder, source_certificate.conclusion)
    )
    integration_remainder = combined_certificate.conclusion

    result = _trim(
        transseries_from_expression(
            sp.expand(primitive),
            x,
            point=point,
            complete=integration_remainder.is_exact,
            remainder=integration_remainder,
        ),
        max(terms * max(1, len(source.terms)), terms),
    )
    md = dict(result.metadata)
    md.setdefault("operation_provenance", []).append(
        PropertyProvenance(
            "asymptotic.asymptotic_integrate", reference="Shackell §5.2.2 / Theorem 14"
        )
    )
    md.setdefault("remainder_certificates", []).extend((source_certificate, combined_certificate))
    return TransseriesExpansion.from_terms(
        result.variable,
        result.point,
        result.terms,
        center=result.center,
        complete=result.remainder.is_exact,
        metadata=md,
        remainder=result.remainder,
    )


def asymptotic_integral(*args, **kwargs) -> TransseriesExpansion:
    """Alias for :func:`asymptotic_integrate` using noun-style operator naming.

    The implementation is intentionally a zero-overhead forwarding wrapper so
    users who prefer ``asymptotic_sum``/``asymptotic_integral`` symmetry do not
    need to learn a second integration algorithm or result type.
    """

    return asymptotic_integrate(*args, **kwargs)
