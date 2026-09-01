"""Power simplification with explicit analytic, formal, and exact-branch semantics.

SymPy's ``powsimp(..., force=True)`` applies identities that are valid in a
formal power group but need not preserve principal complex branches.  Analytic
code must therefore use :func:`analytic_powsimp`; only representation layers
whose powers are formal monomials should use :func:`formal_powsimp`.

When an analytic calculation genuinely needs PowerExpand-style identities,
:func:`power_expand_exact` keeps the identities exact by inserting principal-
argument winding corrections.  This is the SymPy analogue of a branch-correct
``PowerExpandExact`` rather than another spelling of ``force=True``.
"""

from __future__ import annotations

import sympy as sp


def analytic_powsimp(expr: sp.Expr) -> sp.Expr:
    """Combine powers only when SymPy can justify the identity analytically."""

    return sp.powsimp(sp.sympify(expr), force=False)


def formal_powsimp(expr: sp.Expr) -> sp.Expr:
    """Canonicalize an expression that is *entirely* a formal monomial.

    This operation is intentionally PowerExpand-like.  It is suitable only for
    internal formal scale/monomial coordinates whose algebra defines these
    power identities.  User coefficients, exact residuals, limits, domains,
    and proof obligations must not be passed to this function.

    Mixed coefficient/monomial expressions should use :func:`mixed_powsimp` so
    forced identities never cross the analytic/formal representation boundary.
    """

    return sp.powsimp(sp.sympify(expr), force=True)


def mixed_powsimp(coefficient: sp.Expr, monomial: sp.Expr) -> sp.Expr:
    """Combine an analytic coefficient with a canonical formal monomial safely.

    The formal part is canonicalized independently, while the coefficient and
    final product are simplified only with branch-valid identities.  This is
    the robust replacement for applying ``powsimp(force=True)`` to a whole
    transseries term.
    """

    coefficient = analytic_powsimp(sp.sympify(coefficient))
    monomial = formal_powsimp(sp.sympify(monomial))
    return analytic_powsimp(coefficient * monomial)


def _assumption_truth(predicate: sp.Expr, assumptions: sp.Expr) -> bool | None:
    """Return a conservative truth value under explicit assumptions."""

    try:
        refined = sp.refine(predicate, assumptions)
    except (TypeError, ValueError, NotImplementedError):
        refined = predicate
    if refined is sp.S.true:
        return True
    if refined is sp.S.false:
        return False
    try:
        answer = sp.ask(predicate, assumptions)
    except (TypeError, ValueError, NotImplementedError):
        return None
    return answer if answer in (True, False) else None


def _refined_arg(expr: sp.Expr, assumptions: sp.Expr) -> sp.Expr:
    value = sp.arg(expr)
    try:
        value = sp.refine(value, assumptions)
    except (TypeError, ValueError, NotImplementedError):
        pass
    return sp.simplify(value)


def _refined_floor(expr: sp.Expr, assumptions: sp.Expr) -> sp.Expr:
    value = sp.floor(expr)
    try:
        value = sp.refine(value, assumptions)
    except (TypeError, ValueError, NotImplementedError):
        pass
    return sp.simplify(value)


def _principal_winding(angle: sp.Expr, assumptions: sp.Expr) -> sp.Expr:
    """Integer k with principal(angle) = angle + 2*pi*k.

    SymPy's principal argument convention is ``(-pi, pi]``.  The floor formula
    therefore handles the negative-real boundary correctly: ``angle = pi``
    gives zero winding while ``angle = -pi`` gives one.
    """

    return _refined_floor((sp.pi - angle) / (2 * sp.pi), assumptions)


def _known_nonzero(expr: sp.Expr, assumptions: sp.Expr) -> bool:
    if expr.is_zero is False:
        return True
    return _assumption_truth(sp.Q.nonzero(expr), assumptions) is True


def _guard_zeros(
    original: sp.Expr,
    transformed: sp.Expr,
    bases: tuple[sp.Expr, ...],
    assumptions: sp.Expr,
) -> sp.Expr:
    """Keep the original expression on unresolved zero loci.

    Argument/logarithm correction formulae require nonzero bases.  Instead of
    silently assuming that, retain the original expression on any unresolved
    zero locus.  Under nonzero assumptions the guard disappears entirely.
    """

    unresolved = tuple(base for base in bases if not _known_nonzero(base, assumptions))
    if not unresolved:
        return transformed
    zero_locus = sp.Or(*(sp.Eq(base, 0, evaluate=False) for base in unresolved))
    decision = _assumption_truth(zero_locus, assumptions)
    if decision is False:
        return transformed
    if decision is True:
        return original
    return sp.Piecewise((original, zero_locus), (transformed, True), evaluate=False)


def _expand_log_exact(argument: sp.Expr, assumptions: sp.Expr) -> sp.Expr | None:
    if isinstance(argument, sp.Mul):
        factors = tuple(argument.args)
        angle = sp.Add(*(_refined_arg(factor, assumptions) for factor in factors))
        winding = _principal_winding(angle, assumptions)
        log_factors: list[sp.Expr] = []
        for factor in factors:
            if isinstance(factor, sp.Pow):
                expanded_factor = _expand_log_exact(factor, assumptions)
                log_factors.append(sp.log(factor) if expanded_factor is None else expanded_factor)
            else:
                log_factors.append(sp.log(factor))
        transformed = sp.Add(
            *log_factors,
            2 * sp.pi * sp.I * winding,
        )
        return _guard_zeros(sp.log(argument), sp.simplify(transformed), factors, assumptions)

    if isinstance(argument, sp.Pow):
        base, exponent = argument.as_base_exp()
        logarithmic_phase = sp.im(exponent * sp.log(base))
        try:
            logarithmic_phase = sp.refine(logarithmic_phase, assumptions)
        except (TypeError, ValueError, NotImplementedError):
            pass
        winding = _principal_winding(logarithmic_phase, assumptions)
        transformed = exponent * sp.log(base) + 2 * sp.pi * sp.I * winding
        return _guard_zeros(sp.log(argument), sp.simplify(transformed), (base,), assumptions)
    return None


def _expand_power_exact(power: sp.Pow, assumptions: sp.Expr) -> sp.Expr | None:
    base, exponent = power.as_base_exp()
    if isinstance(base, sp.Mul):
        factors = tuple(base.args)
        angle = sp.Add(*(_refined_arg(factor, assumptions) for factor in factors))
        winding = _principal_winding(angle, assumptions)
        transformed = sp.Mul(*(factor**exponent for factor in factors)) * sp.exp(
            2 * sp.pi * sp.I * exponent * winding
        )
        return _guard_zeros(power, analytic_powsimp(transformed), factors, assumptions)

    if isinstance(base, sp.Pow):
        inner_base, inner_exponent = base.as_base_exp()
        logarithmic_phase = sp.im(inner_exponent * sp.log(inner_base))
        try:
            logarithmic_phase = sp.refine(logarithmic_phase, assumptions)
        except (TypeError, ValueError, NotImplementedError):
            pass
        winding = _principal_winding(logarithmic_phase, assumptions)
        transformed = inner_base ** (inner_exponent * exponent) * sp.exp(
            2 * sp.pi * sp.I * exponent * winding
        )
        return _guard_zeros(power, analytic_powsimp(transformed), (inner_base,), assumptions)
    return None


def power_expand_exact(
    expr: sp.Expr,
    assumptions: sp.Expr = sp.S.true,
    *,
    expand_logs: bool = True,
    expand_powers: bool = True,
) -> sp.Expr:
    """Expand logarithms and powers with exact principal-branch corrections.

    Unlike ``powsimp(..., force=True)`` or ``expand_log(..., force=True)``, this
    operation does not discard winding information.  Product and nested-power
    identities receive explicit ``2*pi*I`` correction indices expressed with
    ``arg`` and ``floor``.  Explicit assumptions are used to collapse those
    indices whenever possible.  If a correction formula requires a nonzero
    base and nonzeroness is undecidable, a ``Piecewise`` guard preserves the
    original expression on the zero locus.

    The transformation is a finite structural pass rather than a heuristic
    fixed point, so it is deterministic and cannot repeatedly expand its own
    correction terms.
    """

    assumptions = sp.sympify(assumptions)

    def visit(node: sp.Expr) -> sp.Expr:
        node = sp.sympify(node)
        if not node.args:
            return node
        new_args = tuple(visit(argument) for argument in node.args)
        try:
            rebuilt = node.func(*new_args)
        except (TypeError, ValueError):
            rebuilt = node.xreplace(dict(zip(node.args, new_args, strict=True)))

        if expand_logs and rebuilt.func is sp.log and len(rebuilt.args) == 1:
            expanded = _expand_log_exact(rebuilt.args[0], assumptions)
            if expanded is not None:
                return expanded
        if expand_powers and isinstance(rebuilt, sp.Pow):
            expanded = _expand_power_exact(rebuilt, assumptions)
            if expanded is not None:
                return expanded
        return rebuilt

    return analytic_powsimp(visit(sp.sympify(expr)))
