from __future__ import annotations

from dataclasses import dataclass
from functools import reduce

import sympy as sp

from ._integer_utils import integer_lcm as _lcm
from ._symbolic_errors import SYMBOLIC_ERRORS
from ._symbolic_policy import bounded_limit, bounded_solve_one
from .context import AsymptoticContext, context_for
from .function_properties import PropertyDecision, nested_branch_safety_decisions
from .puiseux import BranchChoice, PuiseuxSeries, PuiseuxTerm, _extract_puiseux_terms


@dataclass(frozen=True)
class ReversionBranch:
    original: sp.Expr
    variable: sp.Symbol
    inverse_variable: sp.Symbol
    point: sp.Expr
    target_point: sp.Expr
    leading_exponent: sp.Rational
    leading_coefficient: sp.Expr
    series: PuiseuxSeries
    choice: BranchChoice
    branch_decisions: tuple[PropertyDecision, ...] = ()

    def truncate(self, n: int | None = None) -> sp.Expr:
        return self.series.truncate(n)


def _extract_terms(expr: sp.Expr, x: sp.Symbol) -> tuple[PuiseuxTerm, ...]:
    return _extract_puiseux_terms(expr, x, error_context="reversion")


def _leading_zero_series(expr: sp.Expr, x: sp.Symbol) -> tuple[sp.Rational, sp.Expr]:
    lead = sp.expand(expr).as_leading_term(x)
    exponent = sp.sympify(lead.as_powers_dict().get(x, 0))
    if not exponent.is_Rational:
        raise NotImplementedError("series reversion requires a rational leading exponent")
    exponent = sp.Rational(exponent)
    if exponent <= 0:
        raise ValueError("local reversion requires f(x)-f(0) to vanish with positive order")
    coefficient = sp.simplify(lead / x**exponent)
    return exponent, coefficient


def _series_coefficients_in_t(expr: sp.Expr, t: sp.Symbol) -> dict[sp.Rational, sp.Expr]:
    expanded = sp.expand(expr)
    grouped = {}
    for term in sp.Add.make_args(expanded):
        power = sp.sympify(term.as_powers_dict().get(t, 0))
        if not power.is_Rational:
            continue
        power = sp.Rational(power)
        coeff = sp.simplify(term / t**power)
        grouped[power] = sp.expand(grouped.get(power, 0) + coeff)
    return grouped


def _lift_inverse_branch(
    local_f: sp.Expr,
    x: sp.Symbol,
    y: sp.Symbol,
    exponent: sp.Rational,
    leading_coefficient: sp.Expr,
    root: sp.Expr,
    *,
    terms: int,
    context: AsymptoticContext,
) -> sp.Expr:
    # f(x) ~ a*x**(p/q). Put x=t**q*u, y=t**p.  The inverse has
    # x ~ root*t**q, where a*root**(p/q)=1.
    """Lift one inverse branch coefficient by coefficient and track its residual order."""
    p, q = int(exponent.p), int(exponent.q)
    t = sp.Dummy("t", positive=True)
    u = sp.sympify(root)
    emitted = 1
    k = 1

    while emitted < terms and k <= 256:
        a_k = sp.Dummy(f"a{k}")
        trial = t**q * (u + a_k * t**k)
        # We only need enough local terms to identify the first coefficient
        # containing a_k. SymPy is used as a local Taylor/Puiseux expander here;
        # the coefficient-solving loop itself is demand-driven and has no fixed
        # oversampling ratio.
        residual = sp.series(local_f.xreplace({x: trial}) - t**p, t, 0, p + q + k + 4).removeO()
        grouped = _series_coefficients_in_t(residual, t)
        solved = None
        for power in sorted(grouped):
            coeff = sp.expand(grouped[power])
            if not coeff.has(a_k):
                continue
            roots = bounded_solve_one(coeff, a_k) or ()
            if roots:
                solved = sp.simplify(roots[0])
            break
        if solved is not None:
            if context.is_zero(solved) is not True:
                u = sp.expand(u + solved * t**k)
            mapped = sp.expand((t**q * u).xreplace({t: y ** sp.Rational(1, p)}))
            try:
                emitted = len(_extract_terms(mapped, y))
            except SYMBOLIC_ERRORS:
                pass
        k += 1
    return sp.expand((t**q * u).xreplace({t: y ** sp.Rational(1, p)}))


def series_reversion(
    expr: sp.Expr,
    variable: sp.Symbol,
    inverse_variable: sp.Symbol | None = None,
    *,
    point: sp.Expr = 0,
    terms: int = 6,
    branch: int | None = None,
    context: AsymptoticContext | None = None,
) -> tuple[ReversionBranch, ...] | ReversionBranch:
    """Revert a local (possibly Puiseux) series ``y=f(x)``.

    The algorithm ramifies the source and target coordinates so a rational
    leading exponent becomes integral, chooses every algebraic leading inverse
    coefficient unless ``branch`` is supplied, and then lifts coefficients
    recursively. Exact zero decisions are delegated to ``AsymptoticContext``.
    """

    expr = sp.sympify(expr)
    y = inverse_variable or sp.Symbol("y", positive=True)
    ctx = context_for(variable, point, context)
    branch_decisions = nested_branch_safety_decisions(expr, variable, point)
    if point != 0:
        u = sp.Dummy("u")
        base = sp.simplify(expr.subs(variable, point))
        local = sp.expand(expr.xreplace({variable: point + u}) - base)
        shifted = series_reversion(
            local, u, y, point=0, terms=terms, branch=branch, context=AsymptoticContext(u, point=0)
        )
        seq = (shifted,) if isinstance(shifted, ReversionBranch) else shifted
        out = []
        for item in seq:
            terms2 = tuple(PuiseuxTerm(t.exponent, t.coefficient) for t in item.series.terms)
            series = PuiseuxSeries(
                point + item.series.expr,
                y,
                base,
                terms2,
                item.series.ramification_index,
                item.choice,
            )
            out.append(
                ReversionBranch(
                    expr,
                    variable,
                    y,
                    point,
                    base,
                    item.leading_exponent,
                    item.leading_coefficient,
                    series,
                    item.choice,
                    branch_decisions + item.branch_decisions,
                )
            )
        if branch is not None:
            return out[0]
        return tuple(out)

    f0 = sp.simplify(expr.subs(variable, 0))
    if f0.has(sp.nan, sp.zoo, sp.oo, -sp.oo) or f0 is sp.nan:
        limit = bounded_limit(expr, variable, 0, direction="+", allow_general=True)
        if limit is not None:
            f0 = sp.simplify(limit)
    local_f = sp.expand(expr - f0)
    r, a = _leading_zero_series(local_f, variable)
    c = sp.Dummy("c")
    leading_eq = sp.together(a * c**r - 1)
    roots = bounded_solve_one(leading_eq, c) or ()
    if not roots:
        # For rational powers SymPy can occasionally leave the equation unsolved.
        roots = (sp.simplify(a ** (-1 / r)),)

    selected = list(enumerate(roots))
    if branch is not None:
        if branch < 0 or branch >= len(selected):
            raise IndexError("reversion branch index out of range")
        selected = [selected[branch]]

    result = []
    for index, root in selected:
        choice = BranchChoice(index=index, label=f"inverse-branch-{index}")
        formal = _lift_inverse_branch(local_f, variable, y, r, a, root, terms=terms, context=ctx)
        pterms = _extract_terms(formal, y)[:terms]
        ram = reduce(_lcm, (int(t.exponent.q) for t in pterms), 1)
        ps = PuiseuxSeries(sp.expand(formal + point), y, f0, pterms, ram, choice)
        result.append(
            ReversionBranch(
                expr,
                variable,
                y,
                point,
                f0,
                sp.Rational(1, 1) / r,
                sp.simplify(root),
                ps,
                choice,
                branch_decisions,
            )
        )
    if branch is not None:
        return result[0]
    return tuple(result)


def inverse_asymptotic(
    expr: sp.Expr,
    variable: sp.Symbol,
    inverse_variable: sp.Symbol | None = None,
    *,
    point: sp.Expr = sp.oo,
    terms: int = 6,
    branch: int | None = 0,
    context: AsymptoticContext | None = None,
    assumptions: sp.Expr | bool = sp.S.true,
    allow_unknown_properties: bool = False,
) -> ReversionBranch | tuple[ReversionBranch, ...]:
    """Asymptotically invert ``y=f(x)`` at a finite point or infinity.

    Infinite inversion is reduced exactly to local reversion by reciprocal
    coordinates. If ``f(x)->oo`` we revert ``1/f(1/u)`` against ``z=1/y``;
    if ``f(x)->0`` we revert ``f(1/u)`` against ``y``. The returned expression
    is mapped back to the original inverse variable.
    """

    y = inverse_variable or sp.Symbol("y", positive=True)
    if point not in (sp.oo, -sp.oo):
        return series_reversion(
            expr, variable, y, point=point, terms=terms, branch=branch, context=context
        )

    ctx = context_for(variable, point, context)
    lim = ctx.limit(expr)
    # Exponential-height inverses are not Puiseux after reciprocal
    # localization. Route them through the Chapter-7 log-height reduction and
    # Ecalle iteration instead of forcing a power-series reversion.
    if lim in (sp.oo, -sp.oo) and sp.sympify(expr).has(sp.exp):
        from .general_ops import inverse_logexp

        return inverse_logexp(
            expr,
            variable,
            y,
            point=point,
            terms=terms,
            assumptions=assumptions,
            allow_unknown_properties=allow_unknown_properties,
        )
    u = sp.Dummy("u", positive=True)
    sign = 1 if point is sp.oo else -1
    transformed = sp.simplify(expr.xreplace({variable: sign / u}))

    if lim in (sp.oo, -sp.oo):
        z = sp.Dummy("z", positive=True)
        local = sp.simplify(1 / transformed)
        rev = series_reversion(
            local, u, z, terms=terms, branch=branch, context=AsymptoticContext(u, point=0)
        )
        seq = (rev,) if isinstance(rev, ReversionBranch) else rev
        out = []
        for item in seq:
            inv_u = sp.expand(sign / item.series.truncate())
            mapped = sp.series(inv_u.xreplace({z: 1 / y}), y, sp.oo, terms).removeO()
            pterms = tuple(
                sorted(_extract_terms(mapped, y), key=lambda t: t.exponent, reverse=True)
            )
            ram = reduce(_lcm, (int(t.exponent.q) for t in pterms), 1)
            ps = PuiseuxSeries(mapped, y, sp.oo, pterms, ram, item.choice)
            out.append(
                ReversionBranch(
                    expr,
                    variable,
                    y,
                    point,
                    lim,
                    item.leading_exponent,
                    item.leading_coefficient,
                    ps,
                    item.choice,
                    item.branch_decisions,
                )
            )
        return out[0] if branch is not None else tuple(out)

    if lim == 0:
        rev = series_reversion(
            transformed, u, y, terms=terms, branch=branch, context=AsymptoticContext(u, point=0)
        )
        seq = (rev,) if isinstance(rev, ReversionBranch) else rev
        out = []
        for item in seq:
            mapped = sp.expand(sign / item.series.truncate())
            pterms = tuple(
                sorted(_extract_terms(mapped, y), key=lambda t: t.exponent, reverse=True)
            )
            ram = reduce(_lcm, (int(t.exponent.q) for t in pterms), 1)
            ps = PuiseuxSeries(mapped, y, sp.S.Zero, pterms, ram, item.choice)
            out.append(
                ReversionBranch(
                    expr,
                    variable,
                    y,
                    point,
                    lim,
                    -item.leading_exponent,
                    item.leading_coefficient,
                    ps,
                    item.choice,
                    item.branch_decisions,
                )
            )
        return out[0] if branch is not None else tuple(out)

    raise NotImplementedError("inverse asymptotics requires f to tend to 0 or directed infinity")
