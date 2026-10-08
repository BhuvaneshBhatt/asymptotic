"""Small, proof-aware integer-shift simplification independent of limit dispatch.

Rules use identical non-shifted arguments and a bounded parameter span.
No logarithm-of-gamma, general hypergeometric search, or branch-changing power
identities are used. Time budgets are cooperative, backed by structural caps.
"""

import time
from dataclasses import dataclass

import sympy as sp

from ._recurrence_families import (
    EXTRA_HEADS,
    KELVIN,
    POLYNOMIALS,
    TWO_BASIS,
    HypergeometricU,
    LegendreQ,
    RegularizedHypergeometric2F1,
    WhittakerM,
    WhittakerW,
    function_at,
    next_basis,
)
from ._recurrence_families import signatures as extra_signatures
from .limit_models import LimitEvidence


@dataclass(frozen=True)
class RecurrenceSimplificationResult:
    original: sp.Expr
    expression: sp.Expr
    evidence: tuple[LimitEvidence, ...] = ()
    steps: int = 0
    stopped: str | None = None


_HEADS = (
    sp.gamma,
    sp.polygamma,
    sp.uppergamma,
    sp.expint,
    sp.besselj,
    sp.bessely,
    sp.besseli,
    sp.besselk,
    sp.hankel1,
    sp.hankel2,
    sp.jn,
    sp.yn,
    sp.zeta,
    *EXTRA_HEADS,
)


def simplify_recurrences(
    expr,
    *,
    assumptions=True,
    variables=(),
    target=(),
    max_shift=4,
    max_steps=24,
    max_ops=240,
    budget_seconds=0.05,
    require_defined_germ=False,
    return_result=False,
):
    """Reduce nearby integer-shifted functions to a common recurrence basis.

    Without a limit context, every new denominator must be proved nonzero.
    A one-variable real limit context additionally admits nonzero polynomial
    germs on a punctured neighborhood. Hurwitz shifts require positive argument
    germs. Unresolved parameter strata, large shifts and growth are declined.
    Limit dispatch sets require_defined_germ to reject fixed, unresolved
    gamma/polygamma and zeta pole strata. Standalone identities are meromorphic.
    Accepted rewrites are identities on the original common function domain;
    no value is assigned at removed poles. ``return_result`` exposes evidence.
    """
    from .instrumentation import record_symbolic_event

    record_symbolic_event("recurrence_calls")
    original = sp.sympify(expr)
    current = original
    if max_shift < 0 or max_steps < 0 or max_ops < 0 or budget_seconds < 0:
        raise ValueError("recurrence budgets must be nonnegative")
    if isinstance(variables, sp.Symbol):
        variables = (variables,)
    else:
        variables = tuple(variables)
    if isinstance(target, (sp.Basic, int, float)):
        target = (sp.sympify(target),)
    else:
        target = tuple(map(sp.sympify, target))
    target = tuple(sp.Rational(p) if isinstance(p, sp.Float) else p for p in target)
    started = time.monotonic()
    steps = 0
    evidence = []
    stopped = None

    def finish():
        result = RecurrenceSimplificationResult(
            original, current, tuple(evidence), steps, stopped
        )
        return result if return_result else current

    if not isinstance(original, sp.Expr) or not original.has(*_HEADS):
        return finish()
    if sp.count_ops(original) > max_ops:
        stopped = "input operation cap"
        return finish()
    if budget_seconds == 0 or max_steps == 0:
        stopped = "rewrite budget"
        return finish()
    clauses = sp.And.make_args(sp.sympify(assumptions))

    def nonzero(z):
        if z.is_zero is False:
            return True
        if z.is_Mul and all(nonzero(f) for f in z.args):
            return True
        if sp.Ne(z, 0) in clauses or sp.Gt(z, 0) in clauses or sp.Lt(z, 0) in clauses:
            return True
        if len(variables) != 1 or len(target) != 1:
            return False
        x, p = variables[0], target[0]
        if not z.has(x):
            return False
        if sp.count_ops(z) > 32 or any(
            p.exp.is_Integer and abs(p.exp) > 2 for p in z.atoms(sp.Pow)
        ):
            return False
        if z.is_rational_function(x) and not z.is_polynomial(x):
            numerator, denominator = sp.fraction(sp.cancel(z))
            # Both polynomial germs must be nonzero: no removed pole is
            # assigned a value, and each has only finitely many zeros.
            return nonzero(numerator) and nonzero(denominator)
        try:
            q = sp.Poly(z, x)
        except sp.PolynomialError:
            return False
        if q.degree() > 2:
            return False
        if p in (sp.oo, -sp.oo):
            return q.LC().is_zero is False
        # The first nonzero Taylor coefficient proves an isolated zero;
        # unknown earlier coefficients cannot be set to zero.
        for k in range(q.degree() + 1):
            c = sp.diff(z, x, k).subs(x, p)
            if c.is_zero is True:
                continue
            return c.is_zero is False
        return False

    def positive(z):
        if z.is_positive is True or sp.Gt(z, 0) in clauses:
            return True
        for clause in clauses:
            if isinstance(clause, sp.StrictGreaterThan):
                difference = clause.lhs - clause.rhs
            elif isinstance(clause, sp.StrictLessThan):
                difference = clause.rhs - clause.lhs
            else:
                continue
            if difference == z:
                return True
        if len(variables) != 1 or len(target) != 1:
            return False
        x, p = variables[0], target[0]
        if sp.count_ops(z) > 32 or any(
            p.exp.is_Integer and abs(p.exp) > 2 for p in z.atoms(sp.Pow)
        ):
            return False
        try:
            q = sp.Poly(z, x)
        except sp.PolynomialError:
            return False
        if q.degree() > 1:
            return False
        if p is sp.oo:
            return q.LC().is_positive is True
        if p is -sp.oo:
            return (
                (-q.LC()).is_positive is True
                if q.degree() == 1
                else q.LC().is_positive is True
            )
        return z.subs(x, p).is_positive is True

    def gamma_defined(z):
        if positive(z) or z.is_real is False or z.is_integer is False:
            return True
        if len(variables) != 1 or len(target) != 1 or not z.has(variables[0]):
            return False
        x = variables[0]
        if sp.count_ops(z) > 32 or any(
            p.exp.is_Integer and abs(p.exp) > 2 for p in z.atoms(sp.Pow)
        ):
            return False
        try:
            q = sp.Poly(z, x)
        except sp.PolynomialError:
            return False
        return 1 <= q.degree() <= 2 and q.LC().is_zero is False

    def expansion_size(node):
        if node.is_Add:
            return min(257, sum(expansion_size(a) for a in node.args))
        if node.is_Mul:
            size = 1
            for a in node.args:
                size *= expansion_size(a)
                if size > 256:
                    return 257
            return size
        if node.is_Pow and node.exp.is_Integer:
            if abs(node.exp) > 8:
                return 257
            return min(257, expansion_size(node.base) ** abs(int(node.exp)))
        return 1

    def signature(a):
        if a.func is sp.gamma:
            return (), a.args[0]
        if a.func is sp.polygamma:
            m, z = a.args
            if not m.is_Integer or not 0 <= m <= 8:
                return None
            return (m,), z
        if a.func is sp.zeta:
            if len(a.args) != 2:
                return None
            return (a.args[0],), a.args[1]
        if len(a.args) == 2:
            return (a.args[1],), a.args[0]
        return None

    def shift(p, q):
        cp, rp = p.as_coeff_Add()
        cq, rq = q.as_coeff_Add()
        delta = cp - cq
        return int(delta) if rp == rq and delta.is_Integer else None

    for head in _HEADS:
        # Recurrence collection targets additive/rational fixed-order forms.
        # Preserve compact powered gamma products for the Stirling dispatcher.
        if any(
            p.base.has(head) and (not p.exp.is_Integer or abs(p.exp) > 8)
            for p in current.atoms(sp.Pow)
        ):
            continue
        atoms = sorted(current.atoms(head), key=sp.default_sort_key)
        if len(atoms) > 16:
            stopped = "function occurrence cap"
            continue
        groups = []
        for atom in atoms:
            sigs = extra_signatures(atom) if head in EXTRA_HEADS else [signature(atom)]
            for sig in sigs:
                if sig is None:
                    continue
                key, parameter = sig
                placed = False
                for group in groups:
                    if group[0][1] != key:
                        continue
                    difference = shift(parameter, group[0][2])
                    if difference is not None:
                        group.append((atom, key, parameter))
                        placed = True
                        break
                if not placed:
                    groups.append([(atom, key, parameter)])
        for group in groups:
            if len(group) < 2:
                continue
            if time.monotonic() - started >= budget_seconds:
                stopped = "time budget"
                return finish()
            origin = group[0][2]
            group.sort(key=lambda a: shift(a[2], origin))
            anchor, key, a0 = group[0]
            span = shift(group[-1][2], a0)
            if require_defined_germ:
                if head in (sp.gamma, sp.polygamma) and not all(
                    gamma_defined(g[2]) for g in group
                ):
                    continue
                if head is sp.zeta and not nonzero(key[0] - 1):
                    continue
            # Do not erase a potentially empty germ through cancellation of a
            # parameter-only denominator. Gamma has no zeros on its domain.
            if require_defined_germ:
                numerator, denominator = sp.fraction(current)
                unsafe = False
                for factor in sp.Mul.make_args(denominator):
                    base, _ = factor.as_base_exp()
                    if not base.has(*variables) and not nonzero(base):
                        unsafe = True
                        break
                if unsafe:
                    continue
            if span > max_shift:
                continue
            if head in (*POLYNOMIALS, sp.subfactorial, sp.Ynm):
                # Native polynomial heads have a discrete nonnegative degree
                # contract, not an unspecified analytic continuation in degree.
                if a0.is_integer is not True or not positive(a0 + 1):
                    continue
                if head in (sp.assoc_legendre, sp.Ynm):
                    m = key[0]
                    if (
                        m.is_integer is not True
                        or m.is_nonnegative is not True
                        or not positive(a0 - m + 1)
                    ):
                        continue
            if head is LegendreQ:
                m, z = key
                if (
                    m.is_integer is not True
                    or m.is_nonnegative is not True
                    or not positive(a0 + 1)
                ):
                    continue
                if not positive(1 - z) or not positive(1 + z):
                    continue
            if head in KELVIN and (a0.is_real is not True or not positive(key[0])):
                continue
            if head in (HypergeometricU, WhittakerM, WhittakerW) and not nonzero(
                key[-1]
            ):
                continue
            if head is sp.lerchphi and not positive(a0):
                continue
            if head is sp.harmonic and not positive(a0 + 1):
                continue
            if require_defined_germ:
                if head is sp.factorial and not all(
                    gamma_defined(g[2] + 1) for g in group
                ):
                    continue
                if head in (sp.rf, sp.ff):
                    base = key[0]
                    if not gamma_defined(base if head is sp.rf else base + 1):
                        continue
                if head is WhittakerM and not gamma_defined(2 * key[0] + 1):
                    continue
                if head is RegularizedHypergeometric2F1 and not positive(1 - key[-1]):
                    continue
                if head is sp.lowergamma and not all(
                    gamma_defined(g[2]) for g in group
                ):
                    continue
                if head is sp.harmonic and not all(
                    gamma_defined(g[2] + 1) for g in group
                ):
                    continue
                if head is sp.hyper:
                    if key[0] in ("1f1_a", "2f1_a"):
                        denominator = key[1] if key[0] == "1f1_a" else key[2]
                        if not gamma_defined(denominator):
                            continue
                    elif not all(gamma_defined(g[2]) for g in group):
                        continue
                    if key[0] in ("2f1_a", "2f1_c") and not positive(1 - key[-1]):
                        continue
                if head is sp.lerchphi and not positive(1 - sp.Abs(key[0])):
                    continue
            if steps + span > max_steps:
                stopped = "step budget"
                return finish()
            replacements = {}
            basis = {0: anchor}
            conditions = []
            z = key[0] if key else None
            if head in (
                sp.besselj,
                sp.bessely,
                sp.besseli,
                sp.besselk,
                sp.hankel1,
                sp.hankel2,
                sp.jn,
                sp.yn,
            ):
                if not nonzero(z):
                    continue
                conditions.append(sp.Ne(z, 0))
                basis[1] = head(a0 + 1, z)
            if head in TWO_BASIS:
                basis[1] = function_at(head, key, a0 + 1)
            valid = True
            for k in range(1, span + 1):
                if head in EXTRA_HEADS:

                    def guard(den, conditions=conditions):
                        if not nonzero(den):
                            return False
                        conditions.append(sp.Ne(den, 0))
                        return True

                    try:
                        basis[k] = next_basis(head, key, a0, k, basis, guard)
                    except (ValueError, TypeError, ZeroDivisionError):
                        valid = False
                        break
                elif head is sp.gamma:
                    basis[k] = (a0 + k - 1) * basis[k - 1]
                elif head is sp.polygamma:
                    m = key[0]
                    den = a0 + k - 1
                    if not nonzero(den):
                        valid = False
                        break
                    conditions.append(sp.Ne(den, 0))
                    basis[k] = basis[k - 1] + (-1) ** m * sp.factorial(m) / den ** (
                        m + 1
                    )
                elif head is sp.uppergamma:
                    if not nonzero(z):
                        valid = False
                        break
                    conditions.append(sp.Ne(z, 0))
                    basis[k] = (a0 + k - 1) * basis[k - 1] + z ** (a0 + k - 1) * sp.exp(
                        -z
                    )
                elif head is sp.expint:
                    den = a0 + k - 1
                    if not nonzero(den) or not nonzero(z):
                        valid = False
                        break
                    conditions.extend((sp.Ne(den, 0), sp.Ne(z, 0)))
                    basis[k] = (sp.exp(-z) - z * basis[k - 1]) / den
                elif head is sp.zeta:
                    den = a0 + k - 1
                    if not positive(den):
                        valid = False
                        break
                    conditions.append(sp.Gt(den, 0))
                    basis[k] = basis[k - 1] - den ** (-key[0])
                elif k >= 2:
                    coefficient = (
                        (2 * (a0 + k - 1) + 1) / z
                        if head in (sp.jn, sp.yn)
                        else 2 * (a0 + k - 1) / z
                    )
                    if head is sp.besseli:
                        basis[k] = basis[k - 2] - coefficient * basis[k - 1]
                    elif head is sp.besselk:
                        basis[k] = basis[k - 2] + coefficient * basis[k - 1]
                    else:
                        basis[k] = coefficient * basis[k - 1] - basis[k - 2]
                if sp.count_ops(basis[k]) > max_ops:
                    valid = False
                    stopped = "growth cap"
                    break
            if not valid:
                continue
            for atom, _, parameter in group:
                offset = shift(parameter, a0)
                if atom != basis[offset]:
                    replacements[atom] = basis[offset]
            if not replacements:
                continue
            candidate = current.xreplace(replacements)
            if sp.count_ops(candidate) > max_ops:
                stopped = "growth cap"
                continue
            # Collect exact coefficients without expanding special functions.
            markers = {
                f: sp.Dummy("recurrence_basis") for f in candidate.atoms(sp.Function)
            }
            rational = candidate.xreplace(markers)
            if time.monotonic() - started >= budget_seconds:
                stopped = "time budget"
                return finish()
            if sp.count_ops(rational) <= max_ops:
                # Restrict coefficient collection to rational expressions;
                # large compact powers must never be polynomial-expanded.
                if expansion_size(rational) > 256:
                    stopped = "coefficient expansion cap"
                    continue
                try:
                    candidate = sp.cancel(rational).xreplace(
                        {v: k for k, v in markers.items()}
                    )
                except (sp.PolynomialError, ValueError, NotImplementedError):
                    continue
            if sp.count_ops(candidate) > max_ops or candidate.has(sp.nan, sp.zoo):
                stopped = "growth or singular-value cap"
                continue
            if candidate == current:
                continue
            current = candidate
            steps += span
            evidence.append(
                LimitEvidence(
                    "integer_shift_recurrence_" + head.__name__,
                    "Exact same-argument integer-shift recurrence; common original function domain, no pole filling; guards: "
                    + str(tuple(dict.fromkeys(conditions))),
                    tuple(replacements.items()),
                    candidate,
                )
            )
    return finish()
