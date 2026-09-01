"""Newton--MRV polynomial solving over computable Hardy/log-exp coefficients.

This module is an expert backend for :func:`asymptotic.solve.asymptotic_solve`.
It deliberately avoids constructing exact algebraic roots of the original
parameter-dependent polynomial.  Instead it values each coefficient in the
package's transseries/Hardy monomial group, constructs lower Newton balances,
and recursively lifts smaller corrections.  For real output a separate
asymptotic Sturm sequence certifies the eventual number of distinct real roots
when all coefficient signs are decidable.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import sympy as sp

from ._power_simplify import analytic_powsimp, mixed_powsimp
from ._symbolic_errors import SYMBOLIC_ERRORS
from ._symbolic_policy import bounded_ask
from .context import AsymptoticContext, GrowthComparison, context_for
from .dominant import (
    TransseriesBalanceCandidate,
    transseries_dominant_balance_candidates,
)
from .stratification import AsymptoticStratification


@dataclass(frozen=True)
class MRVNewtonStep:
    """One globally checked Newton balance in the Hardy/transseries field."""

    prefix_before: sp.Expr
    correction_monomial: sp.Expr
    coefficient: sp.Expr
    coefficient_equation: sp.Expr
    common_monomial: sp.Expr
    dominant_powers: tuple[int, ...]
    char_multiplicity: int

    @property
    def correction(self) -> sp.Expr:
        return mixed_powsimp(self.coefficient, self.correction_monomial)

    def replay(self) -> bool | None:
        """Verify that the recorded coefficient solves its characteristic equation."""

        balance_symbols = [
            symbol
            for symbol in self.coefficient_equation.free_symbols
            if symbol.name == "__balance_c"
        ]
        if len(balance_symbols) != 1:
            return None
        residual = sp.simplify(self.coefficient_equation.subs(balance_symbols[0], self.coefficient))
        if residual == 0 or residual.is_zero is True:
            return True
        if residual.is_zero is False:
            return False
        return None


@dataclass(frozen=True)
class MRVHardyBranch:
    """A finite asymptotic root prefix produced without exact algebraic solving."""

    expression: sp.Expr
    steps: tuple[MRVNewtonStep, ...]
    multiplicity: int
    exact: bool

    @property
    def terms(self) -> int:
        return len(self.steps)


@dataclass(frozen=True)
class AsymptoticSturmCertificate:
    """Eventual real-root count obtained from a Sturm sequence in the root variable."""

    polynomial: sp.Expr
    dependent: sp.Symbol
    parameter: sp.Symbol
    point: sp.Expr
    sequence: tuple[sp.Expr, ...]
    signs_at_minus_infinity: tuple[int, ...]
    signs_at_plus_infinity: tuple[int, ...]
    variations_at_minus_inf: int | None
    variations_at_plus_inf: int | None
    distinct_real_roots: int | None
    certified: bool

    def replay(self) -> bool | None:
        """Replay the combinatorial part of the stored Sturm certificate."""

        if not self.certified:
            return None
        minus = _sign_variations(self.signs_at_minus_infinity)
        plus = _sign_variations(self.signs_at_plus_infinity)
        return (
            minus == self.variations_at_minus_inf
            and plus == self.variations_at_plus_inf
            and minus - plus == self.distinct_real_roots
        )


@dataclass(frozen=True)
class MRVHardySolveResult:
    """Expert result for one polynomial solved by Newton--MRV lifting."""

    branches: tuple[MRVHardyBranch, ...]
    degree: int
    point: sp.Expr
    domain: sp.Set
    complete: bool
    sturm: AsymptoticSturmCertificate | None = None
    zero_root_multiplicity: int = 0
    unresolved_multiplicity: int = 0


def _characteristic_symbol(candidate: TransseriesBalanceCandidate) -> sp.Symbol | None:
    symbols = tuple(
        symbol
        for symbol in candidate.coefficient_equation.free_symbols
        if symbol.name == "__balance_c"
    )
    return symbols[0] if len(symbols) == 1 else None


def _char_multiplicity(
    candidate: TransseriesBalanceCandidate,
    coefficient: sp.Expr,
) -> int:
    symbol = _characteristic_symbol(candidate)
    if symbol is None:
        return 1
    polynomial = sp.expand(candidate.coefficient_equation)
    degree = sp.degree(polynomial, symbol)
    if not isinstance(degree, int) or degree < 1:
        return 1
    multiplicity = 0
    derivative = polynomial
    for _ in range(degree + 1):
        value = sp.simplify(derivative.subs(symbol, coefficient))
        if value != 0 and value.is_zero is not True:
            break
        multiplicity += 1
        derivative = sp.diff(derivative, symbol)
    return max(1, multiplicity)


def _newton_step(
    prefix: sp.Expr,
    candidate: TransseriesBalanceCandidate,
    coefficient: sp.Expr,
) -> MRVNewtonStep:
    return MRVNewtonStep(
        prefix_before=prefix,
        correction_monomial=candidate.monomial,
        coefficient=coefficient,
        coefficient_equation=candidate.coefficient_equation,
        common_monomial=candidate.common_monomial,
        dominant_powers=tuple(term.dependent_power for term in candidate.dominant_terms),
        char_multiplicity=_char_multiplicity(candidate, coefficient),
    )


def _branch_real_decision(
    expression: sp.Expr,
    assumptions: sp.Expr,
) -> bool | None:
    try:
        answer = bounded_ask(sp.Q.real(expression), assumptions)
    except (TypeError, ValueError, NotImplementedError):
        answer = None
    if answer in (True, False):
        return answer
    imaginary = sp.refine(sp.im(expression), assumptions)
    if imaginary == 0 or imaginary.is_zero is True:
        return True
    if imaginary.is_zero is False:
        return False
    return None


def _deduplicate_branches(
    branches: list[MRVHardyBranch],
    context: AsymptoticContext,
) -> tuple[MRVHardyBranch, ...]:
    unique: list[MRVHardyBranch] = []
    for branch in branches:
        duplicate = False
        for prior in unique:
            difference = analytic_powsimp(branch.expression - prior.expression)
            if context.is_zero(difference) is True:
                duplicate = True
                break
        if not duplicate:
            unique.append(branch)
    return tuple(unique)


def _candidate_tuple(
    equation: sp.Expr,
    dependent: sp.Symbol,
    parameter: sp.Symbol,
    *,
    point: sp.Expr,
    context: AsymptoticContext,
    smaller_than: sp.Expr | None,
    assumptions: sp.Expr,
) -> tuple[TransseriesBalanceCandidate, ...]:
    candidates = transseries_dominant_balance_candidates(
        equation,
        dependent,
        parameter,
        context=context,
        point=point,
        smaller_than=smaller_than,
        corrections_must_vanish=False,
        assumptions=assumptions,
        stratify_parameters=False,
    )
    if isinstance(candidates, AsymptoticStratification):
        return ()
    return candidates


def _lift_mrv_hardy_branches(
    polynomial: sp.Expr,
    dependent: sp.Symbol,
    parameter: sp.Symbol,
    *,
    point: sp.Expr,
    terms: int,
    assumptions: sp.Expr,
    context: AsymptoticContext,
) -> tuple[MRVHardyBranch, ...]:
    """Recursively lift MRV/Hardy Newton branches and account for unresolved multiplicity."""
    correction = sp.Dummy("_mrv_delta")
    output: list[MRVHardyBranch] = []

    def recurse(
        prefix: sp.Expr,
        previous_monomial: sp.Expr | None,
        steps: tuple[MRVNewtonStep, ...],
        depth: int,
    ) -> None:
        """Lift one Hardy branch until solved, truncated, or unsupported."""
        residual = analytic_powsimp(sp.expand(polynomial.subs(dependent, prefix)))
        if context.is_zero(residual) is True:
            multiplicity = steps[-1].char_multiplicity if steps else 1
            output.append(MRVHardyBranch(prefix, steps, multiplicity, True))
            return
        if depth >= terms:
            multiplicity = steps[-1].char_multiplicity if steps else 1
            output.append(MRVHardyBranch(prefix, steps, multiplicity, False))
            return

        translated = analytic_powsimp(sp.expand(polynomial.subs(dependent, prefix + correction)))
        candidates = _candidate_tuple(
            translated,
            correction,
            parameter,
            point=point,
            context=context,
            smaller_than=previous_monomial,
            assumptions=assumptions,
        )
        if not candidates:
            if steps:
                multiplicity = steps[-1].char_multiplicity
                output.append(MRVHardyBranch(prefix, steps, multiplicity, False))
            return

        advanced = False
        for candidate in candidates:
            if previous_monomial is not None:
                relation, _ = context.compare_growth(candidate.monomial, previous_monomial)
                if relation is not GrowthComparison.SMALLER:
                    continue
            for coefficient in candidate.coefficients:
                step = _newton_step(prefix, candidate, coefficient)
                next_prefix = analytic_powsimp(prefix + step.correction)
                if context.is_zero(next_prefix - prefix) is True:
                    continue
                advanced = True
                recurse(
                    next_prefix,
                    candidate.monomial,
                    steps + (step,),
                    depth + 1,
                )
        if not advanced and steps:
            multiplicity = steps[-1].char_multiplicity
            output.append(MRVHardyBranch(prefix, steps, multiplicity, False))

    recurse(sp.S.Zero, None, (), 0)
    return _deduplicate_branches(output, context)


def _sign_variations(signs: tuple[int, ...]) -> int:
    nonzero = tuple(sign for sign in signs if sign != 0)
    return sum(left != right for left, right in pairwise(nonzero))


def _sturm_infinity_signs(
    sequence: tuple[sp.Expr, ...],
    dependent: sp.Symbol,
    context: AsymptoticContext,
    assumptions: sp.Expr,
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    minus: list[int] = []
    plus: list[int] = []
    for member in sequence:
        member = sp.refine(sp.sympify(member), assumptions)
        try:
            poly = sp.Poly(member, dependent)
        except sp.PolynomialError:
            return None
        if poly.is_zero:
            minus.append(0)
            plus.append(0)
            continue
        leading = sp.refine(poly.LC(), assumptions)
        sign = context.eventual_sign(leading)
        if sign not in (-1, 0, 1):
            return None
        plus.append(sign)
        minus.append(sign if poly.degree() % 2 == 0 else -sign)
    return tuple(minus), tuple(plus)


def _asymptotic_reduce_polynomial(
    polynomial: sp.Expr,
    dependent: sp.Symbol,
    context: AsymptoticContext,
) -> sp.Expr:
    """Drop leading coefficients proved identically zero in the asymptotic germ."""

    polynomial = sp.expand(sp.sympify(polynomial))
    try:
        poly = sp.Poly(polynomial, dependent)
    except sp.PolynomialError:
        return polynomial
    if poly.is_zero:
        return sp.S.Zero
    coefficients = dict(poly.terms())
    degrees = sorted((powers[0] for powers in coefficients), reverse=True)
    kept: list[tuple[tuple[int], sp.Expr]] = []
    found_lead = False
    for degree in degrees:
        coefficient = coefficients[(degree,)]
        if not found_lead and context.is_zero(coefficient) is True:
            continue
        found_lead = True
        kept.append(((degree,), coefficient))
    if not kept:
        return sp.S.Zero
    return sp.Poly.from_dict(dict(kept), dependent).as_expr()


def _asymptotic_sturm_sequence(
    polynomial: sp.Expr,
    dependent: sp.Symbol,
    context: AsymptoticContext,
) -> tuple[sp.Expr, ...]:
    """Build a Sturm sequence while reducing germ-zero leading coefficients.

    Ordinary symbolic polynomial remainder arithmetic can retain a nominal
    leading coefficient that is identically zero only in the asymptotic germ.
    Removing those coefficients after every remainder mirrors the Hardy-field
    Sturm construction and avoids incorrect eventual degree/sign data.
    """

    first = _asymptotic_reduce_polynomial(polynomial, dependent, context)
    if first == 0:
        return ()
    second = _asymptotic_reduce_polynomial(sp.diff(first, dependent), dependent, context)
    sequence = [first]
    if second == 0:
        return tuple(sequence)
    while second != 0:
        sequence.append(second)
        try:
            remainder = -sp.rem(first, second, dependent)
        except (sp.PolynomialError, ValueError, TypeError, *SYMBOLIC_ERRORS):
            return ()
        remainder = _asymptotic_reduce_polynomial(remainder, dependent, context)
        first, second = second, remainder
    return tuple(sequence)


def asymptotic_sturm_certificate(
    polynomial: sp.Expr,
    dependent: sp.Symbol,
    parameter: sp.Symbol,
    *,
    point: sp.Expr = sp.oo,
    assumptions: sp.Expr = sp.S.true,
    context: AsymptoticContext | None = None,
) -> AsymptoticSturmCertificate:
    """Count eventual distinct real roots by coefficient-sign Sturm analysis.

    The ordinary Sturm sequence is formed exactly in ``dependent``.  Its
    coefficients remain expressions in the asymptotic parameter.  Their signs
    at ``parameter -> point`` are then decided by :class:`AsymptoticContext`.
    Unknown coefficient signs make the certificate explicitly uncertified.
    """

    polynomial = sp.expand(sp.sympify(polynomial))
    ctx = context_for(parameter, point, context)
    sequence = _asymptotic_sturm_sequence(polynomial, dependent, ctx)
    signs = _sturm_infinity_signs(sequence, dependent, ctx, assumptions) if sequence else None
    if signs is None:
        return AsymptoticSturmCertificate(
            polynomial, dependent, parameter, point, sequence, (), (), None, None, None, False
        )
    minus, plus = signs
    left = _sign_variations(minus)
    right = _sign_variations(plus)
    return AsymptoticSturmCertificate(
        polynomial,
        dependent,
        parameter,
        point,
        sequence,
        minus,
        plus,
        left,
        right,
        left - right,
        True,
    )


def mrv_hardy_polynomial_solve(
    polynomial: sp.Expr,
    dependent: sp.Symbol,
    parameter: sp.Symbol,
    *,
    point: sp.Expr = sp.oo,
    terms: int = 6,
    domain: sp.Set = sp.S.Complexes,
    assumptions: sp.Expr = sp.S.true,
    context: AsymptoticContext | None = None,
) -> MRVHardySolveResult:
    """Solve a univariate polynomial by MRV Newton balances and recursive lifting.

    Coefficients may contain logarithmic/exponential Hardy-field expressions;
    exact roots of the original polynomial are never requested.  ``complete``
    means that branch counting is independently certified: polynomial degree
    counting is used over the complexes, while real output requires a certified
    asymptotic Sturm count.  An incomplete result is still useful to expert
    callers, but :func:`asymptotic_solve` only prefers it automatically when
    completeness has been established.
    """

    if not isinstance(dependent, sp.Symbol) or not isinstance(parameter, sp.Symbol):
        raise TypeError("dependent and parameter must be symbols")
    if terms < 1:
        raise ValueError("terms must be positive")
    polynomial = sp.expand(sp.sympify(polynomial))
    try:
        poly = sp.Poly(polynomial, dependent)
    except sp.PolynomialError as exc:
        raise ValueError(
            "MRV Hardy solving requires a polynomial in the dependent variable"
        ) from exc
    if poly.degree() < 1:
        return MRVHardySolveResult((), max(0, poly.degree()), point, domain, True, None)

    ctx = context_for(parameter, point, context)

    # Strip an exact/identically-zero root before Newton lifting.  This mirrors
    # the reduction step used by classical MRV-Hardy solvers and prevents a
    # zero characteristic root from being discarded merely because Newton
    # corrections are required to have a nonzero leading coefficient.
    zero_multiplicity = 0
    for exponent in range(poly.degree() + 1):
        coefficient = poly.coeff_monomial(dependent**exponent)
        if ctx.is_zero(coefficient) is True:
            zero_multiplicity += 1
            continue
        break
    reduced_polynomial = polynomial
    if zero_multiplicity:
        reduced_polynomial = sp.cancel(polynomial / dependent**zero_multiplicity)

    branches = list(
        _lift_mrv_hardy_branches(
            reduced_polynomial,
            dependent,
            parameter,
            point=point,
            terms=terms,
            assumptions=assumptions,
            context=ctx,
        )
    )
    if zero_multiplicity:
        branches.insert(0, MRVHardyBranch(sp.S.Zero, (), zero_multiplicity, True))
    branches = tuple(branches)

    sturm = None
    if domain == sp.S.Reals:
        real_branches = tuple(
            branch
            for branch in branches
            if _branch_real_decision(branch.expression, assumptions) is True
        )
        sturm = asymptotic_sturm_certificate(
            polynomial,
            dependent,
            parameter,
            point=point,
            assumptions=assumptions,
            context=ctx,
        )
        complete = bool(
            sturm.certified
            and sturm.distinct_real_roots is not None
            and len(real_branches) == sturm.distinct_real_roots
        )
        unresolved = (
            max(0, (sturm.distinct_real_roots or 0) - len(real_branches))
            if sturm.certified
            else poly.degree()
        )
        return MRVHardySolveResult(
            real_branches,
            poly.degree(),
            point,
            domain,
            complete,
            sturm,
            zero_multiplicity,
            unresolved,
        )

    multiplicity_total = sum(branch.multiplicity for branch in branches)
    complete = domain == sp.S.Complexes and multiplicity_total == poly.degree()
    unresolved = max(0, poly.degree() - multiplicity_total)
    return MRVHardySolveResult(
        branches,
        poly.degree(),
        point,
        domain,
        complete,
        sturm,
        zero_multiplicity,
        unresolved,
    )
