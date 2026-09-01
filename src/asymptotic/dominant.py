from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import sympy as sp

from ._power_simplify import analytic_powsimp, formal_powsimp, mixed_powsimp
from ._symbolic_errors import SYMBOLIC_ERRORS
from ._symbolic_policy import bounded_solve_one
from .canonical import canonical_equal
from .context import AsymptoticContext, GrowthComparison, context_for
from .parameter_auto import (
    automatic_parameter_stratification,
    parameter_symbols,
    specialize_expression,
)
from .stratification import AsymptoticStratification
from .transseries import TransseriesValuation, transseries_valuation


@dataclass(frozen=True)
class BalanceTerm:
    dependent_power: int
    valuation: sp.Rational
    leading_coefficient: sp.Expr
    coefficient: sp.Expr


@dataclass(frozen=True)
class DominantBalanceCertificate:
    """Replayable proof that a candidate is a global dominant balance."""

    exponent: sp.Rational
    terms: tuple[BalanceTerm, ...]
    weighted_valuations: tuple[sp.Rational, ...]
    minimum: sp.Rational
    dominant_indices: tuple[int, ...]
    coefficient_equation: sp.Expr

    def replay(self) -> bool:
        weighted = tuple(
            sp.Rational(sp.simplify(t.valuation + t.dependent_power * self.exponent))
            for t in self.terms
        )
        if weighted != self.weighted_valuations or min(weighted) != self.minimum:
            return False
        active = tuple(i for i, value in enumerate(weighted) if value == self.minimum)
        if active != self.dominant_indices or len(active) < 2:
            return False
        c = sp.Symbol("__balance_c")
        equation = sp.expand(
            sum(
                self.terms[i].leading_coefficient * c ** self.terms[i].dependent_power
                for i in active
            )
        )
        return canonical_equal(equation, self.coefficient_equation)


@dataclass(frozen=True)
class DominantBalanceCandidate:
    exponent: sp.Rational
    valuation: sp.Rational
    dominant_terms: tuple[BalanceTerm, ...]
    coefficient_equation: sp.Expr
    coefficients: tuple[sp.Expr, ...]
    certificate: DominantBalanceCertificate | None = None

    def replay(self) -> bool | None:
        return None if self.certificate is None else self.certificate.replay()


@dataclass(frozen=True)
class DominantBalanceBranch:
    """A recursively lifted dominant-balance branch.

    ``series`` is the current exact Puiseux prefix in ``variable``.  ``path``
    records every Newton/dominant-balance stage used to obtain it, including
    later polygons exposed only after translating by earlier terms.
    """

    series: sp.Expr
    path: tuple[DominantBalanceCandidate, ...]
    leading_coefficients: tuple[sp.Expr, ...]
    complete: bool = False
    monomials: tuple[sp.Expr, ...] = ()


def rational_valuation(expr: sp.Expr, variable: sp.Symbol) -> tuple[sp.Rational, sp.Expr] | None:
    """Return the exact rational valuation and leading coefficient at ``variable -> 0``."""

    expr = sp.sympify(expr)
    if expr == 0:
        return None
    try:
        # Factor powers of the asymptotic variable first.  SymPy may otherwise
        # leave expressions such as ``a*x + x`` additive, obscuring the exact
        # valuation even though it is simply ``x*(a + 1)``.
        normalized = sp.factor_terms(sp.expand(expr), variable)
        lead = normalized.as_leading_term(variable)
        exponent = sp.sympify(lead.as_powers_dict().get(variable, 0))
        if not exponent.is_Rational:
            return None
        exponent = sp.Rational(exponent)
        coeff = sp.simplify(lead / variable**exponent)
        if variable in coeff.free_symbols:
            return None
        return exponent, coeff
    except SYMBOLIC_ERRORS:
        return None


def polynomial_balance_terms(
    equation: sp.Expr,
    dependent: sp.Symbol,
    variable: sp.Symbol,
    *,
    valuation: Callable[
        [sp.Expr, sp.Symbol], tuple[sp.Rational, sp.Expr] | None
    ] = rational_valuation,
) -> tuple[BalanceTerm, ...]:
    """Return exact balance terms when the equation is polynomial in ``dependent``."""

    poly = sp.Poly(sp.expand(sp.sympify(equation)), dependent)
    out = []
    for (power,), coeff in poly.terms():
        val = valuation(coeff, variable)
        if val is None:
            continue
        alpha, lead = val
        out.append(BalanceTerm(power, alpha, sp.simplify(lead), coeff))
    return tuple(out)


def dependent_taylor_balance_terms(
    equation: sp.Expr,
    dependent: sp.Symbol,
    variable: sp.Symbol,
    *,
    degree: int = 8,
    context: AsymptoticContext | None = None,
    valuation: Callable[
        [sp.Expr, sp.Symbol], tuple[sp.Rational, sp.Expr] | None
    ] = rational_valuation,
) -> tuple[BalanceTerm, ...]:
    """Build a local Taylor jet in the dependent correction variable.

    This extends dominant balance to equations that are not polynomial in the
    dependent variable.  It is intended for a correction ``dependent -> 0``;
    the caller translates a nonzero branch center before invoking it.  Each
    Taylor coefficient remains an exact expression in the asymptotic variable,
    and only coefficients with exactly decidable rational valuation are used.
    """

    if degree < 1:
        raise ValueError("degree must be positive")
    ctx = context_for(variable, 0, context)
    equation = sp.sympify(equation)
    out = []
    derivative = equation
    factorial = sp.S.One
    for power in range(degree + 1):
        if power:
            derivative = sp.diff(derivative, dependent)
            factorial *= power
        try:
            coeff = sp.simplify(derivative.subs(dependent, 0) / factorial)
        except SYMBOLIC_ERRORS:
            continue
        zero = ctx.is_zero(coeff)
        if zero is True:
            continue
        val = valuation(coeff, variable)
        if val is None:
            continue
        alpha, lead = val
        out.append(BalanceTerm(power, alpha, sp.simplify(lead), coeff))
    return tuple(out)


def _candidate_from_terms(
    terms: tuple[BalanceTerm, ...],
    variable: sp.Symbol,
    *,
    context: AsymptoticContext,
    minimum_exponent: sp.Rational | None = None,
) -> tuple[DominantBalanceCandidate, ...]:
    """Construct one dominant-balance candidate from a selected lower-envelope face."""
    if len(terms) < 2:
        return ()
    c = sp.Symbol("__balance_c")
    found = {}
    for i, left in enumerate(terms):
        for right in terms[i + 1 :]:
            if left.dependent_power == right.dependent_power:
                continue
            r = sp.simplify(
                (right.valuation - left.valuation) / (left.dependent_power - right.dependent_power)
            )
            if not r.is_Rational:
                continue
            r = sp.Rational(r)
            if minimum_exponent is not None and r <= minimum_exponent:
                continue
            weighted = [sp.simplify(t.valuation + t.dependent_power * r) for t in terms]
            if not all(v.is_Rational for v in weighted):
                continue
            minimum = min(sp.Rational(v) for v in weighted)
            dominant = tuple(t for t, v in zip(terms, weighted) if sp.Rational(v) == minimum)
            if len(dominant) < 2:
                continue
            coeff_eq = sp.expand(
                sum(t.leading_coefficient * c**t.dependent_power for t in dominant)
            )
            roots = bounded_solve_one(coeff_eq, c) or ()
            unique = []
            for root in roots:
                root = sp.simplify(root)
                duplicate = any(canonical_equal(root, prior) for prior in unique)
                if not duplicate:
                    unique.append(root)
            active = tuple(i for i, value in enumerate(weighted) if sp.Rational(value) == minimum)
            certificate = DominantBalanceCertificate(
                exponent=r,
                terms=terms,
                weighted_valuations=tuple(sp.Rational(v) for v in weighted),
                minimum=minimum,
                dominant_indices=active,
                coefficient_equation=coeff_eq,
            )
            found[r] = DominantBalanceCandidate(
                exponent=r,
                valuation=minimum,
                dominant_terms=dominant,
                coefficient_equation=coeff_eq,
                coefficients=tuple(unique),
                certificate=certificate,
            )
    return tuple(found[r] for r in sorted(found))


def dominant_balance_candidates(
    equation: sp.Expr,
    dependent: sp.Symbol,
    variable: sp.Symbol,
    *,
    context: AsymptoticContext | None = None,
    valuation: Callable[
        [sp.Expr, sp.Symbol], tuple[sp.Rational, sp.Expr] | None
    ] = rational_valuation,
    taylor_degree: int = 8,
    minimum_exponent: sp.Rational | None = None,
    assumptions: sp.Expr | bool = sp.S.true,
    stratify_parameters: bool = True,
    max_parameter_splits: int = 6,
) -> (
    tuple[DominantBalanceCandidate, ...]
    | AsymptoticStratification[tuple[DominantBalanceCandidate, ...]]
):
    """Find exact Newton-style dominant balances.

    Polynomial equations use their exact coefficient set.  For genuinely
    transcendental dependence on ``dependent``, an exact local Taylor jet in
    the correction variable is used.  Candidate exponents are still accepted
    only after the tied terms are verified to attain the global minimum.
    """

    equation = sp.sympify(equation)
    ctx = context_for(variable, 0, context)
    try:
        polynomial = bool(equation.is_polynomial(dependent))
    except SYMBOLIC_ERRORS:
        polynomial = False
    if polynomial:
        terms = polynomial_balance_terms(equation, dependent, variable, valuation=valuation)
    else:
        terms = dependent_taylor_balance_terms(
            equation,
            dependent,
            variable,
            degree=taylor_degree,
            context=ctx,
            valuation=valuation,
        )

    generic_candidates = _candidate_from_terms(
        terms, variable, context=ctx, minimum_exponent=minimum_exponent
    )

    if stratify_parameters:
        parameters = parameter_symbols(equation, (variable, dependent))
        if parameters:
            coefficient_values = tuple(
                coefficient
                for candidate in generic_candidates
                for coefficient in candidate.coefficients
            )
            coefficient_differences = tuple(
                sp.simplify(left - right)
                for candidate in generic_candidates
                for i, left in enumerate(candidate.coefficients)
                for right in candidate.coefficients[i + 1 :]
            )
            structural = (
                tuple(term.leading_coefficient for term in terms)
                + coefficient_values
                + coefficient_differences
            )

            def evaluate(condition: sp.Expr) -> tuple[DominantBalanceCandidate, ...]:
                specialized = specialize_expression(equation, condition, parameters=parameters)
                result = dominant_balance_candidates(
                    specialized,
                    dependent,
                    variable,
                    context=AsymptoticContext(variable, point=ctx.point),
                    valuation=valuation,
                    taylor_degree=taylor_degree,
                    minimum_exponent=minimum_exponent,
                    assumptions=condition,
                    stratify_parameters=False,
                    max_parameter_splits=max_parameter_splits,
                )
                if isinstance(result, AsymptoticStratification):
                    raise TypeError("unstratified solver returned a stratification")
                return result

            stratified = automatic_parameter_stratification(
                structural,
                evaluate,
                parameters=parameters,
                assumptions=assumptions,
                max_splits=max_parameter_splits,
                provenance_source="asymptotic.dominant_balance",
            )
            if stratified is not None:
                return stratified

    return generic_candidates


def _prefix_term_count(expr: sp.Expr, variable: sp.Symbol) -> int:
    expr = sp.expand(expr)
    if expr == 0:
        return 0
    exponents = set()
    for term in sp.Add.make_args(expr):
        power = sp.sympify(term.as_powers_dict().get(variable, 0))
        if power.is_Rational:
            exponents.add(sp.Rational(power))
    return len(exponents)


def lift_dominant_balance_branches(
    equation: sp.Expr,
    dependent: sp.Symbol,
    variable: sp.Symbol,
    *,
    terms: int = 6,
    center: sp.Expr = 0,
    context: AsymptoticContext | None = None,
    taylor_degree: int = 8,
    max_depth: int = 32,
    corrections_must_vanish: bool = True,
) -> tuple[DominantBalanceBranch, ...]:
    """Recursively translate and re-run dominant balance until branches split.

    This is the shared Newton--Puiseux/implicit lifting engine.  Repeated roots
    on one Newton edge are *not* forced through a linear coefficient solve.
    Instead the selected leading term is translated out, a new residual
    equation in a correction variable is formed, and a new dominant balance is
    computed.  Thus later Newton polygons can split a singular branch.

    For transcendental equations the same recursion uses local Taylor jets in
    each correction variable, so analytic equations such as ``sin(y)-x=0`` and
    ``exp(y)-1-x=0`` are handled without first solving them explicitly.
    """

    if terms < 1:
        raise ValueError("terms must be positive")
    equation = sp.sympify(equation)
    center = sp.sympify(center)
    ctx = context_for(variable, 0, context)
    output = []

    def recurse(
        prefix: sp.Expr,
        minimum_exponent: sp.Rational | None,
        path: tuple[DominantBalanceCandidate, ...],
        coeffs: tuple[sp.Expr, ...],
        depth: int,
    ) -> None:
        """Lift one Puiseux balance path while residual order improves."""
        residual = sp.simplify(equation.subs(dependent, prefix))
        if residual == 0 or residual.is_zero is True:
            output.append(DominantBalanceBranch(sp.expand(prefix), path, coeffs, True))
            return
        if _prefix_term_count(sp.expand(prefix), variable) >= terms:
            output.append(DominantBalanceBranch(sp.expand(prefix), path, coeffs, False))
            return
        if depth >= max_depth:
            output.append(DominantBalanceBranch(sp.expand(prefix), path, coeffs, False))
            return

        correction = sp.Dummy(f"delta_{depth}")
        shifted = sp.simplify(equation.subs(dependent, prefix + correction))
        balances = dominant_balance_candidates(
            shifted,
            correction,
            variable,
            context=ctx,
            taylor_degree=taylor_degree,
            minimum_exponent=minimum_exponent,
            stratify_parameters=False,
        )
        progressed = False
        for balance in balances:
            exponent = balance.exponent
            if minimum_exponent is not None and exponent <= minimum_exponent:
                continue
            # A Taylor jet is local in the correction variable, so corrections
            # for transcendental dependence must vanish at the expansion point.
            if not shifted.is_polynomial(correction) and exponent <= 0:
                continue
            for c in balance.coefficients:
                c = sp.simplify(c)
                if c == 0 or c.is_zero is True:
                    continue
                new_prefix = sp.expand(prefix + c * variable**exponent)
                if sp.simplify(new_prefix - prefix) == 0:
                    continue
                progressed = True
                recurse(
                    new_prefix,
                    exponent,
                    path + (balance,),
                    coeffs + (sp.simplify(c),),
                    depth + 1,
                )
        if not progressed and prefix != center:
            output.append(DominantBalanceBranch(sp.expand(prefix), path, coeffs, False))

    start_min = sp.Rational(0) if corrections_must_vanish else None
    recurse(center, start_min, (), (), 0)

    # Deduplicate branches by certified symbolic equality when possible.
    unique = []
    for branch in output:
        duplicate = any(sp.simplify(branch.series - prior.series) == 0 for prior in unique)
        if not duplicate:
            unique.append(branch)
    return tuple(unique)


@dataclass(frozen=True)
class TransseriesBalanceTerm:
    """A dependent-variable Taylor term with a generalized valuation."""

    dependent_power: int
    valuation: TransseriesValuation
    coefficient: sp.Expr


@dataclass(frozen=True)
class TransseriesBalanceCandidate:
    """Dominant balance whose correction scale is a general monomial."""

    monomial: sp.Expr
    dominant_terms: tuple[TransseriesBalanceTerm, ...]
    coefficient_equation: sp.Expr
    coefficients: tuple[sp.Expr, ...]
    common_monomial: sp.Expr

    @property
    def exponent(self) -> sp.Rational | None:
        """Return the Puiseux exponent when this monomial is a pure power."""
        symbols = tuple(self.monomial.free_symbols)
        if len(symbols) != 1:
            return None
        variable = symbols[0]
        power = sp.sympify(self.monomial.as_powers_dict().get(variable, 0))
        remainder = formal_powsimp(self.monomial / variable**power)
        if power.is_Rational and variable not in remainder.free_symbols:
            return sp.Rational(power)
        return None


def transseries_balance_terms(
    equation: sp.Expr,
    dependent: sp.Symbol,
    variable: sp.Symbol,
    *,
    degree: int = 8,
    point: sp.Expr = 0,
    context: AsymptoticContext | None = None,
) -> tuple[TransseriesBalanceTerm, ...]:
    """Return Taylor terms valued in the package's generalized monomial group.

    Polynomial dependence is exact.  Otherwise a local Taylor jet in the
    dependent correction variable is used, exactly as in the Puiseux engine,
    but coefficient valuations may now be powers, logs, exponentials, or mixed
    exp-log monomials.
    """

    if degree < 1:
        raise ValueError("degree must be positive")
    equation = sp.sympify(equation)
    ctx = context_for(variable, point, context)
    raw = []
    try:
        polynomial = bool(equation.is_polynomial(dependent))
    except SYMBOLIC_ERRORS:
        polynomial = False
    if polynomial:
        poly = sp.Poly(sp.expand(equation), dependent)
        raw = [(power, coeff) for (power,), coeff in poly.terms()]
    else:
        derivative = equation
        factorial = sp.S.One
        for power in range(degree + 1):
            if power:
                derivative = sp.diff(derivative, dependent)
                factorial *= power
            try:
                coeff = sp.simplify(derivative.subs(dependent, 0) / factorial)
            except SYMBOLIC_ERRORS:
                continue
            raw.append((power, coeff))

    out = []
    for power, coeff in raw:
        if ctx.is_zero(coeff) is True:
            continue
        val = transseries_valuation(coeff, variable, point=point, context=ctx)
        if val is None:
            continue
        out.append(TransseriesBalanceTerm(power, val, coeff))
    return tuple(out)


def _same_monomial_class(
    left: sp.Expr,
    right: sp.Expr,
    context: AsymptoticContext,
) -> bool:
    if sp.simplify(left - right) == 0:
        return True
    relation, _ = context.compare_growth(left, right)
    return relation is GrowthComparison.SAME_ORDER


def _transseries_candidate(
    terms: tuple[TransseriesBalanceTerm, ...],
    monomial: sp.Expr,
    variable: sp.Symbol,
    *,
    context: AsymptoticContext,
) -> TransseriesBalanceCandidate | None:
    """Build transseries balance candidates from valued coefficient terms and characteristic roots."""
    weighted = [
        formal_powsimp(term.valuation.monomial * monomial**term.dependent_power) for term in terms
    ]
    if not weighted:
        return None

    # Pick a maximal asymptotic magnitude.  Unknown comparisons make the
    # candidate uncertified and are rejected rather than guessed.
    common = weighted[0]
    for value in weighted[1:]:
        relation, _ = context.compare_growth(value, common)
        if relation is GrowthComparison.LARGER:
            common = value
        elif relation is GrowthComparison.UNKNOWN:
            return None

    dominant_indices = []
    ratios = {}
    for index, value in enumerate(weighted):
        relation, ratio = context.compare_growth(value, common)
        if relation is GrowthComparison.LARGER:
            return None
        if relation is GrowthComparison.UNKNOWN:
            return None
        if relation is GrowthComparison.SAME_ORDER:
            if ratio is None:
                try:
                    ratio = context.limit(value / common)
                except SYMBOLIC_ERRORS:
                    return None
            if ratio is None or ratio.is_finite is not True or ratio.is_zero is not False:
                return None
            dominant_indices.append(index)
            ratios[index] = sp.simplify(ratio)

    if len(dominant_indices) < 2:
        return None

    c = sp.Symbol("__balance_c")
    dominant = tuple(terms[i] for i in dominant_indices)
    coeff_eq = sp.expand(
        sum(
            terms[i].valuation.leading_coefficient * ratios[i] * c ** terms[i].dependent_power
            for i in dominant_indices
        )
    )
    roots = bounded_solve_one(coeff_eq, c)
    if roots is None:
        # A high-degree characteristic polynomial can still factor into small
        # residue-field pieces.  Solve those factors separately before giving
        # up; this captures cases such as c**5+1 without invoking a potentially
        # expensive general algebraic-root isolation routine.
        factored_roots: list[sp.Expr] = []
        try:
            _content, factors = sp.factor_list(coeff_eq, c)
        except (sp.PolynomialError, NotImplementedError, ValueError):
            factors = ()
        for factor, _multiplicity in factors:
            factor_roots = bounded_solve_one(factor, c)
            if factor_roots is None:
                factored_roots = []
                break
            factored_roots.extend(factor_roots)
        roots = tuple(factored_roots)
    roots = roots or ()
    unique = []
    for root in roots:
        root = sp.simplify(root)
        if context.is_zero(root) is True:
            continue
        if not any(context.is_zero(root - prior) is True for prior in unique):
            unique.append(root)
    if not unique:
        return None
    return TransseriesBalanceCandidate(
        monomial=formal_powsimp(monomial),
        dominant_terms=dominant,
        coefficient_equation=coeff_eq,
        coefficients=tuple(unique),
        common_monomial=formal_powsimp(common),
    )


def transseries_dominant_balance_candidates(
    equation: sp.Expr,
    dependent: sp.Symbol,
    variable: sp.Symbol,
    *,
    context: AsymptoticContext | None = None,
    taylor_degree: int = 8,
    point: sp.Expr = 0,
    smaller_than: sp.Expr | None = None,
    corrections_must_vanish: bool = True,
    assumptions: sp.Expr | bool = sp.S.true,
    stratify_parameters: bool = True,
    max_parameter_splits: int = 6,
) -> (
    tuple[TransseriesBalanceCandidate, ...]
    | AsymptoticStratification[tuple[TransseriesBalanceCandidate, ...]]
):
    """Find dominant balances in an expression-valued transseries domain.

    For a pair ``a_i delta**i`` and ``a_j delta**j`` with leading monomials
    ``m_i`` and ``m_j``, the candidate correction monomial is obtained from
    ``m_i*M**i ~ m_j*M**j``.  Every candidate is then checked against *all*
    terms using the same exact growth-comparison service used by scale
    discovery.  Newton--Puiseux balance is therefore the special case in which
    every ``M`` is a rational power of the local variable.
    """

    ctx = context_for(variable, point, context)
    terms = transseries_balance_terms(
        equation,
        dependent,
        variable,
        degree=taylor_degree,
        point=point,
        context=ctx,
    )
    if stratify_parameters:
        parameters = parameter_symbols(equation, (variable, dependent))
        if parameters:
            structural = tuple(term.valuation.leading_coefficient for term in terms)

            def evaluate(condition: sp.Expr) -> tuple[TransseriesBalanceCandidate, ...]:
                specialized = specialize_expression(equation, condition, parameters=parameters)
                result = transseries_dominant_balance_candidates(
                    specialized,
                    dependent,
                    variable,
                    context=AsymptoticContext(variable, point=point),
                    taylor_degree=taylor_degree,
                    point=point,
                    smaller_than=smaller_than,
                    corrections_must_vanish=corrections_must_vanish,
                    assumptions=condition,
                    stratify_parameters=False,
                    max_parameter_splits=max_parameter_splits,
                )
                if isinstance(result, AsymptoticStratification):
                    raise TypeError("unstratified solver returned a stratification")
                return result

            stratified = automatic_parameter_stratification(
                structural,
                evaluate,
                parameters=parameters,
                assumptions=assumptions,
                max_splits=max_parameter_splits,
                provenance_source="asymptotic.transseries_dominant_balance",
            )
            if stratified is not None:
                return stratified

    if len(terms) < 2:
        return ()

    found = []
    for i, left in enumerate(terms):
        for right in terms[i + 1 :]:
            power_delta = left.dependent_power - right.dependent_power
            if power_delta == 0:
                continue
            ratio = sp.powsimp(right.valuation.monomial / left.valuation.monomial)
            monomial = sp.powsimp(ratio ** sp.Rational(1, power_delta))
            if corrections_must_vanish:
                lim = ctx.limit(sp.Abs(monomial))
                if lim != 0:
                    continue
            if smaller_than is not None:
                relation, _ = ctx.compare_growth(monomial, smaller_than)
                if relation is not GrowthComparison.SMALLER:
                    continue

            candidate = _transseries_candidate(terms, monomial, variable, context=ctx)
            if candidate is None:
                continue
            duplicate = any(
                _same_monomial_class(candidate.monomial, old.monomial, ctx) for old in found
            )
            if not duplicate:
                found.append(candidate)
    return tuple(found)


def lift_transseries_balance_branches(
    equation: sp.Expr,
    dependent: sp.Symbol,
    variable: sp.Symbol,
    *,
    terms: int = 6,
    center: sp.Expr = 0,
    context: AsymptoticContext | None = None,
    taylor_degree: int = 8,
    max_depth: int = 32,
    corrections_must_vanish: bool = True,
) -> tuple[DominantBalanceBranch, ...]:
    """Recursively lift branches with arbitrary ordered asymptotic monomials."""

    if terms < 1:
        raise ValueError("terms must be positive")
    equation = sp.sympify(equation)
    center = sp.sympify(center)
    ctx = context_for(variable, 0, context)
    output = []

    def recurse(
        prefix: sp.Expr,
        previous_monomial: sp.Expr | None,
        path: tuple[TransseriesBalanceCandidate, ...],
        coeffs: tuple[sp.Expr, ...],
        monomials: tuple[sp.Expr, ...],
        depth: int,
    ) -> None:
        """Lift one transseries balance path through successively smaller scales."""
        residual = analytic_powsimp(sp.simplify(equation.subs(dependent, prefix)))
        if ctx.is_zero(residual) is True:
            # The branch container accepts either balance family, so exact
            # termination can use the same result type as incomplete branches.
            output.append(  # type: ignore[arg-type]
                DominantBalanceBranch(sp.expand(prefix), path, coeffs, True, monomials)
            )
            return
        center_terms = 0 if ctx.is_zero(center) is True else 1
        if len(monomials) + center_terms >= terms or depth >= max_depth:
            output.append(  # type: ignore[arg-type]
                DominantBalanceBranch(sp.expand(prefix), path, coeffs, False, monomials)
            )
            return

        correction = sp.Dummy(f"tau_{depth}")
        shifted = sp.powsimp(sp.simplify(equation.subs(dependent, prefix + correction)))
        balances = transseries_dominant_balance_candidates(
            shifted,
            correction,
            variable,
            context=ctx,
            taylor_degree=taylor_degree,
            smaller_than=previous_monomial,
            corrections_must_vanish=corrections_must_vanish,
            stratify_parameters=False,
        )
        progressed = False
        for balance in balances:
            for coefficient in balance.coefficients:
                term = mixed_powsimp(coefficient, balance.monomial)
                new_prefix = analytic_powsimp(sp.expand(prefix + term))
                if ctx.is_zero(new_prefix - prefix) is True:
                    continue
                progressed = True
                recurse(
                    new_prefix,
                    balance.monomial,
                    path + (balance,),  # type: ignore[arg-type]
                    coeffs + (sp.simplify(coefficient),),
                    monomials + (balance.monomial,),
                    depth + 1,
                )
        if not progressed and monomials:
            output.append(  # type: ignore[arg-type]
                DominantBalanceBranch(sp.expand(prefix), path, coeffs, False, monomials)
            )

    recurse(center, None, (), (), (), 0)

    unique = []
    for branch in output:
        if not any(ctx.is_zero(branch.series - prior.series) is True for prior in unique):
            unique.append(branch)
    return tuple(unique)
