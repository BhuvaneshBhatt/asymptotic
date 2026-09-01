from __future__ import annotations

from dataclasses import dataclass
from functools import reduce

import sympy as sp

from ._integer_utils import integer_lcm as _lcm
from ._power_simplify import analytic_powsimp, formal_powsimp
from .context import AsymptoticContext, context_for
from .dominant import (
    DominantBalanceCandidate,
    TransseriesBalanceCandidate,
    dominant_balance_candidates,
    lift_transseries_balance_branches,
    transseries_balance_terms,
)
from .function_properties.semantics import entails
from .parameter_auto import (
    automatic_parameter_stratification,
    parameter_symbols,
    specialize_expression,
)
from .puiseux import BranchChoice, PuiseuxSeries, PuiseuxTerm, _extract_puiseux_terms
from .remainder import AsymptoticRemainder
from .stratification import AsymptoticStratification
from .transseries import TransseriesExpansion, TransseriesTerm

BalanceCandidate = DominantBalanceCandidate | TransseriesBalanceCandidate
SeriesLike = PuiseuxSeries | TransseriesExpansion


@dataclass(frozen=True)
class ImplicitSingularityProfile:
    """Local multiplicity/scaling data for an implicit branch center.

    ``multiplicity`` is the first nonzero dependent derivative order of the
    local equation at the requested center.  A value greater than one marks a
    singular implicit root and triggers Newton--Puiseux/scaling lifting.
    ``turning_point`` is certified when the dependent Jacobian vanishes but
    the first variable derivative does not.
    """

    equation: sp.Expr
    dependent: sp.Symbol
    variable: sp.Symbol
    point: sp.Expr
    dependent_limit: sp.Expr
    multiplicity: int | None
    jacobian: sp.Expr
    variable_derivative: sp.Expr
    singular: bool | None
    turning_point: bool | None
    discriminant: sp.Expr | None = None
    scaling_exponents: tuple[sp.Rational, ...] = ()

    @property
    def requires_blowup(self) -> bool:
        """Whether a singular Newton--Puiseux/scaling lift is certified necessary."""

        return self.multiplicity is not None and self.multiplicity > 1


def _localize_implicit_equation(
    equation: sp.Expr,
    dependent: sp.Symbol,
    variable: sp.Symbol,
    *,
    point: sp.Expr,
    dependent_limit: sp.Expr,
) -> tuple[sp.Expr, sp.Symbol, sp.Symbol]:
    """Translate an implicit problem to ``u -> 0, delta -> 0`` coordinates."""

    point = sp.sympify(point)
    dependent_limit = sp.sympify(dependent_limit)
    u = sp.Dummy("implicit_u", positive=True)
    delta = sp.Dummy("implicit_delta")
    if point in (sp.oo, -sp.oo):
        sign = 1 if point is sp.oo else -1
        transformed = sp.together(
            sp.sympify(equation)
            .subs(dependent, dependent_limit + delta)
            .xreplace({variable: sign / u})
        )
        transformed, _ = sp.fraction(transformed)
    elif point == 0:
        transformed = (
            sp.sympify(equation).subs(dependent, dependent_limit + delta).xreplace({variable: u})
        )
    else:
        transformed = (
            sp.sympify(equation)
            .subs(dependent, dependent_limit + delta)
            .xreplace({variable: point + u})
        )
    return sp.expand(transformed), u, delta


def implicit_singularity_profile(
    equation: sp.Expr,
    dependent: sp.Symbol,
    variable: sp.Symbol,
    *,
    point: sp.Expr = 0,
    dependent_limit: sp.Expr = 0,
    taylor_degree: int = 8,
    assumptions: sp.Expr | bool = sp.S.true,
) -> ImplicitSingularityProfile:
    """Diagnose local root multiplicity, turning points, and Puiseux scales.

    The routine is deliberately local and bounded.  For polynomial dependence
    it also records a small-degree discriminant; for analytic dependence it
    searches dependent derivatives only through ``taylor_degree``.
    """

    equation = sp.sympify(equation)
    local, u, delta = _localize_implicit_equation(
        equation, dependent, variable, point=point, dependent_limit=dependent_limit
    )
    ctx = AsymptoticContext(u, point=0)
    assumptions = sp.sympify(assumptions)

    def zero_decision(value: sp.Expr) -> bool | None:
        value = sp.simplify(value)
        if not value.free_symbols:
            return ctx.is_zero(value)
        eq = entails(sp.Eq(value, 0), assumptions)
        if eq is True:
            return True
        ne = entails(sp.Ne(value, 0), assumptions)
        if ne is True:
            return False
        return None

    derivatives = []
    derivative = local
    multiplicity: int | None = None
    for order in range(taylor_degree + 1):
        if order:
            derivative = sp.diff(derivative, delta)
        value = sp.simplify(derivative.subs({delta: 0, u: 0}))
        derivatives.append(value)
        if order >= 1:
            zero = zero_decision(value)
            if zero is False:
                multiplicity = order
                break
            if zero is None:
                break
    jacobian = (
        derivatives[1]
        if len(derivatives) > 1
        else sp.simplify(sp.diff(local, delta).subs({delta: 0, u: 0}))
    )
    variable_derivative = sp.simplify(sp.diff(local, u).subs({delta: 0, u: 0}))
    jzero = zero_decision(jacobian)
    singular = True if jzero is True else False if jzero is False else None
    xzero = zero_decision(variable_derivative)
    turning = True if singular is True and xzero is False else False if singular is False else None

    discriminant = None
    try:
        poly = sp.Poly(local, delta)
        if 2 <= poly.degree() <= 6 and sp.count_ops(local) <= 80:
            discriminant = sp.factor(sp.discriminant(poly.as_expr(), delta))
    except (sp.PolynomialError, NotImplementedError, ValueError):
        pass

    scaling_exponents: tuple[sp.Rational, ...] = ()
    if singular is True:
        try:
            balances = dominant_balance_candidates(
                local, delta, u, context=ctx, taylor_degree=taylor_degree, stratify_parameters=False
            )
            if isinstance(balances, AsymptoticStratification):
                raise TypeError("unstratified balance search returned a stratification")
            scaling_exponents = tuple(sorted({b.exponent for b in balances if b.exponent > 0}))
        except (NotImplementedError, ValueError, TypeError):
            pass

    return ImplicitSingularityProfile(
        equation,
        dependent,
        variable,
        sp.sympify(point),
        sp.sympify(dependent_limit),
        multiplicity,
        jacobian,
        variable_derivative,
        singular,
        turning,
        discriminant,
        scaling_exponents,
    )


@dataclass(frozen=True)
class ImplicitAsymptoticBranch:
    equation: sp.Expr
    dependent: sp.Symbol
    variable: sp.Symbol
    balance: BalanceCandidate
    leading_coefficient: sp.Expr
    series: SeriesLike
    choice: BranchChoice
    balance_path: tuple[BalanceCandidate, ...] = ()
    complete: bool = False
    singularity: ImplicitSingularityProfile | None = None
    method: str = "dominant-balance"

    @property
    def residual(self) -> sp.Expr:
        return sp.expand(self.equation.subs(self.dependent, self.series.truncate()))

    @property
    def point(self) -> sp.Expr:
        return self.series.point

    @property
    def remainder(self) -> AsymptoticRemainder:
        if self.complete:
            return AsymptoticRemainder.exact_zero(
                self.variable, self.point, source="complete implicit branch"
            )
        return AsymptoticRemainder.unknown(
            self.variable,
            self.point,
            source="implicit prefix; solution error requires an implicit remainder theorem",
        )

    @property
    def is_transseries(self) -> bool:
        return isinstance(self.series, TransseriesExpansion)

    def truncate(self, terms: int | None = None) -> sp.Expr:
        return self.series.truncate(terms)

    def asymptotic_element(self):
        """View this implicit branch through the common asymptotic algebra."""
        from .algebra import asymptotic_element

        return asymptotic_element(self, self.variable, point=self.point)


def _extract_terms(expr: sp.Expr, x: sp.Expr) -> tuple[PuiseuxTerm, ...]:
    return _extract_puiseux_terms(
        expr,
        x,
        error_context="implicit",
        constant_coefficients=True,
    )


def _power_monomial_exponent(monomial: sp.Expr, variable: sp.Symbol) -> sp.Rational | None:
    monomial = formal_powsimp(sp.sympify(monomial))
    power = sp.sympify(monomial.as_powers_dict().get(variable, 0))
    if not power.is_Rational:
        return None
    remainder = formal_powsimp(monomial / variable**power)
    if variable in remainder.free_symbols:
        return None
    # A variable-independent multiplier belongs in the coefficient rather than
    # the monomial.  Accept it here so transformed coordinates remain robust.
    return sp.Rational(power)


def _build_series(
    expression: sp.Expr,
    variable: sp.Symbol,
    point: sp.Expr,
    choice: BranchChoice,
    coefficients: tuple[sp.Expr, ...],
    monomials: tuple[sp.Expr, ...],
    *,
    center: sp.Expr = 0,
) -> SeriesLike:
    exponents = tuple(_power_monomial_exponent(m, variable) for m in monomials)
    if monomials and all(exponent is not None for exponent in exponents):
        pterms = tuple(
            PuiseuxTerm(sp.Rational(exponent), sp.simplify(coefficient))
            for exponent, coefficient in zip(exponents, coefficients)
            if exponent is not None
        )
        # A nonzero dependent center is the exponent-zero Puiseux term.
        if center != 0:
            pterms = (PuiseuxTerm(sp.S.Zero, sp.sympify(center)),) + pterms
        pterms = tuple(sorted(pterms, key=lambda term: term.exponent))
        ram = reduce(_lcm, (int(t.exponent.q) for t in pterms), 1)
        return PuiseuxSeries(expression, variable, point, pterms, ram, choice)

    tterms = tuple(
        TransseriesTerm(sp.simplify(c), formal_powsimp(m)) for c, m in zip(coefficients, monomials)
    )
    return TransseriesExpansion(
        expression=expression,
        variable=variable,
        point=point,
        terms=tterms,
        center=sp.sympify(center),
    )


def _local_implicit_asymptotic(
    equation: sp.Expr,
    dependent: sp.Symbol,
    variable: sp.Symbol,
    *,
    terms: int,
    dependent_limit: sp.Expr,
    context: AsymptoticContext,
    taylor_degree: int,
    max_depth: int,
    corrections_must_vanish: bool = True,
    singularity: ImplicitSingularityProfile | None = None,
) -> tuple[ImplicitAsymptoticBranch, ...]:
    """Solve a translated local implicit equation using simple-root or Newton/Puiseux lifting."""
    if singularity is not None and singularity.multiplicity is not None:
        taylor_degree = max(taylor_degree, singularity.multiplicity + 2)
    lifted = lift_transseries_balance_branches(
        equation,
        dependent,
        variable,
        terms=terms,
        center=dependent_limit,
        context=context,
        taylor_degree=taylor_degree,
        max_depth=max_depth,
        corrections_must_vanish=corrections_must_vanish,
    )
    branches = []
    for index, item in enumerate(lifted):
        if not item.path or not item.leading_coefficients or not item.monomials:
            continue
        choice = BranchChoice(index=index, label=f"implicit-branch-{index}")
        series = _build_series(
            item.series,
            variable,
            sp.S.Zero,
            choice,
            item.leading_coefficients,
            item.monomials,
            center=dependent_limit,
        )
        branches.append(
            ImplicitAsymptoticBranch(
                equation=equation,
                dependent=dependent,
                variable=variable,
                balance=item.path[0],
                leading_coefficient=item.leading_coefficients[0],
                series=series,
                choice=choice,
                balance_path=item.path,
                complete=item.complete,
                singularity=singularity,
                method=(
                    "newton-puiseux-blowup"
                    if singularity is not None and singularity.requires_blowup
                    else "regular-dominant-balance"
                ),
            )
        )
    return tuple(branches)


def _map_branch(
    item: ImplicitAsymptoticBranch,
    *,
    equation: sp.Expr,
    dependent: sp.Symbol,
    variable: sp.Symbol,
    point: sp.Expr,
    old_variable: sp.Symbol,
    replacement: sp.Expr,
    dependent_limit: sp.Expr,
    assumptions: sp.Expr | bool = sp.S.true,
) -> ImplicitAsymptoticBranch:
    """Map a local implicit branch back to the original coordinates without changing its certificate."""
    mapped_expr = analytic_powsimp(
        sp.expand(item.series.truncate().xreplace({old_variable: replacement}))
    )

    if isinstance(item.series, PuiseuxSeries):
        try:
            pterms = _extract_terms(mapped_expr, variable)
            # At infinity conventional presentation uses decreasing powers.
            if point in (sp.oo, -sp.oo):
                pterms = tuple(sorted(pterms, key=lambda t: t.exponent, reverse=True))
            ram = reduce(_lcm, (int(t.exponent.q) for t in pterms), 1)
            series: SeriesLike = PuiseuxSeries(
                mapped_expr, variable, point, pterms, ram, item.choice
            )
        except NotImplementedError:
            mapped_terms = tuple(
                TransseriesTerm(
                    term.coefficient,
                    formal_powsimp(
                        (old_variable**term.exponent).xreplace({old_variable: replacement})
                    ),
                )
                for term in item.series.terms
                if term.exponent != 0 or dependent_limit == 0
            )
            series = TransseriesExpansion(
                mapped_expr, variable, point, mapped_terms, dependent_limit
            )
    else:
        mapped_terms = tuple(
            TransseriesTerm(
                term.coefficient,
                formal_powsimp(term.monomial.xreplace({old_variable: replacement})),
            )
            for term in item.series.terms
        )
        series = TransseriesExpansion(
            mapped_expr,
            variable,
            point,
            mapped_terms,
            item.series.center,
        )

    try:
        mapped_profile = implicit_singularity_profile(
            equation,
            dependent,
            variable,
            point=point,
            dependent_limit=dependent_limit,
            assumptions=assumptions,
        )
    except (NotImplementedError, TypeError, ValueError):
        mapped_profile = item.singularity
    return ImplicitAsymptoticBranch(
        equation,
        dependent,
        variable,
        item.balance,
        item.leading_coefficient,
        series,
        item.choice,
        item.balance_path,
        item.complete,
        mapped_profile,
        item.method,
    )


def _implicit_structural_coefficients(
    equation: sp.Expr,
    dependent: sp.Symbol,
    variable: sp.Symbol,
    point: sp.Expr,
    taylor_degree: int,
    dependent_limit: sp.Expr = 0,
) -> tuple[sp.Expr, ...]:
    """Leading coefficient data whose vanishing can change an implicit balance."""

    point = sp.sympify(point)
    local_variable = variable
    local_equation = sp.sympify(equation)
    if point in (sp.oo, -sp.oo):
        local_variable = sp.Dummy("u", positive=True)
        sign = 1 if point is sp.oo else -1
        transformed = sp.together(local_equation.xreplace({variable: sign / local_variable}))
        local_equation, _ = sp.fraction(transformed)
    elif point != 0:
        local_variable = sp.Dummy("u", positive=True)
        local_equation = sp.simplify(local_equation.xreplace({variable: point + local_variable}))
    correction = sp.Dummy("implicit_structural_delta")
    local_equation = sp.expand(
        local_equation.subs(dependent, sp.sympify(dependent_limit) + correction)
    )
    try:
        terms = transseries_balance_terms(
            local_equation,
            correction,
            local_variable,
            degree=taylor_degree,
            point=0,
            context=AsymptoticContext(local_variable, point=0),
        )
    except (NotImplementedError, ValueError):
        terms = ()
    structural = [term.valuation.leading_coefficient for term in terms]
    # Center derivatives detect parameter values at which a simple implicit
    # root becomes multiple, including roots centered away from zero.
    derivative = local_equation
    for order in range(1, min(taylor_degree, 6) + 1):
        derivative = sp.diff(derivative, correction)
        value = sp.simplify(derivative.subs({correction: 0, local_variable: 0}))
        if value.free_symbols - {local_variable}:
            structural.append(value)
    try:
        poly = sp.Poly(local_equation, correction)
        if 2 <= poly.degree() <= 6 and sp.count_ops(local_equation) <= 80:
            disc = sp.factor(sp.discriminant(poly.as_expr(), correction).subs(local_variable, 0))
            if disc != 0:
                structural.append(disc)
    except (sp.PolynomialError, NotImplementedError, ValueError):
        pass
    return tuple(structural)


def implicit_asymptotic(
    equation: sp.Expr,
    dependent: sp.Symbol,
    variable: sp.Symbol,
    *,
    point: sp.Expr = 0,
    dependent_limit: sp.Expr = 0,
    terms: int = 6,
    context: AsymptoticContext | None = None,
    taylor_degree: int = 8,
    max_depth: int = 32,
    corrections_must_vanish: bool = True,
    assumptions: sp.Expr | bool = sp.S.true,
    stratify_parameters: bool = True,
    max_parameter_splits: int = 6,
) -> (
    tuple[ImplicitAsymptoticBranch, ...]
    | AsymptoticStratification[tuple[ImplicitAsymptoticBranch, ...]]
):
    """Construct Puiseux or generalized-transseries implicit branches.

    Dominant balance uses the same exp-log monomial hierarchy as multiseries.
    Pure rational-power branches use ``PuiseuxSeries``; logarithmic or
    exponential corrections use ``TransseriesExpansion``.
    """

    equation = sp.sympify(equation)
    dependent_limit = sp.sympify(dependent_limit)
    if terms < 1:
        raise ValueError("terms must be positive")

    if stratify_parameters:
        parameters = parameter_symbols(equation, (variable, dependent))
        if parameters:
            structural = _implicit_structural_coefficients(
                equation, dependent, variable, point, taylor_degree, dependent_limit
            )
            try:
                generic_probe = implicit_asymptotic(
                    equation,
                    dependent,
                    variable,
                    point=point,
                    dependent_limit=dependent_limit,
                    terms=terms,
                    context=context,
                    taylor_degree=taylor_degree,
                    max_depth=max_depth,
                    corrections_must_vanish=corrections_must_vanish,
                    assumptions=assumptions,
                    stratify_parameters=False,
                    max_parameter_splits=max_parameter_splits,
                )
                if isinstance(generic_probe, AsymptoticStratification):
                    raise TypeError("unstratified implicit probe returned a stratification")
                branch_coefficients = [branch.leading_coefficient for branch in generic_probe]
                for branch in generic_probe:
                    for balance in branch.balance_path:
                        branch_coefficients.extend(balance.coefficients)
                branch_coefficients.extend(
                    sp.simplify(left.leading_coefficient - right.leading_coefficient)
                    for i, left in enumerate(generic_probe)
                    for right in generic_probe[i + 1 :]
                )
                structural = structural + tuple(branch_coefficients)
            except (NotImplementedError, ValueError, TypeError):
                pass

            def evaluate(condition: sp.Expr) -> tuple[ImplicitAsymptoticBranch, ...]:
                specialized = specialize_expression(equation, condition, parameters=parameters)
                result = implicit_asymptotic(
                    specialized,
                    dependent,
                    variable,
                    point=point,
                    dependent_limit=dependent_limit,
                    terms=terms,
                    taylor_degree=taylor_degree,
                    max_depth=max_depth,
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
                provenance_source="asymptotic.implicit",
            )
            if stratified is not None:
                return stratified

    if point in (sp.oo, -sp.oo):
        u = sp.Dummy("u", positive=True)
        sign = 1 if point is sp.oo else -1
        transformed = sp.together(equation.xreplace({variable: sign / u}))
        num, _ = sp.fraction(transformed)
        local = implicit_asymptotic(
            num,
            dependent,
            u,
            point=0,
            dependent_limit=dependent_limit,
            terms=terms,
            context=AsymptoticContext(u, point=0),
            taylor_degree=taylor_degree,
            max_depth=max_depth,
            corrections_must_vanish=False,
            assumptions=assumptions,
            stratify_parameters=False,
            max_parameter_splits=max_parameter_splits,
        )
        return tuple(
            _map_branch(
                item,
                equation=equation,
                dependent=dependent,
                variable=variable,
                point=point,
                old_variable=u,
                replacement=sign / variable,
                dependent_limit=dependent_limit,
                assumptions=assumptions,
            )
            for item in local
        )

    if point != 0:
        u = sp.Dummy("u", positive=True)
        transformed = sp.simplify(equation.xreplace({variable: point + u}))
        local = implicit_asymptotic(
            transformed,
            dependent,
            u,
            point=0,
            dependent_limit=dependent_limit,
            terms=terms,
            context=AsymptoticContext(u, point=0),
            taylor_degree=taylor_degree,
            max_depth=max_depth,
            corrections_must_vanish=corrections_must_vanish,
            assumptions=assumptions,
            stratify_parameters=False,
            max_parameter_splits=max_parameter_splits,
        )
        return tuple(
            _map_branch(
                item,
                equation=equation,
                dependent=dependent,
                variable=variable,
                point=point,
                old_variable=u,
                replacement=variable - point,
                dependent_limit=dependent_limit,
                assumptions=assumptions,
            )
            for item in local
        )

    ctx = context_for(variable, 0, context)
    profile = implicit_singularity_profile(
        equation,
        dependent,
        variable,
        point=0,
        dependent_limit=dependent_limit,
        taylor_degree=taylor_degree,
        assumptions=assumptions,
    )
    return _local_implicit_asymptotic(
        equation,
        dependent,
        variable,
        terms=terms,
        dependent_limit=dependent_limit,
        context=ctx,
        taylor_degree=taylor_degree,
        max_depth=max_depth,
        corrections_must_vanish=corrections_must_vanish,
        singularity=profile,
    )
