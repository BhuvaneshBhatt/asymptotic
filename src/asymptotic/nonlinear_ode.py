"""Dominant balances for nonlinear differential-polynomial equations."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from ._linear_ode_operator import linear_operator_coefficients
from ._power_simplify import analytic_powsimp, formal_powsimp, mixed_powsimp
from ._symbolic_policy import bounded_limit, bounded_solve_one, bounded_solve_system
from ._symbolic_primitives import certification_primitive
from .dominant import rational_valuation
from .parameter_auto import (
    automatic_parameter_stratification,
    parameter_symbols,
    specialize_expression,
)
from .remainder_theorems import (
    certify_frechet_inverse_operator_remainder,
    certify_nonlinear_lifting_remainder,
)
from .stratification import AsymptoticStratification
from .transseries import TransseriesExpansion, transseries_from_expression


@dataclass(frozen=True)
class DifferentialBalanceTerm:
    coefficient: sp.Expr
    coefficient_valuation: sp.Rational
    dependent_degree: int
    derivative_weight: int
    jet_powers: tuple[int, ...]

    def valuation_at(self, exponent: sp.Expr) -> sp.Expr:
        return sp.simplify(
            self.coefficient_valuation + self.dependent_degree * exponent - self.derivative_weight
        )


@dataclass(frozen=True)
class NonlinearDifferentialBalance:
    exponent: sp.Expr
    valuation: sp.Expr
    characteristic_poly: sp.Expr
    coefficient_symbol: sp.Symbol
    terms: tuple[DifferentialBalanceTerm, ...]
    roots: tuple[sp.Expr, ...]


@dataclass(frozen=True)
class DifferentialTransseriesStep:
    """One certified Newton-style correction in a nonlinear ODE branch.

    ``local_exponent`` is the power in the local coordinate ``h`` (or in the
    reciprocal coordinate at infinity).  ``term`` is expressed in the original
    independent variable.  Residual valuations are local-coordinate valuations
    before and after accepting the correction when they can be certified.
    """

    local_exponent: sp.Rational | None
    coefficient: sp.Expr
    local_term: sp.Expr
    term: sp.Expr
    balance: NonlinearDifferentialBalance | None
    residual_order_before: sp.Rational | None
    residual_order_after: sp.Rational | None
    correction_kind: str = "power"
    logarithmic_power: int = 0
    exponential_phase: sp.Expr | None = None
    free_parameter: sp.Symbol | None = None


@dataclass(frozen=True)
class NonlinearDifferentialTransseriesBranch:
    """A recursively lifted formal branch of a nonlinear differential equation."""

    series: sp.Expr
    local_series: sp.Expr
    point: sp.Expr
    local_coordinate: sp.Expr
    local_parameter: sp.Symbol
    transseries: TransseriesExpansion
    steps: tuple[DifferentialTransseriesStep, ...]
    residual: sp.Expr
    residual_valuation: sp.Rational | None
    complete: bool
    limitation: str | None = None

    def asymptotic_element(self):
        """View this ODE-generated branch through the common field protocol."""
        from .algebra import asymptotic_element

        return asymptotic_element(self)

    @property
    def terms(self) -> tuple[sp.Expr, ...]:
        return tuple(step.term for step in self.steps)

    @property
    def coefficients(self) -> tuple[sp.Expr, ...]:
        return tuple(step.coefficient for step in self.steps)


def _jet_symbols(function: sp.FunctionClass, x: sp.Symbol, order: int) -> tuple[sp.Symbol, ...]:
    return tuple(sp.Dummy(f"Y{k}") for k in range(order + 1))


def _falling(alpha: sp.Expr, k: int) -> sp.Expr:
    return sp.prod(alpha - j for j in range(k))


def _local_balance_terms(
    equation: sp.Expr,
    function: sp.FunctionClass,
    variable: sp.Symbol,
) -> tuple[DifferentialBalanceTerm, ...]:
    """Extract valuation-affine terms from a differential polynomial at x=0."""

    equation = sp.expand(sp.sympify(equation))
    y = function(variable)
    derivatives = [d for d in equation.atoms(sp.Derivative) if d.expr == y]
    order = max((d.derivative_count for d in derivatives), default=0)
    jets = _jet_symbols(function, variable, order)
    repl: dict[sp.Expr, sp.Symbol] = {y: jets[0]}
    for k in range(1, order + 1):
        repl[sp.diff(y, variable, k)] = jets[k]
    algebraic = sp.expand(equation.xreplace(repl))
    try:
        poly = sp.Poly(algebraic, *jets)
    except sp.PolynomialError as exc:
        raise NotImplementedError("equation must be polynomial in y and its derivatives") from exc

    result = []
    for powers, coeff in poly.terms():
        val = rational_valuation(coeff, variable)
        if val is None:
            raise NotImplementedError(f"coefficient {coeff} has no finite rational valuation at 0")
        valuation, leading = val
        degree = sum(powers)
        weight = sum(k * power for k, power in enumerate(powers))
        result.append(
            DifferentialBalanceTerm(
                sp.simplify(leading),
                sp.Rational(valuation),
                degree,
                weight,
                tuple(int(p) for p in powers),
            )
        )
    return tuple(result)


def _local_dominant_balances(
    equation: sp.Expr,
    function: sp.FunctionClass,
    variable: sp.Symbol,
    *,
    minimum_exponent: sp.Rational | None = None,
) -> tuple[NonlinearDifferentialBalance, ...]:
    """Find power-law balances ``y ~ c*x**alpha`` for a nonlinear ODE.

    Candidate exponents are intersections of the affine valuations of all
    differential-polynomial terms.  At each lower-envelope intersection, the
    corresponding exact characteristic polynomial in ``c`` is constructed.
    """

    terms = _local_balance_terms(equation, function, variable)
    candidates = set()
    for i, left in enumerate(terms):
        for right in terms[i + 1 :]:
            denominator = left.dependent_degree - right.dependent_degree
            if denominator == 0:
                continue
            alpha = sp.simplify(
                (
                    right.coefficient_valuation
                    - right.derivative_weight
                    - left.coefficient_valuation
                    + left.derivative_weight
                )
                / denominator
            )
            if alpha.is_Rational:
                alpha = sp.Rational(alpha)
                if minimum_exponent is None or alpha > minimum_exponent:
                    candidates.add(alpha)

    c = sp.Symbol("c", nonzero=True)
    out = []
    for alpha in sorted(candidates, key=sp.default_sort_key):
        values = [term.valuation_at(alpha) for term in terms]
        # At a rational candidate exponent every valuation is rational, so use
        # mathematical order rather than SymPy's structural default_sort_key.
        minimum = min(values)
        active = tuple(term for term, value in zip(terms, values) if value == minimum)
        if len(active) < 2:
            continue
        characteristic = sp.S.Zero
        for term in active:
            factor = sp.S.One
            for k, power in enumerate(term.jet_powers):
                if power:
                    factor *= (c * _falling(alpha, k)) ** power
            characteristic += term.coefficient * factor
        characteristic = sp.factor(characteristic)
        roots = tuple(root for root in (bounded_solve_one(characteristic, c) or ()) if root != 0)
        if roots:
            out.append(
                NonlinearDifferentialBalance(
                    alpha,
                    sp.simplify(minimum),
                    characteristic,
                    c,
                    active,
                    roots,
                )
            )
    return tuple(out)


def _change_independent_variable(
    equation: sp.Expr,
    function: sp.FunctionClass,
    variable: sp.Symbol,
    parameter: sp.Symbol,
    variable_map: sp.Expr,
) -> tuple[sp.Expr, sp.FunctionClass]:
    """Transform a scalar differential expression under ``x=variable_map(t)``."""

    yx = function(variable)
    v = sp.Function(f"{function.__name__}_local")
    vt = v(parameter)
    derivatives = [d for d in equation.atoms(sp.Derivative) if d.expr == yx]
    order = max((d.derivative_count for d in derivatives), default=0)
    dxdt = sp.diff(variable_map, parameter)
    if dxdt == 0:
        raise ValueError("independent-variable map must have nonzero derivative")
    transformed_derivatives: dict[int, sp.Expr] = {0: vt}
    current = vt
    for k in range(1, order + 1):
        current = sp.simplify(sp.diff(current, parameter) / dxdt)
        transformed_derivatives[k] = current
    # xreplace must replace derivatives before the bare x inside them.
    transformed = equation
    for k in range(order, 0, -1):
        transformed = transformed.xreplace({sp.diff(yx, variable, k): transformed_derivatives[k]})
    transformed = transformed.xreplace({yx: vt, variable: variable_map})
    return sp.together(sp.expand(transformed)), v


def nonlinear_differential_balance_terms(
    equation: sp.Expr,
    function: sp.FunctionClass,
    variable: sp.Symbol,
    *,
    point: sp.Expr = 0,
) -> tuple[DifferentialBalanceTerm, ...]:
    """Extract nonlinear differential balance terms at a finite point or infinity."""

    point = sp.sympify(point)
    if point == 0:
        return _local_balance_terms(equation, function, variable)
    u = sp.Dummy("u", positive=True)
    if point in (sp.oo, -sp.oo):
        sign = 1 if point is sp.oo else -1
        transformed, local_function = _change_independent_variable(
            equation, function, variable, u, sign / u
        )
    else:
        transformed, local_function = _change_independent_variable(
            equation, function, variable, u, point + u
        )
    numerator, _ = sp.fraction(sp.together(transformed))
    return _local_balance_terms(sp.expand(numerator), local_function, u)


def nonlinear_differential_dominant_balances(
    equation: sp.Expr,
    function: sp.FunctionClass,
    variable: sp.Symbol,
    *,
    point: sp.Expr = 0,
    assumptions: sp.Expr | bool = sp.S.true,
    stratify_parameters: bool = True,
    max_parameter_splits: int = 6,
) -> (
    tuple[NonlinearDifferentialBalance, ...]
    | AsymptoticStratification[tuple[NonlinearDifferentialBalance, ...]]
):
    """Find power-law balances for a nonlinear differential polynomial.

    At infinity the equation is transformed exactly to the reciprocal local
    coordinate before balancing, so derivative chain-rule factors are retained.
    Returned exponents are expressed in the original variable convention.
    """

    point = sp.sympify(point)
    if stratify_parameters:
        local_equation, local_function, h, _ = _local_problem(equation, function, variable, point)
        parameters = parameter_symbols(local_equation, (h,))
        if parameters:
            try:
                balance_terms = _local_balance_terms(local_equation, local_function, h)
                generic_local = _local_dominant_balances(local_equation, local_function, h)
                structural = tuple(term.coefficient for term in balance_terms) + tuple(
                    root for balance in generic_local for root in balance.roots
                )
            except NotImplementedError:
                structural = ()

            def evaluate(condition: sp.Expr) -> tuple[NonlinearDifferentialBalance, ...]:
                specialized = specialize_expression(equation, condition, parameters=parameters)
                result = nonlinear_differential_dominant_balances(
                    specialized,
                    function,
                    variable,
                    point=point,
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
                provenance_source="asymptotic.nonlinear_differential_balance",
            )
            if stratified is not None:
                return stratified

    if point == 0:
        return _local_dominant_balances(equation, function, variable)
    u = sp.Dummy("u", positive=True)
    if point in (sp.oo, -sp.oo):
        sign = 1 if point is sp.oo else -1
        transformed, local_function = _change_independent_variable(
            equation, function, variable, u, sign / u
        )
        numerator, _ = sp.fraction(sp.together(transformed))
        local = _local_dominant_balances(sp.expand(numerator), local_function, u)
        return tuple(
            NonlinearDifferentialBalance(
                -item.exponent,
                item.valuation,
                item.characteristic_poly,
                item.coefficient_symbol,
                item.terms,
                item.roots,
            )
            for item in local
        )
    transformed, local_function = _change_independent_variable(
        equation, function, variable, u, point + u
    )
    numerator, _ = sp.fraction(sp.together(transformed))
    return _local_dominant_balances(sp.expand(numerator), local_function, u)


def _substitute_dependent_expression(
    equation: sp.Expr,
    function: sp.FunctionClass,
    variable: sp.Symbol,
    replacement: sp.Expr,
) -> sp.Expr:
    """Substitute an explicit dependent expression, including all derivatives."""

    equation = sp.sympify(equation)
    y = function(variable)
    derivatives = [d for d in equation.atoms(sp.Derivative) if d.expr == y]
    order = max((d.derivative_count for d in derivatives), default=0)
    transformed = equation
    for k in range(order, 0, -1):
        transformed = transformed.xreplace(
            {sp.diff(y, variable, k): sp.diff(replacement, variable, k)}
        )
    transformed = transformed.xreplace({y: replacement})
    return analytic_powsimp(sp.together(sp.expand(transformed)))


def _residual_valuation(expr: sp.Expr, variable: sp.Symbol) -> sp.Rational | None:
    expr = sp.cancel(sp.together(sp.sympify(expr)))
    if expr == 0 or expr.is_zero is True:
        return None
    value = rational_valuation(expr, variable)
    return None if value is None else sp.Rational(value[0])


def _frechet_linearization(
    equation: sp.Expr,
    function: sp.FunctionClass,
    variable: sp.Symbol,
    prefix: sp.Expr,
    correction_function: sp.FunctionClass,
) -> sp.Expr:
    """Return the Fréchet derivative at ``prefix`` applied to a correction.

    The nonlinear equation is differentiated with respect to an auxiliary
    scalar parameter after substituting ``y = prefix + eps*delta``.  This
    retains all derivative couplings exactly and produces the homogeneous
    scalar linearized ODE governing beyond-all-orders perturbations.
    """

    eps = sp.Dummy("eps")
    delta = correction_function(variable)
    shifted = _substitute_dependent_expression(equation, function, variable, prefix + eps * delta)
    return sp.powsimp(sp.expand(sp.diff(shifted, eps).subs(eps, 0)))


def _logarithmic_valuation(
    expr: sp.Expr,
    variable: sp.Symbol,
) -> tuple[sp.Rational, sp.Expr] | None:
    """Return the lowest power of ``variable`` allowing polynomial log factors.

    The second result is the complete coefficient of that power as an
    expression in ``log(variable)``.  This extends :func:`rational_valuation`
    just enough for resonant power-log corrections without pretending that
    logarithms have an ordinary rational valuation.
    """

    expr = sp.expand(sp.sympify(expr))
    if expr == 0 or expr.is_zero is True:
        return None
    logh = sp.log(variable)
    pieces = []
    for term in sp.Add.make_args(expr):
        power = sp.S.Zero
        coefficient = sp.S.One
        for factor in sp.Mul.make_args(term):
            if factor == variable:
                power += 1
                continue
            base, exponent = factor.as_base_exp()
            if base == variable and exponent.is_Rational:
                power += exponent
                continue
            coefficient *= factor
        if not power.is_Rational:
            return None
        # Any remaining non-log dependence on h means this term is not in the
        # finite power-log sector handled by this helper.
        remainder_symbols = coefficient.free_symbols - {variable}
        _ = remainder_symbols
        replaced = coefficient.xreplace({logh: sp.Dummy("L")})
        if variable in replaced.free_symbols:
            return None
        pieces.append((sp.Rational(power), coefficient))
    minimum = min(power for power, _ in pieces)
    lead = sp.expand(sum(coeff for power, coeff in pieces if power == minimum))
    if lead == 0:
        # Cancellation may reveal a later power; retry after simplification.
        simplified = sp.expand(sp.cancel(sp.together(expr)))
        if simplified != expr:
            return _logarithmic_valuation(simplified, variable)
        return None
    return minimum, lead


def _linear_operator_coefficients(
    linearized: sp.Expr,
    correction_function: sp.FunctionClass,
    variable: sp.Symbol,
) -> tuple[sp.Expr, ...]:
    """Extract coefficients of a homogeneous scalar linearized ODE."""

    extracted = linear_operator_coefficients(linearized, correction_function(variable), variable)
    if extracted is None:
        raise NotImplementedError(
            "linearized equation is not a homogeneous scalar linear correction operator"
        )
    coefficients, _ = extracted
    return coefficients


def _candidate_log_corrections(
    equation: sp.Expr,
    function: sp.FunctionClass,
    variable: sp.Symbol,
    prefix: sp.Expr,
    *,
    minimum_exponent: sp.Rational | None,
    max_log_power: int = 4,
) -> tuple[tuple[sp.Rational, sp.Expr, int, sp.Expr, sp.Expr], ...]:
    """Find resonant corrections ``c*h**alpha*log(h)**k``.

    Candidates are generated by matching the rational power of the current
    residual against the affine power produced by each coefficient of the
    Fréchet linearization.  We then substitute the full power-log ansatz into
    the original nonlinear equation and solve all coefficients of the leading
    log-polynomial simultaneously.  This catches the standard resonance where
    the pure power characteristic coefficient vanishes but a logarithmic
    derivative supplies the missing term.
    """

    residual = _substitute_dependent_expression(equation, function, variable, prefix)
    residual_data = _logarithmic_valuation(residual, variable)
    if residual_data is None:
        return ()
    residual_power, _ = residual_data
    correction_function = sp.Function("delta_log")
    linearized = _frechet_linearization(equation, function, variable, prefix, correction_function)
    try:
        coefficients = _linear_operator_coefficients(linearized, correction_function, variable)
    except NotImplementedError:
        return ()

    candidate_exponents = set()
    for order, coefficient in enumerate(coefficients):
        if coefficient == 0:
            continue
        valuation = rational_valuation(coefficient, variable)
        if valuation is None:
            continue
        alpha = sp.simplify(residual_power - sp.Rational(valuation[0]) + order)
        if alpha.is_Rational:
            alpha = sp.Rational(alpha)
            if minimum_exponent is None or alpha > minimum_exponent:
                candidate_exponents.add(alpha)

    c = sp.Dummy("c", nonzero=True)
    logh = sp.log(variable)
    out = []
    for alpha in sorted(candidate_exponents, key=sp.default_sort_key):
        for log_power in range(1, max_log_power + 1):
            ansatz = mixed_powsimp(c, variable**alpha * logh**log_power)
            trial = _substitute_dependent_expression(
                equation, function, variable, sp.expand(prefix + ansatz)
            )
            lead = _logarithmic_valuation(trial, variable)
            if lead is None:
                continue
            power_after, coefficient_after = lead
            # We require cancellation of the old leading power; merely changing
            # its logarithmic coefficient is not a certified improvement.
            if power_after > residual_power:
                roots = (c,)
            elif power_after < residual_power:
                continue
            else:
                L = sp.Dummy("L")
                lead_poly_expr = sp.expand(coefficient_after.xreplace({logh: L}))
                try:
                    poly = sp.Poly(lead_poly_expr, L)
                except sp.PolynomialError:
                    continue
                equations = [sp.Eq(coeff, 0) for coeff in poly.all_coeffs()]
                solved_system = bounded_solve_system(equations, (c,)) or ()
                roots = tuple(solution[c] for solution in solved_system if c in solution)
            for root in roots:
                root = sp.simplify(root)
                if root == 0 or root.has(c):
                    continue
                local_term = sp.powsimp(root * variable**alpha * logh**log_power)
                residual_after = _substitute_dependent_expression(
                    equation, function, variable, sp.expand(prefix + local_term)
                )
                after = _logarithmic_valuation(residual_after, variable)
                if (
                    residual_after == 0
                    or residual_after.is_zero is True
                    or (after is not None and after[0] > residual_power)
                ):
                    out.append((alpha, root, log_power, local_term, residual_after))
        if out:
            # Prefer the smallest logarithmic degree at the first viable power.
            break
    return tuple(out)


def _first_order_exponential_modes(
    equation: sp.Expr,
    function: sp.FunctionClass,
    variable: sp.Symbol,
    prefix: sp.Expr,
) -> tuple[tuple[sp.Expr, sp.Expr], ...]:
    """Return exact first-order homogeneous exponential modes of the linearization.

    For ``a1*d' + a0*d = 0`` we construct
    ``d = C*exp(integral(-a0/a1, h))``.  Only modes whose phase is nonconstant
    are returned here; regular power modes remain the job of Newton lifting.
    """

    correction_function = sp.Function("delta_exp")
    linearized = _frechet_linearization(equation, function, variable, prefix, correction_function)
    try:
        coefficients = _linear_operator_coefficients(linearized, correction_function, variable)
    except NotImplementedError:
        return ()
    if len(coefficients) != 2 or coefficients[1] == 0:
        return ()
    phase_derivative = sp.cancel(sp.together(-coefficients[0] / coefficients[1]))
    phase = certification_primitive(phase_derivative, variable)
    if phase is None or phase == 0:
        return ()
    phase = sp.simplify(phase)
    if variable not in phase.free_symbols:
        return ()
    # ``phase = a*log(h)`` is only an ordinary power mode h**a.  Reserve the
    # exponential-correction path for genuinely beyond-power scales.
    phase_without_logs = sp.expand(phase).subs(sp.log(variable), 0)
    if variable not in phase_without_logs.free_symbols:
        return ()
    # A residual analytic phase such as h**2 does not create a
    # beyond-all-orders scale: exp(a*log(h) + h**2) is still a power times an
    # analytic unit.  In the rational/Laurent sector we can decide divergence
    # from the valuation directly, avoiding a general symbolic limit.
    phase_data = rational_valuation(phase_without_logs, variable)
    if phase_data is None or sp.Rational(phase_data[0]) >= 0:
        return ()
    leading = sp.sympify(phase_data[1])
    if leading.is_positive is not True and leading.is_negative is not True:
        return ()
    return ((phase, sp.exp(phase)),)


def _odeanalysis_exponential_modes(
    equation: sp.Expr,
    function: sp.FunctionClass,
    variable: sp.Symbol,
    prefix: sp.Expr,
    *,
    terms: int,
) -> tuple[tuple[sp.Expr, sp.Expr], ...]:
    """Obtain formal exponential modes from the optional ``odeanalysis`` bridge.

    This is intentionally imported lazily so the core package remains usable
    without its ``ode`` extra.  Only the stable ``FormalODEData`` interchange
    schema is consumed.
    """

    correction_function = sp.Function("delta_formal")
    linearized = _frechet_linearization(equation, function, variable, prefix, correction_function)
    if linearized == 0:
        return ()
    try:
        from odeanalysis.interchange import formal_ode_data
    except ImportError:
        return ()
    try:
        data = formal_ode_data(
            linearized,
            correction_function,
            variable,
            point=0,
            terms=max(2, terms),
            include_stokes=False,
        )
    except (NotImplementedError, ValueError, ZeroDivisionError):
        return ()
    out = []
    for block in data.blocks:
        phase = sp.simplify(block.exponential_polynomial)
        if phase == 0:
            continue
        for expression in block.expressions:
            if expression == 0:
                continue
            out.append((phase, analytic_powsimp(sp.sympify(expression))))
    return tuple(out)


def _lift_exponential_parameter_series(
    equation: sp.Expr,
    function: sp.FunctionClass,
    variable: sp.Symbol,
    prefix: sp.Expr,
    mode: sp.Expr,
    parameter: sp.Symbol,
    *,
    max_terms: int,
) -> tuple[tuple[sp.Expr, sp.Expr, int], ...]:
    """Lift a free exponential mode through nonlinear powers of its parameter.

    Starting with ``C*m``, seek descendants ``a_n*C**n*m**n`` by cancelling
    the first nonzero coefficient in the exact residual polynomial in ``C``.
    This is a conservative but useful nonlinear transseries recursion: it is
    exact whenever the descendant amplitude lies in the multiplicative tower
    generated by the leading formal mode.  Failure simply stops the tower.
    """

    if max_terms <= 1:
        return ()
    current = analytic_powsimp(prefix + mixed_powsimp(parameter, mode))
    descendants = []
    for n in range(2, max_terms + 1):
        residual = sp.expand(
            _substitute_dependent_expression(equation, function, variable, current)
        )
        if residual == 0 or residual.is_zero is True:
            break
        try:
            poly = sp.Poly(residual, parameter)
        except sp.PolynomialError:
            break
        degrees = sorted(degree for (degree,), coeff in poly.terms() if coeff != 0)
        if not degrees:
            break
        target_degree = degrees[0]
        if target_degree != n:
            break
        a = sp.Dummy(f"a{n}")
        candidate = mixed_powsimp(a * parameter**n, formal_powsimp(mode**n))
        trial = sp.expand(
            _substitute_dependent_expression(equation, function, variable, current + candidate)
        )
        try:
            trial_poly = sp.Poly(trial, parameter)
        except sp.PolynomialError:
            break
        coefficient = sp.factor(trial_poly.coeff_monomial(parameter**n))
        expanded_coefficient = sp.expand(coefficient)
        linear_coefficient = expanded_coefficient.coeff(a)
        remainder = sp.expand(expanded_coefficient - linear_coefficient * a)
        if (
            linear_coefficient != 0
            and a not in linear_coefficient.free_symbols
            and a not in remainder.free_symbols
        ):
            roots = (sp.cancel(-remainder / linear_coefficient),)
        elif sp.count_ops(expanded_coefficient) <= 24:
            roots = bounded_solve_one(expanded_coefficient, a) or ()
        else:
            roots = ()
        accepted = None
        for root in roots:
            root = sp.simplify(root)
            if root.has(variable) or root.has(parameter) or root == 0:
                continue
            accepted = root
            break
        if accepted is None:
            break
        term = mixed_powsimp(accepted * parameter**n, formal_powsimp(mode**n))
        current = analytic_powsimp(sp.expand(current + term))
        descendants.append((accepted, term, n))
    return tuple(descendants)


def _candidate_exponential_corrections(
    equation: sp.Expr,
    function: sp.FunctionClass,
    variable: sp.Symbol,
    prefix: sp.Expr,
    *,
    terms: int,
) -> tuple[tuple[sp.Expr, sp.Expr, sp.Symbol, sp.Expr], ...]:
    """Find exponentially small free perturbations of a constructed branch."""

    modes = _odeanalysis_exponential_modes(equation, function, variable, prefix, terms=terms)
    if not modes:
        modes = _first_order_exponential_modes(equation, function, variable, prefix)
    out = []
    seen = set()
    for index, (phase, mode) in enumerate(modes):
        # A formal exponential correction is relevant only if it tends to zero
        # along the positive local coordinate.  SymPy limits are used as a
        # conservative certificate; unresolved modes are not guessed.
        limit = bounded_limit(sp.Abs(mode), variable, 0, direction="+")
        if limit != 0:
            continue
        key = sp.srepr(sp.simplify(mode))
        if key in seen:
            continue
        seen.add(key)
        parameter = sp.Symbol(f"C{index + 1}")
        local_term = mixed_powsimp(parameter, mode)
        out.append((sp.simplify(phase), sp.simplify(mode), parameter, local_term))
    return tuple(out)


def _local_problem(
    equation: sp.Expr,
    function: sp.FunctionClass,
    variable: sp.Symbol,
    point: sp.Expr,
) -> tuple[sp.Expr, sp.FunctionClass, sp.Symbol, sp.Expr]:
    """Return ``(equation, function, h, h(x))`` for a local zero-coordinate problem."""

    point = sp.sympify(point)
    if point == 0:
        return sp.sympify(equation), function, variable, variable
    h = sp.Dummy("h", positive=True)
    if point in (sp.oo, -sp.oo):
        sign = sp.S.One if point is sp.oo else -sp.S.One
        transformed, local_function = _change_independent_variable(
            equation, function, variable, h, sign / h
        )
        local_to_original = sign / variable
    else:
        transformed, local_function = _change_independent_variable(
            equation, function, variable, h, point + h
        )
        local_to_original = variable - point
    numerator, _ = sp.fraction(sp.together(transformed))
    return sp.expand(numerator), local_function, h, sp.sympify(local_to_original)


def nonlinear_differential_transseries(
    equation: sp.Expr,
    function: sp.FunctionClass,
    variable: sp.Symbol,
    *,
    point: sp.Expr = 0,
    terms: int = 6,
    max_depth: int = 32,
    assumptions: sp.Expr | bool = sp.S.true,
    stratify_parameters: bool = True,
    max_parameter_splits: int = 6,
) -> (
    tuple[NonlinearDifferentialTransseriesBranch, ...]
    | AsymptoticStratification[tuple[NonlinearDifferentialTransseriesBranch, ...]]
):
    """Recursively lift nonlinear differential dominant balances.

    The equation is first put in a local coordinate ``h -> 0`` (using the
    exact reciprocal transformation at infinity).  A leading balance
    ``v ~ c*h**alpha`` is selected, translated out, and the full differential
    equation is rebuilt in a new correction function.  The process is then
    repeated with strictly increasing local exponents.

    This is the differential analogue of recursive Newton--Puiseux lifting:
    repeated or singular leading balances are not linearized prematurely.
    Every accepted correction is checked by substitution into the *original*
    local differential equation, and residual valuations are recorded when
    they are rationally decidable.

    After the rational-power Newton sector is exhausted, the lifter probes the
    Fréchet linearization automatically.  Resonant forcing is tested with exact
    power-log ansätze ``c*h**alpha*log(h)**k``; exponentially small homogeneous
    modes are obtained from the stable ``odeanalysis`` formal-data interface
    when available, with an exact first-order fallback.  Exponential modes carry
    an explicit free transseries parameter rather than silently fixing a Stokes
    constant.
    """

    if terms < 1:
        raise ValueError("terms must be positive")
    if max_depth < 1:
        raise ValueError("max_depth must be positive")

    point = sp.sympify(point)
    local_equation, local_function, h, h_of_x = _local_problem(equation, function, variable, point)

    if stratify_parameters:
        parameters = parameter_symbols(local_equation, (h,))
        if parameters:
            try:
                balance_terms = _local_balance_terms(local_equation, local_function, h)
                generic_local = _local_dominant_balances(local_equation, local_function, h)
                structural = tuple(term.coefficient for term in balance_terms) + tuple(
                    root for balance in generic_local for root in balance.roots
                )
            except NotImplementedError:
                structural = ()

            def evaluate(condition: sp.Expr) -> tuple[NonlinearDifferentialTransseriesBranch, ...]:
                specialized = specialize_expression(equation, condition, parameters=parameters)
                result = nonlinear_differential_transseries(
                    specialized,
                    function,
                    variable,
                    point=point,
                    terms=terms,
                    max_depth=max_depth,
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
                provenance_source="asymptotic.nonlinear_differential_lifting",
            )
            if stratified is not None:
                return stratified

    initial = _local_dominant_balances(local_equation, local_function, h)
    output = []

    def original_expr(local_expr: sp.Expr) -> sp.Expr:
        if h == variable:
            return analytic_powsimp(sp.expand(local_expr))
        return analytic_powsimp(sp.expand(local_expr.subs(h, h_of_x)))

    def emit(
        prefix: sp.Expr,
        steps: tuple[DifferentialTransseriesStep, ...],
        *,
        complete: bool,
        limitation: str | None,
    ) -> None:
        """Record a lifted branch after replaying its residual against the local equation."""
        residual = _substitute_dependent_expression(local_equation, local_function, h, prefix)
        prefix_expr = analytic_powsimp(sp.expand(prefix))
        ts = transseries_from_expression(prefix_expr, h, point=0, complete=complete)
        # A simple-root remainder theorem is valid automatically for the
        # algebraic (order-zero) subset.  Differential inverse-operator bounds
        # are not guessed: they remain UNKNOWN unless the residual is exact.
        if not complete:
            if not local_equation.atoms(sp.Derivative):
                linearized = sp.simplify(
                    sp.diff(local_equation, local_function(h)).subs(local_function(h), prefix_expr)
                )
                certificate = certify_nonlinear_lifting_remainder(residual, linearized, h, 0)
            else:
                delta_head = sp.Function("__asymptotic_delta")
                linearized = _frechet_linearization(
                    local_equation, local_function, h, prefix_expr, delta_head
                )
                certificate = certify_frechet_inverse_operator_remainder(
                    residual, linearized, delta_head, h, 0
                )
            if certificate.conclusion.is_certified:
                md = dict(ts.metadata)
                md.setdefault("remainder_certificates", []).append(certificate)
                ts = TransseriesExpansion.from_terms(
                    ts.variable,
                    ts.point,
                    ts.terms,
                    center=ts.center,
                    complete=certificate.conclusion.is_exact,
                    metadata=md,
                    remainder=certificate.conclusion,
                )
        output.append(
            NonlinearDifferentialTransseriesBranch(
                series=original_expr(prefix),
                local_series=prefix_expr,
                point=point,
                local_coordinate=h_of_x if h != variable else variable,
                local_parameter=h,
                transseries=ts,
                steps=steps,
                residual=analytic_powsimp(sp.simplify(residual)),
                residual_valuation=_residual_valuation(residual, h),
                complete=complete,
                limitation=limitation,
            )
        )

    def recurse(
        prefix: sp.Expr,
        previous_exponent: sp.Rational,
        steps: tuple[DifferentialTransseriesStep, ...],
        depth: int,
    ) -> None:
        """Lift successively smaller differential corrections and enforce residual improvement."""
        residual_before = _substitute_dependent_expression(
            local_equation, local_function, h, prefix
        )
        if residual_before == 0 or residual_before.is_zero is True:
            # The constructed particular branch is exact, but a differential
            # equation may still possess exponentially small homogeneous
            # perturbations invisible to every algebraic order.  Probe the
            # Fréchet linearization before declaring the formal family closed.
            exponential = _candidate_exponential_corrections(
                local_equation,
                local_function,
                h,
                prefix,
                terms=max(2, terms - len(steps)),
            )
            if exponential and len(steps) < terms and depth < max_depth:
                emit(prefix, steps, complete=True, limitation=None)
                for phase, mode, parameter, local_term in exponential:
                    new_prefix = analytic_powsimp(sp.expand(prefix + local_term))
                    residual_after = _substitute_dependent_expression(
                        local_equation, local_function, h, new_prefix
                    )
                    exp_steps: tuple[DifferentialTransseriesStep, ...] = (
                        DifferentialTransseriesStep(
                            local_exponent=None,
                            coefficient=parameter,
                            local_term=local_term,
                            term=original_expr(local_term),
                            balance=None,
                            residual_order_before=None,
                            residual_order_after=_residual_valuation(residual_after, h),
                            correction_kind="exponential",
                            exponential_phase=phase,
                            free_parameter=parameter,
                        ),
                    )
                    descendants = _lift_exponential_parameter_series(
                        local_equation,
                        local_function,
                        h,
                        prefix,
                        mode,
                        parameter,
                        max_terms=max(1, terms - len(steps)),
                    )
                    for coefficient, descendant_term, power in descendants:
                        before = _substitute_dependent_expression(
                            local_equation, local_function, h, new_prefix
                        )
                        new_prefix = sp.powsimp(sp.expand(new_prefix + descendant_term))
                        after = _substitute_dependent_expression(
                            local_equation, local_function, h, new_prefix
                        )
                        exp_steps += (
                            DifferentialTransseriesStep(
                                local_exponent=None,
                                coefficient=sp.simplify(coefficient * parameter**power),
                                local_term=descendant_term,
                                term=original_expr(descendant_term),
                                balance=None,
                                residual_order_before=_residual_valuation(before, h),
                                residual_order_after=_residual_valuation(after, h),
                                correction_kind="exponential",
                                exponential_phase=sp.simplify(power * phase),
                                free_parameter=parameter,
                            ),
                        )
                    final_residual = _substitute_dependent_expression(
                        local_equation, local_function, h, new_prefix
                    )
                    exact = final_residual == 0 or final_residual.is_zero is True
                    emit(
                        new_prefix,
                        steps + exp_steps,
                        complete=exact,
                        limitation=(
                            None
                            if exact
                            else "exponentially small correction tower certified from "
                            "the Fréchet linearization; the requested truncation or "
                            "current multiplicative-mode ansatz leaves a smaller "
                            "beyond-all-orders residual"
                        ),
                    )
                return
            emit(prefix, steps, complete=True, limitation=None)
            return
        if len(steps) >= terms or depth >= max_depth:
            emit(prefix, steps, complete=False, limitation="term/depth limit reached")
            return

        correction_function = sp.Function(f"delta_{depth}")
        delta = correction_function(h)
        shifted = _substitute_dependent_expression(
            local_equation, local_function, h, prefix + delta
        )
        balances = _local_dominant_balances(
            shifted,
            correction_function,
            h,
            minimum_exponent=previous_exponent,
        )
        before_val = _residual_valuation(residual_before, h)
        progressed = False

        for balance in balances:
            exponent = sp.Rational(balance.exponent)
            for coefficient in balance.roots:
                coefficient = sp.simplify(coefficient)
                if coefficient == 0 or coefficient.is_zero is True:
                    continue
                local_term = mixed_powsimp(coefficient, h**exponent)
                new_prefix = analytic_powsimp(sp.expand(prefix + local_term))
                if sp.simplify(new_prefix - prefix) == 0:
                    continue
                residual_after = _substitute_dependent_expression(
                    local_equation, local_function, h, new_prefix
                )
                after_val = _residual_valuation(residual_after, h)
                # A genuine correction must either solve the equation exactly or
                # improve the rational residual valuation when both valuations
                # are decidable.  If valuation is undecidable, keep the exact
                # balance certificate rather than guessing an improvement.
                if before_val is not None and after_val is not None and after_val <= before_val:
                    continue
                progressed = True
                step = DifferentialTransseriesStep(
                    local_exponent=exponent,
                    coefficient=coefficient,
                    local_term=local_term,
                    term=original_expr(local_term),
                    balance=balance,
                    residual_order_before=before_val,
                    residual_order_after=after_val,
                )
                recurse(
                    new_prefix,
                    exponent,
                    steps + (step,),
                    depth + 1,
                )

        if not progressed:
            logarithmic = _candidate_log_corrections(
                local_equation,
                local_function,
                h,
                prefix,
                minimum_exponent=previous_exponent,
            )
            for exponent, coefficient, log_power, local_term, residual_after in logarithmic:
                progressed = True
                after_data = _logarithmic_valuation(residual_after, h)
                step = DifferentialTransseriesStep(
                    local_exponent=exponent,
                    coefficient=coefficient,
                    local_term=local_term,
                    term=original_expr(local_term),
                    balance=None,
                    residual_order_before=before_val,
                    residual_order_after=(None if after_data is None else after_data[0]),
                    correction_kind="logarithmic",
                    logarithmic_power=log_power,
                )
                new_prefix = analytic_powsimp(sp.expand(prefix + local_term))
                if residual_after == 0 or residual_after.is_zero is True:
                    recurse(new_prefix, exponent, steps + (step,), depth + 1)
                else:
                    emit(
                        new_prefix,
                        steps + (step,),
                        complete=False,
                        limitation=(
                            "resonant logarithmic correction certified; subsequent "
                            "mixed power-log lifting is only partially recursive"
                        ),
                    )
            if not progressed:
                emit(
                    prefix,
                    steps,
                    complete=False,
                    limitation=(
                        "no smaller power or resonant logarithmic correction could be "
                        "certified; the next scale may be exponential or outside the "
                        "current differential lifting domain"
                    ),
                )

    for balance in initial:
        exponent = sp.Rational(balance.exponent)
        for coefficient in balance.roots:
            coefficient = sp.simplify(coefficient)
            if coefficient == 0 or coefficient.is_zero is True:
                continue
            local_term = mixed_powsimp(coefficient, h**exponent)
            residual_after = _substitute_dependent_expression(
                local_equation, local_function, h, local_term
            )
            first = DifferentialTransseriesStep(
                local_exponent=exponent,
                coefficient=coefficient,
                local_term=local_term,
                term=original_expr(local_term),
                balance=balance,
                residual_order_before=None,
                residual_order_after=_residual_valuation(residual_after, h),
            )
            recurse(local_term, exponent, (first,), 1)

    if not initial:
        # Some resonant equations have no nonzero pure-power characteristic
        # root at all: the leading solution is already power-log (for example
        # ``h*y' - y = h``).  Seed the same resonance detector from zero.
        logarithmic = _candidate_log_corrections(
            local_equation,
            local_function,
            h,
            sp.S.Zero,
            minimum_exponent=None,
        )
        for exponent, coefficient, log_power, local_term, residual_after in logarithmic:
            step = DifferentialTransseriesStep(
                local_exponent=exponent,
                coefficient=coefficient,
                local_term=local_term,
                term=original_expr(local_term),
                balance=None,
                residual_order_before=_residual_valuation(local_equation, h),
                residual_order_after=_residual_valuation(residual_after, h),
                correction_kind="logarithmic",
                logarithmic_power=log_power,
            )
            recurse(local_term, exponent, (step,), 1)

    # Deduplicate branches by exact symbolic equality when possible.
    unique = []
    for branch in output:
        if not any(sp.simplify(branch.series - old.series) == 0 for old in unique):
            unique.append(branch)
    return tuple(unique)
