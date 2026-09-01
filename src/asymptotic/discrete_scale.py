"""Discrete factorial/exponential asymptotic scales and Newton lifting.

The native recurrence backend represents a scalar sequence on scales of the form

    Gamma(n + 1)**kappa * lambda_**n * exp(phase(n))
        * n**theta * log(n)**log_power.

For polynomial-coefficient linear recurrences, the upper Newton polygon fixes
``kappa`` and the edge characteristic polynomial fixes ``lambda_``. Simple
roots use ordinary Birkhoff--Trjitzinsky lifting. Repeated constant-coefficient
roots use exact polynomial Jordan chains, while supported repeated
variable-coefficient roots use a secondary Newton polygon to discover
stretched-exponential phases and ramified correction lattices.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import sympy as sp

from ._symbolic_policy import bounded_solve_one

# Reuse internal asymptotic coordinates across calls.  Fresh ``Dummy`` objects
# become distinct keys in SymPy's process-global expression caches; creating
# them for every recurrence lift caused severe same-process test degradation.
# Module-private Dummies retain collision safety while keeping cache keys stable.
_EDGE_LAMBDA = sp.Dummy("lambda")
_BT_T = sp.Dummy("t", positive=True)
_BT_LOCAL = sp.Dummy("s", positive=True)
_BT_THETA = sp.Dummy("theta")
_SECONDARY_U = sp.Dummy("u")
_SECONDARY_V = sp.Dummy("v")
_BT_COEFFICIENTS = tuple(sp.Dummy(f"c{m}") for m in range(1, 65))


def _bt_coefficients(count: int) -> tuple[sp.Symbol, ...]:
    """Return stable private coefficient symbols for one finite BT lift."""

    if count < 0:
        raise ValueError("coefficient count must be nonnegative")
    if count > len(_BT_COEFFICIENTS):
        raise ValueError("BT lifting supports at most 64 correction coefficients")
    return _BT_COEFFICIENTS[:count]


@dataclass(frozen=True)
class DiscreteAsymptoticScale:
    """One factorial/exponential/power scale for a sequence at ``n -> +oo``."""

    index: sp.Symbol
    factorial_power: sp.Expr = sp.S.Zero
    exponential_base: sp.Expr = sp.S.One
    power: sp.Expr = sp.S.Zero
    phase: sp.Expr = sp.S.Zero
    log_power: sp.Expr = sp.S.Zero

    def __post_init__(self) -> None:
        if not isinstance(self.index, sp.Symbol):
            raise TypeError("index must be a Symbol")
        for name in ("factorial_power", "exponential_base", "power", "phase", "log_power"):
            object.__setattr__(self, name, sp.sympify(getattr(self, name)))

    @property
    def expression(self) -> sp.Expr:
        n = self.index
        return sp.powsimp(
            sp.gamma(n + 1) ** self.factorial_power
            * self.exponential_base**n
            * sp.exp(self.phase)
            * n**self.power
            * sp.log(n) ** self.log_power,
            force=False,
        )

    def ratio(self, shift: int) -> sp.Expr:
        """Return the exact shifted-scale ratio ``scale(n+shift)/scale(n)``."""

        if int(shift) != shift:
            raise ValueError("shift must be an integer")
        n = self.index
        j = int(shift)
        return sp.powsimp(
            self.expression.subs(n, n + j) / self.expression,
            force=False,
        )


@dataclass(frozen=True)
class DiscreteNewtonEdge:
    """A balanced upper Newton edge of a polynomial-coefficient recurrence."""

    factorial_power: sp.Expr
    height: sp.Expr
    active_shifts: tuple[int, ...]
    characteristic: sp.Expr
    characteristic_symbol: sp.Symbol


@dataclass(frozen=True)
class DiscreteAsymptoticBranch:
    """A finite Birkhoff--Trjitzinsky branch attached to one Newton edge root."""

    scale: DiscreteAsymptoticScale
    coefficients: tuple[sp.Expr, ...]
    expression: sp.Expr
    edge: DiscreteNewtonEdge
    characteristic_root: sp.Expr
    characteristic_mult: int
    residual_order: sp.Expr | None = None
    lattice_step: sp.Rational = sp.S.One
    secondary_mult: int = 1
    transport_mult: int = 1
    tertiary_mult: int = 1

    def replay_characteristic(self) -> bool | None:
        value = sp.expand(
            self.edge.characteristic.subs(self.edge.characteristic_symbol, self.characteristic_root)
        )
        if value == 0 or value.is_zero is True:
            return True
        if value.is_zero is False:
            return False
        return None

    def replay_residual(self, data: LinearRecurrenceData) -> bool | None:
        """Recompute the normalized residual order for this finite branch."""

        measured = _measure_branch_residual(data, self)
        if measured is None or self.residual_order is None:
            return None
        return bool(measured == self.residual_order)


@dataclass(frozen=True)
class _RecurrencePolynomialTerm:
    """Cached polynomial metadata for one normalized shift coefficient."""

    shift: int
    expression: sp.Expr
    polynomial: sp.Poly
    degree: int
    leading_coefficient: sp.Expr


@dataclass(frozen=True)
class LinearRecurrenceData:
    """Normalized homogeneous scalar linear recurrence data.

    ``polynomial_terms`` stores the exact ``Poly`` objects, degrees, and leading
    coefficients used by Newton construction and lifting so hot paths do not
    repeatedly rebuild polynomial metadata from expressions.
    """

    coefficients: tuple[tuple[int, sp.Expr], ...]
    index: sp.Symbol
    order: int
    polynomial_terms: tuple[_RecurrencePolynomialTerm, ...] = ()
    forcing: sp.Expr = sp.S.Zero

    def __post_init__(self) -> None:
        if self.polynomial_terms:
            return
        terms = []
        for shift, expression in self.coefficients:
            poly = sp.Poly(expression, self.index)
            terms.append(
                _RecurrencePolynomialTerm(
                    shift=shift,
                    expression=expression,
                    polynomial=poly,
                    degree=int(poly.degree()),
                    leading_coefficient=poly.LC(),
                )
            )
        object.__setattr__(self, "polynomial_terms", tuple(terms))


def _shift_of(term: sp.Expr, sequence: sp.Expr, index: sp.Symbol) -> int | None:
    if getattr(term, "func", None) != sequence.func or len(term.args) != 1:
        return None
    delta = sp.expand(term.args[0] - index)
    if delta.is_integer is True and delta.is_number:
        return int(delta)
    return None


def linear_recurrence_data(
    recurrence: sp.Expr | sp.Equality,
    sequence: sp.Expr,
    index: sp.Symbol,
) -> LinearRecurrenceData:
    """Extract a homogeneous linear recurrence with integer shifts.

    Rational coefficient functions are cleared to polynomial coefficients.
    The smallest shift is translated to zero, so the returned order is the
    largest normalized shift.
    """

    expr = sp.sympify(recurrence)
    if isinstance(expr, sp.Equality):
        expr = expr.lhs - expr.rhs
    expr = sp.expand(expr)
    sequence = sp.sympify(sequence)
    atoms = sorted(
        (atom for atom in expr.atoms(sp.Function) if getattr(atom, "func", None) == sequence.func),
        key=str,
    )
    if not atoms:
        raise ValueError("recurrence does not contain the requested sequence")

    raw: dict[int, sp.Expr] = {}
    rebuilt = sp.S.Zero
    for atom in atoms:
        shift = _shift_of(atom, sequence, index)
        if shift is None:
            raise NotImplementedError("native discrete lifting requires integer shifts")
        coefficient = sp.diff(expr, atom)
        if coefficient.has(sequence.func):
            raise NotImplementedError("native discrete lifting requires a linear recurrence")
        raw[shift] = sp.expand(raw.get(shift, 0) + coefficient)
        rebuilt += coefficient * atom
    forcing = sp.expand(expr - rebuilt)

    minimum = min(raw)
    shifted = {
        shift - minimum: sp.together(coeff.subs(index, index - minimum))
        for shift, coeff in raw.items()
    }
    shifted_forcing = sp.together(forcing.subs(index, index - minimum))
    common_den = sp.S.One
    for coeff in (*shifted.values(), shifted_forcing):
        common_den = sp.lcm(common_den, sp.denom(coeff))
    polynomial: dict[int, sp.Expr] = {}
    metadata: dict[int, _RecurrencePolynomialTerm] = {}
    for shift, coeff in shifted.items():
        value = sp.cancel(coeff * common_den)
        try:
            poly = sp.Poly(value, index)
        except sp.PolynomialError as exc:
            raise NotImplementedError(
                "native discrete Newton lifting requires rational coefficient functions"
            ) from exc
        expression = sp.expand(poly.as_expr())
        if expression != 0:
            polynomial[shift] = expression
            metadata[shift] = _RecurrencePolynomialTerm(
                shift=shift,
                expression=expression,
                polynomial=poly,
                degree=int(poly.degree()),
                leading_coefficient=poly.LC(),
            )
    if len(polynomial) < 2:
        raise NotImplementedError("recurrence has fewer than two nonzero shift coefficients")
    order = max(polynomial)
    coefficients = tuple(sorted(polynomial.items()))
    terms = tuple(metadata[shift] for shift, _ in coefficients)
    normalized_forcing = sp.expand(sp.cancel(shifted_forcing * common_den))
    return LinearRecurrenceData(coefficients, index, order, terms, normalized_forcing)


def discrete_newton_edges(data: LinearRecurrenceData) -> tuple[DiscreteNewtonEdge, ...]:
    """Construct all balanced upper Newton edges of a recurrence."""

    points = [(term.shift, term.degree, term.leading_coefficient) for term in data.polynomial_terms]

    candidates: set[sp.Expr] = set()
    for (i, di, _), (j, dj, _) in combinations(points, 2):
        if i != j:
            candidates.add(sp.Rational(di - dj, j - i))

    z = _EDGE_LAMBDA
    edges: list[DiscreteNewtonEdge] = []
    for kappa in sorted(candidates, key=sp.default_sort_key):
        heights = [(sp.Rational(degree) + kappa * shift, shift, lc) for shift, degree, lc in points]
        height = max(item[0] for item in heights)
        active = [(shift, lc) for value, shift, lc in heights if value == height]
        if len(active) < 2:
            continue
        characteristic = sp.expand(sum(lc * z**shift for shift, lc in active))
        edges.append(
            DiscreteNewtonEdge(
                sp.sympify(kappa),
                sp.sympify(height),
                tuple(shift for shift, _ in active),
                characteristic,
                z,
            )
        )
    return tuple(edges)


def _root_multiplicity(polynomial: sp.Expr, symbol: sp.Symbol, root: sp.Expr) -> int:
    try:
        degree = int(sp.Poly(polynomial, symbol).degree())
    except (sp.PolynomialError, TypeError, ValueError):
        return 1
    derivative = sp.expand(polynomial)
    for multiplicity in range(1, degree + 1):
        derivative = sp.diff(derivative, symbol)
        value = sp.expand(derivative.subs(symbol, root))
        if value != 0 and value.is_zero is not True:
            return multiplicity
    return max(1, degree)


def _edge_roots(edge: DiscreteNewtonEdge) -> tuple[tuple[sp.Expr, int], ...]:
    z = edge.characteristic_symbol
    try:
        roots = sp.roots(edge.characteristic, z)
    except (sp.PolynomialError, TypeError, ValueError):
        roots = {}
    if roots and sum(int(mult) for mult in roots.values()) == sp.degree(edge.characteristic, z):
        return tuple((sp.sympify(root), int(mult)) for root, mult in roots.items() if root != 0)
    solved = bounded_solve_one(edge.characteristic, z, allow_general=True) or ()
    out = []
    for root in solved:
        if root != 0:
            out.append((sp.sympify(root), _root_multiplicity(edge.characteristic, z, root)))
    return tuple(out)


def _coefficient_equations(
    data: LinearRecurrenceData,
    edge: DiscreteNewtonEdge,
    root: sp.Expr,
    terms: int,
) -> tuple[sp.Symbol, tuple[sp.Symbol, ...], list[sp.Expr]]:
    """Build inverse-power coefficient equations after Newton normalization."""

    n = data.index
    t = _BT_T
    theta = _BT_THETA
    coeffs = _bt_coefficients(max(1, terms) - 1)
    series_coeffs = (sp.S.One,) + coeffs
    residual = sp.S.Zero

    for term_data in data.polynomial_terms:
        shift = term_data.shift
        degree = term_data.degree
        normalized_coeff = sp.expand(term_data.expression.subs(n, 1 / t) * t**degree)
        gap = sp.cancel(edge.height - degree - edge.factorial_power * shift)
        if gap.is_integer is not True or gap.is_nonnegative is not True:
            raise NotImplementedError(
                "fractionally spaced Newton levels require a ramified discrete lift"
            )
        factorial_ratio = sp.prod(1 + q * t for q in range(1, shift + 1)) ** edge.factorial_power
        power_ratio = (1 + shift * t) ** theta
        shifted_series = sum(
            series_coeffs[m] * t**m * (1 + shift * t) ** (-m) for m in range(len(series_coeffs))
        )
        term = (
            t ** int(gap)
            * normalized_coeff
            * root**shift
            * factorial_ratio
            * power_ratio
            * shifted_series
        )
        residual += sp.series(term, t, 0, terms + 3).removeO()

    expanded = sp.expand(residual)
    equations: dict[sp.Expr, sp.Expr] = {}
    for term in sp.Add.make_args(expanded):
        exponent = term.as_powers_dict().get(t, sp.S.Zero)
        coefficient = sp.expand(term / t**exponent)
        equations[exponent] = sp.expand(equations.get(exponent, 0) + coefficient)
    ordered = [equations[key] for key in sorted(equations, key=sp.default_sort_key)]
    return theta, coeffs, ordered


def _linear_equation_solution(
    equation: sp.Expr,
    target: sp.Symbol,
) -> sp.Expr | None:
    """Solve a coefficient equation only when it is exactly linear in target."""

    try:
        poly = sp.Poly(sp.expand(equation), target)
    except sp.PolynomialError:
        return None
    if poly.degree() != 1:
        return None
    linear = poly.nth(1)
    constant = poly.nth(0)
    if linear == 0 or linear.is_zero is True:
        return None
    return sp.cancel(-constant / linear)


def _solve_lift_equations(
    theta: sp.Symbol,
    coeffs: tuple[sp.Symbol, ...],
    equations: list[sp.Expr],
    terms: int,
) -> tuple[sp.Expr, tuple[sp.Expr, ...]] | None:
    """Solve successive nonresonant BT equations by linear coefficient extraction."""

    unknowns = [theta, *coeffs[: max(0, terms - 1)]]
    solved: dict[sp.Symbol, sp.Expr] = {}
    for equation in equations:
        current = sp.expand(equation.subs(solved))
        if current == 0 or current.is_zero is True:
            continue
        pending = [symbol for symbol in unknowns if symbol not in solved and current.has(symbol)]
        if not pending:
            if current.is_zero is False:
                return None
            continue
        target = pending[0]
        value = _linear_equation_solution(current, target)
        if value is None:
            return None
        solved[target] = sp.cancel(value.subs(solved))
        if len(solved) == len(unknowns):
            break
    if theta not in solved:
        return None
    values = tuple(sp.cancel(solved.get(symbol, 0)) for symbol in coeffs[: max(0, terms - 1)])
    return sp.cancel(solved[theta]), values


def _solve_ramified_transport_branches(
    theta: sp.Symbol,
    coeffs: tuple[sp.Symbol, ...],
    equations: list[sp.Expr],
    terms: int,
) -> tuple[tuple[sp.Expr, tuple[sp.Expr, ...], int], ...]:
    """Solve a ramified transport hierarchy, allowing a polynomial indicial equation.

    The ordinary BT solver assumes the first unresolved equation is linear in
    ``theta``.  At a repeated secondary Newton root that transport equation can
    instead be polynomial.  We solve its exact roots (with multiplicities) and
    then continue each branch through the usual linear correction hierarchy.
    """

    theta_equation = None
    prefix: list[sp.Expr] = []
    for equation in equations:
        current = sp.expand(equation)
        if current == 0 or current.is_zero is True:
            continue
        if current.has(theta):
            theta_equation = current
            break
        prefix.append(current)
    if theta_equation is None or any(item.is_zero is False for item in prefix):
        return ()
    try:
        polynomial = sp.Poly(theta_equation, theta)
    except sp.PolynomialError:
        return ()
    if polynomial.degree() < 1 or polynomial.degree() > 8:
        return ()
    roots = sp.roots(polynomial.as_expr(), theta)
    if not roots or sum(int(mult) for mult in roots.values()) != polynomial.degree():
        solved = bounded_solve_one(polynomial.as_expr(), theta, allow_general=True) or ()
        roots = {sp.sympify(value): 1 for value in solved}
    out = []
    for theta_value, multiplicity in roots.items():
        substituted = [sp.expand(eq.subs(theta, theta_value)) for eq in equations]
        unknowns = list(coeffs[: max(0, terms - 1)])
        solved_coeffs: dict[sp.Symbol, sp.Expr] = {}
        failed = False
        for equation in substituted:
            current = sp.expand(equation.subs(solved_coeffs))
            if current == 0 or current.is_zero is True:
                continue
            pending = [
                symbol for symbol in unknowns if symbol not in solved_coeffs and current.has(symbol)
            ]
            if not pending:
                if current.is_zero is False:
                    failed = True
                    break
                continue
            target = pending[0]
            value = _linear_equation_solution(current, target)
            if value is None:
                failed = True
                break
            solved_coeffs[target] = sp.cancel(value.subs(solved_coeffs))
        if failed:
            continue
        corrections = tuple(sp.cancel(solved_coeffs.get(symbol, 0)) for symbol in unknowns)
        out.append((sp.cancel(theta_value), corrections, int(multiplicity)))
    return tuple(out)


def _gap_for_term(
    edge: DiscreteNewtonEdge,
    term: _RecurrencePolynomialTerm,
) -> sp.Expr:
    return sp.cancel(edge.height - term.degree - edge.factorial_power * term.shift)


def _phase_monomial(phase: sp.Expr, index: sp.Symbol) -> tuple[sp.Expr, sp.Rational] | None:
    """Return ``(coefficient, exponent)`` for a single power phase ``c*n**alpha``."""

    phase = sp.expand_power_base(sp.sympify(phase), force=False)
    coefficient, exponent = phase.as_coeff_exponent(index)
    if coefficient * index**exponent != phase or index in coefficient.free_symbols:
        return None
    if exponent.is_Rational is not True:
        return None
    return sp.sympify(coefficient), sp.Rational(exponent)


def _phase_ratio_series(
    phase: sp.Expr,
    index: sp.Symbol,
    shift: int,
    local: sp.Symbol,
    denominator: int,
    order: int,
) -> sp.Expr:
    """Expand ``exp(Phi(n+j)-Phi(n))`` on a ramified local lattice."""

    if phase == 0:
        return sp.S.One
    delta = sp.S.Zero
    for term in sp.Add.make_args(sp.expand(phase)):
        monomial = _phase_monomial(term, index)
        if monomial is None:
            raise NotImplementedError(
                "residual replay supports finite sums of power stretched-exponential phases"
            )
        coefficient, exponent = monomial
        delta += (
            coefficient
            * local ** (-denominator * exponent)
            * ((1 + shift * local**denominator) ** exponent - 1)
        )
    delta = sp.series(delta, local, 0, order).removeO()
    return sp.series(sp.exp(delta), local, 0, order).removeO()


def _normalized_residual_series(
    data: LinearRecurrenceData,
    branch: DiscreteAsymptoticBranch,
    series_order: int,
) -> sp.Expr:
    """Return the recurrence residual divided by its Newton leading scale.

    The local variable is ``s = n**(-1/D)`` where ``1/D`` is the branch's
    correction-lattice step.  This covers ordinary inverse powers and ramified
    Birkhoff--Trjitzinsky lattices with the same replay code.
    """

    n = data.index
    step = sp.Rational(branch.lattice_step)
    if step.p != 1:
        raise NotImplementedError("branch lattice steps must be reciprocal integers")
    denominator = int(step.q)
    local = _BT_LOCAL
    edge = branch.edge
    scale = branch.scale
    residual = sp.S.Zero

    for term_data in data.polynomial_terms:
        shift = term_data.shift
        normalized_coeff = sp.expand(
            term_data.expression.subs(n, local ** (-denominator))
            * local ** (denominator * term_data.degree)
        )
        gap = _gap_for_term(edge, term_data)
        gap_power = sp.cancel(denominator * gap)
        if gap_power.is_integer is not True or gap_power.is_nonnegative is not True:
            raise NotImplementedError("branch lattice does not resolve the Newton level spacing")

        factorial_ratio = sp.series(
            sp.prod(1 + q * local**denominator for q in range(1, shift + 1))
            ** scale.factorial_power,
            local,
            0,
            series_order,
        ).removeO()
        power_ratio = sp.series(
            (1 + shift * local**denominator) ** scale.power,
            local,
            0,
            series_order,
        ).removeO()
        phase_ratio = _phase_ratio_series(scale.phase, n, shift, local, denominator, series_order)
        if scale.log_power == 0:
            log_ratio = sp.S.One
        else:
            log_ratio = sp.series(
                (sp.log(local ** (-denominator) + shift) / sp.log(local ** (-denominator)))
                ** scale.log_power,
                local,
                0,
                series_order,
            ).removeO()

        shifted_correction = sp.S.Zero
        for m, coefficient in enumerate(branch.coefficients):
            if m >= series_order:
                break
            shifted_factor = sum(
                sp.binomial(-sp.Rational(m, denominator), k) * (shift * local**denominator) ** k
                for k in range((series_order - m - 1) // denominator + 1)
            )
            shifted_correction += coefficient * local**m * shifted_factor
        term = _multiply_truncated(
            (
                local ** int(gap_power),
                normalized_coeff,
                scale.exponential_base**shift,
                factorial_ratio,
                power_ratio,
                phase_ratio,
                log_ratio,
                shifted_correction,
            ),
            local,
            series_order,
        )
        residual = _truncate_local_series(residual + term, local, series_order)
    return sp.expand(residual)


def _measure_branch_residual(
    data: LinearRecurrenceData,
    branch: DiscreteAsymptoticBranch,
) -> sp.Expr | None:
    """Measure the first nonzero normalized residual order in powers of ``1/n``."""

    n = data.index
    if branch.scale.phase == 0:
        exact = sp.S.Zero
        for shift, coefficient in data.coefficients:
            exact += coefficient * branch.expression.subs(n, n + shift)
        try:
            exact = sp.combsimp(sp.cancel(exact))
        except (sp.PolynomialError, TypeError, ValueError, ZeroDivisionError):
            pass
        if exact == 0 or exact.is_zero is True:
            return sp.oo

    denominator = int(sp.Rational(branch.lattice_step).q)
    order = max(8, len(branch.coefficients) + 3 * denominator + 3)
    series = _normalized_residual_series(data, branch, order)
    if series == 0 or series.is_zero is True:
        return None
    local = next((symbol for symbol in series.free_symbols if symbol.name == "s"), None)
    if local is None:
        return sp.S.Zero
    exponents = []
    for term in sp.Add.make_args(series):
        exponent = term.as_powers_dict().get(local, sp.S.Zero)
        coefficient = sp.expand(term / local**exponent)
        if coefficient != 0 and coefficient.is_zero is not True:
            exponents.append(exponent)
    if not exponents:
        return None
    local_order = min(exponents, key=sp.default_sort_key)
    return sp.Rational(local_order, denominator)


def _constant_coefficient_repeated_branches(
    data: LinearRecurrenceData,
    edge: DiscreteNewtonEdge,
    root: sp.Expr,
    multiplicity: int,
) -> tuple[DiscreteAsymptoticBranch, ...]:
    """Return the exact Jordan-chain branches for a repeated constant root."""

    if edge.factorial_power != 0 or not all(term.degree == 0 for term in data.polynomial_terms):
        return ()
    n = data.index
    branches = []
    for power in range(multiplicity):
        scale = DiscreteAsymptoticScale(
            n,
            exponential_base=root,
            power=sp.Integer(power),
        )
        expression = sp.powsimp(scale.expression, force=False)
        branches.append(
            DiscreteAsymptoticBranch(
                scale=scale,
                coefficients=(sp.S.One,),
                expression=expression,
                edge=edge,
                characteristic_root=root,
                characteristic_mult=multiplicity,
                residual_order=sp.oo,
                lattice_step=sp.S.One,
            )
        )
    return tuple(branches)


def _tertiary_phase_by_residual(
    data: LinearRecurrenceData,
    edge: DiscreteNewtonEdge,
    root: sp.Expr,
    multiplicity: int,
    phase: sp.Expr,
    q: sp.Rational,
    secondary_multiplicity: int,
) -> tuple[tuple[sp.Expr, int], ...]:
    """Discover the next phase from the residual valuation of a repeated secondary root."""
    if secondary_multiplicity <= 1:
        return ()
    n = data.index
    d0 = int(q.q)
    base_step = sp.Rational(1, d0)
    base_scale = DiscreteAsymptoticScale(
        n, factorial_power=edge.factorial_power, exponential_base=root, phase=phase
    )
    base = DiscreteAsymptoticBranch(
        scale=base_scale,
        coefficients=(sp.S.One,),
        expression=base_scale.expression,
        edge=edge,
        characteristic_root=root,
        characteristic_mult=multiplicity,
        lattice_step=base_step,
        secondary_mult=secondary_multiplicity,
    )
    try:
        residual = _normalized_residual_series(data, base, max(8, multiplicity + 6))
    except (NotImplementedError, TypeError, ValueError, sp.PolynomialError):
        return ()
    orders = []
    for term in sp.Add.make_args(sp.expand(residual)):
        exponent = term.as_powers_dict().get(_BT_LOCAL, sp.S.Zero)
        coefficient = sp.expand(term / _BT_LOCAL**exponent)
        if exponent.is_integer is True and coefficient != 0 and coefficient.is_zero is not True:
            orders.append(int(exponent))
    if not orders:
        return ()
    leading = min(orders)
    secondary_order = sp.Rational(multiplicity) * d0 * q
    excess_local = sp.Rational(leading) - secondary_order
    if excess_local <= 0:
        return ()
    eta = sp.cancel(excess_local / (d0 * secondary_multiplicity))
    beta = sp.cancel(1 - q - eta)
    if beta.is_Rational is not True or beta <= 0:
        return ()
    beta = sp.Rational(beta)
    denominator = int(sp.ilcm(d0, int(beta.q)))
    c = sp.Dummy("tertiary_c")
    candidate_phase = sp.expand(phase + c * n**beta)
    scale = DiscreteAsymptoticScale(
        n, factorial_power=edge.factorial_power, exponential_base=root, phase=candidate_phase
    )
    probe = DiscreteAsymptoticBranch(
        scale=scale,
        coefficients=(sp.S.One,),
        expression=scale.expression,
        edge=edge,
        characteristic_root=root,
        characteristic_mult=multiplicity,
        lattice_step=sp.Rational(1, denominator),
        secondary_mult=secondary_multiplicity,
    )
    try:
        refined = _normalized_residual_series(data, probe, max(12, 2 * denominator + 6))
    except (NotImplementedError, TypeError, ValueError, sp.PolynomialError):
        return ()
    grouped: dict[int, sp.Expr] = {}
    for term in sp.Add.make_args(sp.expand(refined)):
        exponent = term.as_powers_dict().get(_BT_LOCAL, sp.S.Zero)
        if exponent.is_integer is not True:
            continue
        k = int(exponent)
        grouped[k] = sp.expand(grouped.get(k, 0) + term / _BT_LOCAL**k)
    for k in sorted(grouped):
        coefficient = sp.expand(grouped[k])
        if coefficient == 0 or coefficient.is_zero is True:
            continue
        if not coefficient.has(c):
            return ()
        try:
            polynomial = sp.Poly(coefficient, c)
        except sp.PolynomialError:
            return ()
        roots = sp.roots(polynomial.as_expr(), c)
        if not roots:
            return ()
        return tuple(
            (sp.expand(phase + value * n**beta), int(mult))
            for value, mult in roots.items()
            if value != 0 and value.is_zero is not True
        )
    return ()


def _secondary_newton_phases(
    data: LinearRecurrenceData,
    edge: DiscreteNewtonEdge,
    root: sp.Expr,
    multiplicity: int,
) -> tuple[tuple[sp.Expr, sp.Rational, int, int], ...]:
    """Discover secondary and one-level tertiary stretched-exponential phases."""

    if multiplicity <= 1:
        return ()
    n = data.index
    t = _BT_T
    u = _SECONDARY_U
    max_t_order = multiplicity + 5
    max_u_order = multiplicity + 2
    residual = sp.S.Zero

    for term_data in data.polynomial_terms:
        shift = term_data.shift
        normalized_coeff = sp.expand(term_data.expression.subs(n, 1 / t) * t**term_data.degree)
        gap = _gap_for_term(edge, term_data)
        if gap.is_nonnegative is not True:
            return ()
        factorial_ratio = sp.prod(1 + j * t for j in range(1, shift + 1)) ** edge.factorial_power
        factorial_ratio = sp.series(factorial_ratio, t, 0, max_t_order).removeO()
        shift_phase = sum((shift * u) ** b / sp.factorial(b) for b in range(max_u_order + 1))
        residual += t**gap * normalized_coeff * root**shift * factorial_ratio * shift_phase

    monomials: dict[tuple[sp.Rational, int], sp.Expr] = {}
    for term in sp.Add.make_args(sp.expand(residual)):
        powers = term.as_powers_dict()
        a = powers.get(t, sp.S.Zero)
        b = powers.get(u, sp.S.Zero)
        if a.is_Rational is not True or b.is_integer is not True or b.is_nonnegative is not True:
            continue
        a = sp.Rational(a)
        b_int = int(b)
        coefficient = sp.expand(term / (t**a * u**b_int))
        key = (a, b_int)
        monomials[key] = sp.expand(monomials.get(key, 0) + coefficient)

    candidates: set[sp.Rational] = set()
    for (a, b), coefficient in monomials.items():
        if coefficient == 0 or coefficient.is_zero is True:
            continue
        if a > 0 and b < multiplicity:
            q = sp.Rational(a, multiplicity - b)
            if 0 < q < 1:
                candidates.add(q)

    v = _SECONDARY_V
    phases: list[tuple[sp.Expr, sp.Rational, int, int]] = []
    for q in sorted(candidates, key=sp.default_sort_key):
        target_weight = multiplicity * q
        weights = [
            a + b * q
            for (a, b), coefficient in monomials.items()
            if coefficient != 0 and coefficient.is_zero is not True
        ]
        if not weights or min(weights, key=sp.default_sort_key) != target_weight:
            continue
        edge_terms = [
            coefficient * v**b
            for (a, b), coefficient in monomials.items()
            if coefficient != 0 and coefficient.is_zero is not True and a + b * q == target_weight
        ]
        if len(edge_terms) < 2:
            continue
        polynomial = sp.Poly(sp.expand(sum(edge_terms)), v)
        roots = sp.roots(polynomial.as_expr(), v)
        if not roots:
            continue
        alpha = sp.S.One - q
        for value, root_mult in roots.items():
            if value == 0:
                continue
            phase = sp.cancel(value / alpha) * n**alpha
            corrections = ()
            if int(root_mult) > 1:
                corrections = _tertiary_phase_by_residual(
                    data, edge, root, multiplicity, phase, q, int(root_mult)
                )
            if corrections:
                for refined_phase, tertiary_mult in corrections:
                    phases.append((refined_phase, q, int(root_mult), tertiary_mult))
            else:
                phases.append((phase, q, int(root_mult), 1))
    return tuple(phases)


def _truncate_local_series(expr: sp.Expr, local: sp.Symbol, order: int) -> sp.Expr:
    """Drop expanded nonnegative local powers at or above ``order``."""

    expanded = sp.expand(expr)
    kept = []
    for term in sp.Add.make_args(expanded):
        exponent = term.as_powers_dict().get(local, sp.S.Zero)
        if exponent.is_integer is True and 0 <= int(exponent) < order:
            kept.append(term)
    return sp.expand(sum(kept, sp.S.Zero))


def _multiply_truncated(
    factors: tuple[sp.Expr, ...],
    local: sp.Symbol,
    order: int,
) -> sp.Expr:
    """Multiply already-expanded local series with truncation after each factor."""

    value = sp.S.One
    for factor in factors:
        value = _truncate_local_series(value * factor, local, order)
    return value


def _ramified_coefficient_equations(
    data: LinearRecurrenceData,
    edge: DiscreteNewtonEdge,
    root: sp.Expr,
    phase: sp.Expr,
    lattice_step: sp.Rational,
    terms: int,
) -> tuple[sp.Symbol, tuple[sp.Symbol, ...], list[sp.Expr]]:
    """Build BT equations on a fractional inverse-power lattice."""

    n = data.index
    step = sp.Rational(lattice_step)
    if step.p != 1:
        raise NotImplementedError("ramified lifts require a reciprocal-integer lattice")
    denominator = int(step.q)
    local = _BT_LOCAL
    theta = _BT_THETA
    coeffs = _bt_coefficients(max(1, terms) - 1)
    series_coeffs = (sp.S.One,) + coeffs
    expansion_order = max(8, terms + 2 * denominator + 4)
    residual = sp.S.Zero

    for term_data in data.polynomial_terms:
        shift = term_data.shift
        normalized_coeff = sp.expand(
            term_data.expression.subs(n, local ** (-denominator))
            * local ** (denominator * term_data.degree)
        )
        gap = _gap_for_term(edge, term_data)
        gap_power = sp.cancel(denominator * gap)
        if gap_power.is_integer is not True or gap_power.is_nonnegative is not True:
            raise NotImplementedError("secondary lattice does not resolve the primary Newton gaps")
        factorial_ratio = sp.series(
            sp.prod(1 + q * local**denominator for q in range(1, shift + 1))
            ** edge.factorial_power,
            local,
            0,
            expansion_order,
        ).removeO()
        phase_ratio = _phase_ratio_series(phase, n, shift, local, denominator, expansion_order)
        power_ratio = sum(
            sp.binomial(theta, k) * (shift * local**denominator) ** k
            for k in range((expansion_order - 1) // denominator + 1)
        )
        shifted_series = sp.S.Zero
        for m, series_coefficient in enumerate(series_coeffs):
            shifted_factor = sum(
                sp.binomial(-sp.Rational(m, denominator), k) * (shift * local**denominator) ** k
                for k in range((expansion_order - m - 1) // denominator + 1)
            )
            shifted_series += series_coefficient * local**m * shifted_factor
        term = _multiply_truncated(
            (
                local ** int(gap_power),
                normalized_coeff,
                root**shift,
                factorial_ratio,
                phase_ratio,
                power_ratio,
                shifted_series,
            ),
            local,
            expansion_order,
        )
        residual = _truncate_local_series(residual + term, local, expansion_order)

    expanded = sp.expand(residual)
    equations: dict[int, sp.Expr] = {}
    for term in sp.Add.make_args(expanded):
        exponent = term.as_powers_dict().get(local, sp.S.Zero)
        if exponent.is_integer is not True:
            continue
        coefficient = sp.expand(term / local**exponent)
        exponent_int = int(exponent)
        equations[exponent_int] = sp.expand(equations.get(exponent_int, 0) + coefficient)
    ordered = [equations[key] for key in sorted(equations)]
    return theta, coeffs, ordered


def _ramified_branches(
    data: LinearRecurrenceData,
    edge: DiscreteNewtonEdge,
    root: sp.Expr,
    multiplicity: int,
    terms: int,
) -> tuple[DiscreteAsymptoticBranch, ...]:
    """Lift repeated Newton roots through secondary and tertiary phase edges."""

    n = data.index
    branches: list[DiscreteAsymptoticBranch] = []
    for phase, q, secondary_multiplicity, tertiary_multiplicity in _secondary_newton_phases(
        data, edge, root, multiplicity
    ):
        denominators = [int(q.q)]
        for term_data in data.polynomial_terms:
            gap = sp.Rational(_gap_for_term(edge, term_data))
            denominators.append(int(gap.q))
        for phase_term in sp.Add.make_args(sp.expand(phase)):
            monomial = _phase_monomial(phase_term, n)
            if monomial is not None:
                denominators.append(int(monomial[1].q))
        denominator = int(sp.ilcm(*denominators))
        base_step = sp.Rational(1, denominator)
        lattice_step = base_step
        active_phase = phase
        if True:
            denominator = int(sp.Rational(lattice_step).q)
            try:
                theta_symbol, coeff_symbols, equations = _ramified_coefficient_equations(
                    data, edge, root, active_phase, lattice_step, terms
                )
            except (NotImplementedError, ValueError, TypeError):
                continue
            transport = _solve_ramified_transport_branches(
                theta_symbol, coeff_symbols, equations, terms
            )
            if not transport:
                solved = _solve_lift_equations(theta_symbol, coeff_symbols, equations, terms)
                if solved is None:
                    continue
                transport = ((solved[0], solved[1], 1),)

            for theta, corrections, transport_multiplicity in transport:
                scale = DiscreteAsymptoticScale(
                    n,
                    factorial_power=edge.factorial_power,
                    exponential_base=root,
                    power=theta,
                    phase=active_phase,
                )
                correction = sp.S.One + sum(
                    corrections[m - 1] * n ** (-sp.Rational(m, denominator))
                    for m in range(1, len(corrections) + 1)
                )
                expression = sp.powsimp(scale.expression * correction, force=False)
                branch = DiscreteAsymptoticBranch(
                    scale=scale,
                    coefficients=(sp.S.One, *corrections),
                    expression=expression,
                    edge=edge,
                    characteristic_root=root,
                    characteristic_mult=multiplicity,
                    residual_order=None,
                    lattice_step=lattice_step,
                    secondary_mult=secondary_multiplicity,
                    transport_mult=transport_multiplicity,
                    tertiary_mult=tertiary_multiplicity,
                )
                measured = _measure_branch_residual(data, branch)
                branches.append(
                    DiscreteAsymptoticBranch(
                        scale=branch.scale,
                        coefficients=branch.coefficients,
                        expression=branch.expression,
                        edge=branch.edge,
                        characteristic_root=branch.characteristic_root,
                        characteristic_mult=branch.characteristic_mult,
                        residual_order=measured,
                        lattice_step=branch.lattice_step,
                        secondary_mult=secondary_multiplicity,
                        transport_mult=transport_multiplicity,
                        tertiary_mult=tertiary_multiplicity,
                    )
                )
                if transport_multiplicity > 1 and measured is not None:
                    for log_power in range(1, transport_multiplicity):
                        log_scale = DiscreteAsymptoticScale(
                            n,
                            factorial_power=edge.factorial_power,
                            exponential_base=root,
                            power=theta,
                            phase=active_phase,
                            log_power=log_power,
                        )
                        log_expression = sp.powsimp(log_scale.expression * correction, force=False)
                        log_branch = DiscreteAsymptoticBranch(
                            scale=log_scale,
                            coefficients=(sp.S.One, *corrections),
                            expression=log_expression,
                            edge=edge,
                            characteristic_root=root,
                            characteristic_mult=multiplicity,
                            residual_order=None,
                            lattice_step=lattice_step,
                            secondary_mult=secondary_multiplicity,
                            transport_mult=transport_multiplicity,
                            tertiary_mult=tertiary_multiplicity,
                        )
                        log_measured = _measure_branch_residual(data, log_branch)
                        if log_measured is not None and log_measured >= measured:
                            branches.append(
                                DiscreteAsymptoticBranch(
                                    scale=log_branch.scale,
                                    coefficients=log_branch.coefficients,
                                    expression=log_branch.expression,
                                    edge=log_branch.edge,
                                    characteristic_root=log_branch.characteristic_root,
                                    characteristic_mult=log_branch.characteristic_mult,
                                    residual_order=log_measured,
                                    lattice_step=log_branch.lattice_step,
                                    secondary_mult=secondary_multiplicity,
                                    transport_mult=transport_multiplicity,
                                    tertiary_mult=tertiary_multiplicity,
                                )
                            )
    return tuple(branches)


def _apply_recurrence_operator(data: LinearRecurrenceData, expression: sp.Expr) -> sp.Expr:
    """Apply the normalized homogeneous recurrence operator to an expression."""

    n = data.index
    return sp.expand(
        sum(coefficient * expression.subs(n, n + shift) for shift, coefficient in data.coefficients)
    )


def _rational_shift_ratios(
    data: LinearRecurrenceData, expression: sp.Expr
) -> dict[int, sp.Expr] | None:
    """Return rational shifted ratios for one hypergeometric-style expression."""

    expression = sp.sympify(expression)
    if expression == 0 or expression.is_zero is True:
        return None
    n = data.index
    ratios: dict[int, sp.Expr] = {}
    for shift, _coefficient in data.coefficients:
        ratio = expression.subs(n, n + shift) / expression
        if ratio.has(sp.gamma, sp.factorial, sp.rf, sp.binomial):
            try:
                ratio = sp.combsimp(ratio)
            except (TypeError, ValueError, NotImplementedError, sp.PolynomialError):
                pass
        try:
            ratio = sp.cancel(ratio)
        except (TypeError, ValueError, sp.PolynomialError):
            return None
        if ratio.is_rational_function(n) is not True:
            return None
        ratios[shift] = ratio
    return ratios


def _log_resonant_multiplier(
    data: LinearRecurrenceData,
    target: sp.Expr,
    ratios: dict[int, sp.Expr],
    *,
    terms: int,
) -> sp.Expr | None:
    """Lift a simple first-order resonance, including logarithmic corrections.

    A finite ansatz ``n**p*(c_log*log(n) + c0 + c1/n + ...)`` is substituted
    into the reduced multiplier equation.  Its Laurent/log coefficients form a
    linear system.  This covers polynomial antidifferences such as ``Delta y=1``
    as well as harmonic resonance ``Delta y=1/n`` without invoking a general
    recurrence solver.
    """

    if data.order != 1:
        return None
    n = data.index

    def op(multiplier: sp.Expr) -> sp.Expr:
        return sp.cancel(
            sum(
                coefficient * ratios[shift] * multiplier.subs(n, n + shift)
                for shift, coefficient in data.coefficients
            )
        )

    local = sp.Dummy("t", positive=True)
    for power in range(-2, 5):
        log_coeff = sp.Dummy("rlog")
        coeffs = tuple(sp.Dummy(f"r{k}") for k in range(max(2, terms + 1)))
        unknowns = (log_coeff, *coeffs)
        multiplier = n**power * (
            log_coeff * sp.log(n) + sum(coeffs[k] / n**k for k in range(len(coeffs)))
        )
        try:
            transformed = sp.expand(op(multiplier).subs(n, 1 / local) - 1)
            expansion = sp.series(transformed, local, 0, max(terms + abs(power) + 8, 12)).removeO()
            expansion = sp.expand(expansion)
        except (TypeError, ValueError, NotImplementedError, sp.PolynomialError):
            continue

        equations: list[sp.Expr] = []
        # Split by powers of log(t), then by Laurent powers of t.  Because
        # log(n)=-log(t), all equations remain linear in the ansatz coefficients.
        try:
            log_poly = sp.Poly(expansion, sp.log(local))
            log_parts = log_poly.all_coeffs()
        except sp.PolynomialError:
            log_parts = [expansion]
        for part in log_parts:
            grouped: dict[sp.Expr, sp.Expr] = {}
            for term in sp.Add.make_args(sp.expand(part)):
                exponent = term.as_powers_dict().get(local, sp.S.Zero)
                coefficient = sp.expand(term / local**exponent)
                grouped[exponent] = sp.expand(grouped.get(exponent, 0) + coefficient)
            equations.extend(
                value
                for _, value in sorted(
                    grouped.items(), key=lambda item: sp.default_sort_key(item[0])
                )
                if value != 0
            )
        if not equations:
            continue
        # A finite asymptotic ansatz only matches a finite prefix; imposing all
        # generated tail coefficients would incorrectly demand an exact antidifference.
        equations = equations[: max(2, min(len(equations), terms + 1))]
        try:
            matrix, rhs = sp.linear_eq_to_matrix(equations, unknowns)
            solution_set = sp.linsolve((matrix, rhs), unknowns)
        except (TypeError, ValueError, sp.PolynomialError):
            continue
        if solution_set is sp.EmptySet:
            continue
        rows = list(solution_set)
        if len(rows) != 1:
            continue
        row = rows[0]
        # Free ansatz coefficients represent homogeneous additions; set them to
        # zero to select a particular solution deterministically.
        free = set().union(*(value.free_symbols for value in row)) & set(unknowns)
        substitutions = {symbol: sp.S.Zero for symbol in free}
        values = tuple(sp.simplify(value.subs(substitutions)) for value in row)
        result = sp.expand(multiplier.subs(dict(zip(unknowns, values))))
        if result.free_symbols & set(unknowns):
            continue
        # Reject a candidate that failed to match the requested reduced RHS to
        # the computed truncation order.
        try:
            defect = sp.series(op(result).subs(n, 1 / local) - 1, local, 0, max(terms, 2)).removeO()
        except (TypeError, ValueError, NotImplementedError, sp.PolynomialError):
            continue
        if sp.expand(defect) != 0:
            continue
        return result
    return None


def inhomogeneous_particular_solution(
    data: LinearRecurrenceData,
    *,
    terms: int = 6,
) -> tuple[sp.Expr, sp.Expr] | None:
    """Construct a native asymptotic particular solution for simple forcing.

    The primary route writes ``y = g*m(n)``, with ``g=-forcing``, and solves a
    rational linear difference equation for a Laurent multiplier.  It covers
    first-order rational recurrences and simple linear hypergeometric forcing.
    If the multiplier operator is resonant, a first-order logarithmic ansatz is
    tried before returning ``None``.
    """

    if terms < 1:
        raise ValueError("terms must be positive")
    if data.forcing == 0 or data.forcing.is_zero is True:
        return None
    n = data.index
    target = -sp.sympify(data.forcing)
    ratios = _rational_shift_ratios(data, target)
    if ratios is None:
        return None

    def multiplier_operator(multiplier: sp.Expr) -> sp.Expr:
        return sp.cancel(
            sum(
                coefficient * ratios[shift] * multiplier.subs(n, n + shift)
                for shift, coefficient in data.coefficients
            )
        )

    base = multiplier_operator(sp.S.One)
    if base == 0 or base.is_zero is True:
        multiplier = _log_resonant_multiplier(data, target, ratios, terms=terms)
        if multiplier is None:
            return None
        candidate = sp.powsimp(target * multiplier, force=False)
        residual = sp.expand(_apply_recurrence_operator(data, candidate) + data.forcing)
        try:
            residual = sp.cancel(sp.combsimp(residual))
        except (TypeError, ValueError, NotImplementedError, sp.PolynomialError):
            pass
        return candidate, residual

    try:
        base_num, base_den = sp.fraction(sp.cancel(base))
        base_degree = int(sp.Poly(base_num, n).degree()) - int(sp.Poly(base_den, n).degree())
    except (sp.PolynomialError, TypeError, ValueError):
        return None
    leading_power = -base_degree
    local = sp.Dummy("t", positive=True)
    coeffs = tuple(sp.Dummy(f"p{k}") for k in range(max(1, terms)))
    multiplier = sum(coeffs[k] * n ** (leading_power - k) for k in range(len(coeffs)))
    try:
        transformed = multiplier_operator(multiplier).subs(n, 1 / local)
        expansion = sp.series(
            transformed - 1, local, 0, max(terms + abs(leading_power) + 4, 8)
        ).removeO()
    except (TypeError, ValueError, NotImplementedError, sp.PolynomialError):
        return None
    equations: dict[sp.Expr, sp.Expr] = {}
    for term in sp.Add.make_args(sp.expand(expansion)):
        exponent = term.as_powers_dict().get(local, sp.S.Zero)
        coefficient = sp.expand(term / local**exponent)
        equations[exponent] = sp.expand(equations.get(exponent, 0) + coefficient)
    solved: dict[sp.Symbol, sp.Expr] = {}
    for exponent in sorted(equations, key=sp.default_sort_key):
        current = sp.expand(equations[exponent].subs(solved))
        if current == 0 or current.is_zero is True:
            continue
        pending = [symbol for symbol in coeffs if symbol not in solved and current.has(symbol)]
        if not pending:
            if current.is_zero is False:
                break
            continue
        value = _linear_equation_solution(current, pending[0])
        if value is None:
            return None
        solved[pending[0]] = sp.cancel(value.subs(solved))
        if len(solved) == len(coeffs):
            break
    if not solved or coeffs[0] not in solved:
        return None
    multiplier = sp.expand(multiplier.subs({c: solved.get(c, 0) for c in coeffs}))
    candidate = sp.powsimp(target * multiplier, force=False)
    residual = sp.expand(_apply_recurrence_operator(data, candidate) + data.forcing)
    if residual.has(sp.gamma, sp.factorial, sp.rf, sp.binomial):
        try:
            residual = sp.combsimp(residual)
        except (TypeError, ValueError, NotImplementedError, sp.PolynomialError):
            pass
    try:
        residual = sp.cancel(residual)
    except (TypeError, ValueError, sp.PolynomialError):
        pass
    return candidate, residual


def birkhoff_trjitzinsky_branches(
    data: LinearRecurrenceData,
    *,
    terms: int = 6,
) -> tuple[DiscreteAsymptoticBranch, ...]:
    """Lift Newton-edge roots to discrete asymptotic branches.

    Simple roots use ordinary inverse-power Birkhoff--Trjitzinsky lifting.
    Repeated constant-coefficient roots are expanded into their exact polynomial
    Jordan chain. Repeated variable-coefficient roots are passed to a secondary
    Newton analysis that can discover stretched-exponential phases and ramified
    inverse-power lattices.
    """

    if terms < 1:
        raise ValueError("terms must be positive")
    n = data.index
    branches: list[DiscreteAsymptoticBranch] = []
    for edge in discrete_newton_edges(data):
        for root, multiplicity in _edge_roots(edge):
            if multiplicity > 1:
                constant = _constant_coefficient_repeated_branches(data, edge, root, multiplicity)
                if constant:
                    branches.extend(constant)
                    continue
                branches.extend(_ramified_branches(data, edge, root, multiplicity, terms))
                continue

            try:
                theta_symbol, coeff_symbols, equations = _coefficient_equations(
                    data, edge, root, terms
                )
            except NotImplementedError:
                continue
            solved = _solve_lift_equations(theta_symbol, coeff_symbols, equations, terms)
            if solved is None:
                continue
            theta, corrections = solved
            scale = DiscreteAsymptoticScale(
                n,
                factorial_power=edge.factorial_power,
                exponential_base=root,
                power=theta,
            )
            correction = sp.S.One + sum(
                corrections[m - 1] / n**m for m in range(1, len(corrections) + 1)
            )
            expression = sp.powsimp(scale.expression * correction, force=False)
            branch = DiscreteAsymptoticBranch(
                scale=scale,
                coefficients=(sp.S.One, *corrections),
                expression=expression,
                edge=edge,
                characteristic_root=root,
                characteristic_mult=multiplicity,
                residual_order=None,
                lattice_step=sp.S.One,
            )
            measured = _measure_branch_residual(data, branch)
            branches.append(
                DiscreteAsymptoticBranch(
                    scale=branch.scale,
                    coefficients=branch.coefficients,
                    expression=branch.expression,
                    edge=branch.edge,
                    characteristic_root=branch.characteristic_root,
                    characteristic_mult=branch.characteristic_mult,
                    residual_order=measured,
                    lattice_step=branch.lattice_step,
                )
            )
    return tuple(branches)
