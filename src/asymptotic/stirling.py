"""Certified positive-real Stirling and log-Gamma normalization."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from ._power_simplify import analytic_powsimp, power_expand_exact
from ._symbolic_policy import bounded_ask, bounded_limit
from .instrumentation import record_symbolic_event
from .remainder import AsymptoticRemainder


@dataclass(frozen=True)
class StirlingNormalization:
    """A finite Stirling normalization with an explicit positive-real remainder."""

    expression: sp.Expr
    remainder: AsymptoticRemainder
    argument: sp.Expr
    terms: int
    kind: str
    absolute_error_bound: sp.Expr | None = None
    log_error_bound: sp.Expr | None = None
    relative_error_bound: sp.Expr | None = None
    log_expression: sp.Expr | None = None

    @property
    def certified(self) -> bool:
        """Whether the normalization carries a theorem-backed error statement."""

        return self.remainder.is_certified

    def replay(self) -> bool | None:
        """Replay the positivity and explicit-bound obligations of the normalization."""
        if not self.certified:
            return None
        if self.argument.is_positive is False:
            return False
        bounds = (self.absolute_error_bound, self.log_error_bound, self.relative_error_bound)
        return any(bound is not None for bound in bounds)


def _positive_argument(z: sp.Expr, assumptions: sp.Expr = sp.S.true) -> bool:
    if z.is_positive is True or bounded_ask(sp.Q.positive(z), assumptions):
        return True
    # Gamma arguments in PMFs are commonly ``m + 1`` with a support
    # obligation m >= 0; ask() does not always infer this affine implication.
    if z.is_Add:
        for term in z.args:
            if term.is_Integer and term > 0:
                rest = z - term
                if rest.is_nonnegative is True or bounded_ask(sp.Q.nonnegative(rest), assumptions):
                    return True
    return False


def _loggamma_series(z: sp.Expr, terms: int) -> tuple[sp.Expr, sp.Expr]:
    # log Gamma(z) = (z-1/2)log(z)-z+log(2pi)/2
    # + sum B_{2r}/(2r(2r-1)z^(2r-1)) + R_m.
    # For z>0 the classical Stieltjes bound gives
    # |R_m| <= |B_{2m}|/(2m(2m-1) z^(2m-1)).
    base = (z - sp.Rational(1, 2)) * sp.log(z) - z + sp.log(2 * sp.pi) / 2
    corr = sp.S.Zero
    for r in range(1, terms):
        corr += sp.bernoulli(2 * r) / (2 * r * (2 * r - 1) * z ** (2 * r - 1))
    m = max(1, terms)
    bound = abs(sp.bernoulli(2 * m)) / (2 * m * (2 * m - 1) * z ** (2 * m - 1))
    return analytic_powsimp(base + corr), analytic_powsimp(bound)


def certified_loggamma(
    argument: sp.Expr,
    *,
    parameter: sp.Symbol,
    point: sp.Expr = sp.oo,
    terms: int = 3,
    assumptions: sp.Expr = sp.S.true,
) -> StirlingNormalization:
    """Normalize ``loggamma(argument)`` on the certified positive real branch."""

    z = sp.sympify(argument)
    if point is not sp.oo:
        raise NotImplementedError("certified Stirling normalization targets +infinity")
    if terms < 1:
        raise ValueError("terms must be positive")
    if not _positive_argument(z, assumptions):
        raise ValueError("Stirling certification requires a provably positive real argument")
    expression, bound = _loggamma_series(z, terms)
    remainder = AsymptoticRemainder.big_o(
        bound, parameter, point, source="positive-real Stieltjes log-Gamma remainder bound"
    )
    record_symbolic_event("loggamma_normalizations")
    return StirlingNormalization(
        expression,
        remainder,
        z,
        terms,
        "loggamma",
        absolute_error_bound=bound,
        log_expression=expression,
    )


def certified_logfactorial(
    argument: sp.Expr,
    *,
    parameter: sp.Symbol,
    point: sp.Expr = sp.oo,
    terms: int = 3,
    assumptions: sp.Expr = sp.S.true,
) -> StirlingNormalization:
    """Normalize ``log(argument!)`` via ``loggamma(argument + 1)``."""

    n = sp.sympify(argument)
    result = certified_loggamma(
        n + 1,
        parameter=parameter,
        point=point,
        terms=terms,
        assumptions=assumptions,
    )
    record_symbolic_event("factorial_normalizations")
    return StirlingNormalization(
        result.expression,
        result.remainder,
        n,
        terms,
        "logfactorial",
        absolute_error_bound=result.absolute_error_bound,
        log_expression=result.expression,
    )


def normalize_positive_pmf(
    pmf: sp.Expr,
    *,
    variable: sp.Symbol,
    parameter: sp.Symbol,
    point: sp.Expr = sp.oo,
    terms: int = 3,
    assumptions: sp.Expr = sp.S.true,
) -> StirlingNormalization:
    """Rewrite factorial/binomial PMF factors to a certified exponential scale.

    The input must be provably positive. Factorials and binomial coefficients
    are converted through log-Gamma. The returned exponential uses only
    positive-real logarithms; no PowerExpand-like branch assumptions are made.
    """

    expr = sp.sympify(pmf)

    # Convert binomial and factorial to Gamma ratios before taking the log.
    expanded = expr.replace(
        lambda e: e.func == sp.binomial and len(e.args) == 2,
        lambda e: (
            sp.gamma(e.args[0] + 1)
            / (sp.gamma(e.args[1] + 1) * sp.gamma(e.args[0] - e.args[1] + 1))
        ),
    )
    expanded = expanded.replace(
        lambda e: e.func == sp.factorial and len(e.args) == 1,
        lambda e: sp.gamma(e.args[0] + 1),
    )

    # Build the logarithm multiplicatively so Gamma factors remain isolated.
    factors = sp.Mul.make_args(expanded)
    log_terms: list[sp.Expr] = []
    bounds: list[sp.Expr] = []
    for factor in factors:
        base, exponent = factor.as_base_exp()
        if base.func == sp.gamma:
            norm = certified_loggamma(
                base.args[0], parameter=parameter, point=point, terms=terms, assumptions=assumptions
            )
            log_terms.append(exponent * norm.expression)
            bounds.append(abs(exponent) * norm.remainder.scale)
        elif base.is_positive is True or bounded_ask(sp.Q.positive(base), assumptions):
            log_terms.append(exponent * sp.log(base))
        elif not factor.has(variable, parameter):
            # Parameter-independent positive constants may be accepted from
            # SymPy assumptions; otherwise branch safety wins over convenience.
            if factor.is_positive is not True:
                raise ValueError("PMF normalization requires provably positive factors")
            log_terms.append(sp.log(factor))
        else:
            raise ValueError("PMF normalization requires provably positive factors")
    normalized_log = analytic_powsimp(sp.Add(*log_terms))
    if normalized_log.has(sp.gamma, sp.factorial, sp.binomial):
        raise NotImplementedError(
            "PMF contains Gamma/factorial structure not safely isolated in logarithmic form"
        )

    normalized = sp.exp(normalized_log)
    # If the accumulated log error satisfies |Delta| <= B, then the exact PMF
    # is normalized*exp(Delta) and therefore has the explicit relative bound
    # |exact/normalized - 1| <= exp(B)-1.  Recording that bound is stronger
    # than merely saying O(B), especially before a lattice scaling such as
    # k=n*x has established that B actually tends to zero.
    log_bound = analytic_powsimp(sp.Add(*bounds)) if bounds else sp.S.Zero
    relative_bound = analytic_powsimp(sp.exp(log_bound) - 1)
    if relative_bound.is_zero is True:
        remainder = AsymptoticRemainder.exact_zero(
            parameter, point, source="exact positive PMF logarithmic normalization"
        )
    else:
        remainder = AsymptoticRemainder.big_o(
            relative_bound,
            parameter,
            point,
            source="positive-real Stirling PMF relative-error bound",
        )
    record_symbolic_event("pmf_normalizations")
    return StirlingNormalization(
        normalized,
        remainder,
        expanded,
        terms,
        "positive-pmf",
        log_error_bound=log_bound,
        relative_error_bound=relative_bound,
        log_expression=normalized_log,
    )


@dataclass(frozen=True)
class ScaledStirlingLatticeForm:
    """Scaled exponential form obtained from a certified PMF normalization.

    ``phase`` is independent of the asymptotic parameter and the original
    lattice summand is represented, after ``variable = parameter*x``, as
    ``amplitude * exp(-parameter*phase)`` to the requested algebraic order.
    The extra factor ``parameter`` converting the lattice sum to an interior
    Riemann/Laplace integral is already included in ``amplitude``.
    """

    variable: sp.Symbol
    phase: sp.Expr
    amplitude: sp.Expr
    log_series: sp.Expr
    assumptions: sp.Expr
    normalization: StirlingNormalization


def _open_interval_assumptions(domain: sp.Interval, variable: sp.Symbol) -> sp.Expr:
    pieces: list[sp.Expr] = []
    if domain.start is not -sp.oo:
        pieces.append(sp.Q.positive(variable - domain.start))
    if domain.end is not sp.oo:
        pieces.append(sp.Q.positive(domain.end - variable))
    return sp.And(*pieces) if pieces else sp.S.true


def scaled_stirling_lattice_form(
    normalization: StirlingNormalization,
    *,
    variable: sp.Symbol,
    parameter: sp.Symbol,
    domain: sp.Interval,
    terms: int = 4,
) -> ScaledStirlingLatticeForm:
    """Expose the entropy phase of a positive Stirling-normalized lattice PMF.

    This routine avoids a generic ``series(..., parameter, oo)`` on the full
    PMF.  It changes to ``h=1/parameter`` first, applies branch-correct logarithm
    identities on the open scaled support, and expands the *log PMF*.  That is
    both faster and more robust for the cancellations of ``n*log(n)`` that
    occur in Binomial/Poisson-type masses.
    """

    if normalization.log_expression is None:
        raise ValueError("Stirling normalization does not retain its logarithmic form")
    if terms < 1:
        raise ValueError("terms must be positive")
    x = sp.Dummy(f"{variable}_scaled", real=True)
    h = sp.Dummy("_lattice_h", positive=True)
    assumptions = _open_interval_assumptions(domain, x)
    scaled_log = normalization.log_expression.xreplace({variable: parameter * x})
    scaled_log = scaled_log.xreplace({parameter: 1 / h})
    scaled_log = power_expand_exact(scaled_log, assumptions & sp.Q.positive(h))
    try:
        log_series = sp.series(scaled_log, h, 0, max(3, terms + 2)).removeO()
    except (ValueError, TypeError, NotImplementedError) as exc:
        raise NotImplementedError("could not expand the scaled logarithmic PMF") from exc
    log_series = sp.expand(log_series)
    slope = analytic_powsimp(log_series.coeff(h, -1))
    if slope == 0 or h in slope.free_symbols:
        raise NotImplementedError("scaled PMF does not expose a nonzero order-parameter phase")
    phase = analytic_powsimp(-slope)
    offset = analytic_powsimp(log_series - slope / h)

    log_h = sp.log(h)
    log_power = analytic_powsimp(sp.expand(offset).coeff(log_h))
    regular = analytic_powsimp(offset - log_power * log_h)
    # Exponentiating the regular h-series after the entropy term has been
    # removed is cheap; unlike exponentiating the original log PMF it cannot
    # recreate the cancelled n*log(n) structure.
    try:
        raw_regular_exp = sp.series(sp.exp(regular), h, 0, max(2, terms + 1)).removeO()
    except (ValueError, TypeError, NotImplementedError) as exc:
        raise NotImplementedError("could not expand the scaled Stirling amplitude") from exc
    regular_exp = sp.S.Zero
    for power in range(max(2, terms + 1)):
        coefficient = sp.expand(raw_regular_exp).coeff(h, power)
        if coefficient == 0:
            continue
        try:
            coefficient = sp.factor(sp.cancel(coefficient))
        except (ValueError, TypeError, NotImplementedError, sp.PolynomialError):
            coefficient = analytic_powsimp(coefficient)
        regular_exp += coefficient * h**power
    amplitude_h = analytic_powsimp(h**log_power * regular_exp / h)
    amplitude = analytic_powsimp(amplitude_h.xreplace({h: 1 / parameter}))
    return ScaledStirlingLatticeForm(x, phase, amplitude, log_series, assumptions, normalization)


@dataclass(frozen=True)
class StirlingLocalMassExpansion:
    """Pointwise PMF expansion at ``location = parameter*scale + O(1)``."""

    expression: sp.Expr
    log_series: sp.Expr
    scale: sp.Expr
    lattice_offset: sp.Expr
    rate: sp.Expr
    log_prefactor: sp.Expr
    truncation_scale: sp.Expr
    normalization: StirlingNormalization


def _bounded_lattice_offset(location: sp.Expr, scale: sp.Expr, parameter: sp.Symbol) -> bool:
    offset = analytic_powsimp(location - parameter * scale)
    if parameter not in offset.free_symbols:
        return not offset.has(sp.oo, -sp.oo, sp.zoo, sp.nan)
    if location.func in (sp.floor, sp.ceiling):
        base_offset = analytic_powsimp(location.args[0] - parameter * scale)
        if parameter not in base_offset.free_symbols:
            return True
        try:
            base_limit = bounded_limit(sp.Abs(base_offset), parameter, sp.oo, allow_general=True)
        except (ValueError, TypeError, NotImplementedError):
            base_limit = None
        if base_limit is not None and getattr(base_limit, "is_finite", None) is True:
            return True
    try:
        offset_limit = bounded_limit(sp.Abs(offset), parameter, sp.oo, allow_general=True)
    except (ValueError, TypeError, NotImplementedError):
        return False
    return getattr(offset_limit, "is_finite", None) is True


def stirling_local_mass_expansion(
    normalization: StirlingNormalization,
    *,
    location: sp.Expr,
    variable: sp.Symbol,
    parameter: sp.Symbol,
    terms: int = 4,
) -> StirlingLocalMassExpansion:
    """Expand a normalized lattice mass while retaining bounded rounding shifts.

    The bounded offset is temporarily replaced by an auxiliary real symbol.
    This permits locations such as ``floor(a*n)`` whose fractional-part
    correction has no ordinary limit, and substitutes the exact correction
    back only after the ``1/n`` expansion is formed.
    """

    if normalization.log_expression is None:
        raise ValueError("Stirling normalization does not retain its logarithmic form")
    location = sp.sympify(location)
    try:
        scale = bounded_limit(location / parameter, parameter, sp.oo, allow_general=True)
    except (ValueError, TypeError, NotImplementedError) as exc:
        raise NotImplementedError("could not determine the moving lattice scale") from exc
    if isinstance(scale, sp.Limit) or scale in (sp.oo, -sp.oo, sp.zoo, sp.nan):
        raise NotImplementedError("moving lattice point is not linear-scale")
    offset = analytic_powsimp(location - parameter * scale)
    if not _bounded_lattice_offset(location, scale, parameter):
        raise NotImplementedError("lattice displacement is not bounded")

    h = sp.Dummy("_local_mass_h", positive=True)
    delta = sp.Dummy("_lattice_delta", real=True)
    local_log = normalization.log_expression.xreplace(
        {variable: scale / h + delta, parameter: 1 / h}
    )
    local_log = power_expand_exact(local_log, sp.Q.positive(h))
    try:
        log_series = sp.expand(sp.series(local_log, h, 0, max(3, terms + 2)).removeO())
    except (ValueError, TypeError, NotImplementedError) as exc:
        raise NotImplementedError("could not form the local Stirling log-mass expansion") from exc
    slope = analytic_powsimp(log_series.coeff(h, -1))
    rest = analytic_powsimp(sp.expand(log_series - slope / h))
    log_h = sp.log(h)
    log_power = analytic_powsimp(sp.expand(rest).coeff(log_h))
    regular = analytic_powsimp(rest - log_power * log_h)
    try:
        regular_exp = sp.series(sp.exp(regular), h, 0, max(2, terms + 1)).removeO()
    except (ValueError, TypeError, NotImplementedError) as exc:
        raise NotImplementedError("could not exponentiate the local Stirling correction") from exc
    regular_exp = sp.Add(
        *(
            sp.factor(sp.cancel(sp.expand(regular_exp).coeff(h, power))) * h**power
            for power in range(max(2, terms + 1))
            if sp.expand(regular_exp).coeff(h, power) != 0
        )
    )
    expression = analytic_powsimp(
        sp.exp(parameter * slope)
        * parameter ** (-log_power)
        * regular_exp.xreplace({h: 1 / parameter})
    )
    expression = expression.xreplace({delta: offset})
    log_prefactor = rest.xreplace({delta: offset, h: 1 / parameter})
    log_prefactor = log_prefactor.xreplace({sp.log(1 / parameter): -sp.log(parameter)})
    log_series = log_series.xreplace({delta: offset, h: 1 / parameter})
    return StirlingLocalMassExpansion(
        expression,
        log_series,
        scale,
        offset,
        analytic_powsimp(-slope),
        log_prefactor,
        parameter ** (-terms),
        normalization,
    )


def stirling_sqrt_local_mass_expansion(
    normalization: StirlingNormalization,
    *,
    location: sp.Expr,
    variable: sp.Symbol,
    parameter: sp.Symbol,
    terms: int = 4,
) -> StirlingLocalMassExpansion:
    """Expand a local mass with an ``O(sqrt(parameter))`` displacement.

    This is the natural local-limit/moderate-deviation scale.  A top-level
    ``floor`` or ``ceiling`` is split into its smooth moving center plus a
    bounded rounding variable before the ``parameter**(-1/2)`` expansion, so
    lattice parity information is retained without treating it as smooth.
    """

    if normalization.log_expression is None:
        raise ValueError("Stirling normalization does not retain its logarithmic form")
    if terms < 1:
        raise ValueError("terms must be positive")
    location = sp.sympify(location)
    if location.func in (sp.floor, sp.ceiling):
        smooth_location = location.args[0]
        rounding = analytic_powsimp(location - smooth_location)
    else:
        smooth_location = location
        rounding = sp.S.Zero
    try:
        scale = bounded_limit(smooth_location / parameter, parameter, sp.oo, allow_general=True)
    except (ValueError, TypeError, NotImplementedError) as exc:
        raise NotImplementedError("could not determine the moving lattice scale") from exc
    if isinstance(scale, sp.Limit) or scale in (sp.oo, -sp.oo, sp.zoo, sp.nan):
        raise NotImplementedError("moving lattice point is not linear-scale")
    smooth_offset = analytic_powsimp(smooth_location - parameter * scale)
    offset = analytic_powsimp(location - parameter * scale)
    if rounding == 0 and _bounded_lattice_offset(location, scale, parameter):
        raise NotImplementedError("bounded shifts belong to stirling_local_mass_expansion")
    try:
        sqrt_shift = bounded_limit(
            smooth_offset / sp.sqrt(parameter), parameter, sp.oo, allow_general=True
        )
    except (ValueError, TypeError, NotImplementedError) as exc:
        raise NotImplementedError("could not determine square-root displacement") from exc
    if isinstance(sqrt_shift, sp.Limit) or sqrt_shift in (sp.oo, -sp.oo, sp.zoo, sp.nan):
        raise NotImplementedError("local displacement grows faster than sqrt(parameter)")
    eps = sp.Dummy("_local_mass_eps", positive=True)
    delta = sp.Dummy("_lattice_rounding", real=True)
    smooth_eps = smooth_location.xreplace({parameter: eps**-2})
    local_position = smooth_eps + (delta if rounding != 0 else 0)
    local_log = normalization.log_expression.xreplace({parameter: eps**-2})
    local_log = local_log.xreplace({variable: local_position})
    local_log = power_expand_exact(local_log, sp.Q.positive(eps))
    expansion_order = max(6, 2 * terms + 4)
    try:
        log_series = sp.expand(sp.series(local_log, eps, 0, expansion_order).removeO())
    except (ValueError, TypeError, NotImplementedError) as exc:
        raise NotImplementedError("could not form square-root local Stirling expansion") from exc

    slope = analytic_powsimp(log_series.coeff(eps, -2))
    rest = analytic_powsimp(sp.expand(log_series - slope / eps**2))
    singular = analytic_powsimp(rest.coeff(eps, -1) / eps)
    regular_with_log = analytic_powsimp(sp.expand(rest - singular))
    log_eps = sp.log(eps)
    log_power = analytic_powsimp(sp.expand(regular_with_log).coeff(log_eps))
    regular = analytic_powsimp(regular_with_log - log_power * log_eps)
    try:
        raw_regular_exp = sp.series(sp.exp(regular), eps, 0, max(3, 2 * terms + 1)).removeO()
    except (ValueError, TypeError, NotImplementedError) as exc:
        raise NotImplementedError("could not exponentiate square-root local correction") from exc
    regular_exp = sp.S.Zero
    expanded_exp = sp.expand(raw_regular_exp)
    for power in range(max(3, 2 * terms + 1)):
        coefficient = expanded_exp.coeff(eps, power)
        if coefficient == 0:
            continue
        try:
            coefficient = sp.factor(sp.cancel(coefficient))
        except (ValueError, TypeError, NotImplementedError, sp.PolynomialError):
            coefficient = analytic_powsimp(coefficient)
        regular_exp += coefficient * eps**power

    substitutions = {eps: parameter ** sp.Rational(-1, 2)}
    if rounding != 0:
        substitutions[delta] = rounding
    singular_parameter = singular.xreplace(substitutions)
    regular_parameter = regular_exp.xreplace(substitutions)
    expression = analytic_powsimp(
        sp.exp(parameter * slope)
        * sp.exp(singular_parameter)
        * parameter ** (-log_power / 2)
        * regular_parameter
    )
    log_prefactor = rest.xreplace(substitutions)
    log_prefactor = log_prefactor.xreplace(
        {sp.log(parameter ** sp.Rational(-1, 2)): -sp.log(parameter) / 2}
    )
    final_log_series = log_series.xreplace(substitutions)
    return StirlingLocalMassExpansion(
        expression,
        final_log_series,
        scale,
        offset,
        analytic_powsimp(-slope),
        log_prefactor,
        parameter ** (-sp.Rational(terms, 2)),
        normalization,
    )
