"""High-level statistical asymptotic transforms built from probability primitives."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp
from sympy.stats import density
from sympy.stats import quantile as exact_quantile
from sympy.stats.rv import RandomSymbol

from ._power_simplify import analytic_powsimp
from ._symbolic_policy import bounded_ask, bounded_limit, bounded_solve_one
from .context import AsymptoticContext
from .probability import (
    StatisticalAsymptoticResult,
    _restrict_piecewise_to_support,
    _support,
    _support_assumptions,
    asymptotic_expectation,
    asymptotic_probability,
)
from .remainder import AsymptoticRemainder
from .roots import asymptotic_root
from .stirling import (
    normalize_positive_pmf,
    stirling_local_mass_expansion,
    stirling_sqrt_local_mass_expansion,
)
from .sums import asymptotic_sum
from .transseries import transseries_from_expression


@dataclass(frozen=True)
class StatisticalTransformResult:
    expression: sp.Expr
    parameter: sp.Symbol
    point: sp.Expr
    method: str
    status: str
    sources: tuple[object, ...] = ()
    conditions: tuple[sp.Expr, ...] = ()
    remainder: AsymptoticRemainder | None = None

    @property
    def certified(self) -> bool:
        return self.status in {"EXACT", "CERTIFIED"}

    def truncate(self, terms: int | None = None) -> sp.Expr:
        return self.expression


@dataclass(frozen=True)
class LogProbabilityResult:
    """Logarithmic probability asymptotic retaining rate and prefactor pieces."""

    expression: sp.Expr
    rate: sp.Expr | None
    log_prefactor: sp.Expr | None
    parameter: sp.Symbol
    point: sp.Expr
    status: str
    probability: StatisticalAsymptoticResult

    @property
    def certified(self) -> bool:
        return self.status in {"EXACT", "CERTIFIED"}


@dataclass(frozen=True)
class AsymptoticModeResult:
    """Continuous saddle and lattice-corrected mode information."""

    expression: sp.Expr
    continuous_location: sp.Expr
    lattice_candidates: tuple[sp.Expr, ...]
    rounding_correction: sp.Expr
    parameter: sp.Symbol
    point: sp.Expr
    status: str
    method: str
    conditions: tuple[sp.Expr, ...] = ()

    @property
    def certified(self) -> bool:
        return self.status in {"EXACT", "CERTIFIED"}


def _expr(result: StatisticalAsymptoticResult) -> sp.Expr:
    # Exact special functions must not be run through an inappropriate
    # Poincare truncation merely because the threshold remains symbolic.
    return result.expression if result.status == "EXACT" else result.truncate()


def _status(*results: object) -> str:
    statuses = [getattr(r, "status", "FORMAL") for r in results]
    if statuses and all(s == "EXACT" for s in statuses):
        return "EXACT"
    if statuses and all(s in {"EXACT", "CERTIFIED"} for s in statuses):
        return "CERTIFIED"
    if any(s == "UNKNOWN" for s in statuses):
        return "UNKNOWN"
    return "FORMAL"


def _guard_terms(terms: int) -> int:
    """Return a small composition guard for cancellation-sensitive transforms."""
    if terms < 1:
        raise ValueError("terms must be positive")
    return terms + max(2, terms // 2)


def _truncate_composite(
    expression: sp.Expr, parameter: sp.Symbol, point: sp.Expr, terms: int
) -> sp.Expr:
    """Truncate a composed statistical expression only after cancellation."""
    expression = sp.expand(sp.sympify(expression))
    if parameter not in expression.free_symbols:
        return expression
    try:
        return sp.series(expression, parameter, point, terms).removeO()
    except (ValueError, TypeError, NotImplementedError):
        return expression


def _support_subset_condition(source: sp.Set, target: sp.Set) -> tuple[bool | None, sp.Expr]:
    """Decide support containment and return its exact set-theoretic obligation."""
    decision = source.is_subset(target)
    if decision in (True, False):
        return decision, sp.S.true if decision else sp.S.false
    missing = source - target
    condition = sp.Eq(missing, sp.S.EmptySet, evaluate=False)
    return None, condition


def asymptotic_moment(
    expr: sp.Expr,
    random_symbol: RandomSymbol | None = None,
    *,
    order: int,
    parameter: sp.Symbol,
    point: sp.Expr = sp.oo,
    terms: int = 4,
    central: bool = False,
) -> StatisticalTransformResult:
    """Return a raw or central moment using the asymptotic expectation pipeline."""
    if not isinstance(order, int) or order < 0:
        raise ValueError("order must be a nonnegative integer")
    if central:
        work_terms = _guard_terms(terms)
        mean = asymptotic_expectation(
            expr, random_symbol, parameter=parameter, point=point, terms=work_terms
        )
        centered = (sp.sympify(expr) - _expr(mean)) ** order
        moment = asymptotic_expectation(
            centered, random_symbol, parameter=parameter, point=point, terms=work_terms
        )
        value = (
            _expr(moment)
            if moment.status == "EXACT"
            else _truncate_composite(_expr(moment), parameter, point, terms)
        )
        return StatisticalTransformResult(
            value, parameter, point, "central-moment", _status(mean, moment), (mean, moment)
        )
    moment = asymptotic_expectation(
        sp.sympify(expr) ** order, random_symbol, parameter=parameter, point=point, terms=terms
    )
    return StatisticalTransformResult(
        _expr(moment), parameter, point, "raw-moment", moment.status, (moment,)
    )


def asymptotic_variance(
    expr: sp.Expr,
    random_symbol: RandomSymbol | None = None,
    *,
    parameter: sp.Symbol,
    point: sp.Expr = sp.oo,
    terms: int = 4,
) -> StatisticalTransformResult:
    """Compute variance through a guarded centered expectation.

    Centering before the final expectation avoids loss of requested precision
    when large raw moments cancel. Exact centered results are never truncated.
    """
    work_terms = _guard_terms(terms)
    mean = asymptotic_expectation(
        expr, random_symbol, parameter=parameter, point=point, terms=work_terms
    )
    centered = (sp.sympify(expr) - _expr(mean)) ** 2
    variance = asymptotic_expectation(
        centered, random_symbol, parameter=parameter, point=point, terms=work_terms
    )
    out = (
        _expr(variance)
        if variance.status == "EXACT"
        else _truncate_composite(_expr(variance), parameter, point, terms)
    )
    return StatisticalTransformResult(
        out,
        parameter,
        point,
        "variance-centered-expectation",
        _status(mean, variance),
        (mean, variance),
    )


def asymptotic_covariance(
    left: sp.Expr, right: sp.Expr, *, parameter: sp.Symbol, point: sp.Expr = sp.oo, terms: int = 4
) -> StatisticalTransformResult:
    """Compute covariance through guarded centered expectations.

    Marginal means are evaluated with guard terms and cancellation occurs
    before the final truncation, avoiding subtraction of large raw moments.
    """
    work_terms = _guard_terms(terms)
    ml = asymptotic_expectation(left, parameter=parameter, point=point, terms=work_terms)
    mr = asymptotic_expectation(right, parameter=parameter, point=point, terms=work_terms)
    centered = (sp.sympify(left) - _expr(ml)) * (sp.sympify(right) - _expr(mr))
    covariance = asymptotic_expectation(
        centered, parameter=parameter, point=point, terms=work_terms
    )
    out = (
        _expr(covariance)
        if covariance.status == "EXACT"
        else _truncate_composite(_expr(covariance), parameter, point, terms)
    )
    return StatisticalTransformResult(
        out,
        parameter,
        point,
        "covariance-centered-expectation",
        _status(ml, mr, covariance),
        (ml, mr, covariance),
    )


def asymptotic_mgf(
    expr: sp.Expr,
    random_symbol: RandomSymbol | None = None,
    *,
    transform_variable: sp.Symbol,
    parameter: sp.Symbol,
    point: sp.Expr = sp.oo,
    terms: int = 4,
) -> StatisticalTransformResult:
    """Compute the asymptotic moment-generating function by expectation reduction."""
    r = asymptotic_expectation(
        sp.exp(transform_variable * sp.sympify(expr)),
        random_symbol,
        parameter=parameter,
        point=point,
        terms=terms,
    )
    return StatisticalTransformResult(_expr(r), parameter, point, "mgf", r.status, (r,))


def asymptotic_characteristic_function(
    expr: sp.Expr,
    random_symbol: RandomSymbol | None = None,
    *,
    transform_variable: sp.Symbol,
    parameter: sp.Symbol,
    point: sp.Expr = sp.oo,
    terms: int = 4,
) -> StatisticalTransformResult:
    """Compute the asymptotic characteristic function through complex expectation."""
    r = asymptotic_expectation(
        sp.exp(sp.I * transform_variable * sp.sympify(expr)),
        random_symbol,
        parameter=parameter,
        point=point,
        terms=terms,
    )
    return StatisticalTransformResult(
        _expr(r), parameter, point, "characteristic-function", r.status, (r,)
    )


def asymptotic_cgf(
    expr: sp.Expr,
    random_symbol: RandomSymbol | None = None,
    *,
    transform_variable: sp.Symbol,
    parameter: sp.Symbol,
    point: sp.Expr = sp.oo,
    terms: int = 4,
) -> StatisticalTransformResult:
    """Compute the cumulant-generating function from a branch-safe MGF result."""
    mgf = asymptotic_mgf(
        expr,
        random_symbol,
        transform_variable=transform_variable,
        parameter=parameter,
        point=point,
        terms=terms,
    )
    exact_value = sp.log(mgf.expression)
    value = exact_value
    truncated = False
    try:
        candidate = sp.series(value, parameter, point, terms).removeO()
        truncated = sp.simplify(candidate - exact_value) != 0
        value = candidate
    except (ValueError, TypeError, NotImplementedError):
        pass
    status = "FORMAL" if truncated and mgf.status != "UNKNOWN" else mgf.status
    return StatisticalTransformResult(value, parameter, point, "cgf-from-mgf", status, (mgf,))


def asymptotic_cumulant(
    expr: sp.Expr,
    random_symbol: RandomSymbol | None = None,
    *,
    order: int,
    parameter: sp.Symbol,
    transform_variable: sp.Symbol | None = None,
    point: sp.Expr = sp.oo,
    terms: int = 4,
) -> StatisticalTransformResult:
    """Extract an asymptotic cumulant by differentiating the asymptotic CGF."""
    if not isinstance(order, int) or order < 1:
        raise ValueError("order must be a positive integer")
    t = transform_variable or sp.Dummy("t", real=True)
    cgf = asymptotic_cgf(
        expr, random_symbol, transform_variable=t, parameter=parameter, point=point, terms=terms
    )
    value = sp.diff(cgf.expression, t, order).subs(t, 0)
    return StatisticalTransformResult(
        sp.simplify(value), parameter, point, f"cumulant-{order}", cgf.status, (cgf,)
    )


def asymptotic_cdf(
    random_symbol: RandomSymbol,
    threshold: sp.Expr,
    *,
    parameter: sp.Symbol,
    point: sp.Expr = sp.oo,
    terms: int = 4,
    method: str = "auto",
) -> StatisticalTransformResult:
    """Compute an asymptotic cumulative distribution function at a threshold."""
    r = asymptotic_probability(
        random_symbol <= threshold,
        random_symbol,
        parameter=parameter,
        point=point,
        terms=terms,
        method=method,
    )
    return StatisticalTransformResult(_expr(r), parameter, point, "cdf", r.status, (r,))


def asymptotic_survival(
    random_symbol: RandomSymbol,
    threshold: sp.Expr,
    *,
    parameter: sp.Symbol,
    point: sp.Expr = sp.oo,
    terms: int = 4,
    method: str = "auto",
) -> StatisticalTransformResult:
    """Compute an asymptotic survival function for a one-sided upper-tail event."""
    r = asymptotic_probability(
        random_symbol > threshold,
        random_symbol,
        parameter=parameter,
        point=point,
        terms=terms,
        method=method,
    )
    return StatisticalTransformResult(_expr(r), parameter, point, "survival", r.status, (r,))


def asymptotic_quantile(
    random_symbol: RandomSymbol,
    probability: sp.Expr,
    *,
    parameter: sp.Symbol,
    quantile_variable: sp.Symbol | None = None,
    point: sp.Expr = sp.oo,
    terms: int = 4,
) -> StatisticalTransformResult:
    """Return the generalized-inverse quantile ``inf{x: F(x) >= p}``.

    Exact distribution quantiles are preferred because they preserve the
    generalized-inverse convention on atoms.  Equality-based inversion is
    used only for non-lattice distributions, where a monotone continuous CDF
    may be inverted locally without changing the quantile definition.
    """
    probability = sp.sympify(probability)
    if probability.is_real is False:
        raise ValueError("probability must be real")
    if probability.is_number and not (0 <= probability <= 1):
        raise ValueError("probability must lie in [0, 1]")

    try:
        exact = exact_quantile(random_symbol)(probability)
    except (ValueError, TypeError, NotImplementedError):
        exact = None
    if exact is not None and not sp.sympify(exact).has(sp.nan):
        return StatisticalTransformResult(
            sp.sympify(exact), parameter, point, "generalized-inverse-quantile", "EXACT"
        )

    support = _support(random_symbol)
    if support.is_subset(sp.S.Integers) is True:
        return StatisticalTransformResult(
            sp.Function("QuantileAsymptotic")(probability),
            parameter,
            point,
            "generalized-inverse-quantile",
            "UNKNOWN",
        )

    q = quantile_variable or sp.Dummy("q", real=True)
    cdf = asymptotic_cdf(random_symbol, q, parameter=parameter, point=point, terms=terms)
    sols = bounded_solve_one(sp.Eq(cdf.expression, probability), q, allow_general=True) or ()
    real_sols = [sol for sol in sols if bounded_ask(sp.Q.real(sol)) is not False]
    if len(real_sols) == 1:
        return StatisticalTransformResult(
            sp.sympify(real_sols[0]), parameter, point, "quantile-inversion", cdf.status, (cdf,)
        )
    try:
        root = asymptotic_root(
            cdf.expression - probability,
            q,
            parameter=parameter,
            point=point,
            terms=terms,
            domain=sp.S.Reals,
            branch=0,
        )
        return StatisticalTransformResult(
            sp.sympify(root),
            parameter,
            point,
            "quantile-asymptotic-root",
            "FORMAL",
            (cdf,),
        )
    except (ValueError, TypeError, NotImplementedError, IndexError):
        return StatisticalTransformResult(
            sp.Function("QuantileAsymptotic")(probability),
            parameter,
            point,
            "quantile-inversion",
            "UNKNOWN",
            (cdf,),
        )


def asymptotic_rate_function(
    probability_result: StatisticalAsymptoticResult
    | StatisticalTransformResult
    | LogProbabilityResult,
    *,
    parameter: sp.Symbol,
    speed: sp.Expr | None = None,
    point: sp.Expr = sp.oo,
) -> StatisticalTransformResult:
    """Extract the leading large-deviation rate from a probability asymptotic.

    ``LogProbabilityResult`` carries the rate explicitly, so that route avoids
    taking a logarithm of a truncated tiny probability and preserves lattice
    prefactors separately from the exponential speed.
    """
    if isinstance(probability_result, LogProbabilityResult) and probability_result.rate is not None:
        return StatisticalTransformResult(
            sp.sympify(probability_result.rate),
            parameter,
            point,
            "large-deviation-rate/log-probability",
            probability_result.status,
            (probability_result,),
        )
    v = sp.sympify(probability_result.expression)
    s = sp.sympify(speed if speed is not None else parameter)
    rate_expr = -sp.log(v) / s
    rate = bounded_limit(rate_expr, parameter, point, allow_general=True)
    if rate is None:
        rate = rate_expr
    status = (
        "CERTIFIED"
        if getattr(probability_result, "certified", False) and not rate.has(sp.Limit)
        else "FORMAL"
    )
    return StatisticalTransformResult(
        rate, parameter, point, "large-deviation-rate", status, (probability_result,)
    )


def asymptotic_product(
    factor: sp.Expr,
    variable: sp.Symbol,
    lower: sp.Expr,
    upper: sp.Expr,
    *,
    parameter: sp.Symbol,
    point: sp.Expr = sp.oo,
    terms: int = 4,
    assumptions: sp.Expr = sp.S.true,
    method: str = "auto",
) -> StatisticalTransformResult:
    """Compute a positive asymptotic product by reducing its logarithm to a sum."""
    factor = sp.sympify(factor)
    if not (factor.is_positive is True or bounded_ask(sp.Q.positive(factor), assumptions)):
        raise ValueError(
            "asymptotic_product requires a provably positive factor for branch-safe logarithms"
        )
    summed = asymptotic_sum(
        sp.log(factor),
        variable,
        lower,
        upper,
        parameter=parameter,
        point=point,
        terms=terms,
        method=method,
    )
    value = sp.exp(summed.expression)
    return StatisticalTransformResult(
        value,
        parameter,
        point,
        "product-via-log-sum",
        summed.status,
        (summed,),
        remainder=summed.remainder,
    )


def _density_on_support(random_symbol: RandomSymbol, variable: sp.Expr) -> sp.Expr:
    raw = sp.sympify(density(random_symbol)(variable))
    if isinstance(raw, sp.Piecewise):
        nonzero = [value for value, _condition in raw.args if value != 0]
        if nonzero:
            return sp.sympify(nonzero[0])
    return raw


def _log_positive_asymptotic(expression: sp.Expr, parameter: sp.Symbol, terms: int) -> sp.Expr:
    """Take a logarithm while extracting manifest positive exponential factors."""

    expression = sp.sympify(expression)
    pieces: list[sp.Expr] = []
    residual = sp.S.One
    for factor in sp.Mul.make_args(expression):
        if factor.func is sp.exp and factor.args[0].is_real is not False:
            pieces.append(factor.args[0])
            continue
        base, exponent = factor.as_base_exp()
        if base == parameter and exponent.is_real is True:
            pieces.append(exponent * sp.log(parameter))
            continue
        if factor.is_positive is True:
            pieces.append(sp.log(factor))
            continue
        residual *= factor
    if residual != 1:
        pieces.append(sp.log(residual))
    value = sp.Add(*pieces) if pieces else sp.log(expression)
    try:
        value = sp.series(value, parameter, sp.oo, max(2, terms)).removeO()
    except (ValueError, TypeError, NotImplementedError):
        pass
    return sp.expand(value)


def asymptotic_log_probability(
    event: sp.Expr,
    random_symbol: RandomSymbol | None = None,
    *,
    parameter: sp.Symbol,
    point: sp.Expr = sp.oo,
    terms: int = 4,
    method: str = "auto",
) -> LogProbabilityResult:
    """Return ``log P(event)`` without first numerically underflowing a tiny tail.

    Besides the logarithmic expansion, the result records the leading
    large-deviation rate ``-log(P)/parameter`` and the residual log-prefactor.
    """

    probability = asymptotic_probability(
        event, random_symbol, parameter=parameter, point=point, terms=terms, method=method
    )
    lattice_tail = probability.normalization
    rate = None
    log_prefactor = None
    if type(lattice_tail).__name__ == "BinomialLatticeTailExpansion":
        local_log = lattice_tail.local_mass.log_series
        try:
            factor_log = sp.series(
                sp.log(lattice_tail.lattice_factor), parameter, sp.oo, max(2, terms)
            ).removeO()
        except (ValueError, TypeError, NotImplementedError):
            factor_log = sp.log(lattice_tail.lattice_factor)
        value = analytic_powsimp(local_log + factor_log)
        if point is sp.oo:
            value = value.xreplace({sp.log(1 / parameter): -sp.log(parameter)})
            rate = lattice_tail.local_mass.rate
            log_prefactor = analytic_powsimp(lattice_tail.local_mass.log_prefactor + factor_log)
    else:
        value = _log_positive_asymptotic(_expr(probability), parameter, terms)
        if point is sp.oo:
            candidate = bounded_limit(-value / parameter, parameter, sp.oo, allow_general=True)
            if (
                candidate is not None
                and not isinstance(candidate, sp.Limit)
                and candidate not in (sp.oo, -sp.oo, sp.zoo, sp.nan)
            ):
                rate = sp.simplify(candidate)
                log_prefactor = sp.expand(value + parameter * rate)
    return LogProbabilityResult(
        value, rate, log_prefactor, parameter, point, probability.status, probability
    )


def _distribution_mode_data(random_symbol: RandomSymbol):
    distribution = random_symbol.pspace.distribution
    name = type(distribution).__name__
    if name == "BinomialDistribution" and len(distribution.args) >= 4:
        count, probability, success, failure = distribution.args[:4]
        if success == 1 and failure == 0:
            continuous = sp.simplify(count * probability)
            exact_parameter = sp.simplify((count + 1) * probability)
            primary = sp.floor(exact_parameter)
            alternate = sp.ceiling(exact_parameter) - 1
            return continuous, primary, (primary, alternate), "binomial-lattice-mode"
    if name == "PoissonDistribution" and distribution.args:
        mean = distribution.args[0]
        primary = sp.floor(mean)
        alternate = sp.ceiling(mean) - 1
        return mean, primary, (primary, alternate), "poisson-lattice-mode"
    return None


def asymptotic_mode(
    random_symbol: RandomSymbol,
    *,
    parameter: sp.Symbol,
    point: sp.Expr = sp.oo,
    terms: int = 4,
) -> AsymptoticModeResult:
    """Return an asymptotic mode with explicit lattice-rounding information."""

    if not isinstance(random_symbol, RandomSymbol):
        raise TypeError("random_symbol must be a SymPy RandomSymbol")
    special = _distribution_mode_data(random_symbol)
    if special is not None:
        continuous, primary, candidates, method = special
        unique = tuple(dict.fromkeys(candidates))
        return AsymptoticModeResult(
            primary,
            continuous,
            unique,
            sp.simplify(primary - continuous),
            parameter,
            point,
            "EXACT",
            method,
        )

    z = sp.Dummy(str(random_symbol.symbol), real=True)
    weight = _density_on_support(random_symbol, z)
    log_weight = sp.log(weight)
    derivative = sp.simplify(sp.diff(log_weight, z))
    candidates = bounded_solve_one(sp.Eq(derivative, 0), z, allow_general=True) or ()
    maxima = []
    for candidate in candidates:
        second = sp.simplify(sp.diff(log_weight, z, 2).subs(z, candidate))
        if second.is_negative is True:
            maxima.append(candidate)
    if not maxima:
        raise NotImplementedError("could not determine a dominant mode saddle")
    continuous = maxima[0]
    if random_symbol.pspace.is_Discrete:
        primary = sp.floor(continuous + sp.Rational(1, 2))
        lattice = (primary,)
    else:
        primary = continuous
        lattice = (continuous,)
    return AsymptoticModeResult(
        primary,
        continuous,
        lattice,
        sp.simplify(primary - continuous),
        parameter,
        point,
        "FORMAL",
        "density-saddle-mode",
    )


def asymptotic_map(random_symbol: RandomSymbol, **kwargs) -> AsymptoticModeResult:
    """Alias of :func:`asymptotic_mode` for maximum-a-posteriori calculations."""

    return asymptotic_mode(random_symbol, **kwargs)


def _standard_binomial_parameters(random_symbol: RandomSymbol):
    distribution = random_symbol.pspace.distribution
    if type(distribution).__name__ != "BinomialDistribution" or len(distribution.args) < 4:
        return None
    count, probability, success, failure = distribution.args[:4]
    if success != 1 or failure != 0:
        return None
    return count, probability


def _binomial_entropy_expansion(
    count: sp.Expr, probability: sp.Expr, parameter: sp.Symbol, terms: int
) -> sp.Expr | None:
    if count != parameter:
        return None
    p = sp.sympify(probability)
    q = 1 - p
    if not (
        (p.is_positive is True or bounded_ask(sp.Q.positive(p)))
        and (q.is_positive is True or bounded_ask(sp.Q.positive(q)))
    ):
        return None
    value = sp.log(2 * sp.pi * sp.E * parameter * p * q) / 2
    if terms >= 2:
        value -= (1 - 2 * p) ** 2 / (12 * parameter * p * q)
    return value


def asymptotic_entropy(
    random_symbol: RandomSymbol,
    *,
    parameter: sp.Symbol,
    point: sp.Expr = sp.oo,
    terms: int = 4,
    method: str = "auto",
) -> StatisticalTransformResult:
    """Compute Shannon entropy (or differential entropy) asymptotically."""

    binomial = _standard_binomial_parameters(random_symbol)
    if binomial is not None:
        expansion = _binomial_entropy_expansion(*binomial, parameter, terms)
        if expansion is not None:
            remainder = AsymptoticRemainder.big_o(
                parameter**-2 if terms >= 2 else parameter**-1,
                parameter,
                point,
                source="formal Binomial entropy expansion",
            )
            return StatisticalTransformResult(
                expansion, parameter, point, "binomial-entropy", "FORMAL", remainder=remainder
            )
    weight = _density_on_support(random_symbol, random_symbol)
    result = asymptotic_expectation(
        -sp.log(weight), random_symbol, parameter=parameter, point=point, terms=terms, method=method
    )
    return StatisticalTransformResult(
        _expr(result),
        parameter,
        point,
        "entropy",
        result.status,
        (result,),
        remainder=result.remainder,
    )


def asymptotic_cross_entropy(
    reference: RandomSymbol,
    target: RandomSymbol,
    *,
    parameter: sp.Symbol,
    point: sp.Expr = sp.oo,
    terms: int = 4,
    method: str = "auto",
) -> StatisticalTransformResult:
    """Compute ``-E_reference[log q(X)]`` for two distributions."""

    left_binomial = _standard_binomial_parameters(reference)
    right_binomial = _standard_binomial_parameters(target)
    if (
        left_binomial is not None
        and right_binomial is not None
        and left_binomial[0] == right_binomial[0]
    ):
        count, p = left_binomial
        _count, q = right_binomial
        entropy = _binomial_entropy_expansion(count, p, parameter, terms)
        if entropy is not None:
            divergence = count * (p * sp.log(p / q) + (1 - p) * sp.log((1 - p) / (1 - q)))
            return StatisticalTransformResult(
                entropy + divergence,
                parameter,
                point,
                "binomial-cross-entropy",
                "FORMAL",
            )
    source_support = _support(reference)
    target_support = _support(target)
    subset, support_condition = _support_subset_condition(source_support, target_support)
    if subset is False:
        return StatisticalTransformResult(sp.oo, parameter, point, "cross-entropy", "EXACT")
    target_weight = _density_on_support(target, reference)
    result = asymptotic_expectation(
        -sp.log(target_weight),
        reference,
        parameter=parameter,
        point=point,
        terms=terms,
        method=method,
    )
    conditions = () if subset is True else (support_condition,)
    return StatisticalTransformResult(
        _expr(result),
        parameter,
        point,
        "cross-entropy",
        result.status,
        (result,),
        conditions,
        result.remainder,
    )


def asymptotic_kl_divergence(
    reference: RandomSymbol,
    target: RandomSymbol,
    *,
    parameter: sp.Symbol,
    point: sp.Expr = sp.oo,
    terms: int = 4,
    method: str = "auto",
) -> StatisticalTransformResult:
    """Compute Kullback--Leibler divergence asymptotically."""

    left_binomial = _standard_binomial_parameters(reference)
    right_binomial = _standard_binomial_parameters(target)
    if (
        left_binomial is not None
        and right_binomial is not None
        and left_binomial[0] == right_binomial[0]
    ):
        count, p = left_binomial
        _count, q = right_binomial
        value = count * (p * sp.log(p / q) + (1 - p) * sp.log((1 - p) / (1 - q)))
        return StatisticalTransformResult(
            value, parameter, point, "binomial-kl-divergence", "EXACT"
        )
    p_weight = _density_on_support(reference, reference)
    q_weight = _density_on_support(target, reference)
    source_support = _support(reference)
    target_support = _support(target)
    subset, support_condition = _support_subset_condition(source_support, target_support)
    if subset is False:
        return StatisticalTransformResult(sp.oo, parameter, point, "kl-divergence", "EXACT")
    result = asymptotic_expectation(
        sp.log(p_weight / q_weight),
        reference,
        parameter=parameter,
        point=point,
        terms=terms,
        method=method,
    )
    conditions = () if subset is True else (support_condition,)
    return StatisticalTransformResult(
        _expr(result),
        parameter,
        point,
        "kl-divergence",
        result.status,
        (result,),
        conditions,
        result.remainder,
    )


def asymptotic_cumulative_hazard(
    random_symbol: RandomSymbol,
    threshold: sp.Expr,
    *,
    parameter: sp.Symbol,
    point: sp.Expr = sp.oo,
    terms: int = 4,
    method: str = "auto",
) -> StatisticalTransformResult:
    """Compute cumulative hazard ``-log S(threshold)`` asymptotically."""

    log_survival = asymptotic_log_probability(
        random_symbol > threshold,
        random_symbol,
        parameter=parameter,
        point=point,
        terms=terms,
        method=method,
    )
    return StatisticalTransformResult(
        -log_survival.expression,
        parameter,
        point,
        "cumulative-hazard",
        log_survival.status,
        (log_survival,),
    )


def asymptotic_hazard(
    random_symbol: RandomSymbol,
    threshold: sp.Expr,
    *,
    parameter: sp.Symbol,
    point: sp.Expr = sp.oo,
    terms: int = 4,
    method: str = "auto",
) -> StatisticalTransformResult:
    """Compute discrete or continuous hazard from local mass/density and survival."""

    support = _support(random_symbol)
    if support.is_subset(sp.S.Integers) is True:
        survival = asymptotic_probability(
            random_symbol >= threshold,
            random_symbol,
            parameter=parameter,
            point=point,
            terms=terms,
            method=method,
        )
    else:
        survival = asymptotic_survival(
            random_symbol, threshold, parameter=parameter, point=point, terms=terms, method=method
        )
    local = asymptotic_local_limit(
        random_symbol, threshold, parameter=parameter, point=point, terms=terms
    )
    value = sp.simplify(local.expression / survival.expression)
    return StatisticalTransformResult(
        value, parameter, point, "hazard", _status(local, survival), (local, survival)
    )


def asymptotic_factorial_moment(
    random_symbol: RandomSymbol,
    *,
    order: int,
    parameter: sp.Symbol,
    point: sp.Expr = sp.oo,
    terms: int = 4,
) -> StatisticalTransformResult:
    """Compute the falling-factorial moment ``E[(X)_order]``."""

    if not isinstance(order, int) or order < 0:
        raise ValueError("order must be a nonnegative integer")
    binomial = _standard_binomial_parameters(random_symbol)
    if binomial is not None:
        count, probability = binomial
        value = sp.prod(count - j for j in range(order)) * probability**order
        return StatisticalTransformResult(
            sp.simplify(value), parameter, point, f"binomial-factorial-moment-{order}", "EXACT"
        )
    distribution = random_symbol.pspace.distribution
    if type(distribution).__name__ == "PoissonDistribution" and distribution.args:
        return StatisticalTransformResult(
            distribution.args[0] ** order,
            parameter,
            point,
            f"poisson-factorial-moment-{order}",
            "EXACT",
        )
    falling = sp.prod(random_symbol - j for j in range(order))
    result = asymptotic_expectation(
        falling, random_symbol, parameter=parameter, point=point, terms=terms
    )
    return StatisticalTransformResult(
        _expr(result), parameter, point, f"factorial-moment-{order}", result.status, (result,)
    )


def asymptotic_pgf(
    random_symbol: RandomSymbol,
    *,
    transform_variable: sp.Symbol,
    parameter: sp.Symbol,
    point: sp.Expr = sp.oo,
    terms: int = 4,
) -> StatisticalTransformResult:
    """Compute a probability-generating function asymptotically."""

    binomial = _standard_binomial_parameters(random_symbol)
    if binomial is not None:
        count, probability = binomial
        value = (1 - probability + probability * transform_variable) ** count
        return StatisticalTransformResult(value, parameter, point, "binomial-pgf", "EXACT")
    distribution = random_symbol.pspace.distribution
    if type(distribution).__name__ == "PoissonDistribution" and distribution.args:
        value = sp.exp(distribution.args[0] * (transform_variable - 1))
        return StatisticalTransformResult(value, parameter, point, "poisson-pgf", "EXACT")
    result = asymptotic_expectation(
        transform_variable**random_symbol,
        random_symbol,
        parameter=parameter,
        point=point,
        terms=terms,
    )
    return StatisticalTransformResult(
        _expr(result), parameter, point, "pgf", result.status, (result,)
    )


def asymptotic_local_limit(
    random_symbol: RandomSymbol,
    location: sp.Expr,
    *,
    parameter: sp.Symbol,
    point: sp.Expr = sp.oo,
    terms: int = 4,
) -> StatisticalTransformResult:
    """Expand the local mass/density at a parameter-dependent location.

    Discrete factorial/Gamma PMFs are first put into the certified positive
    Stirling form, so central and moderate-deviation locations can retain the
    pointwise Stieltjes error certificate.
    """

    location = sp.sympify(location)
    support = _support(random_symbol)
    discrete = support.is_subset(sp.S.Integers) is True
    z = (
        sp.Dummy(str(random_symbol.symbol), integer=True)
        if discrete
        else sp.Dummy(str(random_symbol.symbol), real=True)
    )
    raw = _restrict_piecewise_to_support(sp.sympify(density(random_symbol)(z)), support, z)
    normalization = None
    approximant = raw
    remainder = None
    status = "FORMAL"
    if raw.has(sp.factorial, sp.gamma, sp.binomial):
        try:
            normalization = normalize_positive_pmf(
                raw,
                variable=z,
                parameter=parameter,
                point=point,
                terms=max(2, terms),
                assumptions=_support_assumptions(support, z),
            )
        except (ValueError, NotImplementedError):
            normalization = None
        if normalization is not None:
            approximant = normalization.expression
    localized = sp.simplify(approximant.subs(z, location))
    if normalization is not None and point is sp.oo:
        try:
            local_stirling = stirling_local_mass_expansion(
                normalization,
                location=location,
                variable=z,
                parameter=parameter,
                terms=terms,
            )
            expression = local_stirling.expression
        except (ValueError, TypeError, NotImplementedError):
            try:
                local_stirling = stirling_sqrt_local_mass_expansion(
                    normalization,
                    location=location,
                    variable=z,
                    parameter=parameter,
                    terms=terms,
                )
                expression = local_stirling.expression
            except (ValueError, TypeError, NotImplementedError):
                local_stirling = None
    else:
        local_stirling = None
    if local_stirling is None:
        try:
            expansion = transseries_from_expression(localized, parameter, point=point).prefix(terms)
            expression = expansion.truncate()
        except (ValueError, TypeError, NotImplementedError):
            try:
                expression = sp.series(localized, parameter, point, max(2, terms)).removeO()
            except (ValueError, TypeError, NotImplementedError):
                expression = localized
    if normalization is not None and normalization.relative_error_bound is not None:
        bound = sp.simplify(normalization.relative_error_bound.subs(z, location))
        ctx = AsymptoticContext(parameter, point=point)
        truncation = local_stirling.truncation_scale if local_stirling is not None else sp.S.One
        bound_limit = ctx.limit(bound)
        if bound_limit != 0 and (bound.has(sp.floor) or bound.has(sp.ceiling)):
            # floor(u)-u and ceiling(u)-u are uniformly bounded.  For the
            # positive Stirling certificates used here, replacing rounded
            # moving Gamma arguments by their smooth centers preserves the
            # vanishing inverse-power error scale.
            smooth_bound = bound.replace(
                lambda node: node.func in (sp.floor, sp.ceiling),
                lambda node: node.args[0],
            )
            if ctx.limit(smooth_bound) == 0:
                bound_limit = sp.S.Zero
        if bound_limit == 0 and ctx.limit(truncation) == 0:
            remainder = AsymptoticRemainder.big_o(
                sp.Abs(expression) * (bound + truncation),
                parameter,
                point,
                source=("positive-real Stirling bound plus analytic local-limit Taylor truncation"),
            )
            status = "CERTIFIED"
    condition = sp.Contains(location, support, evaluate=False)
    return StatisticalTransformResult(
        expression,
        parameter,
        point,
        "local-limit",
        status,
        tuple(item for item in (normalization,) if item is not None),
        (condition,),
        remainder,
    )
