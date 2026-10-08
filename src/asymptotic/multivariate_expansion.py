"""Certified weighted multivariate asymptotic expansions near finite real points."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import cache
from math import gcd

import sympy as sp

from ._power_simplify import analytic_powsimp
from ._symbolic_policy import bounded_ask
from .angular_extrema import _angular_squared_extremum, _certified_extremum
from .blowup_geometry import CoverageCertificate
from .limit_primitives import _normalize_variables, normalize_limit_target


def _newton_weight_vectors(expr, variables, target):
    from .blowup_geometry import newton_candidate_rays

    return newton_candidate_rays(expr, tuple(variables), tuple(target))


class ExpansionStatus(Enum):
    CERTIFIED = "certified"
    FORMAL = "formal"
    UNKNOWN = "unknown"


class AsymptoticRelationStatus(Enum):
    """Certification state for a local multivariate asymptotic relation."""

    CERTIFIED = "certified"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class MultivariateAsymptoticRelation:
    """Certified local ``O``, ``o``, or asymptotic-equivalence statement."""

    kind: str
    lhs: sp.Expr
    rhs: sp.Expr
    variables: tuple[sp.Symbol, ...]
    target: tuple[sp.Expr, ...]
    domain: sp.Expr
    status: AsymptoticRelationStatus
    atlas: MultivariateAsymptoticAtlas | None = None
    constant: sp.Expr | None = None
    provider: str = "none"
    statement: str = ""
    certificate_backends: tuple[str, ...] = ()

    @property
    def certified(self) -> bool:
        return self.status is AsymptoticRelationStatus.CERTIFIED


@dataclass(frozen=True)
class UniformRemainderCertificate:
    """Uniform bound ``|remainder| <= C*r**order`` on a weighted angular chart."""

    order: sp.Expr
    constant: sp.Expr
    radius: sp.Expr
    provider: str
    statement: str

    def __post_init__(self):
        order = sp.sympify(self.order)
        constant = sp.sympify(self.constant)
        radius = sp.sympify(self.radius)
        object.__setattr__(self, "order", order)
        object.__setattr__(self, "constant", constant)
        object.__setattr__(self, "radius", radius)
        if order.is_positive is not True:
            raise ValueError("uniform remainder order must be provably positive")
        if constant.is_nonnegative is not True:
            raise ValueError("uniform remainder constant must be provably nonnegative")
        if radius.is_positive is not True:
            raise ValueError("uniform remainder radius must be provably positive")


@dataclass(frozen=True)
class MultivariateAsymptoticTerm:
    order: sp.Expr
    coefficient: sp.Expr


# Public alias for the subsystem-neutral coverage representation.


@dataclass(frozen=True)
class MultivariateAsymptoticExpansion:
    expression: sp.Expr
    variables: tuple[sp.Symbol, ...]
    target: tuple[sp.Expr, ...]
    weights: tuple[int, ...]
    radius_symbol: sp.Symbol
    angular_variables: tuple[sp.Symbol, ...]
    terms: tuple[MultivariateAsymptoticTerm, ...]
    remainder_order: sp.Expr
    status: ExpansionStatus
    remainder_certificate: UniformRemainderCertificate | None = None
    domain: sp.Expr = sp.S.true

    @property
    def certified(self) -> bool:
        return self.status is ExpansionStatus.CERTIFIED

    @property
    def leading_term(self) -> MultivariateAsymptoticTerm | None:
        return self.terms[0] if self.terms else None

    def truncated_expression(self) -> sp.Expr:
        r = self.radius_symbol
        return sp.Add(*(term.coefficient * r**term.order for term in self.terms))

    def uniform_remainder(self):
        """Expose the chart remainder through the shared uniform calculus."""
        from .uniform_remainder import from_weighted_certificate

        return from_weighted_certificate(
            self.remainder_certificate,
            self.radius_symbol,
            uniform_variables=self.angular_variables,
            domain=self.domain,
        )


def _primitive_weights(weights, n):
    if weights is None:
        return None
    values = tuple(int(w) for w in weights)
    if len(values) != n or any(w <= 0 for w in values):
        raise ValueError("weights must be positive integers matching variables")
    g = 0
    for w in values:
        g = gcd(g, w)
    return tuple(w // g for w in values)


def _candidate_weights(expr, variables, target, weights):
    explicit = _primitive_weights(weights, len(variables))
    if explicit is not None:
        return (explicit,)
    found = {(1,) * len(variables)}
    found.update(_newton_weight_vectors(expr, variables, target))
    return tuple(sorted(found, key=lambda w: (sum(w), w)))


def _rational_radial_valuation(expr, r):
    """Exact r-adic valuation for a rational expression polynomial in r."""
    numerator, denominator = sp.fraction(sp.cancel(sp.together(expr)))
    try:
        npoly = sp.Poly(numerator, r)
        dpoly = sp.Poly(denominator, r)
    except (sp.PolynomialError, TypeError, ValueError):
        return None
    nval = min((m[0] for m, _ in npoly.terms()), default=0)
    dval = min((m[0] for m, _ in dpoly.terms()), default=0)
    return int(nval - dval)


def _series_terms(transformed, r, order):
    try:
        ser = sp.series(transformed, r, 0, order + 1).removeO().expand()
    except (TypeError, ValueError, NotImplementedError):
        return None
    terms = {}
    for summand in sp.Add.make_args(ser):
        power = summand.as_powers_dict().get(r, sp.S.Zero)
        if power.is_integer is not True or power < 0 or power > order:
            return None
        coeff = sp.cancel(summand / r**power)
        if coeff.has(r):
            return None
        terms[int(power)] = sp.simplify(terms.get(int(power), 0) + coeff)
    return tuple(
        MultivariateAsymptoticTerm(sp.Integer(k), v)
        for k, v in sorted(terms.items())
        if v != 0
    )


def _uniform_rational_remainder(
    transformed,
    truncation,
    r,
    angular,
    remainder_order,
    radius,
    domain_constraint=sp.S.true,
):
    remainder = sp.cancel(sp.together(transformed - truncation))
    if remainder == 0:
        return UniformRemainderCertificate(
            sp.Integer(remainder_order),
            sp.S.Zero,
            radius,
            "exact_zero_remainder",
            "the truncated expansion is exact",
        )
    scaled = sp.cancel(sp.together(remainder / r**remainder_order))
    numerator, denominator = sp.fraction(scaled)
    try:
        sp.Poly(numerator, r, *angular)
        sp.Poly(denominator, r, *angular)
    except (sp.PolynomialError, TypeError, ValueError):
        return None

    def coefficient_bound(poly_expr):
        poly = sp.Poly(sp.expand(poly_expr), r, *angular)
        return sp.simplify(
            sum(
                sp.Abs(coefficient) * radius ** monomial[0]
                for monomial, coefficient in poly.terms()
            )
        )

    denominator_center = sp.expand(denominator.subs(r, 0))
    if not denominator_center.free_symbols.intersection(angular):
        perturbation_bound = coefficient_bound(denominator - denominator_center)
        lower = sp.simplify(sp.Abs(denominator_center) - perturbation_bound)
        if lower.is_positive is True:
            constant = sp.simplify(coefficient_bound(numerator) / lower)
            return UniformRemainderCertificate(
                sp.Integer(remainder_order),
                constant,
                radius,
                "exact_coefficient_norm",
                "On |u_i|<=1 and 0<=r<=radius, coefficient norms bound the numerator and the denominator perturbation, leaving a positive denominator lower bound.",
            )
    constraints = sp.And(
        sp.Ge(r, 0),
        sp.Le(r, radius),
        sp.Eq(sp.Add(*(u**2 for u in angular)), 1),
        domain_constraint,
    )
    if not denominator.has(r):
        dmin = _angular_squared_extremum(sp.expand(denominator**2), angular, kind="min")
    else:
        dmin = _certified_extremum(
            sp.expand(denominator**2), constraints, (r, *angular), kind="min"
        )
    # Remove an explicit radial monomial before angular optimization.  The
    # remaining radius factor is maximized at the compact interval endpoint.
    npoly = sp.Poly(sp.expand(numerator), r)
    min_r_power = min((monom[0] for monom, _ in npoly.terms()), default=0)
    angular_numerator = (
        sp.cancel(numerator / r**min_r_power) if min_r_power else numerator
    )
    if not angular_numerator.has(r):
        angular_max = _angular_squared_extremum(
            sp.expand(angular_numerator**2), angular, kind="max"
        )
        nmax = (
            (sp.simplify(radius ** (2 * min_r_power) * angular_max[0]), angular_max[1])
            if angular_max
            else None
        )
    else:
        nmax = _certified_extremum(
            sp.expand(numerator**2), constraints, (r, *angular), kind="max"
        )
    if (
        dmin is not None
        and nmax is not None
        and bounded_ask(sp.Q.positive(dmin[0])) is True
        and not nmax[0].has(sp.oo, -sp.oo, sp.zoo, sp.nan)
    ):
        constant = sp.sqrt(sp.simplify(nmax[0] / dmin[0]))
        provider = dmin[1] if dmin[1] == nmax[1] else f"{dmin[1]}+{nmax[1]}"
    else:
        return None
    statement = (
        f"uniform compact-chart optimization bound: min(denominator^2)={sp.sstr(dmin[0])}, max(numerator^2)={sp.sstr(nmax[0])}"
        if dmin is not None and nmax is not None
        else "uniform compact-chart bound from exact coefficient norms"
    )
    return UniformRemainderCertificate(
        sp.Integer(remainder_order), constant, radius, provider, statement
    )


def _remove_simple_radical_removable(expr, r):
    """Rationalize a one-radical removable quotient when exact simplification verifies it."""
    expr = sp.cancel(sp.together(expr))
    numerator, denominator = sp.fraction(expr)
    radicals = [p for p in numerator.atoms(sp.Pow) if p.exp == sp.Rational(1, 2)]
    if len(radicals) != 1 or not denominator.has(r):
        return expr
    radical = radicals[0]
    conjugate = numerator.xreplace({radical: -radical})
    if conjugate == numerator:
        return expr
    candidate = sp.cancel(
        sp.together(
            sp.expand(numerator * conjugate) / sp.expand(denominator * conjugate)
        )
    )
    try:
        if sp.simplify(candidate - expr) == 0:
            return candidate
    except (TypeError, ValueError, NotImplementedError):
        pass
    return expr


def _uniform_univariate_remainder(transformed, truncation, r, remainder_order, radius):
    """Certify a radial remainder using exact univariate calculus when possible."""
    transformed = _remove_simple_radical_removable(transformed, r)
    remainder = sp.simplify(transformed - truncation)
    if remainder == 0:
        return UniformRemainderCertificate(
            sp.Integer(remainder_order),
            sp.S.Zero,
            radius,
            "exact_zero_remainder",
            "the truncated expansion is exact",
        )
    scaled = sp.simplify(remainder / r**remainder_order)
    scaled = _remove_simple_radical_removable(scaled, r)
    # Cheap exact bound for positive square-root denominators.  On r>=0,
    # sqrt(1+r**2)>=1; a denominator polynomial with nonnegative coefficients
    # is therefore bounded below by its value at r=0, radical=1.
    snum, sden = sp.fraction(sp.cancel(scaled))
    radicals = [p for p in sden.atoms(sp.Pow) if p.exp == sp.Rational(1, 2)]
    if len(radicals) == 1 and not (snum.free_symbols - {r}):
        radical = radicals[0]
        try:
            poly = sp.Poly(sp.expand(sden), r, radical)
            if all(bounded_ask(sp.Q.nonnegative(c)) is True for _, c in poly.terms()):
                lower = sp.simplify(sden.subs({r: 0, radical: 1}))
                if bounded_ask(sp.Q.positive(lower)) is True:
                    nbound = (
                        sp.Abs(snum).subs(r, radius) if snum.has(r) else sp.Abs(snum)
                    )
                    return UniformRemainderCertificate(
                        sp.Integer(remainder_order),
                        sp.simplify(nbound / lower),
                        radius,
                        "positive_radical_denominator",
                        "uniform radial bound from sqrt(1+r**2)>=1 and nonnegative denominator coefficients",
                    )
        except (sp.PolynomialError, TypeError, ValueError):
            pass
    # Continuity on the compact radial interval gives a finite bound.  Ask the
    # existing certified optimizer first; it handles algebraic/transcendental
    # univariate objectives through the package's normal exact machinery.
    constraints = sp.And(sp.Ge(r, 0), sp.Le(r, radius))
    maximum = None
    if scaled.is_rational_function(r):
        maximum = _certified_extremum(
            sp.simplify(scaled**2), constraints, (r,), kind="max"
        )
    if maximum is None or maximum[0].has(sp.oo, -sp.oo, sp.zoo, sp.nan):
        try:
            from sympy.calculus.util import maximum as exact_maximum

            derivative = sp.diff(transformed, r, int(remainder_order))
            derivative_bound = exact_maximum(
                sp.Abs(derivative), r, sp.Interval(0, radius)
            )
            if derivative_bound.has(sp.oo, -sp.oo, sp.zoo, sp.nan):
                return None
            constant = sp.simplify(
                derivative_bound / sp.factorial(int(remainder_order))
            )
            return UniformRemainderCertificate(
                sp.Integer(remainder_order),
                constant,
                radius,
                "univariate_taylor_theorem",
                "uniform radial remainder bound from the exact Taylor theorem",
            )
        except (ArithmeticError, TypeError, ValueError, NotImplementedError):
            # For sinc-type removable compositions use the alternating sine
            # remainder directly.  If t=r**k and the -t**2/6 term is retained,
            # |sin(t)/t - 1 + t**2/6| <= |t|**4/120 for |t|<=1.
            for k in range(1, 7):
                t = r**k
                sinc_residual = sp.sin(t) / t - 1
                if sp.simplify(transformed - sinc_residual) != 0:
                    continue
                if (
                    sp.simplify(truncation + t**2 / 6) != 0
                    or bounded_ask(sp.Q.le(radius**k, 1)) is not True
                ):
                    continue
                actual_order = 4 * k
                if actual_order < remainder_order:
                    continue
                constant = sp.simplify(radius ** (actual_order - remainder_order) / 120)
                return UniformRemainderCertificate(
                    sp.Integer(remainder_order),
                    constant,
                    radius,
                    "alternating_sine_remainder",
                    "uniform sinc remainder from the alternating sine Taylor bound",
                )
            return None
    return UniformRemainderCertificate(
        sp.Integer(remainder_order),
        sp.sqrt(sp.simplify(maximum[0])),
        radius,
        maximum[1],
        "certified compact radial remainder bound",
    )


def _reduce_sphere_invariants(expr, angular):
    """Reduce exact occurrences of the angular sphere norm to one.

    This is structural: it uses only the defining blow-up chart
    equation ``sum(u_i**2)=1`` and does not perform heuristic identities.
    """
    sphere = sp.Add(*(u**2 for u in angular))
    expr = sp.factor_terms(sp.sympify(expr))
    radial_symbols = tuple(expr.free_symbols - set(angular))
    previous = None
    while expr != previous:
        previous = expr
        expr = analytic_powsimp(sp.factor_terms(expr))
        expr = expr.replace(
            lambda e: e.is_Add,
            lambda e: sp.factor_terms(sp.collect(e, radial_symbols)).subs(
                sphere, sp.S.One
            ),
        )
        expr = expr.subs(sphere, sp.S.One)
        # Common constructors distribute a scalar radial factor over the sphere.
        expr = expr.replace(
            lambda e: e.is_Add and sp.factor_terms(e).has(sphere),
            lambda e: sp.factor_terms(sp.collect(e, radial_symbols)).subs(
                sphere, sp.S.One
            ),
        )
    return sp.simplify(expr)


def multivariate_expansion(
    expr,
    variables,
    target,
    *,
    order=2,
    weights=None,
    domain=True,
    radius=None,
    return_formal=False,
):
    """Expand on weighted blow-up charts and certify a uniform remainder when possible.

    The chart is ``x_i-a_i = r**w_i*u_i`` with ``sum(u_i**2)=1``.  Polynomial/
    rational transformed expressions receive an exact compact-chart remainder
    certificate through semialgebraic optimization (and the certified symbopt
    fallback used by the limit machinery).  Unsupported cases are explicit
    ``UNKNOWN`` unless ``return_formal=True``.
    """
    expr = sp.sympify(expr)
    variables = _normalize_variables(variables)
    target = normalize_limit_target(variables, target)
    domain = sp.sympify(domain)
    order = int(order)
    if order < 0:
        raise ValueError("order must be nonnegative")
    radius = sp.Rational(1, 2) if radius is None else sp.sympify(radius)
    r = sp.Symbol("r", positive=True)
    angular = tuple(sp.Symbol(f"u{i}", real=True) for i in range(len(variables)))
    best_formal = None
    for w in _candidate_weights(expr, variables, target, weights):
        subs = {
            v: a + r**wi * u
            for v, a, wi, u in zip(variables, target, w, angular, strict=True)
        }
        transformed = sp.cancel(sp.together(expr.subs(subs, simultaneous=True)))
        transformed = _reduce_sphere_invariants(transformed, angular)
        transformed_domain = sp.sympify(domain).subs(subs, simultaneous=True)
        valuation = _rational_radial_valuation(transformed, r)
        series_order = max(order, valuation or 0)
        terms = _series_terms(transformed, r, series_order)
        if terms is None:
            continue
        # A relation may begin beyond the user's display order (e.g. sinc-1 is
        # O(r**4)).  Look a bounded distance ahead so a genuine leading term is
        # not mistaken for an empty expansion.
        while not terms and series_order < max(order + 4, 6):
            series_order += 1
            terms = _series_terms(transformed, r, series_order)
            if terms is None:
                break
        if terms is None:
            continue
        truncation = sp.Add(*(t.coefficient * r**t.order for t in terms))
        remainder_order = series_order + 1
        cert = _uniform_rational_remainder(
            transformed,
            truncation,
            r,
            angular,
            remainder_order,
            radius,
            transformed_domain,
        )
        if cert is None and not (transformed.free_symbols & set(angular)):
            cert = _uniform_univariate_remainder(
                transformed, truncation, r, remainder_order, radius
            )
        result = MultivariateAsymptoticExpansion(
            expr,
            variables,
            target,
            w,
            r,
            angular,
            terms,
            sp.Integer(remainder_order),
            ExpansionStatus.CERTIFIED if cert else ExpansionStatus.FORMAL,
            cert,
            domain,
        )
        if cert:
            return result
        best_formal = result
    if return_formal and best_formal is not None:
        return best_formal
    if best_formal is not None:
        return MultivariateAsymptoticExpansion(
            expr,
            variables,
            target,
            best_formal.weights,
            r,
            angular,
            best_formal.terms,
            best_formal.remainder_order,
            ExpansionStatus.UNKNOWN,
            domain=domain,
        )
    return MultivariateAsymptoticExpansion(
        expr,
        variables,
        target,
        (1,) * len(variables),
        r,
        angular,
        (),
        sp.Integer(order + 1),
        ExpansionStatus.UNKNOWN,
        domain=domain,
    )


@dataclass(frozen=True)
class MultivariateAsymptoticChart:
    """One weighted blow-up expansion restricted to a closed angular sector."""

    expansion: MultivariateAsymptoticExpansion
    sector: sp.Expr
    dominant_coordinate: int

    @property
    def certified(self) -> bool:
        return self.expansion.certified


@dataclass(frozen=True)
class MultivariateAsymptoticAtlas:
    """Finite collection of certified weighted charts with a coverage certificate."""

    expression: sp.Expr
    variables: tuple[sp.Symbol, ...]
    target: tuple[sp.Expr, ...]
    charts: tuple[MultivariateAsymptoticChart, ...]
    coverage_certificate: CoverageCertificate
    domain: sp.Expr = sp.S.true

    @property
    def certified(self) -> bool:
        return (
            self.coverage_certificate.certified
            and bool(self.charts)
            and all(chart.certified for chart in self.charts)
        )

    @property
    def weights(self) -> tuple[tuple[int, ...], ...]:
        return tuple(dict.fromkeys(chart.expansion.weights for chart in self.charts))


def _coordinate_sectors(angular):
    """Closed max-coordinate sectors; their union is the unit sphere."""
    sectors = []
    for i, ui in enumerate(angular):
        sectors.append(
            sp.And(*(sp.Ge(ui**2, uj**2) for j, uj in enumerate(angular) if j != i))
        )
    return tuple(sectors)


def _expansion_for_weight(
    expr, variables, target, order, weight, domain, radius, return_formal
):
    return multivariate_expansion(
        expr,
        variables,
        target,
        order=order,
        weights=weight,
        domain=domain,
        radius=radius,
        return_formal=return_formal,
    )


@cache
def _semialgebraic_sector_cover(dimension):
    angular = tuple(sp.Symbol(f"_cover_u{i}", real=True) for i in range(dimension))
    sphere = sp.Eq(sp.Add(*(u**2 for u in angular)), 1)
    cover = sp.Or(*_coordinate_sectors(angular))
    try:
        from semialg import implies

        result = implies(sphere, cover, angular, return_result=True)
        return bool(getattr(result, "valid", result))
    except (
        ImportError,
        ArithmeticError,
        TypeError,
        ValueError,
        NotImplementedError,
        RuntimeError,
    ):
        return False


def multivariate_atlas(
    expr,
    variables,
    target,
    *,
    order=2,
    weights=None,
    domain=True,
    radius=None,
    include_formal=False,
    first_certified_family=False,
):
    """Build a finite Newton/weighted blow-up atlas with certified angular coverage.

    Each positive weight vector defines a weighted polar blow-up.  The unit angular
    sphere is split into the closed sectors ``u_i**2 >= u_j**2``.  These sectors
    cover the sphere because every finite tuple has a coordinate of maximal absolute
    value.  For positive weights, every nonzero translated point has a unique radial
    coordinate solving ``sum(dx_i**2/r**(2*w_i)) = 1``; hence every complete sector
    family covers a punctured neighborhood.  Only weight families whose expansion is
    certified are admitted to a certified atlas.
    """
    expr = sp.sympify(expr)
    variables = _normalize_variables(variables)
    target = normalize_limit_target(variables, target)
    domain = sp.sympify(domain)
    candidate_weights = _candidate_weights(expr, variables, target, weights)
    if first_certified_family:
        radial = (1,) * len(variables)
        candidate_weights = tuple(
            sorted(candidate_weights, key=lambda w: (w == radial, sum(w), w))
        )
    charts = []
    complete_weights = []
    for weight in candidate_weights:
        expansion = _expansion_for_weight(
            expr, variables, target, int(order), weight, domain, radius, include_formal
        )
        if not expansion.certified and not include_formal:
            continue
        from .blowup_geometry import weighted_spherical_atlas

        geometry = weighted_spherical_atlas(variables, target, weight)
        replacements = dict(
            zip(
                geometry.charts[0].angular_variables,
                expansion.angular_variables,
                strict=True,
            )
        )
        sectors = tuple(
            chart.sector.xreplace(replacements) for chart in geometry.charts
        )
        family = tuple(
            MultivariateAsymptoticChart(expansion, sector, i)
            for i, sector in enumerate(sectors)
        )
        charts.extend(family)
        if expansion.certified and len(family) == len(variables):
            complete_weights.append(weight)
            if first_certified_family:
                break

    certified = False
    provider = "none"
    statement = "no complete certified weighted sector family was obtained"
    if complete_weights:
        try:
            # Max-coordinate sectors cover every finite-dimensional sphere by
            # construction.  No external solver is needed for an unrestricted
            # approach domain.
            cover_ok = bool(charts[0].expansion.angular_variables if charts else ())
            if domain is sp.S.true or domain == sp.S.true:
                approach_ok = True
                dependency = "weighted_blowup_geometry"
            else:
                try:
                    from semialg import point_in_closure
                except (ImportError, AttributeError):
                    from semialg.derived_geometry import point_in_closure

                distance2 = sp.Add(
                    *((v - a) ** 2 for v, a in zip(variables, target, strict=True))
                )
                approach = sp.And(domain, sp.Gt(distance2, 0))
                closure = point_in_closure(
                    approach,
                    dict(zip(variables, target, strict=True)),
                    variables,
                    return_result=True,
                )
                approach_ok = bool(getattr(closure, "in_closure", closure))
                dependency = "semialg+weighted_blowup_geometry"
            certified = cover_ok and approach_ok
            if certified:
                provider = dependency
                statement = (
                    "max-coordinate sectors exactly cover the angular sphere; "
                    "the requested punctured approach domain accumulates at the "
                    "target, so a complete positive-weight blow-up family covers "
                    "all local approaches"
                )
            elif not approach_ok:
                statement = (
                    "target is not in the closure of the punctured approach domain"
                )
            else:
                statement = "the angular sector cover was not certified"
        except (
            ImportError,
            ArithmeticError,
            TypeError,
            ValueError,
            NotImplementedError,
            RuntimeError,
        ):
            statement = "semialg could not certify relative-domain atlas coverage"
    coverage = (
        CoverageCertificate.complete(provider, statement, tuple(range(len(charts))))
        if certified
        else CoverageCertificate.unknown(provider, statement, ("local_approach_cover",))
    )
    return MultivariateAsymptoticAtlas(
        expr, variables, target, tuple(charts), coverage, domain
    )


def _atlas_relation_ratio(lhs, rhs):
    return sp.cancel(sp.together(sp.sympify(lhs) / sp.sympify(rhs)))


def _certified_atlas_bound(atlas, *, return_backends=False):
    """Return a finite uniform bound assembled from every covering chart.

    With ``return_backends=True`` also return the exact certificate providers
    actually used.  This powers the relation-certification corpus without
    changing the public relation semantics.
    """
    if not atlas.certified:
        return None
    bounds = []
    backends = {atlas.coverage_certificate.provider}
    for chart in atlas.charts:
        expansion = chart.expansion
        if not expansion.certified or expansion.remainder_certificate is None:
            return None
        if any(
            bounded_ask(sp.Q.negative(term.order)) is True for term in expansion.terms
        ):
            return None
        coefficient_bound = sp.S.Zero
        angular = expansion.angular_variables
        for term in expansion.terms:
            coefficient = sp.simplify(term.coefficient)
            if not (coefficient.free_symbols & set(angular)):
                coefficient_bound += sp.Abs(coefficient)
                continue
            # On a max-coordinate sector, a positive sum containing an even
            # pure power of the dominant angular coordinate has an immediate
            # denominator lower bound: |u_i|^2 >= 1/n.  Use this before CAD; it
            # is the key scalable certificate for higher-dimensional Newton fans.
            normalized = sp.cancel(coefficient)
            numerator0, denominator0 = sp.fraction(normalized)
            try:
                dpoly = sp.Poly(sp.expand(denominator0), *angular)
                npoly = sp.Poly(sp.expand(numerator0), *angular)
                nonnegative_den = all(
                    bounded_ask(sp.Q.nonnegative(c)) is True for _, c in dpoly.terms()
                )
                dominant_power = None
                dominant_coeff = None
                for monom, coeff0 in dpoly.terms():
                    if (
                        all(
                            e == 0
                            for j, e in enumerate(monom)
                            if j != chart.dominant_coordinate
                        )
                        and monom[chart.dominant_coordinate] > 0
                        and monom[chart.dominant_coordinate] % 2 == 0
                        and bounded_ask(sp.Q.positive(coeff0)) is True
                    ):
                        dominant_power, dominant_coeff = (
                            monom[chart.dominant_coordinate],
                            coeff0,
                        )
                        break
                if len(angular) >= 3 and nonnegative_den and dominant_power is not None:
                    lower = sp.simplify(
                        dominant_coeff
                        * sp.Integer(len(angular)) ** (-sp.Rational(dominant_power, 2))
                    )
                    nbound = sp.Add(*(sp.Abs(c) for _, c in npoly.terms()))
                    coefficient_bound += sp.simplify(nbound / lower)
                    backends.add("sector_dominant_coordinate_bound")
                    continue
            except (sp.PolynomialError, TypeError, ValueError):
                pass

            # Polynomial/rational angular coefficients are optimized exactly on
            # the actual sector.
            # The coefficient is squared, so use the sphere/simplex reduction
            # before falling back to general constrained optimization.  Sector
            # constraints are retained when they are nontrivial; the direct
            # optimizer is then required.
            extremum = _angular_squared_extremum(
                sp.expand(coefficient**2), angular, kind="max", sector=chart.sector
            )
            if extremum is None:
                # A finite O-bound does not require the sharp angular extremum.
                # On the unit sphere every |u_i| <= 1; for a polynomial angular
                # coefficient, the l1 coefficient norm is therefore a rigorous
                # uniform bound.  Keep this as a last exact fallback rather than
                # invoking a general optimizer for an elementary bounded form.
                normalized = sp.cancel(coefficient)
                numerator, denominator = sp.fraction(normalized)
                sphere = sp.Add(*(u**2 for u in angular))
                # Angular coefficients often retain a denominator that is
                # identically one on the sphere.  Remove exact powers of that
                # sphere equation before applying the coefficient-norm bound.
                for power in range(1, 5):
                    if sp.expand(denominator - sphere**power) == 0:
                        normalized = sp.expand(numerator)
                        break
                try:
                    poly = sp.Poly(normalized, *angular)
                except (sp.PolynomialError, TypeError, ValueError):
                    return None
                coefficient_bound += sp.Add(*(sp.Abs(c) for _, c in poly.terms()))
                backends.add("angular_coefficient_norm")
                continue
            coefficient_bound += sp.sqrt(sp.simplify(extremum[0]))
            backends.add(extremum[1])
        cert = expansion.remainder_certificate
        backends.add(cert.provider)
        radial_factor = cert.radius**cert.order
        bounds.append(sp.simplify(coefficient_bound + cert.constant * radial_factor))
    if not bounds:
        return None
    bound = bounds[0]
    for value in bounds[1:]:
        bound = sp.Max(bound, value)
    result = sp.simplify(bound)
    if return_backends:
        return result, tuple(
            sorted(provider for provider in backends if provider and provider != "none")
        )
    return result


def _structurally_nonnegative(expr):
    expr = sp.sympify(expr)
    if bounded_ask(sp.Q.nonnegative(expr)) is True:
        return True
    # sqrt(1+h)-1 >= 0 whenever h>=0.
    if expr.is_Add:
        radicals = [a for a in expr.args if a.is_Pow and a.exp == sp.Rational(1, 2)]
        if len(radicals) == 1 and sp.simplify(expr - (radicals[0] - 1)) == 0:
            return _structurally_nonnegative(radicals[0].base - 1)
        return all(_structurally_nonnegative(a) for a in expr.args)
    return bool(
        expr.is_Pow and expr.exp.is_even is True and expr.exp.is_integer is True
    )


def _composition_relation_shortcut(
    lhs, rhs, variables, target, domain, order, radius, kind
):
    """Certified elementary composition theorems before constructing a costly atlas."""
    lhs = sp.sympify(lhs)
    rhs = sp.sympify(rhs)
    # exp(h)-1 ~ h, sin(h) ~ h, and 1-cos(h)=O(h**2) as h->0.
    if kind == "o":
        if (
            rhs.is_Pow
            and rhs.exp == sp.Rational(1, 2)
            and sp.simplify(lhs - rhs.base) == 0
        ):
            inner = multivariate_little_o(
                lhs,
                sp.S.One,
                variables,
                target,
                domain=domain,
                order=order,
                radius=radius,
            )
            if inner.certified:
                return MultivariateAsymptoticRelation(
                    "o",
                    lhs,
                    rhs,
                    inner.variables,
                    inner.target,
                    inner.domain,
                    AsymptoticRelationStatus.CERTIFIED,
                    inner.atlas,
                    sp.S.Zero,
                    "positive_power_decay",
                    "h=o(sqrt(h)) for nonnegative h->0",
                    tuple(
                        sorted(
                            set(inner.certificate_backends) | {"positive_power_decay"}
                        )
                    ),
                )
        if lhs.func is sp.sin and len(lhs.args) == 1:
            inner = multivariate_little_o(
                lhs.args[0],
                rhs,
                variables,
                target,
                domain=domain,
                order=order,
                radius=radius,
            )
            if inner.certified:
                return MultivariateAsymptoticRelation(
                    "o",
                    lhs,
                    rhs,
                    inner.variables,
                    inner.target,
                    inner.domain,
                    AsymptoticRelationStatus.CERTIFIED,
                    inner.atlas,
                    sp.S.Zero,
                    "composition_sine_lipschitz",
                    "|sin(h)| <= |h| and h=o(rhs) on the certified atlas",
                    tuple(sorted(set(inner.certificate_backends) | {"sine_lipschitz"})),
                )
        # sqrt(1+h)-1 = h/(sqrt(1+h)+1); for h>=0 the denominator is >=2.
        if lhs.is_Add:
            radicals = [a for a in lhs.args if a.is_Pow and a.exp == sp.Rational(1, 2)]
            if len(radicals) == 1 and sp.simplify(lhs - (radicals[0] - 1)) == 0:
                base = sp.expand(radicals[0].base - 1)
                inner = multivariate_little_o(
                    base,
                    rhs,
                    variables,
                    target,
                    domain=domain,
                    order=order,
                    radius=radius,
                )
                if inner.certified and _structurally_nonnegative(base):
                    return MultivariateAsymptoticRelation(
                        "o",
                        lhs,
                        rhs,
                        inner.variables,
                        inner.target,
                        inner.domain,
                        AsymptoticRelationStatus.CERTIFIED,
                        inner.atlas,
                        sp.S.Zero,
                        "composition_positive_square_root",
                        "sqrt(1+h)-1=h/(sqrt(1+h)+1), denominator >=2, and h=o(rhs)",
                        tuple(
                            sorted(
                                set(inner.certificate_backends)
                                | {"positive_square_root_identity"}
                            )
                        ),
                    )
    if (
        kind == "~"
        and lhs.func is sp.sin
        and len(lhs.args) == 1
        and sp.simplify(rhs - lhs.args[0]) == 0
    ):
        inner = multivariate_little_o(
            lhs.args[0],
            sp.S.One,
            variables,
            target,
            domain=domain,
            order=order,
            radius=radius,
        )
        if inner.certified:
            return MultivariateAsymptoticRelation(
                "~",
                lhs,
                rhs,
                inner.variables,
                inner.target,
                inner.domain,
                AsymptoticRelationStatus.CERTIFIED,
                inner.atlas,
                sp.S.Zero,
                "composition_sinc",
                "sin(h)~h because h=o(1)",
                tuple(sorted(set(inner.certificate_backends) | {"sinc_continuity"})),
            )
    if kind == "~" and rhs == 1:
        if lhs.func is sp.exp and len(lhs.args) == 1:
            inner = multivariate_little_o(
                lhs.args[0],
                sp.S.One,
                variables,
                target,
                domain=domain,
                order=order,
                radius=radius,
            )
            if inner.certified:
                return MultivariateAsymptoticRelation(
                    "~",
                    lhs,
                    rhs,
                    inner.variables,
                    inner.target,
                    inner.domain,
                    AsymptoticRelationStatus.CERTIFIED,
                    inner.atlas,
                    sp.S.Zero,
                    "composition_exp_continuity",
                    "exp(h)~1 because h=o(1)",
                    tuple(sorted(set(inner.certificate_backends) | {"exp_continuity"})),
                )
        if lhs.func is sp.cos and len(lhs.args) == 1:
            inner = multivariate_little_o(
                lhs.args[0],
                sp.S.One,
                variables,
                target,
                domain=domain,
                order=order,
                radius=radius,
            )
            if inner.certified:
                return MultivariateAsymptoticRelation(
                    "~",
                    lhs,
                    rhs,
                    inner.variables,
                    inner.target,
                    inner.domain,
                    AsymptoticRelationStatus.CERTIFIED,
                    inner.atlas,
                    sp.S.Zero,
                    "composition_cos_continuity",
                    "cos(h)~1 because h=o(1)",
                    tuple(sorted(set(inner.certificate_backends) | {"cos_continuity"})),
                )
        if lhs.is_Pow and lhs.exp == sp.Rational(1, 2):
            h = sp.expand(lhs.base - 1)
            if _structurally_nonnegative(h):
                inner = multivariate_little_o(
                    h,
                    sp.S.One,
                    variables,
                    target,
                    domain=domain,
                    order=order,
                    radius=radius,
                )
                if inner.certified:
                    return MultivariateAsymptoticRelation(
                        "~",
                        lhs,
                        rhs,
                        inner.variables,
                        inner.target,
                        inner.domain,
                        AsymptoticRelationStatus.CERTIFIED,
                        inner.atlas,
                        sp.S.Zero,
                        "composition_positive_square_root",
                        "sqrt(1+h)~1 because h=o(1) and h>=0",
                        tuple(
                            sorted(
                                set(inner.certificate_backends)
                                | {"positive_square_root_identity"}
                            )
                        ),
                    )
    return None


def multivariate_big_o(
    lhs, rhs, variables, target, *, domain=True, order=2, radius=None
):
    """Certify ``lhs = O(rhs)`` on a common punctured multivariate neighborhood."""
    lhs, rhs = sp.sympify(lhs), sp.sympify(rhs)
    variables = _normalize_variables(variables)
    target = normalize_limit_target(variables, target)
    domain = sp.sympify(domain)
    # Exact norm inequality sqrt(a^2+b^2+...) <= |a|+|b|+... .
    if lhs.is_Pow and lhs.exp == sp.Rational(1, 2):
        terms = sp.Add.make_args(sp.expand(lhs.base))
        roots = []
        valid = True
        for term in terms:
            base, exponent = term.as_base_exp()
            if exponent.is_even is True and exponent.is_integer is True:
                roots.append(sp.Abs(base ** (exponent // 2)))
            else:
                valid = False
                break
        if valid and roots and sp.simplify(rhs - sp.Add(*roots)) == 0:
            return MultivariateAsymptoticRelation(
                "O",
                lhs,
                rhs,
                variables,
                target,
                domain,
                AsymptoticRelationStatus.CERTIFIED,
                None,
                sp.S.One,
                "euclidean_l1_norm_bound",
                "sqrt(sum a_i^2) <= sum |a_i|",
                ("exact_norm_inequality",),
            )
    # Every |x_i| <= sqrt(sum x_j**2); hence any degree-d coordinate
    # monomial is O((sum x_j**2)**(d/2)).
    try:
        lpoly = sp.Poly(lhs, *variables)
        if len(lpoly.terms()) == 1 and rhs.is_Pow:
            monom, coeff0 = lpoly.terms()[0]
            degree = sum(monom)
            sphere_base = sp.Add(*(v**2 for v in variables))
            if sp.simplify(rhs.base - sphere_base) == 0 and rhs.exp == sp.Rational(
                degree, 2
            ):
                return MultivariateAsymptoticRelation(
                    "O",
                    lhs,
                    rhs,
                    variables,
                    target,
                    domain,
                    AsymptoticRelationStatus.CERTIFIED,
                    None,
                    sp.Abs(coeff0),
                    "coordinate_monomial_norm_bound",
                    "|x_i| <= sqrt(sum x_j^2) bounds the coordinate monomial",
                    ("exact_norm_inequality",),
                )
    except (sp.PolynomialError, TypeError, ValueError):
        pass
    ratio = _atlas_relation_ratio(lhs, rhs)
    atlas = multivariate_atlas(
        ratio,
        variables,
        target,
        order=order,
        domain=domain,
        radius=radius,
        first_certified_family=True,
    )
    bounded = _certified_atlas_bound(atlas, return_backends=True)
    bound, backends = bounded if bounded is not None else (None, ())
    if bound is not None and not bound.has(sp.oo, -sp.oo, sp.zoo, sp.nan):
        return MultivariateAsymptoticRelation(
            "O",
            lhs,
            rhs,
            variables,
            target,
            domain,
            AsymptoticRelationStatus.CERTIFIED,
            atlas,
            bound,
            "atlas_uniform_bound",
            "every covering chart has a certified finite uniform bound for |lhs/rhs|",
            backends,
        )
    return MultivariateAsymptoticRelation(
        "O",
        lhs,
        rhs,
        variables,
        target,
        domain,
        AsymptoticRelationStatus.UNKNOWN,
        atlas=atlas,
        statement="a finite bound was not certified on every covering chart",
    )


def multivariate_little_o(
    lhs, rhs, variables, target, *, domain=True, order=2, radius=None
):
    """Certify ``lhs = o(rhs)`` by atlas-wide uniform decay of ``lhs/rhs``."""
    lhs, rhs = sp.sympify(lhs), sp.sympify(rhs)
    variables = _normalize_variables(variables)
    target = normalize_limit_target(variables, target)
    domain = sp.sympify(domain)
    ratio = _atlas_relation_ratio(lhs, rhs)
    atlas = multivariate_atlas(
        ratio,
        variables,
        target,
        order=max(1, int(order)),
        domain=domain,
        radius=radius,
        first_certified_family=True,
    )
    if atlas.certified:
        uniform_decay = True
        for chart in atlas.charts:
            expansion = chart.expansion
            if any(
                term.order == 0 and term.coefficient != 0 for term in expansion.terms
            ):
                uniform_decay = False
                break
            if any(
                bounded_ask(sp.Q.positive(term.order)) is not True
                for term in expansion.terms
            ):
                uniform_decay = False
                break
            cert = expansion.remainder_certificate
            if cert is None or bounded_ask(sp.Q.positive(cert.order)) is not True:
                uniform_decay = False
                break
        if uniform_decay:
            return MultivariateAsymptoticRelation(
                "o",
                lhs,
                rhs,
                variables,
                target,
                domain,
                AsymptoticRelationStatus.CERTIFIED,
                atlas,
                sp.S.Zero,
                "atlas_uniform_decay",
                "every covering chart has only positive radial orders and a uniformly decaying remainder",
                tuple(
                    sorted(
                        {
                            atlas.coverage_certificate.provider,
                            *(
                                chart.expansion.remainder_certificate.provider
                                for chart in atlas.charts
                                if chart.expansion.remainder_certificate is not None
                            ),
                        }
                        - {"none", ""}
                    )
                ),
            )
    shortcut = _composition_relation_shortcut(
        lhs, rhs, variables, target, domain, order, radius, "o"
    )
    if shortcut is not None:
        return shortcut
    return MultivariateAsymptoticRelation(
        "o",
        lhs,
        rhs,
        variables,
        target,
        domain,
        AsymptoticRelationStatus.UNKNOWN,
        atlas=atlas,
        statement="uniform decay was not certified on every covering chart",
    )


def multivariate_equivalent(
    lhs, rhs, variables, target, *, domain=True, order=2, radius=None
):
    """Certify ``lhs ~ rhs`` via the atlas-wide relation ``lhs/rhs - 1 = o(1)``."""
    lhs, rhs = sp.sympify(lhs), sp.sympify(rhs)
    variables_n = _normalize_variables(variables)
    target_n = normalize_limit_target(variables_n, target)
    domain_n = sp.sympify(domain)
    residual = sp.cancel(sp.together(lhs / rhs - 1))
    result = multivariate_little_o(
        residual, sp.S.One, variables, target, domain=domain, order=order, radius=radius
    )
    if not result.certified:
        shortcut = _composition_relation_shortcut(
            lhs, rhs, variables_n, target_n, domain_n, order, radius, "~"
        )
        if shortcut is not None:
            return shortcut
    return MultivariateAsymptoticRelation(
        "~",
        lhs,
        rhs,
        result.variables,
        result.target,
        result.domain,
        result.status,
        result.atlas,
        result.constant,
        result.provider,
        "lhs/rhs - 1 is uniformly o(1) on the certified atlas"
        if result.certified
        else "asymptotic equivalence was not certified on every covering chart",
        result.certificate_backends,
    )


__all__ = [
    "AsymptoticRelationStatus",
    "CoverageCertificate",
    "ExpansionStatus",
    "MultivariateAsymptoticAtlas",
    "MultivariateAsymptoticChart",
    "MultivariateAsymptoticExpansion",
    "MultivariateAsymptoticRelation",
    "MultivariateAsymptoticTerm",
    "UniformRemainderCertificate",
    "multivariate_atlas",
    "multivariate_big_o",
    "multivariate_equivalent",
    "multivariate_expansion",
    "multivariate_little_o",
]
