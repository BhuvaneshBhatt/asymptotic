"""Backend-neutral data model for mathematical function properties.

The records in this module have no dependencies on the rest of ``asymptotic``.
They describe function properties without coupling them to expansion algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import sympy as sp


class ArgumentDomain(str, Enum):
    """Coarse mathematical domains used in function signatures."""

    COMPLEX = "complex"
    REAL = "real"
    POSITIVE_REAL = "positive_real"
    NONNEGATIVE_REAL = "nonnegative_real"
    RATIONAL = "rational"
    INTEGER = "integer"
    POSITIVE_INTEGER = "positive_integer"
    NONNEGATIVE_INTEGER = "nonnegative_integer"
    ANY = "any"


@dataclass(frozen=True)
class ArgumentSpec:
    """One argument slot in a mathematical function signature.

    ``shape`` is intentionally symbolic: ``None`` denotes a scalar, while a
    tuple can represent vector/matrix dimensions.  ``variadic`` represents a
    repeated final slot. This supports variable-arity special-function entries
    without baking a CAS-specific pattern language into the model.
    """

    domain: ArgumentDomain = ArgumentDomain.COMPLEX
    shape: tuple[sp.Expr | int | None, ...] | None = None
    alternatives: tuple[sp.Expr, ...] = ()
    variadic: bool = False


@dataclass(frozen=True)
class ArgumentSignature:
    """One supported argument structure for a function head."""

    arguments: tuple[ArgumentSpec, ...]


@dataclass(frozen=True)
class AssumptionProperties:
    """Sufficient conditions for value-level predicates."""

    integer: sp.Expr | None = None
    rational: sp.Expr | None = None
    real: sp.Expr | None = None
    real_if_defined: sp.Expr | None = None
    positive: sp.Expr | None = None
    negative: sp.Expr | None = None
    nonpositive: sp.Expr | None = None
    nonnegative: sp.Expr | None = None


@dataclass(frozen=True)
class DomainProperties:
    """Real and complex domains for one concrete function expression."""

    arguments: tuple[sp.Expr, ...]
    real_domain: sp.Expr | None = None
    complex_domain: sp.Expr | None = None


@dataclass(frozen=True)
class SingularityLocus:
    """A symbolic singularity/cut locus with optional discontinuity data."""

    condition: sp.Expr
    parameter_condition: sp.Expr = sp.S.true
    jump: sp.Expr | None = None
    multiplicity: sp.Expr | None = None
    note: str | None = None


@dataclass(frozen=True)
class SingularityProperties:
    """Analytic/singularity information for a concrete function expression.

    A locus collection is ``None`` when no information has been registered and
    ``()`` when the registry knows the collection is empty.
    """

    locally_analytic: bool | None = None
    branch_cuts: tuple[SingularityLocus, ...] | None = None
    definition_cuts: tuple[SingularityLocus, ...] | None = None
    poles: tuple[SingularityLocus, ...] | None = None
    essential: tuple[SingularityLocus, ...] | None = None
    branch_points: tuple[SingularityLocus, ...] | None = None


@dataclass(frozen=True)
class DomainEndpoint:
    """One endpoint of a real-domain interval."""

    point: sp.Expr
    interior_limit: sp.Expr | None = None
    value: sp.Expr | None = None


@dataclass(frozen=True)
class DomainInterval:
    """One real-domain interval with endpoint behavior."""

    left: DomainEndpoint
    right: DomainEndpoint


@dataclass(frozen=True)
class Discontinuity:
    """One (possibly parameterized) real discontinuity locus."""

    condition: sp.Expr
    left_limit: sp.Expr | None = None
    value: sp.Expr | None = None
    right_limit: sp.Expr | None = None


@dataclass(frozen=True)
class GlobalExtremum:
    """A global extremal value and where/how it is attained."""

    condition: sp.Expr
    value: sp.Expr
    attained: bool


@dataclass(frozen=True)
class RealUnivariateProperties:
    """Optional real-univariate behavior with other arguments as parameters."""

    variable: sp.Expr
    domain_intervals: tuple[DomainInterval, ...] | None = None
    discontinuities: tuple[Discontinuity, ...] | None = None
    global_monotonicity: int | None = None
    local_minima: sp.Expr | None = None
    local_maxima: sp.Expr | None = None
    global_minimum: GlobalExtremum | None = None
    global_maximum: GlobalExtremum | None = None
    global_convexity: int | None = None
    falling_inflections: sp.Expr | None = None
    rising_inflections: sp.Expr | None = None
    affine_intervals: sp.Expr | None = None
    inflection_singularities: sp.Expr | None = None
    range: sp.Set | sp.Expr | None = None
    injective: bool | None = None


@dataclass(frozen=True)
class FunctionProperties:
    """Composite properties for one concrete mathematical function expression.

    The grouped records are the canonical representation.  Compatibility
    properties expose the original flat ``asymptotic`` API while consumers
    migrate to ``.domain``, ``.assumptions``, ``.singularities``, and
    ``.real_univariate``.
    """

    expression: sp.Expr
    arguments: tuple[sp.Expr, ...]
    argument_signatures: tuple[ArgumentSignature, ...] = ()
    assumptions: AssumptionProperties | None = None
    domain: DomainProperties | None = None
    singularities: SingularityProperties | None = None
    real_univariate: RealUnivariateProperties | None = None

    @property
    def real_domain(self) -> sp.Expr | None:
        return None if self.domain is None else self.domain.real_domain

    @property
    def complex_domain(self) -> sp.Expr | None:
        return None if self.domain is None else self.domain.complex_domain

    @property
    def real_valued(self) -> sp.Expr | None:
        return None if self.assumptions is None else self.assumptions.real

    @property
    def locally_analytic(self) -> bool | None:
        return None if self.singularities is None else self.singularities.locally_analytic

    @property
    def branch_cuts(self) -> tuple[SingularityLocus, ...] | None:
        return None if self.singularities is None else self.singularities.branch_cuts

    @property
    def definition_cuts(self) -> tuple[SingularityLocus, ...] | None:
        return None if self.singularities is None else self.singularities.definition_cuts

    @property
    def poles(self) -> tuple[SingularityLocus, ...] | None:
        return None if self.singularities is None else self.singularities.poles

    @property
    def essential_singularities(self) -> tuple[SingularityLocus, ...] | None:
        return None if self.singularities is None else self.singularities.essential

    @property
    def branch_points(self) -> tuple[SingularityLocus, ...] | None:
        return None if self.singularities is None else self.singularities.branch_points

    @property
    def real_range(self) -> sp.Set | sp.Expr | None:
        return None if self.real_univariate is None else self.real_univariate.range
