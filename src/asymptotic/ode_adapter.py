"""Dependency-light adapter for :mod:`odeanalysis` ``FormalODEData`` objects.

No import from ``odeanalysis`` occurs here.  The adapter intentionally consumes
the versioned public interchange schema by structural attribute access, keeping
``asymptotic`` core independent of the ODE package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sympy as sp

from ._power_simplify import analytic_powsimp
from .monomial import (
    AsymptoticMonomial,
    RamificationModel,
    canonical_parameter_monomial,
)
from .remainder_theorems import (
    GreenOperatorCertificate,
    RemainderTheoremCertificate,
    certify_green_inverse_operator_remainder,
)
from .transseries import TransseriesExpansion, TransseriesTerm


class FormalODEAdapterError(ValueError):
    """Raised when an ODE interchange object cannot be mapped safely."""


@dataclass(frozen=True)
class ODETransseriesBlock:
    """One ODE exponential block converted to native transseries solutions."""

    index: int
    ramification: RamificationModel
    exponential_monomial: AsymptoticMonomial
    semisimple_exponent: sp.ImmutableMatrix
    nilpotent_exponent: sp.ImmutableMatrix
    formal_exponent_matrix: sp.ImmutableMatrix
    solutions: tuple[TransseriesExpansion, ...]
    cover_monodromy: sp.ImmutableMatrix
    has_logarithms: bool

    @property
    def dimension(self) -> int:
        return len(self.solutions)


@dataclass(frozen=True)
class ODETransseriesData:
    """Native-asymptotic view of a ``FormalODEData`` interchange object."""

    schema_version: int
    variable: sp.Symbol
    point: sp.Expr
    ramification_index: int
    blocks: tuple[ODETransseriesBlock, ...]
    cover_monodromy: sp.ImmutableMatrix
    local_monodromy: sp.ImmutableMatrix | None
    stokes: Any | None
    complete: bool
    limitation: str | None

    @property
    def solutions(self) -> tuple[TransseriesExpansion, ...]:
        return tuple(solution for block in self.blocks for solution in block.solutions)

    @property
    def dimension(self) -> int:
        return sum(block.dimension for block in self.blocks)


def _block_exponential_on_parameter(block: Any, ram: RamificationModel) -> sp.Expr:
    h = sp.sympify(block.local_coordinate)
    q_h = sp.sympify(block.local_exponential_polynomial)
    # local_coordinate is the formal h symbol in odeanalysis; h=t**r on the
    # block cover.  xreplace is exact and avoids assumptions about its name.
    return analytic_powsimp(sp.expand(q_h.xreplace({h: ram.parameter**ram.index})))


def _solution_from_vector(
    vector: Any,
    block: Any,
    variable: sp.Symbol,
    point: sp.Expr,
    ram: RamificationModel,
    q_t: sp.Expr,
    stokes: Any | None = None,
) -> TransseriesExpansion:
    source_t = vector.local_parameter
    amp = sp.expand(sp.sympify(vector.amplitude_parameter))
    if source_t != ram.parameter:
        amp = amp.xreplace({source_t: ram.parameter})

    terms = []
    for summand in sp.Add.make_args(amp):
        coefficient, amplitude_monomial = canonical_parameter_monomial(summand, ram)
        full_monomial = AsymptoticMonomial(
            ram,
            sp.expand(q_t + amplitude_monomial.exponential),
            amplitude_monomial.power,
            amplitude_monomial.log_power,
        )
        terms.append(TransseriesTerm(coefficient, full_monomial))

    metadata: dict[str, object] = {
        "source": "odeanalysis.FormalODEData",
        "block_index": int(block.index),
        "source_exponent": sp.sympify(vector.source_exponent),
        "formal_exponent": sp.sympify(vector.exponent),
        "ramified_exponent": sp.sympify(vector.ramified_exponent),
        "logarithmic_degree": int(vector.logarithmic_degree),
        "source_expression": sp.sympify(vector.expression),
    }
    if stokes is not None:
        memberships = []
        for sector in getattr(stokes, "sectors", ()):
            level_index = next(
                (
                    level
                    for level, blocks in enumerate(sector.dominance_levels)
                    if int(block.index) in blocks
                ),
                None,
            )
            if level_index is None:
                continue
            memberships.append(
                {
                    "sector_index": int(sector.index),
                    "dominance_level": level_index,
                    "cover_start_angle": sp.sympify(sector.start_angle),
                    "cover_end_angle": sp.sympify(sector.end_angle),
                    "cover_representative_angle": sp.sympify(sector.representative_angle),
                    "local_representative_angle": getattr(
                        sector, "local_representative_angle", None
                    ),
                    "original_representative_angle": getattr(
                        sector, "original_representative_angle", None
                    ),
                    "sheet": getattr(sector, "sheet", None),
                }
            )
        original_rays = []
        for pair in getattr(stokes, "pairs", ()):
            if int(block.index) not in pair.block_indices:
                continue
            original_rays.extend(getattr(pair, "equal_magnitude_original_angles", ()))
        metadata["stokes_sector_membership"] = tuple(memberships)
        metadata["stokes_original_rays"] = tuple(dict.fromkeys(map(sp.sympify, original_rays)))
        metadata["stokes_common_ramification"] = int(stokes.common_ramification)

    return TransseriesExpansion.from_terms(
        variable, point, terms, complete=False, metadata=metadata
    )


def from_formal_ode_data(data: Any, variable: sp.Symbol) -> ODETransseriesData:
    """Convert ``odeanalysis.FormalODEData`` to native asymptotic objects.

    Only schema version 1 is accepted.  ``variable`` is explicit on
    purpose: the interchange object may contain arbitrary symbolic ODE
    parameters, so inferring the independent variable from free symbols would
    be ambiguous and mathematically unsafe.
    """

    if not isinstance(variable, sp.Symbol):
        raise TypeError("variable must be a Symbol")
    schema_version = getattr(data, "schema_version", None)
    if schema_version != 1:
        raise FormalODEAdapterError(
            f"unsupported FormalODEData schema version {schema_version!r}; expected 1"
        )

    point = sp.sympify(data.point)
    converted = []
    for block in data.blocks:
        source_t = block.local_parameter
        ram = RamificationModel(
            variable,
            point,
            int(block.ramification_index),
            source_t,
        )
        q_t = _block_exponential_on_parameter(block, ram)
        exponential_monomial = AsymptoticMonomial(ram, q_t)
        solutions = tuple(
            _solution_from_vector(
                vector, block, variable, point, ram, q_t, getattr(data, "stokes", None)
            )
            for vector in block.basis_vectors
        )
        converted.append(
            ODETransseriesBlock(
                index=int(block.index),
                ramification=ram,
                exponential_monomial=exponential_monomial,
                semisimple_exponent=sp.ImmutableMatrix(block.semisimple_exponent),
                nilpotent_exponent=sp.ImmutableMatrix(block.nilpotent_exponent),
                formal_exponent_matrix=sp.ImmutableMatrix(block.formal_exponent_matrix),
                solutions=solutions,
                cover_monodromy=sp.ImmutableMatrix(block.cover_monodromy),
                has_logarithms=bool(block.has_logarithms),
            )
        )

    return ODETransseriesData(
        schema_version=1,
        variable=variable,
        point=point,
        ramification_index=int(data.ramification_index),
        blocks=tuple(converted),
        cover_monodromy=sp.ImmutableMatrix(data.cover_monodromy),
        local_monodromy=(
            None if data.local_monodromy is None else sp.ImmutableMatrix(data.local_monodromy)
        ),
        stokes=getattr(data, "stokes", None),
        complete=bool(data.complete),
        limitation=getattr(data, "limitation", None),
    )


def certify_green_operator_data(
    data: Any,
    residual: sp.Expr,
) -> tuple[RemainderTheoremCertificate, GreenOperatorCertificate | None]:
    """Certify an ``odeanalysis`` scalar-operator Green inverse.

    ``data`` is consumed structurally and is expected to follow
    ``FormalODEGreenOperatorData`` schema version 1.  No import from
    :mod:`odeanalysis` occurs, preserving the one-way optional integration.
    Before theorem application the characteristic polynomial is replayed from
    the supplied coefficients; malformed or stale interchange data is rejected.
    """

    schema_version = getattr(data, "schema_version", None)
    if schema_version != 1:
        raise FormalODEAdapterError(
            f"unsupported Green-operator schema version {schema_version!r}; expected 1"
        )
    variable = getattr(data, "variable", None)
    if not isinstance(variable, sp.Symbol):
        raise FormalODEAdapterError("Green-operator data must contain a Symbol variable")
    point = sp.sympify(getattr(data, "point", sp.oo))
    coefficients = tuple(sp.sympify(c) for c in getattr(data, "coefficients", ()))
    order = int(getattr(data, "order", -1))
    if order < 1 or len(coefficients) != order + 1:
        raise FormalODEAdapterError("Green-operator coefficient count does not match its order")

    source_lambda = getattr(data, "characteristic_parameter", None)
    source_polynomial = sp.sympify(getattr(data, "characteristic_poly", sp.nan))
    lam = sp.Symbol("__lambda")
    if isinstance(source_lambda, sp.Symbol) and source_lambda != lam:
        source_polynomial = source_polynomial.xreplace({source_lambda: lam})
    replayed = sp.expand(sum(coefficients[k] * lam**k for k in range(order + 1)))
    if sp.simplify(sp.expand(source_polynomial - replayed)) != 0:
        raise FormalODEAdapterError("Green-operator characteristic polynomial failed replay")

    delta = sp.Function("__green_delta")
    delta_x = delta(variable)
    linearized_operator = sp.expand(
        sum(coefficients[k] * sp.diff(delta_x, variable, k) for k in range(order + 1))
    )
    return certify_green_inverse_operator_remainder(
        sp.sympify(residual),
        linearized_operator,
        delta,
        variable,
        point,
    )
