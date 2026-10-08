"""Dependency-light adapter for :mod:`odeanalysis` ``FormalODEData`` objects.

No import from ``odeanalysis`` occurs here.  The adapter consumes
explicit interchange schemas and the provider's registered record type, keeping
``asymptotic`` core independent of the ODE package.
"""

from __future__ import annotations

import sys
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
                    "cover_representative_angle": sp.sympify(
                        sector.representative_angle
                    ),
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
        metadata["stokes_original_rays"] = tuple(
            dict.fromkeys(map(sp.sympify, original_rays))
        )
        metadata["stokes_common_ramification"] = int(stokes.common_ramification)

    return TransseriesExpansion.from_terms(
        variable, point, terms, complete=False, metadata=metadata
    )


def _interchange_schema(data, record_name):
    schema = getattr(data, "schema_version", None)
    if schema is None:
        # Published records omit the marker. Identity with a registered provider
        # class supplies the contract without importing or modifying that provider.
        provider = sys.modules.get("odeanalysis.interchange")
        native_record = getattr(provider, record_name, None)
        if native_record is not None and type(data) is native_record:
            return 1
    return schema


def from_formal_ode_data(data: Any, variable: sp.Symbol) -> ODETransseriesData:
    """Convert ``odeanalysis.FormalODEData`` to native asymptotic objects.

    Explicit schema 1 records and the provider's native FormalODEData are
    accepted. ``variable`` is explicit on
    purpose: the interchange object may contain arbitrary symbolic ODE
    parameters, so inferring the independent variable from free symbols would
    be ambiguous and mathematically unsafe.
    """

    if not isinstance(variable, sp.Symbol):
        raise TypeError("variable must be a Symbol")
    schema_version = _interchange_schema(data, "FormalODEData")
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
            None
            if data.local_monodromy is None
            else sp.ImmutableMatrix(data.local_monodromy)
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
    the registered ``FormalODEGreenOperatorData`` or an explicit schema-1
    record. No import from
    :mod:`odeanalysis` occurs, preserving the one-way optional integration.
    Before theorem application the characteristic polynomial is replayed from
    the supplied coefficients; malformed or stale interchange data is rejected.
    """

    schema_version = _interchange_schema(data, "FormalODEGreenOperatorData")
    if schema_version != 1:
        raise FormalODEAdapterError(
            f"unsupported Green-operator schema version {schema_version!r}; expected 1"
        )
    variable = getattr(data, "variable", None)
    if not isinstance(variable, sp.Symbol):
        raise FormalODEAdapterError(
            "Green-operator data must contain a Symbol variable"
        )
    point = sp.sympify(getattr(data, "point", sp.oo))
    coefficients = tuple(sp.sympify(c) for c in getattr(data, "coefficients", ()))
    order = int(getattr(data, "order", -1))
    if order < 1 or len(coefficients) != order + 1:
        raise FormalODEAdapterError(
            "Green-operator coefficient count does not match its order"
        )

    source_lambda = getattr(data, "characteristic_parameter", None)
    source_polynomial = sp.sympify(getattr(data, "characteristic_polynomial", sp.nan))
    lam = sp.Symbol("__lambda")
    if isinstance(source_lambda, sp.Symbol) and source_lambda != lam:
        source_polynomial = source_polynomial.xreplace({source_lambda: lam})
    replayed = sp.expand(sum(coefficients[k] * lam**k for k in range(order + 1)))
    if sp.simplify(sp.expand(source_polynomial - replayed)) != 0:
        raise FormalODEAdapterError(
            "Green-operator characteristic polynomial failed replay"
        )

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


@dataclass(frozen=True)
class ODEFormalSeries:
    """Validated consumer view of one scalar formal ODE series."""

    source: Any
    recurrence: Any | None
    residual_certificate: Any | None
    recurrence_values: tuple[Any, ...]


@dataclass(frozen=True)
class ODESectorDomain:
    """Dependency-light view of an ``odeanalysis`` formal sector domain."""

    source: Any
    index: int | None
    cover_start_angle: sp.Expr | None
    cover_end_angle: sp.Expr | None
    local_start_angle: sp.Expr | None
    local_end_angle: sp.Expr | None
    original_start_angle: sp.Expr | None
    original_end_angle: sp.Expr | None
    ramification_index: int | None
    sheet: int | None
    left_open: bool
    right_open: bool
    dominance_levels: tuple[tuple[int, ...], ...]

    def contains_angle(
        self, angle: sp.Expr, *, coordinate: str = "original"
    ) -> bool | None:
        """Return membership in the selected angular coordinate when decidable."""
        starts = {
            "cover": self.cover_start_angle,
            "local": self.local_start_angle,
            "original": self.original_start_angle,
        }
        ends = {
            "cover": self.cover_end_angle,
            "local": self.local_end_angle,
            "original": self.original_end_angle,
        }
        if coordinate not in starts:
            raise ValueError("coordinate must be 'cover', 'local', or 'original'")
        start, end = starts[coordinate], ends[coordinate]
        if start is None or end is None:
            return None
        theta = sp.sympify(angle)
        # Compare on the circle relative to the start.  This handles wrapped
        # sectors without assuming a particular [-pi,pi) representative.
        width = sp.Mod(sp.sympify(end) - sp.sympify(start), 2 * sp.pi)
        delta = sp.Mod(theta - sp.sympify(start), 2 * sp.pi)
        if sp.simplify(delta).is_zero is True:
            return not self.left_open
        if sp.simplify(delta - width).is_zero is True:
            return not self.right_open
        inside = sp.simplify(sp.And(sp.Gt(delta, 0), sp.Lt(delta, width)))
        if inside is sp.S.true:
            return True
        if inside is sp.S.false:
            return False
        return None


@dataclass(frozen=True)
class ODEFormalInterchange:
    """Validated asymptotic-facing contract for modern ``odeanalysis`` data."""

    formal_data: ODETransseriesData
    scalar_series: tuple[ODEFormalSeries, ...]
    domains: tuple[ODESectorDomain, ...]

    @property
    def solutions(self) -> tuple[TransseriesExpansion, ...]:
        return self.formal_data.solutions

    def validated_residuals(self) -> tuple[Any, ...]:
        """Return residual certificates only after independent ``verify()`` replay."""
        out = []
        for item in self.scalar_series:
            cert = item.residual_certificate
            if cert is None:
                continue
            verifier = getattr(cert, "verify", None)
            if not callable(verifier) or verifier() is not True:
                raise FormalODEAdapterError(
                    "formal residual certificate failed verification"
                )
            _independently_validate_residual(cert)
            out.append(cert)
        return tuple(out)

    def domain_for_angle(
        self, angle: sp.Expr, *, coordinate: str = "original"
    ) -> ODESectorDomain | None:
        """Return the unique recorded sector containing ``angle``, if certified."""
        matches = []
        for domain in self.domains:
            verdict = domain.contains_angle(angle, coordinate=coordinate)
            if verdict is True:
                matches.append(domain)
            elif verdict is None:
                return None
        return matches[0] if len(matches) == 1 else None

    def dominance_at(
        self, angle: sp.Expr, *, coordinate: str = "original"
    ) -> tuple[tuple[int, ...], ...] | None:
        """Return the per-level block dominance ranking at an angular point."""
        domain = self.domain_for_angle(angle, coordinate=coordinate)
        return None if domain is None else domain.dominance_levels


def _verified_interchange_object(obj: Any, what: str) -> None:
    verifier = getattr(obj, "verify", None)
    if not callable(verifier):
        raise FormalODEAdapterError(f"{what} does not expose verify()")
    if verifier() is not True:
        raise FormalODEAdapterError(f"{what} failed verification")


def _independently_validate_residual(cert: Any) -> None:
    """Replay exact-zero/valuation claims from exported residual expressions."""
    if not hasattr(cert, "residual"):
        return
    residual = sp.cancel(sp.together(sp.sympify(cert.residual)))
    exact_zero = getattr(cert, "exact_zero", getattr(cert, "is_exact_zero", None))
    if exact_zero is not None:
        actual_zero = residual == 0 or sp.simplify(residual) == 0
        if bool(exact_zero) != actual_zero:
            raise FormalODEAdapterError(
                "formal residual exact-zero claim failed independent verification"
            )
    valuation = getattr(cert, "certified_valuation", None)
    parameter = getattr(cert, "local_parameter", None)
    if valuation is None or not isinstance(parameter, sp.Symbol) or residual == 0:
        return
    try:
        leading = sp.expand(residual.as_leading_term(parameter))
        exponent = leading.as_powers_dict().get(parameter, sp.S.Zero)
        coefficient = sp.simplify(leading / parameter**exponent)
    except (ValueError, NotImplementedError, sp.PoleError):
        raise FormalODEAdapterError(
            "formal residual valuation could not be independently verified"
        ) from None
    if parameter in coefficient.free_symbols or sp.simplify(exponent - valuation) != 0:
        raise FormalODEAdapterError(
            "formal residual valuation failed independent verification"
        )


def _recurrence_values(recurrence: Any) -> tuple[Any, ...]:
    """Read generated recurrence values without guessing mathematical data."""
    for name in ("values", "coefficient_values", "generated_values"):
        value = getattr(recurrence, name, None)
        if value is not None and not callable(value):
            return tuple(value)
    for name in ("generate_values", "recurrence_values"):
        method = getattr(recurrence, name, None)
        if callable(method):
            return tuple(method())
    return ()


def _sector_attr(obj: Any, *names: str) -> Any | None:
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


def _convert_sector_domain(domain: Any) -> ODESectorDomain:
    levels = tuple(
        tuple(map(int, level)) for level in getattr(domain, "dominance_levels", ())
    )
    return ODESectorDomain(
        source=domain,
        index=int(domain.index) if getattr(domain, "index", None) is not None else None,
        cover_start_angle=_sector_attr(domain, "cover_start_angle", "start_angle"),
        cover_end_angle=_sector_attr(domain, "cover_end_angle", "end_angle"),
        local_start_angle=_sector_attr(domain, "local_start_angle"),
        local_end_angle=_sector_attr(domain, "local_end_angle"),
        original_start_angle=_sector_attr(domain, "original_start_angle"),
        original_end_angle=_sector_attr(domain, "original_end_angle"),
        ramification_index=(
            int(_sector_attr(domain, "ramification_index", "ramification"))
            if _sector_attr(domain, "ramification_index", "ramification") is not None
            else None
        ),
        sheet=(
            int(domain.sheet) if getattr(domain, "sheet", None) is not None else None
        ),
        left_open=bool(
            getattr(domain, "left_open", getattr(domain, "start_open", True))
        ),
        right_open=bool(
            getattr(domain, "right_open", getattr(domain, "end_open", True))
        ),
        dominance_levels=levels,
    )


def from_odeanalysis_formal_data(
    data: Any, variable: sp.Symbol
) -> ODEFormalInterchange:
    """Consume the modern ``odeanalysis`` formal-series interchange contract.

    The core formal blocks are converted by :func:`from_formal_ode_data`.
    Optional scalar-series recurrences and residual certificates are accepted
    only after their own ``verify()`` methods succeed.  Sector/domain data is
    preserved in cover, local, and original angular coordinates, including
    ramified sheet and per-level dominance information.
    """
    converted = from_formal_ode_data(data, variable)
    series_sources = tuple(getattr(data, "scalar_series", ()) or ())
    series = []
    for source in series_sources:
        recurrence = getattr(source, "recurrence", None)
        if recurrence is not None:
            _verified_interchange_object(recurrence, "formal coefficient recurrence")
        residual = getattr(
            source, "residual_certificate", getattr(source, "residual", None)
        )
        if residual is not None:
            _verified_interchange_object(residual, "formal residual certificate")
            _independently_validate_residual(residual)
        series.append(
            ODEFormalSeries(
                source,
                recurrence,
                residual,
                _recurrence_values(recurrence) if recurrence is not None else (),
            )
        )

    raw_domains = tuple(getattr(data, "domains", ()) or ())
    if not raw_domains:
        stokes = getattr(data, "stokes", None)
        raw_domains = (
            tuple(
                getattr(stokes, "domains", ()) or getattr(stokes, "sectors", ()) or ()
            )
            if stokes is not None
            else ()
        )
    domains = tuple(_convert_sector_domain(domain) for domain in raw_domains)
    return ODEFormalInterchange(converted, tuple(series), domains)
