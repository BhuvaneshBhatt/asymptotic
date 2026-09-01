"""Computable asymptotic differential fields, shadows, and ghosts.

This module implements the executable part of Shackell's Definitions 18--22:
R_t, I_t, shadow fields/projections, ghosts, and compatible projection maps.
It is intentionally certificate-driven: undecidable growth or zero tests do not
silently become field membership assertions.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import sympy as sp

from ._symbolic_errors import SYMBOLIC_ERRORS
from ._symbolic_policy import bounded_primitive
from .context import AsymptoticContext
from .function_properties import PropertyDecision, PropertyKnowledge, PropertyProvenance


@dataclass(frozen=True)
class GrowthIdealDecision:
    expression: sp.Expr
    scale: sp.Expr
    verdict: bool | None
    exponent_ratio: sp.Expr | None
    reason: str


@dataclass(frozen=True)
class ShadowGhostDecomposition:
    expression: sp.Expr
    shadow: sp.Expr
    ghost: sp.Expr
    scale: sp.Expr
    certified: bool
    decisions: tuple[GrowthIdealDecision, ...] = ()

    def replay(self) -> bool | None:
        if sp.simplify(self.expression - self.shadow - self.ghost) != 0:
            return False
        if not self.certified:
            return None
        return True


class IntegrationConstantLocation(str, Enum):
    """Where an arbitrary constant lives relative to one shadow projection."""

    FIXED = "fixed"
    SHADOW = "shadow"
    GHOST = "ghost"
    CHANGES_LEADING_SCALE = "changes_leading_scale"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class IntegralShadowExtension:
    """Closed-form differential-field extension g' = f with explicit constant.

    ``primitive`` is the zero-constant representative. It may be an unevaluated
    SymPy Integral. ``constant`` is tracked independently because its location
    may change from one shadow field to another.
    """

    variable: sp.Symbol
    integrand: sp.Expr
    primitive: sp.Expr
    constant: sp.Expr
    point: sp.Expr
    leading_monomial: sp.Expr | None = None
    name: str | None = None

    @property
    def expression(self) -> sp.Expr:
        return sp.Add(self.primitive, self.constant, evaluate=False)

    def verify_differential_equation(self) -> bool | None:
        try:
            return sp.simplify(sp.diff(self.primitive, self.variable) - self.integrand) == 0
        except SYMBOLIC_ERRORS:
            return None


@dataclass(frozen=True)
class IntegralShadowProjection:
    """Shackell-style normalized integral shadow/ghost decomposition."""

    extension: IntegralShadowExtension
    scale: sp.Expr
    leading_monomial: sp.Expr
    normalized_expression: sp.Expr
    normalized_shadow: sp.Expr
    normalized_ghost: sp.Expr
    constant_location: IntegrationConstantLocation
    certified: bool
    decisions: tuple[GrowthIdealDecision, ...] = ()
    note: str | None = None

    @property
    def shadow(self) -> sp.Expr:
        return sp.simplify(self.leading_monomial * self.normalized_shadow)

    @property
    def ghost(self) -> sp.Expr:
        return sp.simplify(self.leading_monomial * self.normalized_ghost)

    def replay(self) -> bool | None:
        if (
            sp.simplify(self.normalized_expression - self.normalized_shadow - self.normalized_ghost)
            != 0
        ):
            return False
        if self.extension.verify_differential_equation() is False:
            return False
        return True if self.certified else None


def _log_exponent_ratio(
    expr: sp.Expr, scale: sp.Expr, variable: sp.Symbol, point: sp.Expr
) -> sp.Expr | None:
    """Return lim log|expr|/log|scale| when decidable."""
    expr, scale = sp.sympify(expr), sp.sympify(scale)
    if expr == 0:
        return sp.oo
    ctx = AsymptoticContext(variable, point=point)
    try:
        value = sp.simplify(ctx.limit(sp.log(sp.Abs(expr)) / sp.log(sp.Abs(scale))))
    except SYMBOLIC_ERRORS:
        return None
    if value in (sp.nan, sp.zoo):
        return None
    return value


def moderate_growth_decision(
    expr: sp.Expr, scale: sp.Expr, variable: sp.Symbol, *, point: sp.Expr = sp.oo
) -> GrowthIdealDecision:
    """Decide membership in R_t when the logarithmic growth exponent is finite.

    R_t consists of functions smaller than t**(-epsilon) for every epsilon>0.
    A logarithmic exponent ratio >= 0 certifies this in the supported Hardy/
    log-exp setting. Positive ratio certifies non-membership.
    """
    ratio = _log_exponent_ratio(expr, scale, variable, point)
    if expr == 0 or ratio is sp.oo:
        verdict = True
    elif ratio is -sp.oo:
        verdict = False
    elif ratio is None or ratio.is_real is False:
        verdict = None
    elif ratio.is_number:
        verdict = bool(ratio >= 0)
    else:
        verdict = None
    return GrowthIdealDecision(
        sp.sympify(expr), sp.sympify(scale), verdict, ratio, "R_t moderate-growth test"
    )


def infinitesimal_ideal_decision(
    expr: sp.Expr, scale: sp.Expr, variable: sp.Symbol, *, point: sp.Expr = sp.oo
) -> GrowthIdealDecision:
    """Decide membership in I_t from a positive logarithmic scale exponent."""
    ratio = _log_exponent_ratio(expr, scale, variable, point)
    if expr == 0 or ratio is sp.oo:
        verdict = True
    elif ratio is -sp.oo:
        verdict = False
    elif ratio is None or ratio.is_real is False:
        verdict = None
    elif ratio.is_number:
        verdict = bool(ratio > 0)
    else:
        verdict = None
    return GrowthIdealDecision(
        sp.sympify(expr), sp.sympify(scale), verdict, ratio, "I_t positive-power test"
    )


@dataclass
class ShadowField:
    """A computable shadow field attached to one scale element."""

    variable: sp.Symbol
    scale: sp.Expr
    point: sp.Expr = sp.oo
    name: str | None = None
    zero_test: Callable[[sp.Expr], bool | None] | None = None

    def element(self, expr: sp.Expr):
        """View a field expression through the common asymptotic-element protocol."""
        from .algebra import asymptotic_element

        return asymptotic_element(sp.sympify(expr), self.variable, point=self.point)

    def _zero(self, expr: sp.Expr) -> bool | None:
        expr = sp.simplify(expr)
        if expr == 0 or expr.is_zero is True:
            return True
        if expr.is_zero is False:
            return False
        if self.zero_test is not None:
            return self.zero_test(expr)
        return None

    def project(self, expr: sp.Expr, *, strict: bool = True) -> ShadowGhostDecomposition:
        """Compute the shadow homomorphism on the supported exp-log field.

        The map is recursive on field operations.  In particular eta(exp f)=
        exp(eta(f)) when f has an infinitesimal ghost, matching the exponential
        extension rule; the ghost is always retained in exact closed form.
        """
        expr = sp.sympify(expr)
        moderate = moderate_growth_decision(expr, self.scale, self.variable, point=self.point)
        if moderate.verdict is False:
            raise ValueError("shadow projection is defined only on R_t")
        decisions: list[GrowthIdealDecision] = [moderate]

        def eta(node: sp.Expr) -> sp.Expr:
            node = sp.sympify(node)
            if not node.has(self.variable):
                return node
            d = infinitesimal_ideal_decision(node, self.scale, self.variable, point=self.point)
            decisions.append(d)
            if d.verdict is True:
                return sp.S.Zero
            if node.is_Add:
                return sp.simplify(sp.Add(*(eta(a) for a in node.args)))
            if node.is_Mul:
                return sp.simplify(sp.Mul(*(eta(a) for a in node.args)))
            if node.is_Pow and node.exp.is_integer:
                base = eta(node.base)
                if base == 0 and node.exp.is_negative:
                    if strict:
                        raise ValueError("projection would invert an infinitesimal")
                    return node
                return sp.simplify(base**node.exp)
            if node.func == sp.exp:
                inner = self.project(node.args[0], strict=strict)
                if inner.certified:
                    return sp.exp(inner.shadow)
            # A non-infinitesimal atomic/subfield element is itself part of S.
            if d.verdict is False:
                return node
            if strict:
                raise ValueError(f"cannot certify shadow/ghost status of {node}")
            return node

        shadow = sp.simplify(eta(sp.expand(expr)))
        ghost = sp.simplify(expr - shadow)
        gd = infinitesimal_ideal_decision(ghost, self.scale, self.variable, point=self.point)
        decisions.append(gd)
        certified = moderate.verdict is True and (ghost == 0 or gd.verdict is True)
        if strict and not certified:
            raise ValueError("computed projection did not certify an I_t ghost")
        return ShadowGhostDecomposition(
            expr, shadow, ghost, self.scale, certified, tuple(decisions)
        )

    def shadow(self, expr: sp.Expr, *, strict: bool = True) -> sp.Expr:
        return self.project(expr, strict=strict).shadow

    def ghost(self, expr: sp.Expr, *, strict: bool = True) -> sp.Expr:
        return self.project(expr, strict=strict).ghost

    def relative_derivative(self, a: sp.Expr, b: sp.Expr) -> sp.Expr:
        bd = sp.diff(b, self.variable)
        if self._zero(bd) is True:
            raise ZeroDivisionError("relative derivative denominator has zero derivative")
        return sp.simplify(sp.diff(a, self.variable) / bd)

    def verify_sfii(self, elements: tuple[sp.Expr, ...]) -> PropertyDecision:
        """Finite replay of S intersect I_t={0} on supplied field elements."""
        verdict: bool | None = True
        reasons = []
        for element in elements:
            d = infinitesimal_ideal_decision(element, self.scale, self.variable, point=self.point)
            z = self._zero(element)
            if d.verdict is True and z is False:
                verdict = False
                reasons.append(f"nonzero element {element} lies in I_t")
                break
            if d.verdict is None or z is None:
                verdict = None if verdict is True else verdict
        return PropertyDecision(
            sp.Symbol("shadow_property_SFii"),
            verdict,
            sp.S.true,
            PropertyKnowledge.SUFFICIENT,
            (PropertyProvenance("asymptotic.shadow_field", note="finite SF(ii) replay"),),
            tuple(reasons) or ("no supplied nonzero shadow element was certified infinitesimal",),
        )


@dataclass
class AsymptoticDifferentialField:
    """Finite computable realization of an asymptotic field with shadow maps."""

    variable: sp.Symbol
    scales: tuple[sp.Expr, ...]
    point: sp.Expr = sp.oo
    shadow_fields: tuple[ShadowField, ...] = field(init=False)
    integral_extensions: list[IntegralShadowExtension] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.scales = tuple(map(sp.sympify, self.scales))
        self.shadow_fields = tuple(
            ShadowField(self.variable, t, self.point, name=f"S{i + 1}")
            for i, t in enumerate(self.scales)
        )

    def element(self, expr: sp.Expr):
        """View an expression in this differential field through the common protocol."""
        from .algebra import asymptotic_element

        return asymptotic_element(sp.sympify(expr), self.variable, point=self.point)

    def add_integral_extension(
        self,
        integrand: sp.Expr,
        *,
        primitive: sp.Expr | None = None,
        constant: sp.Expr = 0,
        leading_monomial: sp.Expr | None = None,
        name: str | None = None,
    ) -> IntegralShadowExtension:
        """Adjoin g with g'=integrand while retaining its arbitrary constant."""
        f = sp.sympify(integrand)
        if primitive is None:
            primitive = bounded_primitive(f, self.variable, allow_general=True, risch=True)
            if primitive is None:
                primitive = sp.Integral(f, self.variable)
        ext = IntegralShadowExtension(
            self.variable,
            f,
            sp.sympify(primitive),
            sp.sympify(constant),
            self.point,
            None if leading_monomial is None else sp.sympify(leading_monomial),
            name,
        )
        if ext.verify_differential_equation() is False:
            raise ValueError("primitive does not satisfy g' = integrand")
        self.integral_extensions.append(ext)
        return ext

    def _integral_leading_monomial(self, ext: IntegralShadowExtension) -> sp.Expr | None:
        if ext.leading_monomial is not None:
            return ext.leading_monomial
        if not ext.primitive.has(sp.Integral):
            try:
                from .transseries import transseries_from_expression

                norm = transseries_from_expression(
                    ext.primitive, self.variable, point=self.point
                ).normalized()
                if norm.terms:
                    return norm.terms[0].expression
                if norm.center != 0:
                    return norm.center
            except SYMBOLIC_ERRORS:
                pass
        try:
            from .general_ops import asymptotic_integrate

            norm = asymptotic_integrate(
                ext.integrand, self.variable, point=self.point, terms=1
            ).normalized()
            if norm.terms:
                return norm.terms[0].expression
            if norm.center != 0:
                return norm.center
        except SYMBOLIC_ERRORS:
            return None
        return None

    def project_integral(
        self,
        index: int,
        extension: IntegralShadowExtension,
        *,
        strict: bool = True,
    ) -> IntegralShadowProjection:
        """Project an integral extension with Shackell's normalized formula.

        For leading monomial T, the constructed shadow of g/T is
        ``T**-1 * Integral(T' * eta_i(f/T'), x)``. Integration constants are
        separately classified as shadow, ghost, or leading-scale changing.
        """
        sf = self.shadow_fields[index]
        t = sf.scale
        if not extension.primitive.has(sp.Integral):
            try:
                direct = sf.project(
                    sp.simplify(extension.primitive + extension.constant), strict=True
                )
                if extension.constant == 0:
                    loc = IntegrationConstantLocation.FIXED
                else:
                    cd = infinitesimal_ideal_decision(
                        extension.constant, t, self.variable, point=self.point
                    )
                    loc = (
                        IntegrationConstantLocation.GHOST
                        if cd.verdict is True
                        else IntegrationConstantLocation.SHADOW
                    )
                return IntegralShadowProjection(
                    extension,
                    t,
                    sp.S.One,
                    sp.simplify(extension.primitive + extension.constant),
                    direct.shadow,
                    direct.ghost,
                    loc,
                    direct.certified,
                    direct.decisions,
                    "explicit primitive projected by the existing shadow homomorphism",
                )
            except ValueError:
                pass

        T = self._integral_leading_monomial(extension)
        if T is None or T == 0:
            if strict:
                raise ValueError("could not certify a leading monomial for the integral extension")
            T = sp.S.One
        Td = sp.simplify(sp.diff(T, self.variable))
        if Td == 0:
            if strict:
                raise ValueError("integral leading monomial has zero derivative")
            normalized = sp.simplify(extension.expression / T)
            return IntegralShadowProjection(
                extension,
                t,
                T,
                normalized,
                sp.S.Zero,
                normalized,
                IntegrationConstantLocation.UNRESOLVED,
                False,
                (),
                "T' = 0",
            )
        ratio = sp.simplify(extension.integrand / Td)
        projected_ratio = sf.project(ratio, strict=strict)
        shadow_integrand = sp.simplify(Td * projected_ratio.shadow)
        shadow_primitive = bounded_primitive(shadow_integrand, self.variable)
        if shadow_primitive is None:
            shadow_primitive = sp.Integral(shadow_integrand, self.variable)
        normalized_shadow = sp.simplify(shadow_primitive / T)
        decisions = list(projected_ratio.decisions)
        loc = IntegrationConstantLocation.FIXED
        if extension.constant != 0:
            c_over_t = sp.simplify(extension.constant / T)
            md = moderate_growth_decision(c_over_t, t, self.variable, point=self.point)
            iid = infinitesimal_ideal_decision(c_over_t, t, self.variable, point=self.point)
            decisions.extend((md, iid))
            if md.verdict is False:
                loc = IntegrationConstantLocation.CHANGES_LEADING_SCALE
                if strict:
                    raise ValueError("integration constant changes the leading monomial")
            elif iid.verdict is True:
                loc = IntegrationConstantLocation.GHOST
            elif md.verdict is True and iid.verdict is False:
                loc = IntegrationConstantLocation.SHADOW
                normalized_shadow = sp.simplify(normalized_shadow + c_over_t)
            else:
                loc = IntegrationConstantLocation.UNRESOLVED
                if strict:
                    raise ValueError("integration-constant location is unresolved")
        normalized_expression = sp.simplify(extension.expression / T)
        normalized_ghost = sp.simplify(normalized_expression - normalized_shadow)
        gd = infinitesimal_ideal_decision(normalized_ghost, t, self.variable, point=self.point)
        decisions.append(gd)
        certified = projected_ratio.certified and (normalized_ghost == 0 or gd.verdict is True)
        if strict and not certified:
            raise ValueError("integral shadow formula did not certify an I_t normalized ghost")
        return IntegralShadowProjection(
            extension,
            t,
            T,
            normalized_expression,
            normalized_shadow,
            normalized_ghost,
            loc,
            certified,
            tuple(decisions),
            "Shackell formula (5.15) with explicit integration-constant placement",
        )

    def projection(
        self, index: int, expr: sp.Expr | IntegralShadowExtension, *, strict: bool = True
    ) -> ShadowGhostDecomposition | IntegralShadowProjection:
        if isinstance(expr, IntegralShadowExtension):
            return self.project_integral(index, expr, strict=strict)
        return self.shadow_fields[index].project(expr, strict=strict)

    def shadow(
        self, index: int, expr: sp.Expr | IntegralShadowExtension, *, strict: bool = True
    ) -> sp.Expr:
        return self.projection(index, expr, strict=strict).shadow

    def ghost(
        self, index: int, expr: sp.Expr | IntegralShadowExtension, *, strict: bool = True
    ) -> sp.Expr:
        return self.projection(index, expr, strict=strict).ghost

    def compatibility(self, i: int, j: int, expr: sp.Expr) -> bool | None:
        """Replay eta_i,j o eta_j = eta_i on the supported common domain."""
        if i >= j:
            raise ValueError("require i < j")
        try:
            direct = self.shadow(i, expr)
            nested = self.shadow(i, self.shadow(j, expr))
        except ValueError:
            return None
        return sp.simplify(direct - nested) == 0

    def shadow_expansion(self, expr: sp.Expr, *, max_terms: int = 8) -> tuple[sp.Expr, ...]:
        """Extract a finite recursive shadow expansion, finest scale first."""
        if not self.scales:
            return (sp.sympify(expr),)
        remainder = sp.expand(expr)
        out = []
        finest = len(self.scales) - 1
        for _ in range(max_terms):
            if sp.simplify(remainder) == 0:
                break
            try:
                sh = self.shadow(finest, remainder)
            except ValueError:
                break
            if sh != 0:
                out.append(sh)
                remainder = sp.simplify(remainder - sh)
            else:
                # Preserve the ghost as a closed-form next object rather than
                # pretending a divergent formal series has converged.
                out.append(remainder)
                break
        return tuple(out)

    def differentiate(self, expr: sp.Expr, order: int = 1) -> sp.Expr:
        return sp.diff(expr, self.variable, order)


def asymptotic_differential_field(
    variable: sp.Symbol, scales: tuple[sp.Expr, ...], *, point: sp.Expr = sp.oo
) -> AsymptoticDifferentialField:
    """Construct an asymptotic differential field over the supplied scale generators."""
    return AsymptoticDifferentialField(variable, scales, point)
