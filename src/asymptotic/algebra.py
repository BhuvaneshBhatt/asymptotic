"""Common protocol and algebra for asymptotic field elements.

The package has several specialized representations.  This module provides a
structural interface and a coordinate-aware algebra over them without forcing
those representations to share storage or inheritance.  Native algorithms are
preferred; a finite :class:`TransseriesExpansion` is the conservative common
normal form when two different representations must interact.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import sympy as sp

from ._power_simplify import analytic_powsimp
from .context import AsymptoticContext, GrowthComparison
from .remainder import AsymptoticRemainder, AsymptoticTruncation


@runtime_checkable
class AsymptoticFieldElementProtocol(Protocol):
    """Structural interface implemented by unified asymptotic elements."""

    @property
    def variable(self) -> sp.Symbol: ...

    @property
    def point(self) -> sp.Expr: ...

    @property
    def remainder(self) -> AsymptoticRemainder: ...

    def as_expr(self) -> sp.Expr: ...
    def truncate(self, terms: int | None = None) -> sp.Expr: ...
    def truncation(self, terms: int | None = None) -> AsymptoticTruncation: ...
    def to_transseries(self, terms: int = 6): ...
    def differentiate(self, order: int = 1): ...
    def integrate(self, *, constant: sp.Expr = 0, terms: int = 6): ...
    def compose(self, outer, *, argument: sp.Symbol | None = None, terms: int = 6): ...
    def reciprocal(self, *, terms: int = 6): ...
    def inverse_asymptotic(
        self, inverse_variable: sp.Symbol | None = None, *, terms: int = 6, branch: int | None = 0
    ): ...
    def compare(self, other) -> GrowthComparison: ...


def _call_signature(method):
    """Return a callable signature when Python can inspect ``method``.

    Signature inspection is used only to select an adapter calling convention;
    exceptions raised by the native implementation itself are never swallowed.
    """

    try:
        return inspect.signature(method)
    except (TypeError, ValueError):
        return None


def _accepts_keyword(method, name: str) -> bool:
    signature = _call_signature(method)
    if signature is None:
        return False
    parameter = signature.parameters.get(name)
    if parameter is not None and parameter.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    ):
        return True
    return any(item.kind is inspect.Parameter.VAR_KEYWORD for item in signature.parameters.values())


def _callable_without_args(method) -> bool:
    signature = _call_signature(method)
    if signature is None:
        return False
    try:
        signature.bind()
    except TypeError:
        return False
    return True


def _validate_context_coordinates(
    context: AsymptoticContext | None, variable: sp.Symbol, point: sp.Expr
) -> None:
    """Reject a proof context attached to a different asymptotic germ."""

    if context is None:
        return
    if context.variable != variable or context.point != point:
        raise ValueError("asymptotic context uses different coordinates")


def _native_coordinates(obj, variable: sp.Symbol | None, point: sp.Expr | None):
    """Infer the asymptotic coordinate from a supported native representation."""

    from .multiseries import Multiseries
    from .nested import NestedExpansion
    from .nonlinear_ode import NonlinearDifferentialTransseriesBranch
    from .scale import ScaleElement
    from .transseries import TransseriesExpansion

    if isinstance(obj, TransseriesExpansion):
        return obj.variable, obj.point
    if isinstance(obj, Multiseries):
        return obj.scale.variable, obj.scale.point
    if isinstance(obj, NestedExpansion):
        return obj.variable, obj.point
    if isinstance(obj, NonlinearDifferentialTransseriesBranch):
        return obj.transseries.variable, obj.transseries.point
    if isinstance(obj, ScaleElement):
        if variable is None:
            raise ValueError("variable is required when adapting a standalone ScaleElement")
        return variable, sp.oo if point is None else sp.sympify(point)
    if isinstance(obj, sp.Expr):
        if variable is None:
            raise ValueError("variable is required when adapting a SymPy expression")
        return variable, sp.oo if point is None else sp.sympify(point)

    native_variable = getattr(obj, "variable", None)
    native_point = getattr(obj, "point", None)
    if native_variable is not None:
        return native_variable, sp.oo if native_point is None else native_point
    if variable is None:
        raise TypeError(f"cannot infer an asymptotic variable for {type(obj).__name__}")
    if not (
        hasattr(obj, "expression")
        or hasattr(obj, "expr")
        or callable(getattr(obj, "truncate", None))
    ):
        raise TypeError(f"unsupported asymptotic representation {type(obj).__name__}")
    return variable, sp.oo if point is None else sp.sympify(point)


def _native_expression(obj) -> sp.Expr:
    from .multiseries import Multiseries
    from .nested import NestedExpansion, NestedForm
    from .nonlinear_ode import NonlinearDifferentialTransseriesBranch
    from .scale import ScaleElement
    from .transseries import TransseriesExpansion

    if isinstance(obj, TransseriesExpansion):
        return obj.truncate()
    if isinstance(obj, Multiseries):
        return obj.expr
    if isinstance(obj, NestedExpansion):
        return obj.expr
    if isinstance(obj, NestedForm):
        return obj.exact_expr
    if isinstance(obj, NonlinearDifferentialTransseriesBranch):
        return obj.series
    if isinstance(obj, ScaleElement):
        return obj.expr
    if isinstance(obj, sp.Expr):
        return obj
    # Branch/series objects deliberately expose their finite represented prefix
    # through truncate().  Prefer it to an original generating expression when
    # available so the common algebra never manufactures terms that were not
    # actually computed by the representation.
    truncate = getattr(obj, "truncate", None)
    if callable(truncate) and _callable_without_args(truncate):
        return sp.sympify(truncate())
    if hasattr(obj, "expression"):
        return sp.sympify(obj.expression)
    if hasattr(obj, "expr"):
        return sp.sympify(obj.expr)
    raise TypeError(f"{type(obj).__name__} does not expose an asymptotic expression")


@dataclass(frozen=True)
class AsymptoticAlgebra:
    """Coordinate-aware common algebra for heterogeneous asymptotic objects.

    The algebra owns coercion and the finite transseries normal form used for
    cross-representation operations.  Native algorithms remain preferred for
    unary operations, but binary arithmetic, comparisons, and remainder
    propagation all pass through this single boundary.
    """

    variable: sp.Symbol
    point: sp.Expr = sp.oo
    terms: int = 6
    context: AsymptoticContext | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "point", sp.sympify(self.point))
        if self.terms < 1:
            raise ValueError("terms must be positive")
        if self.context is None:
            object.__setattr__(self, "context", AsymptoticContext(self.variable, self.point))
        else:
            _validate_context_coordinates(self.context, self.variable, self.point)

    def element(self, obj) -> AsymptoticElement:
        """Coerce ``obj`` into this algebra, rejecting coordinate mismatches."""

        if isinstance(obj, AsymptoticElement):
            if obj.variable != self.variable or obj.point != self.point:
                raise ValueError("asymptotic elements use different coordinates")
            if obj.context is self.context:
                return obj
            return AsymptoticElement(obj.native, self.variable, self.point, self.context)
        element = asymptotic_element(
            obj, variable=self.variable, point=self.point, context=self.context
        )
        if element.variable != self.variable or element.point != self.point:
            raise ValueError("asymptotic elements use different coordinates")
        return element

    def normal_form(self, obj, *, terms: int | None = None):
        """Return the finite certified transseries normal form for ``obj``."""

        n = self.terms if terms is None else int(terms)
        if n < 1:
            raise ValueError("terms must be positive")
        return self.element(obj).to_transseries(n)

    def add(self, left, right, *, terms: int | None = None) -> AsymptoticElement:
        n = self.terms if terms is None else int(terms)
        return self.element(self.normal_form(left, terms=n) + self.normal_form(right, terms=n))

    def subtract(self, left, right, *, terms: int | None = None) -> AsymptoticElement:
        n = self.terms if terms is None else int(terms)
        return self.element(self.normal_form(left, terms=n) - self.normal_form(right, terms=n))

    def multiply(self, left, right, *, terms: int | None = None) -> AsymptoticElement:
        n = self.terms if terms is None else int(terms)
        return self.element(self.normal_form(left, terms=n) * self.normal_form(right, terms=n))

    def divide(self, left, right, *, terms: int | None = None) -> AsymptoticElement:
        n = self.terms if terms is None else int(terms)
        quotient = self.normal_form(left, terms=n) / self.normal_form(right, terms=n)
        return self.element(quotient.prefix(n))

    def power(self, value, exponent, *, terms: int | None = None) -> AsymptoticElement:
        n = self.terms if terms is None else int(terms)
        result = self.normal_form(value, terms=n).constant_power(sp.sympify(exponent))
        return self.element(result.prefix(n))

    def compare(self, left, right) -> GrowthComparison:
        lhs = self.element(left)
        rhs = self.element(right)
        if self.context is None:
            raise RuntimeError("asymptotic algebra has no comparison context")
        relation, _ = self.context.compare_growth(lhs.as_expr(), rhs.as_expr())
        return relation

    def truncation(self, value, terms: int | None = None) -> AsymptoticTruncation:
        return self.element(value).truncation(terms)

    def differentiate(self, value, order: int = 1) -> AsymptoticElement:
        return self.element(value)._differentiate_native(order)

    def integrate(
        self, value, *, constant: sp.Expr = 0, terms: int | None = None
    ) -> AsymptoticElement:
        n = self.terms if terms is None else int(terms)
        return self.element(value)._integrate_native(constant=constant, terms=n)

    def compose(
        self,
        inner,
        outer,
        *,
        argument: sp.Symbol | None = None,
        terms: int | None = None,
        assumptions: sp.Expr | bool = sp.S.true,
        allow_unknown_properties: bool = False,
    ) -> AsymptoticElement:
        n = self.terms if terms is None else int(terms)
        return self.element(inner)._compose_native(
            outer,
            argument=argument,
            terms=n,
            assumptions=assumptions,
            allow_unknown_properties=allow_unknown_properties,
        )

    def reciprocal(self, value, *, terms: int | None = None) -> AsymptoticElement:
        n = self.terms if terms is None else int(terms)
        element = self.element(value)
        method = getattr(element.native, "reciprocal", None)
        if callable(method) and _accepts_keyword(method, "terms"):
            return self.element(method(terms=n))
        if callable(method) and _callable_without_args(method):
            return self.element(method())
        return self.element(self.normal_form(element, terms=n).reciprocal(terms=n))

    def inverse_asymptotic(
        self,
        value,
        inverse_variable: sp.Symbol | None = None,
        *,
        terms: int | None = None,
        branch: int | None = 0,
    ):
        n = self.terms if terms is None else int(terms)
        element = self.element(value)
        method = getattr(element.native, "inverse_asymptotic", None)
        if callable(method):
            return method(inverse_variable, terms=n, branch=branch)
        from .reversion import inverse_asymptotic

        return inverse_asymptotic(
            element.as_expr(),
            self.variable,
            inverse_variable,
            point=self.point,
            terms=n,
            branch=branch,
            context=self.context,
        )


@dataclass(frozen=True)
class AsymptoticElement:
    """Uniform algebraic view of any supported asymptotic representation."""

    native: object
    variable: sp.Symbol
    point: sp.Expr = sp.oo
    context: AsymptoticContext | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "point", sp.sympify(self.point))
        if self.context is None:
            object.__setattr__(self, "context", AsymptoticContext(self.variable, self.point))
        else:
            _validate_context_coordinates(self.context, self.variable, self.point)

    @property
    def algebra(self) -> AsymptoticAlgebra:
        """Return a common algebra sharing this element's coordinate/context."""

        return AsymptoticAlgebra(self.variable, self.point, context=self.context)

    def as_expr(self) -> sp.Expr:
        return sp.sympify(_native_expression(self.native))

    @property
    def remainder(self) -> AsymptoticRemainder:
        """Return explicit remainder semantics for the wrapped representation."""
        from .nonlinear_ode import NonlinearDifferentialTransseriesBranch
        from .transseries import TransseriesExpansion

        if isinstance(self.native, TransseriesExpansion):
            if self.native.remainder is None:
                raise RuntimeError("transseries is missing its remainder")
            return self.native.remainder
        if isinstance(self.native, NonlinearDifferentialTransseriesBranch):
            if self.native.transseries.remainder is None:
                raise RuntimeError("ODE transseries is missing its remainder")
            return self.native.transseries.remainder
        native_remainder = getattr(self.native, "remainder", None)
        if isinstance(native_remainder, AsymptoticRemainder):
            return native_remainder

        from .multiseries import Multiseries
        from .nested import NestedExpansion, NestedForm
        from .puiseux import PuiseuxSeries
        from .scale import ScaleElement

        if isinstance(self.native, PuiseuxSeries):
            omitted = analytic_powsimp(sp.expand(self.native.expr - self.native.truncate()))
            if omitted == 0:
                return AsymptoticRemainder.exact_zero(
                    self.variable, self.point, source="complete Puiseux representation"
                )
            return AsymptoticRemainder.unknown(
                self.variable,
                self.point,
                exact_expression=omitted,
                source="finite Puiseux prefix; omitted-tail scale not certified",
            )

        if isinstance(
            self.native, (sp.Expr, Multiseries, NestedExpansion, NestedForm, ScaleElement)
        ):
            return AsymptoticRemainder.exact_zero(
                self.variable,
                self.point,
                source=f"exact native {type(self.native).__name__} representation",
            )
        return AsymptoticRemainder.unknown(
            self.variable,
            self.point,
            source=f"{type(self.native).__name__} does not declare remainder semantics",
        )

    def truncate(self, terms: int | None = None) -> sp.Expr:
        from .multiseries import Multiseries
        from .nested import NestedExpansion
        from .nonlinear_ode import NonlinearDifferentialTransseriesBranch
        from .transseries import TransseriesExpansion

        if isinstance(self.native, TransseriesExpansion):
            return self.native.truncate(terms)
        if isinstance(self.native, Multiseries):
            return self.native.truncate(terms)
        if isinstance(self.native, NestedExpansion):
            if terms is None:
                return self.native.expr
            from .transseries import transseries_from_expression

            return (
                transseries_from_expression(
                    self.native.expr, self.variable, point=self.point, complete=True
                )
                .prefix(max(1, int(terms)))
                .truncate()
            )
        if isinstance(self.native, NonlinearDifferentialTransseriesBranch):
            return self.native.transseries.truncate(terms)
        method = getattr(self.native, "truncate", None)
        if callable(method):
            return sp.sympify(method() if terms is None else method(terms))
        return self.as_expr()

    def truncation(self, terms: int | None = None) -> AsymptoticTruncation:
        from .transseries import TransseriesExpansion

        if isinstance(self.native, TransseriesExpansion):
            return self.native.truncation(terms)
        native_truncation = getattr(self.native, "truncation", None)
        if callable(native_truncation):
            return native_truncation(terms)
        from .nested import NestedExpansion

        if isinstance(self.native, NestedExpansion) and terms is not None:
            from .transseries import transseries_from_expression

            return transseries_from_expression(
                self.native.expr, self.variable, point=self.point, complete=True
            ).truncation(max(1, int(terms)))
        prefix = self.truncate(terms)
        exact_error = analytic_powsimp(sp.expand(self.as_expr() - prefix))
        if exact_error == 0:
            # A native UNKNOWN remainder must never be upgraded merely because
            # its stored finite prefix equals truncate().
            rem = self.remainder
        else:
            rem = AsymptoticRemainder.unknown(
                self.variable,
                self.point,
                exact_expression=exact_error,
                source=f"exact omitted tail of {type(self.native).__name__}; asymptotic scale not certified",
            )
        count = 0 if terms is None else max(0, int(terms))
        return AsymptoticTruncation(prefix, rem, count, count)

    def to_transseries(self, terms: int = 6):
        from .nonlinear_ode import NonlinearDifferentialTransseriesBranch
        from .transseries import TransseriesExpansion, transseries_from_expression

        if terms < 1:
            raise ValueError("terms must be positive")
        if isinstance(self.native, TransseriesExpansion):
            return self.native.prefix(terms)
        if isinstance(self.native, NonlinearDifferentialTransseriesBranch):
            return self.native.transseries.prefix(terms)
        trunc = self.truncation(terms)
        return transseries_from_expression(
            trunc.prefix,
            self.variable,
            point=self.point,
            complete=trunc.remainder.is_exact,
            remainder=trunc.remainder,
        ).prefix(terms)

    def _wrap(self, value):
        if isinstance(value, AsymptoticElement):
            return value
        try:
            x, p = _native_coordinates(value, self.variable, self.point)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"native asymptotic operation returned unsupported {type(value).__name__}"
            ) from exc
        context = self.context if x == self.variable and p == self.point else None
        return AsymptoticElement(value, x, p, context)

    def _differentiate_native(self, order: int = 1):
        if order < 0:
            raise ValueError("order must be nonnegative")
        method = getattr(self.native, "differentiate", None)
        if callable(method):
            if _accepts_keyword(method, "order"):
                return self._wrap(method(order=order))
            return self._wrap(method(order))
        return self._wrap(sp.diff(self.as_expr(), self.variable, order))

    def differentiate(self, order: int = 1):
        return self.algebra.differentiate(self, order)

    def _integrate_native(self, *, constant: sp.Expr = 0, terms: int = 6):
        method = getattr(self.native, "integrate", None)
        if callable(method):
            accepts_constant = _accepts_keyword(method, "constant")
            accepts_terms = _accepts_keyword(method, "terms")
            if accepts_constant:
                kwargs = {"constant": constant}
                if accepts_terms:
                    kwargs["terms"] = terms
                return self._wrap(method(**kwargs))
            if constant == 0 and _callable_without_args(method):
                return self._wrap(method())
        from .general_ops import asymptotic_integrate

        return self._wrap(
            asymptotic_integrate(self.to_transseries(terms), constant=constant, terms=terms)
        )

    def integrate(self, *, constant: sp.Expr = 0, terms: int = 6):
        return self.algebra.integrate(self, constant=constant, terms=terms)

    def _compose_native(
        self,
        outer,
        *,
        argument: sp.Symbol | None = None,
        terms: int = 6,
        assumptions: sp.Expr | bool = sp.S.true,
        allow_unknown_properties: bool = False,
    ):
        method = getattr(self.native, "compose", None)
        if callable(method):
            supports_argument = _accepts_keyword(method, "argument")
            supports_terms = _accepts_keyword(method, "terms")
            supports_assumptions = _accepts_keyword(method, "assumptions")
            supports_unknown = _accepts_keyword(method, "allow_unknown_properties")
            can_preserve_semantics = (
                (argument is None or supports_argument)
                and (assumptions is sp.S.true or supports_assumptions)
                and (not allow_unknown_properties or supports_unknown)
            )
            if can_preserve_semantics:
                kwargs = {}
                if supports_argument:
                    kwargs["argument"] = argument
                if supports_terms:
                    kwargs["terms"] = terms
                if supports_assumptions:
                    kwargs["assumptions"] = assumptions
                if supports_unknown:
                    kwargs["allow_unknown_properties"] = allow_unknown_properties
                return self._wrap(method(outer, **kwargs))
        from .general_ops import compose_transseries

        return self._wrap(
            compose_transseries(
                outer,
                self.to_transseries(terms),
                argument=argument,
                terms=terms,
                assumptions=assumptions,
                allow_unknown_properties=allow_unknown_properties,
            )
        )

    def compose(
        self,
        outer,
        *,
        argument: sp.Symbol | None = None,
        terms: int = 6,
        assumptions: sp.Expr | bool = sp.S.true,
        allow_unknown_properties: bool = False,
    ):
        return self.algebra.compose(
            self,
            outer,
            argument=argument,
            terms=terms,
            assumptions=assumptions,
            allow_unknown_properties=allow_unknown_properties,
        )

    def reciprocal(self, *, terms: int = 6):
        return self.algebra.reciprocal(self, terms=terms)

    def inverse_asymptotic(
        self, inverse_variable: sp.Symbol | None = None, *, terms: int = 6, branch: int | None = 0
    ):
        return self.algebra.inverse_asymptotic(self, inverse_variable, terms=terms, branch=branch)

    def compare(self, other) -> GrowthComparison:
        return self.algebra.compare(self, other)

    def __add__(self, other):
        return self.algebra.add(self, other)

    def __sub__(self, other):
        return self.algebra.subtract(self, other)

    def __mul__(self, other):
        return self.algebra.multiply(self, other)

    def __truediv__(self, other):
        return self.algebra.divide(self, other)

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return self.algebra.subtract(other, self)

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        return self.algebra.divide(other, self)

    def __pow__(self, power):
        return self.algebra.power(self, power)


def asymptotic_element(
    obj,
    variable: sp.Symbol | None = None,
    *,
    point: sp.Expr | None = None,
    context: AsymptoticContext | None = None,
) -> AsymptoticElement:
    """Adapt a supported native representation to the common field protocol."""

    if isinstance(obj, AsymptoticElement):
        requested_var = obj.variable if variable is None else variable
        requested_point = obj.point if point is None else sp.sympify(point)
        if requested_var != obj.variable or requested_point != obj.point:
            raise ValueError("asymptotic element uses different coordinates")
        if context is None or context is obj.context:
            return obj
        return AsymptoticElement(obj.native, obj.variable, obj.point, context)
    x, p = _native_coordinates(obj, variable, point)
    return AsymptoticElement(obj, x, p, context)
