from __future__ import annotations

import sympy as sp

from .algebra import AsymptoticElement, as_element


def differentiate(obj, order: int = 1):
    """Differentiate any supported asymptotic object through the common protocol.

    Native inputs return their native representation; passing an
    :class:`AsymptoticElement` keeps the unified wrapper.
    """
    wrapped = as_element(obj)
    result = wrapped.differentiate(order=order)
    return result if isinstance(obj, AsymptoticElement) else result.native


def integrate(
    obj,
    variable: sp.Symbol | None = None,
    *,
    point: sp.Expr = sp.oo,
    constant: sp.Expr = 0,
    terms: int | None = None,
    assumptions: sp.Expr | bool = sp.S.true,
    allow_unknown_properties: bool = False,
):
    """Integrate an asymptotic representation or asymptotically integrate an expression.

    ``integrate(representation)`` preserves the representation's native calculus
    protocol. ``integrate(expr, variable, ...)`` constructs a scale-aware finite
    asymptotic primitive of a symbolic expression.
    """
    count = 6 if terms is None else int(terms)
    if variable is not None:
        from .general_ops import _integrate_expression

        return _integrate_expression(
            obj,
            variable,
            point=point,
            constant=constant,
            terms=count,
            assumptions=assumptions,
            allow_unknown_properties=allow_unknown_properties,
        )
    if isinstance(obj, sp.Expr):
        raise TypeError("variable is required when integrating a symbolic expression")
    if assumptions is not sp.S.true or allow_unknown_properties:
        from .general_ops import _integrate_expression

        return _integrate_expression(
            obj,
            constant=constant,
            terms=count,
            assumptions=assumptions,
            allow_unknown_properties=allow_unknown_properties,
        )
    wrapped = as_element(obj)
    result = wrapped.integrate(constant=constant, terms=count)
    return result if isinstance(obj, AsymptoticElement) else result.native


def compose(
    outer,
    inner,
    *,
    argument: sp.Symbol | None = None,
    terms: int = 6,
    assumptions: sp.Expr | bool = sp.S.true,
    allow_unknown_properties: bool = False,
):
    """Return the asymptotic composition ``outer(inner)``.

    ``outer`` may be a symbolic expression, callable, or supported asymptotic
    representation. ``inner`` supplies the asymptotic germ substituted into it.
    The argument order follows the standard convention ``(f \\circ g)(x) =
    f(g(x))``.
    """
    wrapped = as_element(inner)
    if callable(outer) and argument is None:
        argument = sp.Dummy("z")
        outer = outer(argument)
    result = wrapped.compose(
        outer,
        argument=argument,
        terms=terms,
        assumptions=assumptions,
        allow_unknown_properties=allow_unknown_properties,
    )
    return result if isinstance(inner, AsymptoticElement) else result.native


def truncate(obj, terms: int | None = None) -> sp.Expr:
    """Return the finite expression obtained by truncating a representation."""
    if isinstance(obj, sp.Expr):
        if terms is not None:
            raise ValueError(
                "request return_result=True to truncate a representation by term count"
            )
        return obj
    method = getattr(obj, "truncate", None)
    if not callable(method):
        raise TypeError(f"{type(obj).__name__} does not support truncation")
    return method(terms)


def leading_term(obj):
    """Return the leading stored asymptotic term of a supported representation."""
    native = obj.native if isinstance(obj, AsymptoticElement) else obj
    method = getattr(native, "leading_term", None)
    if not callable(method):
        raise TypeError(f"{type(native).__name__} does not expose a leading term")
    return method()
