"""Mathematical return values for the package's primary workflows.

Specialist submodules retain their representation and evidence records. Root
functions expose those records only when requested or when no mathematical
value can be extracted without losing unresolved hypotheses or branch data.
"""

from functools import wraps
from inspect import Parameter, signature

import sympy as sp

from . import (
    calculus,
    dsolve,
    expand,
    hyperasymptotics,
    implicit,
    lindstedt,
    local_expansion,
    mellin,
    multiseries,
    nested,
    optimization,
    probability,
    products,
    puiseux,
    regular_perturbation,
    relations,
    reversion,
    roots,
    rsolve,
    solve,
    sums,
)
from .conditional import ConditionalExpression as conditional


def _expression(result):
    if result is None:
        return None
    if (
        getattr(result, "status", None) in {"UNKNOWN", "FORMAL"}
        or result.expression is None
    ):
        return result
    conditions = getattr(result, "conditions", ())
    return conditional(result.expression, sp.And(*conditions))


def _prefix(result):
    if isinstance(result, nested.NestedExpansion):
        # A lazy nested decomposition has no finite truncation independent of
        # its exact terminal residual, so its mathematical structure is retained.
        return result
    if result is None:
        return None
    value = result.truncate()
    branch = getattr(result, "branch", None)
    condition = None if branch is None else branch.condition
    return conditional(value, sp.S.true if condition is None else condition)


def _branches(result):
    # Branch conditions constrain each solution; stripping them would turn
    # parameter-specific roots into unconditional mathematical claims.
    if result.status == "UNKNOWN" or any(
        b.status == "UNKNOWN" for b in result.branches
    ):
        return result
    return tuple(
        {
            variable: conditional(value, sp.And(*branch.conditions))
            for variable, value in branch.solution
        }
        for branch in result.branches
    )


def _inverse(result):
    if isinstance(result, tuple):
        return tuple(_inverse(branch) for branch in result)
    value = result.truncate()
    condition = result.choice.condition
    return conditional(value, sp.S.true if condition is None else condition)


def _implicit(result):
    from .stratification import AsymptoticStratification

    if isinstance(result, AsymptoticStratification):
        # Parameter strata are not interchangeable branches of one expression.
        return result
    return tuple(_inverse(branch) for branch in result)


def _optimization(result):
    if result.status == "UNKNOWN":
        return result
    return conditional(result.optimum_value, sp.And(*result.conditions))


def _perturbation(result):
    return result.approximations if result.complete else result


def _lindstedt(result):
    return result.approximation if result.complete else result


def _differential(result):
    return result.solutions if result.solutions else result


def _root(result):
    return result if isinstance(result, sp.Expr) else _branches(result)


def _calculus(result):
    from .algebra import AsymptoticElement

    if isinstance(result, sp.Expr):
        return result
    if isinstance(result, AsymptoticElement):
        return result.truncate()
    return _prefix(result)


def _function(function, projection):
    """Add a consistent evidence opt-in without changing specialist algorithms."""

    @wraps(function)
    def public(*args, return_result=False, **kwargs):
        result = function(*args, **kwargs)
        return result if return_result else projection(result)

    original = signature(function)
    parameters = list(original.parameters.values())
    position = next(
        (i for i, p in enumerate(parameters) if p.kind is Parameter.VAR_KEYWORD),
        len(parameters),
    )
    parameters.insert(
        position, Parameter("return_result", Parameter.KEYWORD_ONLY, default=False)
    )
    public.__signature__ = original.replace(
        parameters=parameters, return_annotation=Parameter.empty
    )
    public.__annotations__ = {
        k: v for k, v in function.__annotations__.items() if k != "return"
    }
    public.__doc__ = (
        (function.__doc__ or "")
        + "\n\nThe root API returns the mathematical value. Use ``return_result=True`` for the full representation and evidence record. Unresolved or unproved formal calculations retain their records."
    )
    return public


sum = _function(sums.sum, _expression)
product = _function(products.product, _expression)
expectation = _function(probability.expectation, _expression)
probability = _function(probability.probability, _expression)
series = _function(expand.expand, _prefix)
multiseries = _function(multiseries.multiseries, _prefix)
nested_series = _function(nested.nested_series, _prefix)
puiseux_series = _function(puiseux.puiseux_series, _prefix)
local_series = _function(
    local_expansion.local_series, lambda r: None if r is None else r.prefix
)
solve = _function(solve.solve, _branches)
root = _function(roots.root, _root)
dsolve = _function(dsolve.dsolve, _differential)
rsolve = _function(rsolve.rsolve, _expression)
implicit = _function(implicit.implicit, _implicit)
inverse = _function(reversion.inverse, _inverse)
mellin = _function(mellin.mellin, _expression)
hyperasymptotic_series = _function(
    hyperasymptotics.hyperasymptotic_series, lambda r: r.approximation
)
regular_perturbation = _function(
    regular_perturbation.regular_perturbation, _perturbation
)
lindstedt_poincare = _function(lindstedt.lindstedt_poincare, _lindstedt)
maximize = _function(optimization.maximize, _optimization)
minimize = _function(optimization.minimize, _optimization)
relation = _function(relations.relation, lambda r: r.value)
compose = _function(calculus.compose, _calculus)

integrate = _function(calculus.integrate, _calculus)


def leading_term(obj, *, return_result=False):
    """Return the mathematical leading term of an asymptotic representation.

    ``return_result=True`` retains a stored coefficient/monomial record when
    the representation supplies one. An exhausted expansion has leading term 0.
    """
    from .algebra import AsymptoticElement

    native = obj.native if isinstance(obj, AsymptoticElement) else obj
    term = native.leading_term
    term = term() if callable(term) else term
    if return_result or isinstance(term, sp.Expr):
        return term
    if term is None:
        return sp.S.Zero
    from .transseries import TransseriesTerm

    return (
        term.expression
        if isinstance(term, TransseriesTerm)
        else term.as_expr(native.variable)
    )


def differentiate(obj, order=1, *, variable=None, return_result=False):
    """Differentiate an expression or an asymptotic representation.

    A symbolic expression with one free symbol supplies its differentiation
    variable. Specify ``variable`` when the expression contains several symbols.
    ``return_result=True`` retains a native representation when the input has one.
    """
    if isinstance(obj, sp.Expr):
        if variable is None:
            symbols = obj.free_symbols
            if not symbols:
                return sp.diff(obj, sp.Dummy("constant_variable"), order)
            if len(symbols) != 1:
                raise ValueError("variable is required for a multivariate expression")
            variable = next(iter(symbols))
        return sp.diff(obj, variable, order)
    if variable is not None:
        raise TypeError(
            "a representation already specifies its differentiation variable"
        )
    result = calculus.differentiate(obj, order)
    return result if return_result else _calculus(result)
