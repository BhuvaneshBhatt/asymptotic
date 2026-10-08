"""First-class symbolic conditional mathematical results.

This module is the single value-mode adapter for parameter-stratified
asymptotic computations.  Proof mechanisms return structured strata; this
module alone decides how certified mathematical values are rendered.
"""

from __future__ import annotations

import sympy as sp


class ConditionalExpression(sp.Expr):
    """A mathematical value valid when ``condition`` holds."""

    is_commutative = True

    def __new__(cls, value, condition):
        value = sp.sympify(value)
        condition = sp.sympify(condition)
        if condition is sp.S.true:
            return value
        return sp.Expr.__new__(cls, value, condition)

    @property
    def value(self):
        return self.args[0]

    @property
    def condition(self):
        return self.args[1]

    def _sympystr(self, printer):
        return (
            f"ConditionalExpression({printer.doprint(self.value)}, "
            f"{printer.doprint(self.condition)})"
        )


def conditional_expression(value, condition):
    """Construct a simplified :class:`ConditionalExpression`."""
    condition = sp.simplify(sp.sympify(condition))
    if condition is sp.S.true:
        return sp.sympify(value)
    return ConditionalExpression(value, condition)


def _certified_value(result):
    """Extract a mathematical value from any supported structured result.

    The adapter is protocol-based rather than limit-class based:
    any proof mechanism may participate by exposing either ``mathematical_value``
    or a proved/certified ``value``.
    """
    mathematical = getattr(result, "mathematical_value", None)
    if mathematical is not None:
        return mathematical

    status = getattr(result, "status", None)
    status_name = getattr(status, "name", "")
    value = getattr(result, "value", None)
    if status_name in {"PROVED", "CERTIFIED"} and value is not None:
        return sp.sympify(value)
    if getattr(result, "certified", False) and value is not None:
        return sp.sympify(value)

    # Plain symbolic values produced by parameter evaluators are already
    # mathematical results.  Structured objects without proof status are not.
    if isinstance(result, sp.Basic):
        return result
    return None


def stratification_expression(stratification):
    """Render certified parameter strata as an ordinary symbolic result.

    One non-exhaustive certified regime becomes ``ConditionalExpression``.
    Multiple certified regimes become ``Piecewise``.  Unknown/DNE/incomplete
    regimes are never fabricated as values; if they occupy parameter space the
    returned expression remains conditional on the union of proved regimes.
    """
    if getattr(stratification, "assumptions", sp.S.true) is sp.S.false:
        return None
    branches = []
    for stratum in stratification.strata:
        if not getattr(stratum, "complete", True):
            continue
        value = _certified_value(stratum.result)
        if value is None:
            continue
        condition = sp.sympify(stratum.condition)
        branches.append((sp.sympify(value), condition))

    if not branches:
        return None

    if len(branches) == 1:
        value, condition = branches[0]
        if stratification.exhaustive and condition is sp.S.true:
            return value
        return conditional_expression(value, condition)

    # Preserve all proved regimes. SymPy may simplify an exhaustive final
    # branch to True; that is the ordinary canonical Piecewise representation.
    piecewise = sp.Piecewise(*branches)
    covered = sp.simplify_logic(sp.Or(*(condition for _, condition in branches)))
    if stratification.exhaustive and covered is sp.S.true:
        return piecewise
    return conditional_expression(piecewise, covered)


def mathematical_result(result):
    """Universal value-mode projection for asymptotic structured results.

    ``AsymptoticStratification`` is detected by its strata/parameters protocol
    to avoid a dependency cycle. Other proof results use the same certified
    value protocol as individual strata.
    """
    if hasattr(result, "strata") and hasattr(result, "parameters"):
        return stratification_expression(result)
    return _certified_value(result)


__all__ = [
    "ConditionalExpression",
    "conditional_expression",
    "mathematical_result",
    "stratification_expression",
]
