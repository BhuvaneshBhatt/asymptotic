from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import sympy as sp


class ObligationKind(str, Enum):
    """Kinds of facts the sparse evaluator may request from the outer engine."""

    EXPONENTIAL_SCALE = "exponential_scale"
    LOGARITHMIC_SCALE = "logarithmic_scale"
    ZERO_TEST = "zero_test"
    GROWTH_COMPARISON = "growth_comparison"
    COEFFICIENT_EXPANSION = "coefficient_expansion"
    COMPARABILITY_FACTOR = "comparability_factor"
    UNSUPPORTED_NODE = "unsupported_node"
    NONANALYTIC = "nonanalytic"
    DEPTH_LIMIT = "depth_limit"


@dataclass(frozen=True)
class AsymptoticObligation:
    """A missing asymptotic fact required by lazy sparse evaluation.

    ``recoverable`` means the outer multiseries engine has a resolver for this
    kind of request.  A resolver may mutate the scale/tower *or* simply add a
    fact to :class:`AsymptoticKnowledge` and retry the same expression.
    """

    kind: ObligationKind
    node: sp.Expr
    reason: str
    recoverable: bool = False

    @property
    def key(self) -> tuple[Any, ...]:
        return (self.kind, sp.srepr(self.node))


@dataclass(frozen=True)
class ExponentialScaleObligation(AsymptoticObligation):
    divergent_part: sp.Expr = sp.S.Zero
    argument: sp.Expr = sp.S.Zero

    def __init__(self, node: sp.Expr, argument: sp.Expr, divergent_part: sp.Expr) -> None:
        object.__setattr__(self, "kind", ObligationKind.EXPONENTIAL_SCALE)
        object.__setattr__(self, "node", sp.sympify(node))
        object.__setattr__(self, "reason", "exponential argument has negative active-scale powers")
        object.__setattr__(self, "recoverable", True)
        object.__setattr__(self, "argument", sp.sympify(argument))
        object.__setattr__(self, "divergent_part", sp.sympify(divergent_part))

    @property
    def key(self) -> tuple[Any, ...]:
        return (self.kind, sp.srepr(self.argument), sp.srepr(self.divergent_part))


@dataclass(frozen=True)
class LogarithmicScaleObligation(AsymptoticObligation):
    lead_exponent: sp.Expr = sp.S.Zero

    def __init__(self, node: sp.Expr, lead_exponent: sp.Expr) -> None:
        object.__setattr__(self, "kind", ObligationKind.LOGARITHMIC_SCALE)
        object.__setattr__(self, "node", sp.sympify(node))
        object.__setattr__(
            self, "reason", "logarithm of active-scale monomial requires a lower logarithmic scale"
        )
        object.__setattr__(self, "recoverable", True)
        object.__setattr__(self, "lead_exponent", sp.sympify(lead_exponent))

    @property
    def key(self) -> tuple[Any, ...]:
        return (self.kind, sp.srepr(self.node), sp.srepr(self.lead_exponent))


@dataclass(frozen=True)
class ZeroTestObligation(AsymptoticObligation):
    expression: sp.Expr = sp.S.Zero

    def __init__(
        self, node: sp.Expr, expression: sp.Expr, reason: str = "zero equivalence is required"
    ) -> None:
        object.__setattr__(self, "kind", ObligationKind.ZERO_TEST)
        object.__setattr__(self, "node", sp.sympify(node))
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "recoverable", True)
        object.__setattr__(self, "expression", sp.sympify(expression))

    @property
    def key(self) -> tuple[Any, ...]:
        return (self.kind, sp.srepr(self.expression))


@dataclass(frozen=True)
class GrowthComparisonObligation(AsymptoticObligation):
    left: sp.Expr = sp.S.Zero
    right: sp.Expr = sp.S.One
    logarithmic: bool = False

    def __init__(
        self,
        node: sp.Expr,
        left: sp.Expr,
        right: sp.Expr,
        *,
        logarithmic: bool = False,
        reason: str = "growth comparison is required",
    ) -> None:
        object.__setattr__(self, "kind", ObligationKind.GROWTH_COMPARISON)
        object.__setattr__(self, "node", sp.sympify(node))
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "recoverable", True)
        object.__setattr__(self, "left", sp.sympify(left))
        object.__setattr__(self, "right", sp.sympify(right))
        object.__setattr__(self, "logarithmic", bool(logarithmic))

    @property
    def key(self) -> tuple[Any, ...]:
        return (self.kind, sp.srepr(self.left), sp.srepr(self.right), self.logarithmic)


@dataclass(frozen=True)
class CoefficientExpansionObligation(AsymptoticObligation):
    expression: sp.Expr = sp.S.Zero
    lower_level: int = -1
    terms: int = 1

    def __init__(
        self, node: sp.Expr, expression: sp.Expr, lower_level: int, terms: int = 1
    ) -> None:
        object.__setattr__(self, "kind", ObligationKind.COEFFICIENT_EXPANSION)
        object.__setattr__(self, "node", sp.sympify(node))
        object.__setattr__(self, "reason", "a coefficient must be expanded in lower scale levels")
        object.__setattr__(self, "recoverable", True)
        object.__setattr__(self, "expression", sp.sympify(expression))
        object.__setattr__(self, "lower_level", int(lower_level))
        object.__setattr__(self, "terms", max(1, int(terms)))

    @property
    def key(self) -> tuple[Any, ...]:
        return (self.kind, sp.srepr(self.expression), self.lower_level, self.terms)


@dataclass(frozen=True)
class ComparabilityFactorObligation(AsymptoticObligation):
    expression: sp.Expr = sp.S.Zero
    candidates: tuple[sp.Expr, ...] = ()

    def __init__(self, node: sp.Expr, expression: sp.Expr, candidates: tuple[sp.Expr, ...]) -> None:
        object.__setattr__(self, "kind", ObligationKind.COMPARABILITY_FACTOR)
        object.__setattr__(self, "node", sp.sympify(node))
        object.__setattr__(
            self, "reason", "factorization against an existing comparability class is required"
        )
        object.__setattr__(self, "recoverable", True)
        object.__setattr__(self, "expression", sp.sympify(expression))
        object.__setattr__(self, "candidates", tuple(sp.sympify(c) for c in candidates))

    @property
    def key(self) -> tuple[Any, ...]:
        return (self.kind, sp.srepr(self.expression), tuple(sp.srepr(c) for c in self.candidates))


@dataclass(frozen=True)
class UnsupportedNodeObligation(AsymptoticObligation):
    def __init__(self, node: sp.Expr) -> None:
        object.__setattr__(self, "kind", ObligationKind.UNSUPPORTED_NODE)
        object.__setattr__(self, "node", sp.sympify(node))
        object.__setattr__(self, "reason", f"unsupported sparse expression node: {node.func}")
        object.__setattr__(self, "recoverable", False)


@dataclass(frozen=True)
class NonAnalyticObligation(AsymptoticObligation):
    def __init__(self, node: sp.Expr, reason: str) -> None:
        object.__setattr__(self, "kind", ObligationKind.NONANALYTIC)
        object.__setattr__(self, "node", sp.sympify(node))
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "recoverable", False)


@dataclass(frozen=True)
class DepthLimitObligation(AsymptoticObligation):
    def __init__(self, node: sp.Expr) -> None:
        object.__setattr__(self, "kind", ObligationKind.DEPTH_LIMIT)
        object.__setattr__(self, "node", sp.sympify(node))
        object.__setattr__(self, "reason", "lazy sparse recursion depth limit reached")
        object.__setattr__(self, "recoverable", False)


@dataclass
class AsymptoticKnowledge:
    """Resolved obligation facts shared across sparse retries.

    Values are intentionally opaque to this class.  Individual obligation
    handlers define their answer type: bool for zero tests, growth tuples for
    comparisons, expressions for coefficient expansions, and so on.
    """

    _answers: dict[tuple[Any, ...], Any] = field(default_factory=dict)

    def get(self, obligation: AsymptoticObligation, default: Any = None) -> Any:
        return self._answers.get(obligation.key, default)

    def contains(self, obligation: AsymptoticObligation) -> bool:
        return obligation.key in self._answers

    def set(self, obligation: AsymptoticObligation, value: Any) -> None:
        self._answers[obligation.key] = value

    def clear(self) -> None:
        self._answers.clear()
