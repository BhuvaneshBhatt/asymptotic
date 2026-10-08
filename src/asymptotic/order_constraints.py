"""Conservative local asymptotic order constraints."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import sympy as sp


class OrderRelation(Enum):
    BIG_O = "O"
    LITTLE_O = "o"
    THETA = "theta"


@dataclass
class OrderConstraints:
    """Small proof graph for local O/o/Theta relations.

    Relations are only added from certified callers. Closure uses sound
    transitivity rules; absence of a relation is never evidence of its negation.
    """

    relations: dict[tuple[sp.Expr, sp.Expr], OrderRelation] = field(
        default_factory=dict
    )

    def add(self, left, relation: OrderRelation, right) -> None:
        left, right = sp.sympify(left), sp.sympify(right)
        if left == right and relation is not OrderRelation.THETA:
            relation = OrderRelation.THETA
        old = self.relations.get((left, right))
        rank = {
            OrderRelation.BIG_O: 0,
            OrderRelation.THETA: 1,
            OrderRelation.LITTLE_O: 2,
        }
        if old is None or rank[relation] > rank[old]:
            self.relations[(left, right)] = relation
        if relation is OrderRelation.THETA:
            self.relations[(right, left)] = relation

    def relation(self, left, right) -> OrderRelation | None:
        left, right = sp.sympify(left), sp.sympify(right)
        if left == right:
            return OrderRelation.THETA
        direct = self.relations.get((left, right))
        if direct is not None:
            return direct
        # bounded fixed-point transitive search
        frontier = [(left, False)]
        seen = {left}
        while frontier:
            node, strict = frontier.pop(0)
            for (a, b), rel in tuple(self.relations.items()):
                if a != node or b in seen:
                    continue
                new_strict = strict or rel is OrderRelation.LITTLE_O
                if b == right:
                    return OrderRelation.LITTLE_O if new_strict else OrderRelation.BIG_O
                seen.add(b)
                frontier.append((b, new_strict))
        return None

    def multiply(self, a, b, c, d) -> None:
        """Derive ac = O/o(bd) from certified factor relations."""
        r1, r2 = self.relation(a, b), self.relation(c, d)
        if r1 is None or r2 is None:
            return
        rel = (
            OrderRelation.LITTLE_O
            if OrderRelation.LITTLE_O in (r1, r2)
            else OrderRelation.BIG_O
        )
        if r1 is r2 is OrderRelation.THETA:
            rel = OrderRelation.THETA
        self.add(sp.sympify(a) * sp.sympify(c), rel, sp.sympify(b) * sp.sympify(d))
