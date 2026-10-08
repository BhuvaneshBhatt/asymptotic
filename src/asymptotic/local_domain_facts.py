"""Cheap conservative facts about a punctured local approach domain."""

from __future__ import annotations

from dataclasses import dataclass, field

import sympy as sp

from ._symbolic_policy import bounded_assumption_entails
from .order_constraints import OrderConstraints, OrderRelation


@dataclass
class LocalDomainFacts:
    domain: sp.Expr
    assumptions: sp.Expr = sp.S.true
    positive: set[sp.Expr] = field(default_factory=set)
    nonnegative: set[sp.Expr] = field(default_factory=set)
    nonzero: set[sp.Expr] = field(default_factory=set)
    substitutions: dict[sp.Symbol, sp.Expr] = field(default_factory=dict)
    orders: OrderConstraints = field(default_factory=OrderConstraints)

    @classmethod
    def from_domain(cls, domain, assumptions=sp.S.true):
        self = cls(sp.sympify(domain), sp.sympify(assumptions))
        clauses = (
            sp.And.make_args(self.domain)
            if isinstance(self.domain, sp.And)
            else (self.domain,)
        )
        for c in clauses:
            if isinstance(c, sp.StrictGreaterThan) and c.rhs == 0:
                self.positive.add(c.lhs)
                self.nonnegative.add(c.lhs)
                self.nonzero.add(c.lhs)
            elif isinstance(c, sp.GreaterThan) and c.rhs == 0:
                self.nonnegative.add(c.lhs)
            elif isinstance(c, sp.Unequality) and c.rhs == 0:
                self.nonzero.add(c.lhs)
            elif isinstance(c, sp.Equality):
                if isinstance(c.lhs, sp.Symbol) and c.lhs not in c.rhs.free_symbols:
                    self.substitutions[c.lhs] = c.rhs
            elif isinstance(c, (sp.LessThan, sp.StrictLessThan)):
                # |f| <= g proves f=O(g) locally.
                if c.lhs.func is sp.Abs:
                    self.orders.add(c.lhs.args[0], OrderRelation.BIG_O, c.rhs)
        return self

    def entails(self, condition) -> bool | None:
        condition = sp.sympify(condition)
        if condition in sp.And.make_args(self.domain):
            return True
        return bounded_assumption_entails(
            condition, sp.And(self.assumptions, self.domain)
        )
