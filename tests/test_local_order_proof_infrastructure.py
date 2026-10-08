import sympy as sp

from asymptotic.limit_models import LimitStatus, SimultaneousLimitResult
from asymptotic.local_domain_facts import LocalDomainFacts
from asymptotic.order_constraints import OrderConstraints, OrderRelation
from asymptotic.proof_obligations import ObligationKind, ProofObligation


def test_local_domain_extracts_safe_conjunctive_facts():
    x, y = sp.symbols("x y", real=True)
    f = LocalDomainFacts.from_domain(sp.And(x > 0, sp.Abs(y) <= x**2, sp.Ne(y, 0)))
    assert x in f.positive and y in f.nonzero
    assert f.orders.relation(y, x**2) is OrderRelation.BIG_O


def test_order_graph_propagates_strictness():
    f, g, h = sp.symbols("f g h", positive=True)
    o = OrderConstraints()
    o.add(f, OrderRelation.LITTLE_O, g)
    o.add(g, OrderRelation.BIG_O, h)
    assert o.relation(f, h) is OrderRelation.LITTLE_O


def test_unknown_result_can_carry_proof_obligation():
    x = sp.symbols("x")
    q = ProofObligation(ObligationKind.BRANCH_COVERAGE, "prove all local branches")
    r = SimultaneousLimitResult(x, (x,), (0,), LimitStatus.UNKNOWN, obligations=(q,))
    assert r.obligations[0].kind is ObligationKind.BRANCH_COVERAGE
