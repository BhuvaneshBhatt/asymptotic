import sympy as sp

from asymptotic import AsymptoticContext
from asymptotic.obligations import (
    AsymptoticKnowledge,
    ObligationKind,
)
from asymptotic.sparse import (
    ContinuationStatus,
    LazySparseSeries,
)


class DeferredZeroContext(AsymptoticContext):
    def is_zero(self, expr):
        expr = sp.sympify(expr)
        if expr.is_Symbol and expr != self.variable:
            return None
        return super().is_zero(expr)


def test_fact_obligation_resumes_same_sparse_tree():
    z, a = sp.symbols("z a")
    knowledge = AsymptoticKnowledge()
    sparse = LazySparseSeries(sp.log(a + z), z, DeferredZeroContext(z), knowledge=knowledge)
    continuation = sparse.continuation(3)

    assert continuation.run() is None
    obligation = continuation.obligation
    assert obligation is not None
    assert obligation.kind is ObligationKind.ZERO_TEST

    inner_state = next(state for state in sparse.node_states if state.expr == a + z)
    evaluations_before = inner_state.evaluations
    assert inner_state.status is ContinuationStatus.DONE

    knowledge.set(obligation, False)
    result = continuation.resume()

    assert result is not None
    assert continuation.resumes == 1
    assert sparse.resume_count == 1
    # The successful child prefix was not re-evaluated after resumption.
    assert inner_state.evaluations == evaluations_before
    assert sp.simplify(result[0].coefficient - sp.log(a)) == 0
    assert result[0].exponent == 0
    assert result[1].exponent == 1
    assert sp.simplify(result[1].coefficient - 1 / a) == 0
    assert result[2].exponent == 2
    assert sp.simplify(result[2].coefficient + 1 / (2 * a**2)) == 0


def test_one_continuation_can_suspend_on_successive_facts():
    z, a, b = sp.symbols("z a b")
    knowledge = AsymptoticKnowledge()
    expr = sp.log(a + z) + sp.log(b + z)
    sparse = LazySparseSeries(expr, z, DeferredZeroContext(z), knowledge=knowledge)
    continuation = sparse.continuation(2)

    assert continuation.run() is None
    first = continuation.obligation
    assert first is not None
    knowledge.set(first, False)

    # Resuming the same continuation reaches the next unresolved coefficient.
    assert continuation.resume() is None
    second = continuation.obligation
    assert second is not None
    assert second.key != first.key
    knowledge.set(second, False)

    result = continuation.resume()
    assert result is not None
    assert continuation.resumes == 2
    assert sp.simplify(result[0].coefficient - sp.log(a) - sp.log(b)) == 0
    assert sp.simplify(result[1].coefficient - 1 / a - 1 / b) == 0


def test_product_frontier_survives_fact_obligation_resume():
    z, a = sp.symbols("z a")
    knowledge = AsymptoticKnowledge()
    # Preserve factor order so stage 1 builds a heap frontier before stage 2
    # suspends inside log(a + z).
    expr = sp.Mul(1 + z, 1 + z**2, sp.log(a + z), evaluate=False)
    sparse = LazySparseSeries(expr, z, DeferredZeroContext(z), knowledge=knowledge)
    continuation = sparse.continuation(4)

    assert continuation.run() is None
    obligation = continuation.obligation
    assert obligation is not None
    product_state = next(state for state in sparse.node_states if state.expr == expr)
    frontier_before = product_state.payload.get("product:1")
    assert frontier_before is not None

    knowledge.set(obligation, False)
    result = continuation.resume()
    # Parameter-dependent product coefficients may expose further exact-zero
    # obligations. Resolve them conservatively as nonzero for this continuation
    # state regression; the point of the test is frontier identity preservation.
    while result is None and continuation.obligation is not None:
        knowledge.set(continuation.obligation, False)
        result = continuation.resume()
    assert result is not None
    assert product_state.payload.get("product:1") is frontier_before


def test_analytic_frontier_is_reused_when_request_grows():
    z = sp.symbols("z")
    sparse = LazySparseSeries(sp.sin(z + z**2), z, AsymptoticContext(z))
    first = sparse.terms(3)
    assert first is not None
    state = next(state for state in sparse.node_states if state.expr == sp.sin(z + z**2))
    frontier = state.payload.get("analytic:generic:sin")
    assert frontier is not None

    second = sparse.terms(5)
    assert second is not None
    assert state.payload.get("analytic:generic:sin") is frontier
    assert len(second) >= len(first)
