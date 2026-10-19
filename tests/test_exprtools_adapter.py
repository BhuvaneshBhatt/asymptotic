from unittest.mock import patch

import sympy as sp

import asymptotic.instrumentation as instrumentation
from asymptotic.limit_models import LimitEvidence
from asymptotic.recurrence_simplification import (
    RecurrenceSimplificationResult,
    simplify_recurrences,
)
from asymptotic.special_functions import ParabolicCylinderD as D


def test_adapter_preserves_evidence_and_instrumentation():
    events = []
    n, z = sp.symbols("n z")
    e = D(n + 2, z) - z * D(n + 1, z) + (n + 1) * D(n, z)
    with patch.object(instrumentation, "record_symbolic_event", events.append):
        r = simplify_recurrences(e, return_result=True, budget_seconds=1)
    assert r.expression == 0
    assert isinstance(r, RecurrenceSimplificationResult)
    assert isinstance(r.evidence[0], LimitEvidence)
    assert events == ["recurrence_calls"]


def test_context_uses_exprtools_default_oracle_and_preserves_unknown():
    import exprtools

    from asymptotic.context import AsymptoticContext

    x = sp.Symbol("x")
    context = AsymptoticContext(x, simplify_results=False, use_sympy_zero_fallback=False)
    with patch.object(exprtools, "zerotest", return_value=None) as oracle:
        assert context.is_zero(sp.sin(x)) is None
    oracle.assert_called_once_with(
        sp.sin(x), assumptions=sp.true, use_cache=True, confidence="certified"
    )
