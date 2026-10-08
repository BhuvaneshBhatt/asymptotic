from dataclasses import dataclass

from asymptotic import PublicResult, explain
from asymptotic.public_result import ResultExplanation


@dataclass(frozen=True)
class NativeResult:
    status: str = "formal"
    method: str = "test"
    certified: bool = False
    reason: str = "missing hypothesis"


@dataclass(frozen=True)
class ExplainedResult:
    def explain(self) -> ResultExplanation:
        return ResultExplanation("certified", "direct", True, None)


def test_existing_result_records_have_a_stable_explanation_adapter():
    explanation = explain(NativeResult())
    assert explanation.status == "formal"
    assert explanation.method == "test"
    assert explanation.certified is False
    assert explanation.reason == "missing hypothesis"


def test_native_public_result_protocol_is_preferred():
    result = ExplainedResult()
    assert isinstance(result, PublicResult)
    assert explain(result) == ResultExplanation("certified", "direct", True, None)
