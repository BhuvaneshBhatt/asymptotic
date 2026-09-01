"""Execute the durable mathematical reference corpus."""

import pytest

from tests.reference_cases.algebraic import CASES as ALGEBRAIC_CASES
from tests.reference_cases.differential import CASES as DIFFERENTIAL_CASES
from tests.reference_cases.discrete import CASES as DISCRETE_CASES
from tests.reference_cases.elementary import CASES as ELEMENTARY_CASES
from tests.reference_cases.multivariate import CASES as MULTIVARIATE_CASES
from tests.reference_cases.saddles import CASES as SADDLE_CASES

CASES = (
    ELEMENTARY_CASES
    + ALGEBRAIC_CASES
    + MULTIVARIATE_CASES
    + DIFFERENTIAL_CASES
    + DISCRETE_CASES
    + SADDLE_CASES
)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_reference_case(case):
    assert case.check(), f"reference case failed: {case.area}/{case.name} ({case.status.value})"


def test_reference_corpus_contains_all_capability_statuses():
    statuses = {case.status.value for case in CASES}
    assert statuses == {"certified", "formal", "unknown"}
