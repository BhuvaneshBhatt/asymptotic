"""Recovered setup bindings give concrete mathematical reference values."""

import json
from pathlib import Path

import pytest
import sympy as s

from asymptotic import analytic_limit, limit
from asymptotic.limit_models import LimitStatus
from asymptotic.reference_contracts import parse_reference
from asymptotic.reference_normalization import scalar_reference_namespace

ROWS = {
    row["id"][-4:]: row
    for row in json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "data/univariate_limit_reference_cases.json"
        ).read_text()
    )
}


@pytest.mark.parametrize("case,value", [("3533", 1), ("3534", 0)])
def test_recovered_one_sided_values(case, value):
    row = ROWS[case]
    expression = parse_reference(row["expression"], scalar_reference_namespace())
    x = s.Symbol("x")
    assert analytic_limit(expression, x, 0, direction=row["direction"]) == value
    assert expression.subs(x, 0) == 1


def test_recovered_jump():
    expression = parse_reference(
        ROWS["3533"]["expression"], scalar_reference_namespace()
    )
    x = s.Symbol("x")
    assert (
        limit(expression, x, 0, return_result=True).status is LimitStatus.DOES_NOT_EXIST
    )
    undefined = s.Function("f")(x)
    assert limit(undefined, x, 0, return_result=True).status is LimitStatus.UNKNOWN


def test_stirling_constant():
    expression = parse_reference(
        ROWS["3456"]["expression"], scalar_reference_namespace()
    )
    x = s.Symbol("x")
    assert (
        s.simplify(expression.subs(x, 1) - (1 / s.sqrt(2 * s.pi) - 13 / (12 * s.E)))
        == 0
    )


def test_differentiated_stirling_constant():
    expression = parse_reference(
        ROWS["0248"]["expression"], scalar_reference_namespace()
    )
    n = s.Symbol("n")
    leading = s.sqrt(2 * s.pi) * s.exp(s.Rational(1, 12))
    expected = (leading - s.E) * (17 * leading / 12 - s.E * (2 - s.EulerGamma)) / s.E**2
    assert s.simplify(expression.subs(n, 1) - expected) == 0
