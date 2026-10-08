"""Canonical function bindings preserve the reference expressions."""

import json
from pathlib import Path

import pytest
import sympy as sp
from sympy.core.function import AppliedUndef

from asymptotic import analytic_limit
from asymptotic.limit_models import LimitEvidence, LimitStatus
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
NS = scalar_reference_namespace()


@pytest.mark.parametrize("case", ["0223", "0263", "0264", "0287"])
def test_native_special_function_binding(case):
    expression = sp.sympify(ROWS[case]["expression"], locals=NS)
    assert not expression.has(AppliedUndef)
    assert (
        expression.has(sp.rf)
        if case != "0287"
        else expression.has(NS["ParabolicCylinderD"])
    )


@pytest.mark.parametrize("case", ["3361", "3364", "3365", "3366", "3369", "3372"])
def test_directional_lambert_evidence(case):
    row = ROWS[case]
    variable = sp.Symbol(row["variable"])
    expression = sp.sympify(row["expression"], locals=NS)
    point = sp.sympify(row["point"], locals=NS)
    expected = sp.sympify(row["expected"], locals=NS)
    result = analytic_limit(expression, variable, point, return_result=True)
    assert result.status is LimitStatus.PROVED
    assert result.value == expected
    assert result.target == (point,)
    assert isinstance(result.evidence, tuple)
    assert all(isinstance(item, LimitEvidence) for item in result.evidence)
    assert result.evidence[0].value == expected


@pytest.mark.parametrize("point", [sp.oo, -sp.oo])
@pytest.mark.parametrize("direction", ["+", "-"])
def test_real_infinity_direction(point, direction):
    x = sp.Symbol("x", real=True)
    assert analytic_limit(1 / x, x, point, direction=direction) == 0
