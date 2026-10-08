import json
from pathlib import Path

import pytest
import sympy as sp

from asymptotic.complex_domain import ComplexBranchMetadata, ComplexSector
from asymptotic.sectorial_transseries import (
    branch_aware_rewrite,
    compare_sectorial_exponentials,
)

CASES = json.loads(
    (Path(__file__).parent / "data" / "sectorial_transseries_cases.json").read_text()
)


@pytest.mark.parametrize(
    "case", CASES, ids=lambda c: c["family"] + ":" + c["expression"]
)
def test_contract_corpus(case):
    z = sp.Symbol("z")
    env = {"z": z, "pi": sp.pi, "I": sp.I, "exp": sp.exp, "log": sp.log}
    sector = ComplexSector(
        sp.sympify(case["sector_center"], locals=env),
        sp.sympify(case["sector_opening"], locals=env),
    )
    expr = sp.sympify(case["expression"], locals=env)
    expected = case["expected"]
    if case["family"].startswith("branch"):
        cut = sp.sympify(case["branch_cut"], locals=env)
        result = branch_aware_rewrite(
            expr, z, sector=sector, branch=ComplexBranchMetadata(branch_cuts=(cut,))
        )
        assert ("certified" if result.certified else "unknown") == expected
    else:
        result = compare_sectorial_exponentials(expr, sp.S.One, z, sector=sector)
        assert result.relation.value == expected
        assert result.certified == (expected != "unknown")
