import json
import re
from pathlib import Path

import pytest
import sympy as sp

from asymptotic.relative_growth import prove_growth_comparison

CASES = json.loads(
    (
        Path(__file__).parent / "data/multivariate_limit_comprehensive_cases.json"
    ).read_text()
)
CASES = [
    c for c in CASES if c.get("suite") == "growth_scale" and "growth_comparison" in c
]


def parse_condition(text, loc):
    if not text:
        return sp.S.true
    return sp.And(*(sp.sympify(p.strip(), locals=loc) for p in text.split("&")))


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_structural_growth_comparison(case):
    names = set(case["variables"])
    for f in ("expression", "assumptions"):
        names |= set(re.findall(r"\b[A-Za-z_]\w*\b", str(case.get(f, ""))))
    reserved = {"exp", "log", "sqrt", "Abs", "oo", "E", "pi"}
    loc = {n: sp.Symbol(n, real=True) for n in names - reserved}
    for v in case["variables"]:
        loc[v] = sp.Symbol(v, positive=True)
    assumptions = parse_condition(case.get("assumptions"), loc)
    left, right, wanted = case["growth_comparison"]
    proof = prove_growth_comparison(
        sp.sympify(left, locals=loc),
        sp.sympify(right, locals=loc),
        loc[case["variables"][0]],
        assumptions=assumptions,
    )
    assert proof.relation.value == wanted
    assert proof.certified == (wanted != "unknown")
    if proof.certified:
        assert proof.proof is not None
