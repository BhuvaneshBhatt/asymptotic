"""Proof checks for resolved references and the real-parameter contract."""

import json
import subprocess
import sys
from pathlib import Path

import sympy as sp

from asymptotic.reference_normalization import (
    scalar_reference_equal,
    scalar_reference_namespace,
)

ROOT = Path(__file__).resolve().parents[3]
ROWS = {
    r["id"][-4:]: r
    for r in json.loads(
        (ROOT / "tests/data/univariate_limit_reference_cases.json").read_text()
    )
}
NS = scalar_reference_namespace()


def test_real_parameter_strata_exhaustive_and_independently_attained():
    row = ROWS["0032"]
    a = sp.Symbol("a", real=True)
    x = sp.Symbol("x")
    n = sp.Symbol("n", integer=True, positive=True)
    assert row["parameter_domain"] == "Q.real(a)"
    assert sp.simplify(sp.Or(a > 0, sp.Eq(a, 0), a < 0)) is sp.S.true
    assert [c["stratum_id"] for c in row["parameter_cases"]] == [
        "positive",
        "zero",
        "negative",
    ]
    positive = sp.Symbol("b", positive=True)
    e = (positive - sp.sqrt(positive**2 + x * x)) / x
    assert sp.simplify(e + x / (positive + sp.sqrt(positive**2 + x * x))) == 0
    assert sp.limit(-x / (positive + sp.sqrt(positive**2 + x * x)), x, 0) == 0
    zero = -sp.sqrt(x * x) / x
    assert zero.subs(x, 1 / n) == -1 and zero.subs(x, -1 / n) == 1
    negative = (-2 - sp.sqrt(4 + x * x)) / x
    assert sp.limit(negative.subs(x, 1 / n), n, sp.oo) == -sp.oo
    assert sp.limit(negative.subs(x, -1 / n), n, sp.oo) == sp.oo


def test_isolated_audit_retains_each_stratum_and_unresolved_results():
    proc = subprocess.run(
        [
            sys.executable,
            "tools/audit_univariate_corpus.py",
            "--worker",
            "31",
            "--budget",
            "5",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=12,
        check=True,
    )
    record = json.loads(proc.stdout.splitlines()[-1])
    children = record["strata"]
    assert len(children) == 3 and all(c["executed"] for c in children)
    assert children[0]["classification"] == "pass" and children[0]["actual"] == "0"
    if any(c["classification"] == "unsupported_capability" for c in children):
        assert record["classification"] == "unsupported_capability"
    assert record["soft_budget_fired"] is False


def test_tuple_reference_is_passed_scalar_substitution():
    proc = subprocess.run(
        [
            sys.executable,
            "tools/audit_univariate_corpus.py",
            "--worker",
            "507",
            "--budget",
            "5",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=12,
        check=True,
    )
    record = json.loads(proc.stdout.splitlines()[-1])
    assert record["executed"] is True
    assert "object has no attribute 'subs'" not in record.get("detail", "")


def test_removable_quotient_correction_from_independent_derivatives():
    row = ROWS["0325"]
    x = sp.Symbol("x")
    e = sp.sympify(row["expression"], locals=NS)
    n, d = sp.fraction(sp.together(e))
    assert n.subs(x, 0) == d.subs(x, 0) == 0
    assert sp.diff(d, x).subs(x, 0).is_zero is False
    derived = sp.diff(n, x).subs(x, 0) / sp.diff(d, x).subs(x, 0)
    assert scalar_reference_equal(derived, sp.sympify(row["expected"], locals=NS))


def test_decimal_continuity_preserves_represented_source_precision():
    for id in ("1034", "1035"):
        row = ROWS[id]
        x = sp.Symbol("x")
        e = sp.sympify(row["expression"], locals=NS)
        point = sp.Rational(sp.sympify(row["point"]))
        expected = sp.sympify(row["expected"], locals=NS)
        assert scalar_reference_equal(e.subs(x, point), expected)
    assert (
        "0.025" in ROWS["1034"]["expression"] and "12.0" in ROWS["1035"]["expression"]
    )


def test_principal_branch_and_finite_target_reference_values():
    # acoth(z)=(log(1+1/z)-log(1-1/z))/2 on its principal sheet.
    z = sp.Rational(1, 3)
    logs = (sp.log(1 + 1 / z) - sp.log(1 - 1 / z)) / 2
    assert sp.simplify(-sp.I * logs - (-sp.pi / 2 - sp.I * sp.log(2) / 2)) == 0
    assert sp.sympify(ROWS["0685"]["expected"]) == -sp.I * sp.acoth(sp.Rational(1, 3))
    assert sp.Rational(sp.sympify(ROWS["0574"]["point"])) == 100
    assert ROWS["0574"]["expected_kind"] == "value"
    assert ROWS["3529"]["expected_kind"] == "value" and ROWS["3529"]["expected"] == "1"
