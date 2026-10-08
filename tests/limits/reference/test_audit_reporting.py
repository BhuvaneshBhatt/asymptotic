"""Historical value equivalences must not hide new execution failures."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "classification,status,actual,expected,review_kind,retain_observed",
    [
        (
            "unsupported_capability",
            "unknown",
            "None",
            "unsupported_capability",
            "pass",
            False,
        ),
        (
            "timeout_performance_failure",
            "",
            "I*pi",
            "timeout_performance_failure",
            "pass",
            False,
        ),
        ("solver_wrong_result", "proved", "0", "solver_wrong_result", "pass", False),
        ("solver_wrong_result", "proved", "I*pi", "pass", "pass", False),
        (
            "timeout_performance_failure",
            "",
            "None",
            "timeout_performance_failure",
            "bad_reference_or_missing_assumptions",
            True,
        ),
        (
            "solver_wrong_result",
            "proved",
            "0",
            "solver_wrong_result",
            "bad_reference_or_missing_assumptions",
            True,
        ),
    ],
)
def test_equivalence_review_scope(
    tmp_path, classification, status, actual, expected, review_kind, retain_observed
):
    tools = tmp_path / "tools"
    audit = tmp_path / "audit"
    tools.mkdir()
    audit.mkdir()
    shutil.copyfile(
        ROOT / "tools/build_univariate_audit_report.py",
        tools / "build_univariate_audit_report.py",
    )
    rows = [
        dict(
            id=f"univariate_reference_{i:04d}",
            classification="pass",
            status="proved",
            actual="0",
            executed=True,
            reference=dict(
                expression="x",
                variable="x",
                point="0",
                expected_kind="value",
                expected="0",
            ),
        )
        for i in range(1, 3537)
    ]
    rows[149].update(
        classification=classification,
        status=status,
        actual=actual,
        review_required=classification != "pass",
    )
    (audit / "corpus-run.json").write_text(
        json.dumps(dict(complete=True, rows=rows, counts={classification: 1}))
    )
    (audit / "reference-reviews.json").write_text(
        json.dumps(
            dict(
                reviews={
                    "univariate_reference_0150": dict(
                        classification=review_kind,
                        retain_observed_classification=retain_observed,
                        certified_actual="I*pi",
                        review_required=False,
                    ),
                }
            )
        )
    )
    subprocess.run(
        [sys.executable, str(tools / "build_univariate_audit_report.py")],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads((audit / "univariate-audit.json").read_text())
    assert report["rows"][149]["classification"] == expected
    assert report["rows"][149]["review_required"] is (expected != "pass")


@pytest.mark.parametrize(
    "expression,family",
    [
        ("asin(x)", "radicals and elementary branches"),
        ("KelvinKei(0, x)", "Bessel/Airy and related functions"),
        ("ScorerGi(x)", "Bessel/Airy and related functions"),
        ("FresnelF(x)", "integral/error functions"),
        ("AppellF1(1, 2, 3, 4, x, x)", "hypergeometric"),
        ("QPochhammer(x, q)", "q-special functions"),
    ],
)
def test_named_function_classification(expression, family):
    import runpy

    classify = runpy.run_path(str(ROOT / "tools/classify_capability_gaps.py"))[
        "classify"
    ]
    row = classify(dict(id="example", reference=dict(expression=expression)))
    assert row["mathematical_family"] == family


def test_missing_target_binding():
    import runpy

    classify = runpy.run_path(str(ROOT / "tools/classify_capability_gaps.py"))[
        "classify"
    ]
    row = classify(
        dict(id="example", reference=dict(expression="x", point="unbound_argument(1)"))
    )
    assert row["implementation_need"] == "missing source definitions or bindings"
