"""Combine observed execution with explicit mathematical reviews, without xfails."""

import argparse
import csv
import importlib.metadata
import json
import platform
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = (
    "pass",
    "solver_wrong_result",
    "bad_reference_or_missing_assumptions",
    "unsupported_capability",
    "timeout_performance_failure",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="audit/corpus-run.json")
    args = parser.parse_args()
    raw = json.loads((ROOT / args.input).read_text())
    if not raw["complete"] or len(raw["rows"]) != 3536:
        raise ValueError("the report requires all 3,536 rows from a completed run")
    reviews = json.loads((ROOT / "audit/reference-reviews.json").read_text())["reviews"]
    rows = []
    for record in raw["rows"]:
        row = dict(record)
        if row.get("strata"):
            # A shared-budget interruption can precede the worker's aggregate
            # update. Completed child records prove those solver invocations.
            row["observed_executed"] = record.get("executed", False)
            row["executed"] = any(
                child.get("executed", False) for child in row["strata"]
            )
        row["observed_classification"] = record["classification"]
        row["function_heads"] = sorted(
            set(
                re.findall(
                    r"\b([A-Za-z_][A-Za-z_0-9]*)\(", record["reference"]["expression"]
                )
            )
        )
        review = reviews.get(record["id"])
        if review:
            row["mathematical_review"] = review
            row["classification"] = (
                record["classification"]
                if review.get("retain_observed_classification")
                else review["classification"]
            )
            row["review_required"] = (
                record.get("review_required", False)
                if review.get("retain_observed_classification")
                else review.get("review_required", False)
            )
            row["classification_basis"] = (
                "observed execution under source-verified corrected reference"
                if review.get("retain_observed_classification")
                else "independent mathematical/schema review"
            )
            if review.get("resolution_status"):
                row["reference_resolution_status"] = review["resolution_status"]
        else:
            row["classification_basis"] = (
                "observed execution; agreement is not independent validation"
            )
        if review and review.get("classification") == "pass":
            equivalent_observed_value = record.get("status") == "proved" and record.get(
                "actual"
            ) == review.get("certified_actual")
            if record["classification"] != "pass" and not equivalent_observed_value:
                # Equivalence of a specific proved value cannot certify a new
                # UNKNOWN, interrupted run, or changed solver answer.
                row["classification"] = record["classification"]
                row["classification_basis"] = (
                    "observed execution retained; equivalence review does not cover this outcome"
                )
                row["review_required"] = True
        if record["id"] == "univariate_reference_1170" and not (
            review and review.get("comparison_contract_changed")
        ):
            row["unresolved_issue"] = (
                "Solver returns directional -oo; reference expects spherical zoo. The codomain/infinity convention remains unresolved. Preserve this disagreement and expectation; do not xfail or claim a proven reference defect."
            )
        rows.append(row)
    counts = Counter(r["classification"] for r in rows)
    packages = {
        p: importlib.metadata.version(p)
        for p in (
            "sympy",
            "mpmath",
            "python-flint",
            "semialg",
            "exprtest",
            "funcprops",
            "pytest",
            "hypothesis",
        )
    }
    report = {
        k: v
        for k, v in raw.items()
        if k not in ("rows", "counts", "ids_by_classification", "review_required_count")
    }
    report.update(
        schema_version=2,
        counts={c: counts[c] for c in CATEGORIES},
        reference_defect_resolution_counts={
            c: sum(r.get("reference_resolution_status") == c for r in rows)
            for c in (
                "corrected",
                "ready for correction",
                "awaiting source",
                "awaiting convention",
            )
        },
        solver_invocation_count=sum(
            sum(s.get("executed", False) for s in r["strata"])
            if r.get("strata")
            else bool(r.get("executed", False))
            for r in rows
        ),
        ids_by_classification={
            c: [r["id"] for r in rows if r["classification"] == c] for c in CATEGORIES
        },
        observed_execution_counts=raw["counts"],
        executable_rows=sum(r["reference"].get("executable", True) for r in rows),
        explicitly_non_executable_rows=sum(
            not r["reference"].get("executable", True) for r in rows
        ),
        solver_invoked_rows=sum(r.get("executed", False) for r in rows),
        mathematical_reviews=len(reviews),
        corrected_reference_ids=[
            k for k, v in reviews.items() if v.get("reference_changed", False)
        ],
        review_required_count=sum(r.get("review_required", False) for r in rows),
        environment={
            "python": sys.version,
            "platform": platform.platform(),
            "packages": packages,
        },
        rows=rows,
        interpretation="Primary categories retain independently proven defects in the original reference, including corrected rows. observed_classification preserves execution against the corrected corpus. Remaining discrepancies are provisional solver_wrong_result, not independently proved errors. Pass means observed agreement or documented exact equivalence; the entire reference corpus has not received independent mathematical proof. Non-executable rows are inventoried without solver calls. Mapped applications retain their row IDs and shared budgets. Unsupported rows remain visible.",
    )
    if sum(report["counts"].values()) != 3536 or len({r["id"] for r in rows}) != 3536:
        raise ValueError("the audit must partition 3,536 unique reference IDs")
    (ROOT / "audit/univariate-audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    columns = (
        "id",
        "classification",
        "observed_classification",
        "review_required",
        "executed",
        "status",
        "actual",
        "detail",
        "wall_seconds",
    )
    with (ROOT / "audit/univariate-audit.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (ROOT / "audit/requirements-audit.txt").write_text(
        "\n".join(f"{p}=={v}" for p, v in packages.items()) + "\n"
    )
    print(
        json.dumps(
            {
                "counts": report["counts"],
                "solver_invoked_rows": report["solver_invoked_rows"],
                "review_required_count": report["review_required_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
