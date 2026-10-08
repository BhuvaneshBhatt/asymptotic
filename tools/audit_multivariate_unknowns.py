"""Classify capability-matrix UNKNOWN outcomes by engineering cause."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def classify(case, row):
    detail = row.get("detail", "") or ""
    if case.get("suite") == "geometry":
        if case["capability"] == "projective_directions_infinity":
            return "geometry_missing_infinite_target_dispatch"
        if case["role"] == "prerequisite_fail":
            if case["capability"] in {
                "algebraic_multibranch_curves",
                "singular_cusps_tacnodes",
            }:
                return "geometry_missing_singular_curve_certificate"
            if case["capability"] == "weighted_blowup_geometry":
                return "geometry_symbolic_weight_certificate_absent"
            if case["capability"] == "compact_angular_optimization":
                return "geometry_symbolic_angular_certificate_absent"
        return "geometry_geometry_integration"
    if case.get("suite") == "growth_scale":
        if (
            case["role"] == "prerequisite_fail"
            and case.get("expected_kind") == "unknown"
        ):
            return "growth_scale_intentional_scale_boundary"
        growth = row.get("growth_comparison")
        if growth:
            if growth.get("certified"):
                return "growth_scale_global_integration"
            if growth.get("actual") == "unknown":
                return "growth_scale_missing_scale_proof"
        cap = case["capability"]
        if cap in {
            "iterated_logarithms",
            "nested_exponentials",
            "mixed_exp_log_products",
            "scale_composition",
        }:
            return "growth_scale_missing_scale_proof"
        if cap == "cross_variable_growth_cells":
            return "growth_scale_cross_variable_integration"
        if cap == "one_sided_scales":
            return "growth_scale_one_sided_integration"
        return "growth_scale_global_integration"
    if "timeout" in detail:
        return "proof_search_timeout"
    if "Tuple" in detail:
        return "vector_tuple_dispatch"
    if "finite real target" in detail:
        return "infinite_target_not_implemented"
    if "AsymptoticStratification" in detail:
        return "parameter_result_protocol"
    if "PolynomialError" in detail:
        return "symbolic_exponent_dispatch"
    if "Modulo by zero" in detail:
        return "modular_normalization"
    if "not callable" in detail:
        return "function_symbol_parsing"
    if case["role"] == "prerequisite_fail":
        return "intentional_capability_boundary"
    return "other_solver_gap"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="capability-matrix.json")
    ap.add_argument("--output", default="unknown-cause-audit.json")
    ns = ap.parse_args()
    root = Path(__file__).parents[1]
    matrix = json.loads((root / ns.matrix).read_text())
    cases = json.loads(
        (root / "tests/data/multivariate_limit_comprehensive_cases.json").read_text()
    )
    byid = {c["id"]: c for c in cases}
    rows = []
    for r in matrix["cases"]:
        if r["actual"] != "UNKNOWN":
            continue
        c = byid[r["id"]]
        cause = classify(c, r)
        rows.append(
            {
                "id": r["id"],
                "capability": c["capability"],
                "role": c["role"],
                "suite": c.get("suite"),
                "cause": cause,
                "detail": r.get("detail"),
                "growth_comparison": r.get("growth_comparison"),
            }
        )
    counts = Counter(r["cause"] for r in rows)
    growth_scale = Counter(r["cause"] for r in rows if r["suite"] == "growth_scale")
    geometry = Counter(r["cause"] for r in rows if r["suite"] == "geometry")
    out = {
        "unknown_count": len(rows),
        "counts": dict(counts),
        "growth_scale_counts": dict(growth_scale),
        "geometry_counts": dict(geometry),
        "cases": rows,
    }
    (root / ns.output).write_text(json.dumps(out, indent=2) + "\n")
    print(
        json.dumps(
            {
                "unknown_count": len(rows),
                "counts": dict(counts),
                "growth_scale_counts": dict(growth_scale),
                "geometry_counts": dict(geometry),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
