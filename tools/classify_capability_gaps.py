"""Partition every capability gap; distinguish evidence from inferred families."""

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = (
    (
        "polylogarithms",
        r"polylog",
        "integer/noninteger polylogarithm branch, inversion and cancellation theorem",
    ),
    (
        "distribution/statistics",
        r"Distribution|Skewness|Kurtosis|Expectation|Variance|DiracComb|DiracDelta",
        "translate the distribution/statistic and its parameter-dependent existence conditions",
    ),
    (
        "q-special functions",
        r"QPochhammer|QPolyGamma",
        "q-product or q-polygamma definitions with base, convergence and branch conventions",
    ),
    (
        "Lambert-W",
        r"LambertW",
        "signed/nonprincipal branch germ or coupled Lambert-W scale",
    ),
    (
        "gamma and factorial",
        r"gamma|factorial|binomial|\b(?:rf|ff|beta)\(|polygamma|StepFactorialPower",
        "parameter-aware gamma/factorial expansion with controlled cancellation",
    ),
    (
        "orthogonal polynomials and spherical harmonics",
        r"legendre|hermite|chebyshev|laguerre|gegenbauer|\bjacobi\(|Ynm",
        "controlled degree/order asymptotics or a signed endpoint germ with domain contracts",
    ),
    (
        "elliptic and Weierstrass",
        r"elliptic|Weierstrass|Jacobi|nome",
        "local/asymptotic elliptic or Weierstrass branch theorem",
    ),
    (
        "hypergeometric",
        r"hyper|appell|RegularizedHypergeometric",
        "parameter/sector-aware hypergeometric asymptotics",
    ),
    (
        "Bessel/Airy and related functions",
        r"bessel|hankel|airyai|airybi|scorer|Struve|Kelvin|parabolic_cylinder",
        "order/sector-aware special-function germ and remainder control",
    ),
    (
        "zeta and number-theoretic",
        r"zeta|Siegel|harmonic|Barnes|primepi",
        "number-theoretic local/tail expansion with parameter assumptions",
    ),
    (
        "integral/error functions",
        r"\b(?:Ei|li|Ci|Si|Chi|Shi)\(|erf|expint|inverse_error_increment|fresnel|dawson",
        "signed integral-function/error-function branch or tail theorem",
    ),
    (
        "discontinuous/discrete",
        r"floor|ceiling|Mod\(|\bfrac\(|Heaviside|sign\(|Wave|DiracDelta|NearestInteger",
        "one-sided discontinuity or discrete-domain/attained-subsequence theorem",
    ),
    (
        "oscillatory elementary",
        r"\b(?:sin|cos|tan|cot|sec|csc)\(",
        "attained oscillatory cluster/subsequence or pole-avoidance certificate",
    ),
    (
        "radicals and elementary branches",
        r"RealRoot|real_root|cbrt|sqrt|acosh|acos|asin|acsc|asec|acot|atanh|asech|log\(",
        "signed radical/log/inverse-function germ with domain control",
    ),
    (
        "exponential and variable powers",
        r"exp\(|\*\*",
        "parameter/domain-aware exponential or variable-power theorem",
    ),
)


def classify(row):
    reference = row["reference"]
    text = (
        " + ".join(c["expression"] for c in reference["applications"])
        if reference.get("applications")
        else reference["expression"]
    )
    evidence = set(row.get("evidence", []))
    detail = row.get("detail", "")
    heads = sorted(set(re.findall(r"\b([A-Za-z_][A-Za-z_0-9]*)\(", text)))
    if "unbound_argument(" in text or re.search(
        r"\b(Limit|unresolved_expression|unbound_argument|Quantity)\(", text
    ):
        family = "source serialization / held expression"
        need = "recover and parse the original source expression, bindings and nested evaluation semantics"
    elif re.search(r"\b(f|g|a|b)\(", text):
        family = "unspecified user function / sequence"
        need = "definition or local/asymptotic hypotheses for the user function or sequence"
    else:
        family, need = (
            "algebraic/parameter geometry",
            "conditional parameter strata or relative-domain algebraic limit",
        )
        for name, pattern, needed in FAMILIES:
            if re.search(pattern, text, re.IGNORECASE):
                family, need = name, needed
                break
    basis = "inferred from source expression family; not a proof that the reference is sound"
    if not reference.get("executable", True):
        group = "explicitly outside executable contract"
        basis = "explicit executable=False flag; no solver invocation"
    elif "parameter_strata_reference_contract" in evidence:
        group = "missing parameter stratification"
        basis = "certified parameter cells are recorded; the unconditional reference or unresolved complementary cells still need a complete parameter contract"
        need = "resolve complementary parameter cells and compare condition-specific references"
    elif "recurrence_family_germ_prerequisite" in evidence:
        group = "missing certified theorem in solver dispatch"
        basis = "solver prerequisite evidence: exact definitions/recurrences do not certify general function germs"
    elif (
        "undefined_function_prerequisite" in evidence and "StepFactorialPower(" in text
    ):
        group = "missing parameter stratification"
        basis = "registered step-factorial translation preserves unknown pole strata instead of assuming analytic continuity"
        need = "nonzero base for negative integer order, or a branch/pole contract for variable analytic order"
    elif "undefined_function_prerequisite" in evidence:
        group = "untranslated source function / missing function definition"
        basis = "solver prerequisite evidence"
    elif "special_function_cut_boundary_prerequisite" in evidence:
        group = "missing signed special-function branch germ"
        basis = "solver prerequisite evidence"
    elif evidence & {
        "complex_target_prerequisite",
        "complex_ray_pole_prerequisite",
        "complex_ray_germ_prerequisite",
    }:
        group = "unsupported complex target / direction semantics"
        basis = "solver prerequisite evidence: bounded finite rational continuity declined; a branch, pole or approach-domain theorem is still needed"
    elif "periodic_pole_avoidance_prerequisite" in evidence:
        group = "accumulating poles need avoidance/subsequence proof"
        basis = "solver prerequisite evidence"
    elif "parameter_exponential_rate_unresolved" in evidence:
        group = "missing parameter stratification"
        basis = "solver prerequisite evidence"
    elif evidence & {"accumulation_enclosure_only", "cluster_set_semantics"}:
        group = "cluster enclosure lacks attained-limit/nonexistence proof"
        basis = "solver prerequisite evidence"
    elif detail.startswith(
        ("TypeError:", "AttributeError:", "ValueError:", "NotImplementedError:")
    ):
        if "non-real" in detail or "finite" in detail:
            group = "unsupported complex target / direction semantics"
        elif "Tuple" in detail:
            group = "structured/vector source expression not scalar"
        elif "ff" in detail:
            group = "source function arity not translated"
        else:
            group = "parser or solver exception"
        basis = "recorded exception; implementation failure rather than mathematical nonexistence"
    else:
        group = "missing certified theorem in solver dispatch"
    context = " ".join(str(value) for value in reference.values())
    if re.search(
        r"\b(?:unbound_argument|unresolved_expression|Slot)\(", context
    ) or re.search(r"\b(?:f|g)\(", text):
        blocker = "missing source definitions or bindings"
        blocker_basis = "an unbound placeholder or unspecified user function appears in the reference"
    elif re.search(r"\b(?:Quantity|DiracComb|DiracDelta)\(", text):
        blocker = "units or distribution semantics"
        blocker_basis = (
            "the expression needs a non-scalar or distribution interpretation"
        )
    elif (
        re.search(r"\b(?:Limit|Hold)\(", text)
        or text.startswith("(")
        and "," in text
        and family == "algebraic/parameter geometry"
    ):
        blocker = "held evaluation or structured output"
        blocker_basis = (
            "nested evaluation scope or output structure requires a separate contract"
        )
    elif (
        "Interval(" in context
        or reference.get("direction") not in (None, "+", "-", "above", "below", "both")
        or "complex target / direction" in group
    ):
        blocker = "complex approach or cluster-output contract"
        blocker_basis = "non-real direction, complex-target evidence, or an interval-valued expectation"
    elif (
        "RealRoot(" in text
        or "parameter stratification" in group
        or "Contains(" in reference.get("assumptions", "")
    ):
        blocker = "domain or parameter contract"
        blocker_basis = "explicit real-root, discrete-domain, or parameter-stratification obligation"
    elif group == "untranslated source function / missing function definition":
        blocker = "named function definition or branch translation"
        blocker_basis = "undefined-function evidence for a named mathematical function; its convention must be checked before translation"
    else:
        blocker = "certified theorem or recognition"
        blocker_basis = "inferred from the expression; independent proof and, for excluded rows, executable-contract review are still needed"
    return dict(
        id=row["id"],
        reason_group=group,
        implementation_need=blocker,
        implementation_need_basis=blocker_basis,
        mathematical_family=family,
        needed_capability=need,
        basis=basis,
        function_heads=heads,
        execution_contract=(
            "eligible unresolved row"
            if reference.get("executable", True)
            else "excluded by reference flag"
        ),
        executable=reference.get("executable", True),
        solver_invoked=row.get("executed", False),
        evidence=sorted(evidence),
        detail=detail,
        reference=reference,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="audit/univariate-audit.json")
    parser.add_argument("--output", default="audit/capability-classification.json")
    args = parser.parse_args()
    report = json.loads((ROOT / args.input).read_text())
    rows = [
        classify(r)
        for r in report["rows"]
        if r["classification"] == "unsupported_capability"
    ]

    def inventory(key):
        return {
            label: dict(
                count=sum(r[key] == label for r in rows),
                ids=[r["id"] for r in rows if r[key] == label],
            )
            for label in sorted({r[key] for r in rows})
        }

    output = dict(
        total=len(rows),
        source_sha256=report["source_sha256"],
        by_reason=inventory("reason_group"),
        by_family=inventory("mathematical_family"),
        by_implementation_need=inventory("implementation_need"),
        by_execution_contract=inventory("execution_contract"),
        rows=rows,
        interpretation="Every capability row receives exactly one reason and one family. Explicit non-executable flags are distinguished from solver failures. Family-based routing is inferred, not independent mathematical validation; multiple additional capabilities may be needed.",
    )
    if sum(v["count"] for v in output["by_reason"].values()) != len(rows):
        raise ValueError("capability reason groups must partition all rows")
    (ROOT / args.output).write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: {label: v["count"] for label, v in output[k].items()}
                for k in ("by_reason", "by_family", "by_implementation_need")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
