"""Generate theorem-aware advanced metamorphic contracts from the curated corpus.

These records are contracts for specialized runners; they keep
cluster-set/projective/blow-up transformations distinct from scalar SAME_RESULT.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
seeds = json.loads(
    (ROOT / "tests/data/multivariate_limit_comprehensive_cases.json").read_text()
)
rows = []


def add(c, family, oracle, **meta):
    rows.append(
        {
            "id": f"advmeta_{c['id']}_{family}",
            "seed_id": c["id"],
            "capability": c["capability"],
            "family": family,
            "oracle": oracle,
            **meta,
        }
    )


for c in seeds:
    n = len(c["variables"])
    expr = c["expression"]
    dom = c.get("domain", "True")
    cap = c["capability"]
    # exact GL(n,Q): fixed unitriangular determinant-one map
    if n >= 2 and all(t not in ("oo", "-oo") for t in c["target"]):
        add(
            c,
            "glq_unitriangular",
            "SAME_RESULT",
            matrix=[[1, 1] + [0] * (n - 2)]
            + [[0] * i + [1] + [0] * (n - i - 1) for i in range(1, n)],
            determinant="1",
        )
    # equivalent semialgebraic domain contract
    if dom != "True":
        add(
            c,
            "domain_double_negation",
            "SAME_RESULT",
            domain_rewrite=f"Not(Not({dom}))",
        )
    # projective conjugacy
    if any(t in ("oo", "-oo") for t in c["target"]):
        add(c, "projective_reciprocal", "PROJECTIVE_CONJUGATE")
    # vector output pushforward
    if "Tuple(" in expr:
        add(
            c,
            "vector_unimodular_pushforward",
            "PUSHFORWARD_CLUSTER",
            matrix="unitriangular_det_1",
        )
    # piecewise branch refinement / nonaccumulating branch
    if "Piecewise(" in expr:
        add(c, "piecewise_split", "SAME_RESULT")
        add(c, "piecewise_nonaccumulating", "SAME_RESULT")
    # weighted geometry
    if any(
        k in cap
        for k in ("weighted", "anisotropic", "blowup", "cusp", "tacnode", "projective")
    ):
        add(c, "weighted_blowup_conjugacy", "BLOWUP_CONJUGATE")
    # branch side exchange
    if (
        any(k in expr for k in ("arg(", "log(", "asin(", "acos(", "**Rational("))
        or "branch" in cap
    ):
        add(c, "branch_conjugation", "CONJUGATE_SIDE_EXCHANGE")
    # explicit negative mutation for every prerequisite-fail seed
    if c.get("role") == "prerequisite_fail":
        add(
            c,
            "negative_prerequisite",
            "MUST_NOT_CERTIFY",
            mutation="drop_or_reverse_required_prerequisite",
        )
# Deterministic uniqueness and contracts
assert len({r["id"] for r in rows}) == len(rows)
(ROOT / "tests/data/multivariate_limit_advanced_metamorphic_contracts.json").write_text(
    json.dumps(rows, indent=2) + "\n"
)
print(len(rows))
