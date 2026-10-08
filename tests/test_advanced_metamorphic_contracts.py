import json
from pathlib import Path

ROWS = json.loads(
    (
        Path(__file__).with_name("data")
        / "multivariate_limit_advanced_metamorphic_contracts.json"
    ).read_text()
)


def test_advanced_contracts_cover_all_requested_families():
    families = {r["family"] for r in ROWS}
    assert {
        "glq_unitriangular",
        "domain_double_negation",
        "projective_reciprocal",
        "vector_unimodular_pushforward",
        "piecewise_split",
        "piecewise_nonaccumulating",
        "weighted_blowup_conjugacy",
        "branch_conjugation",
        "negative_prerequisite",
    } <= families


def test_negative_metamorphisms_use_must_not_certify():
    neg = [r for r in ROWS if r["family"] == "negative_prerequisite"]
    assert len(neg) == 80
    assert all(r["oracle"] == "MUST_NOT_CERTIFY" for r in neg)


def test_contract_ids_unique():
    assert len({r["id"] for r in ROWS}) == len(ROWS)
