import json
from collections import defaultdict
from pathlib import Path

CASES = json.loads(
    (
        Path(__file__).resolve().parents[2]
        / "data/multivariate_limit_comprehensive_cases.json"
    ).read_text()
)
ROLES = {"prove", "dne_neighbor", "prerequisite_fail", "variant"}


def test_comprehensive_case_ids_unique():
    assert len(CASES) == 320
    assert len({c["id"] for c in CASES}) == len(CASES)


def test_every_capability_is_a_complete_quartet():
    grouped = defaultdict(list)
    for c in CASES:
        grouped[c["capability"]].append(c)
    assert len(grouped) == 80
    for capability, rows in grouped.items():
        assert len(rows) == 4, capability
        assert {r["role"] for r in rows} == ROLES, capability


def test_prerequisite_failures_are_capability_scoped_unknowns():
    for c in CASES:
        if c["role"] == "prerequisite_fail":
            assert c["scope"] == "capability"
            assert c["expected_kind"] == "unknown"


def test_global_cases_have_explicit_mathematical_expectations():
    for c in CASES:
        if c["scope"] == "global":
            assert c["expected_kind"] in {"value", "dne", "conditional", "cluster"}


def test_every_case_has_neutral_mathematical_schema():
    for c in CASES:
        assert {
            "id",
            "capability",
            "role",
            "scope",
            "expression",
            "variables",
            "target",
            "expected_kind",
            "provenance",
        } <= c.keys()


def test_soundness_boundary_quartets_are_present():
    soundness = [
        c for c in CASES if c["provenance"] == "constructed_soundness_boundary"
    ]
    assert len(soundness) == 48
    assert len({c["capability"] for c in soundness}) == 12
    for capability in {c["capability"] for c in soundness}:
        rows = [c for c in soundness if c["capability"] == capability]
        assert {r["role"] for r in rows} == ROLES


def test_growth_scale_quartets_are_present():
    growth_scale = [c for c in CASES if c.get("suite") == "growth_scale"]
    assert len(growth_scale) == 40
    assert len({c["capability"] for c in growth_scale}) == 10
    for capability in {c["capability"] for c in growth_scale}:
        rows = [c for c in growth_scale if c["capability"] == capability]
        assert {r["role"] for r in rows} == ROLES


def test_geometry_quartets_are_present():
    geometry = [c for c in CASES if c.get("suite") == "geometry"]
    assert len(geometry) == 32
    assert len({c["capability"] for c in geometry}) == 8
    for capability in {c["capability"] for c in geometry}:
        rows = [c for c in geometry if c["capability"] == capability]
        assert {r["role"] for r in rows} == ROLES


def test_special_function_quartets_are_present():
    special_functions = [c for c in CASES if c.get("suite") == "special_functions"]
    assert len(special_functions) == 20
    assert len({c["capability"] for c in special_functions}) == 5
    for capability in {c["capability"] for c in special_functions}:
        rows = [c for c in special_functions if c["capability"] == capability]
        assert {r["role"] for r in rows} == ROLES
        assert (
            next(r for r in rows if r["role"] == "dne_neighbor")["expected_kind"]
            == "dne"
        )


def test_vector_geometry_quartets_are_present():
    vector_geometry = [c for c in CASES if c.get("suite") == "vector_geometry"]
    assert len(vector_geometry) == 16
    assert len({c["capability"] for c in vector_geometry}) == 4
    for capability in {c["capability"] for c in vector_geometry}:
        rows = [c for c in vector_geometry if c["capability"] == capability]
        assert {r["role"] for r in rows} == ROLES
