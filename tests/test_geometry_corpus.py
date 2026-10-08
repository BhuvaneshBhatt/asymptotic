import json
from collections import defaultdict
from pathlib import Path

CASES = json.loads(
    (
        Path(__file__).parent / "data/multivariate_limit_comprehensive_cases.json"
    ).read_text()
)
GEOMETRY = [c for c in CASES if c.get("suite") == "geometry"]


def test_geometry_has_eight_complete_geometry_quartets():
    grouped = defaultdict(list)
    for c in GEOMETRY:
        grouped[c["capability"]].append(c)
    assert len(GEOMETRY) == 32 and len(grouped) == 8
    for rows in grouped.values():
        assert {r["role"] for r in rows} == {
            "prove",
            "dne_neighbor",
            "prerequisite_fail",
            "variant",
        }


def test_geometry_is_geometry_weighted_not_textbook_only():
    required = {
        "algebraic_multibranch_curves",
        "singular_cusps_tacnodes",
        "reducible_varieties",
        "projective_directions_infinity",
        "weighted_blowup_geometry",
        "compact_angular_optimization",
        "disconnected_angular_images",
        "higher_dimensional_geometry",
    }
    assert {c["capability"] for c in GEOMETRY} == required
    assert any(len(c["variables"]) == 4 for c in GEOMETRY)
    assert any("Eq(" in c.get("domain", "") for c in GEOMETRY)
