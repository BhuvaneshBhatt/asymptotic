from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_geometry_sources_are_packaged():
    required = (
        "src/asymptotic/puiseux.py",
        "src/asymptotic/blowup_geometry.py",
        "src/asymptotic/singular_expansion_geometry.py",
        "tests/test_newton_puiseux_branches.py",
        "tests/test_multivariate_blowup_geometry.py",
        "tests/test_multivariate_projective_atlas.py",
        "tests/test_singular_expansion.py",
    )
    missing = [relative for relative in required if not (ROOT / relative).is_file()]
    assert missing == []


def test_expansion_components_are_packaged():
    required = (
        "src/asymptotic/expansion_contract.py",
        "src/asymptotic/nscale_stratification.py",
        "src/asymptotic/parameter_uniformity.py",
        "src/asymptotic/matched.py",
    )
    missing = [relative for relative in required if not (ROOT / relative).is_file()]
    assert missing == []
