import ast
from pathlib import Path

import sympy as sp

from asymptotic.multivariate_limits_advanced import (
    AdvancedLimitStatus,
    ClusterSetResult,
    VectorClusterSetResult,
)

CLUSTER_RESULT_TYPES = {
    "ClusterSetResult",
    "VectorClusterSetResult",
    "ProjectiveClusterAtlasResult",
    "JointClusterGeometryResult",
    "ComplexClusterSetResult",
    "ClusterMapResult",
}


def test_certified_cluster_results_require_at_runtime():
    x = sp.symbols("x", real=True)
    scalar = ClusterSetResult(
        x,
        (x,),
        (0,),
        sp.FiniteSet(0),
        0,
        0,
        AdvancedLimitStatus.CERTIFIED,
        "test",
        "missing coverage",
    )
    vector = VectorClusterSetResult(
        (x,),
        (x,),
        (0,),
        sp.FiniteSet(sp.Tuple(0)),
        AdvancedLimitStatus.CERTIFIED,
        "test",
        "missing coverage",
    )
    assert not scalar.certified
    assert not vector.certified


def test_no_certified_cluster_constructor_omits_traceable_coverage():
    source_root = Path(__file__).parents[1] / "src" / "asymptotic"
    missing = []
    for path in source_root.glob("*.py"):
        source = path.read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", None)
            if name not in CLUSTER_RESULT_TYPES:
                continue
            call_source = ast.get_source_segment(source, node) or ""
            if "AdvancedLimitStatus.CERTIFIED" not in call_source:
                continue
            if not any(keyword.arg == "coverage" for keyword in node.keywords):
                missing.append(f"{path.name}:{node.lineno}:{name}")
    assert not missing, "certified cluster constructors without coverage: " + ", ".join(
        missing
    )
