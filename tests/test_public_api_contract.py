from __future__ import annotations

import inspect
import types

import asymptotic
from asymptotic._api_manifest import EXPERT_SUBMODULE_API, INTERNAL_API

EXPECTED_PRIMARY_API = (
    "AsymptoticContext",
    "Circle",
    "DSolveResult",
    "DirectionalInfinity",
    "Evidence",
    "EvidenceStatus",
    "ExactClusterResult",
    "Multiseries",
    "OptimizationResult",
    "ProductResult",
    "PublicResult",
    "RSolveResult",
    "Remainder",
    "RemainderKind",
    "Scale",
    "SolveResult",
    "SquareWave",
    "StatisticalResult",
    "SumResult",
    "TransseriesExpansion",
    "Truncation",
    "__version__",
    "analytic_limit",
    "argmax",
    "argmin",
    "as_element",
    "big_o",
    "cluster_set",
    "complex_limit",
    "complex_ray_limit",
    "compose",
    "differentiate",
    "discover_scale",
    "discrete_limit",
    "dsolve",
    "equivalent",
    "expectation",
    "explain",
    "hyperasymptotic_series",
    "implicit",
    "integrate",
    "inverse",
    "leading_term",
    "limit",
    "lindstedt_poincare",
    "little_o",
    "local_series",
    "maximize",
    "mellin",
    "minimize",
    "multiseries",
    "nested_series",
    "one_sided_limit",
    "path_limit",
    "probability",
    "product",
    "puiseux_series",
    "regular_perturbation",
    "relation",
    "root",
    "rsolve",
    "series",
    "solve",
    "stratified_series",
    "sum",
    "truncate",
)


def test_root_namespace_is_exactly_the_primary_api():
    assert tuple(sorted(asymptotic.__all__)) == EXPECTED_PRIMARY_API
    eager_expert_objects = {
        name
        for name in EXPERT_SUBMODULE_API
        if name in asymptotic.__dict__
        and not isinstance(asymptotic.__dict__[name], types.ModuleType)
    }
    assert not eager_expert_objects
    assert set(INTERNAL_API).isdisjoint(asymptotic.__dict__)


def test_every_primary_api_has_and_documentation():
    for name in EXPECTED_PRIMARY_API:
        assert hasattr(asymptotic, name), name
        obj = getattr(asymptotic, name)
        if name == "__version__":
            assert obj == "0.2.0"
            continue
        assert callable(obj), name
        assert len((inspect.getdoc(obj) or "").strip()) >= 40, name
        if not (inspect.isclass(obj) and issubclass(obj, BaseException)):
            inspect.signature(obj)


def test_expert_api_remains_available_from_defining_submodules():
    from asymptotic.dominant import DominantBalanceCertificate
    from asymptotic.monomial import AsymptoticMonomial
    from asymptotic.obligations import AsymptoticKnowledge
    from asymptotic.remainder_theorems import GreenOperatorCertificate

    assert DominantBalanceCertificate.__module__ == "asymptotic.dominant"
    assert AsymptoticMonomial.__module__ == "asymptotic.monomial"
    assert AsymptoticKnowledge.__module__ == "asymptotic.obligations"
    assert GreenOperatorCertificate.__module__ == "asymptotic.remainder_theorems"
