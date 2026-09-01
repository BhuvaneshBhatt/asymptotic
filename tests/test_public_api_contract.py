from __future__ import annotations

import inspect
import types

import asymptotic
from asymptotic._api_manifest import EXPERT_SUBMODULE_API, INTERNAL_API

EXPECTED_PRIMARY_API = (
    "AsymptoticAlgebra",
    "AsymptoticContext",
    "AsymptoticDSolveResult",
    "AsymptoticElement",
    "AsymptoticOptimizationResult",
    "AsymptoticRSolveResult",
    "AsymptoticRelationResult",
    "AsymptoticRemainder",
    "AsymptoticScale",
    "AsymptoticSolveResult",
    "AsymptoticSumResult",
    "AsymptoticTruncation",
    "GrowthComparison",
    "Multiseries",
    "NestedExpansion",
    "RemainderKind",
    "StatisticalAsymptoticResult",
    "TransseriesExpansion",
    "__version__",
    "airy_uniform_saddle_asymptotic",
    "asymptotic_argmax",
    "asymptotic_argmin",
    "asymptotic_big_o",
    "asymptotic_dsolve",
    "asymptotic_element",
    "asymptotic_equivalent",
    "asymptotic_expectation",
    "asymptotic_integrate",
    "asymptotic_little_o",
    "asymptotic_maximize",
    "asymptotic_minimize",
    "asymptotic_probability",
    "asymptotic_relation",
    "asymptotic_root",
    "asymptotic_rsolve",
    "asymptotic_solve",
    "asymptotic_sum",
    "coalescing_saddle_asymptotic",
    "compose_transseries",
    "differentiate",
    "discover_scale",
    "dominant_balance_candidates",
    "implicit_asymptotic",
    "integrate",
    "inverse_asymptotic",
    "laplace_asymptotic_integral",
    "mrv_decomposition",
    "multiseries",
    "multivariate_dominant_balance_candidates",
    "multivariate_implicit_asymptotics",
    "nested_expansion",
    "puiseux_series",
    "series_reversion",
    "transseries_from_expression",
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


def test_retired_root_exports_are_removed_not_deprecated():
    assert "certify_product_remainder" not in asymptotic.__dict__
    assert "certify_product_remainder" not in dir(asymptotic)
    retired_name = "certify_product_remainder"
    try:
        getattr(asymptotic, retired_name)
    except AttributeError:
        pass
    else:
        raise AssertionError("retired root export unexpectedly remains available")

    from asymptotic.remainder_theorems import certify_product_remainder

    assert callable(certify_product_remainder)


def test_every_primary_api_has_a_direct_import_contract_and_documentation():
    for name in EXPECTED_PRIMARY_API:
        assert hasattr(asymptotic, name), name
        obj = getattr(asymptotic, name)
        if name == "__version__":
            assert obj == "0.53.2"
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
