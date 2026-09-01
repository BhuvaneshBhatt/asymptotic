"""Deterministic pytest shard and symbolic-cost classification.

The release workflow imports this module through ``tools/run_test_shard.py``.
Cheap and moderate modules may share an interpreter within a shard. Expensive
and stateful modules run in fresh pytest subprocesses to prevent process-global
symbolic caches from coupling unrelated mathematical workloads.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TestModule:
    """A test module assigned to one release shard and one cost class."""

    path: str
    cost: str


SHARDS: dict[str, tuple[TestModule, ...]] = {
    "contracts": tuple(
        TestModule(path, "cheap")
        for path in (
            "tests/test_document_text_sanity.py",
            "tests/test_documentation_links.py",
            "tests/test_expensive_symbolic_call_policy.py",
            "tests/test_generated_api_reference.py",
            "tests/test_instrumentation_event_registry.py",
            "tests/test_private_refactor_invariants.py",
            "tests/test_public_api_behavior.py",
            "tests/test_public_api_behavior_coverage.py",
            "tests/test_public_api_contract.py",
            "tests/test_publish_workflow.py",
            "tests/test_repository_coherence.py",
            "tests/test_root_import_boundaries.py",
            "tests/test_suite_layout.py",
            "tests/test_symbolic_instrumentation.py",
            "tests/test_symbolic_policy.py",
        )
    ),
    "core-algebra": tuple(
        TestModule(path, "moderate")
        for path in (
            "tests/test_asymptotic_algebra.py",
            "tests/test_asymptotic_element_protocol.py",
            "tests/test_foundations.py",
            "tests/test_frontier.py",
        )
    ),
    "series-calculus": tuple(
        TestModule(path, "moderate")
        for path in (
            "tests/test_multiseries.py",
            "tests/test_nested.py",
            "tests/test_scale.py",
            "tests/test_local_limit_sqrt_scale.py",
            "tests/test_periodic_scale_calculus.py",
            "tests/test_power_simplification_policy.py",
        )
    ),
    "transseries": (
        TestModule("tests/test_asymptotic_field_shadow.py", "moderate"),
        TestModule("tests/test_general_transseries_operations.py", "moderate"),
        TestModule("tests/test_monomial_transseries_adapter.py", "moderate"),
        TestModule("tests/test_recursive_logexp_transseries.py", "expensive"),
        TestModule("tests/test_transseries_advanced.py", "moderate"),
        TestModule("tests/test_transseries_balance.py", "moderate"),
        TestModule("tests/test_continuations.py", "moderate"),
    ),
    "algebraic": (
        TestModule("tests/test_newton_puiseux_branches.py", "moderate"),
        TestModule("tests/test_reversion_implicit.py", "expensive"),
        TestModule("tests/test_singular_implicit_asymptotics.py", "moderate"),
        TestModule("tests/test_asymptotic_optimization_and_roots.py", "moderate"),
    ),
    "relations-properties": (
        TestModule("tests/test_asymptotic_relations.py", "moderate"),
        TestModule("tests/test_function_properties.py", "moderate"),
        TestModule("tests/test_hardy_mrv_scale.py", "moderate"),
        TestModule("tests/test_hardy_sturm_germ_reduction.py", "moderate"),
        TestModule("tests/test_exprtest_integration.py", "moderate"),
    ),
    "multivariate": (
        TestModule("tests/test_multivariate_parameter_stratification.py", "moderate"),
        TestModule("tests/test_multivariate_weight_cones.py", "moderate"),
        TestModule("tests/test_parameter_stratification_canonical.py", "moderate"),
        TestModule("tests/test_parameter_stratification_provenance.py", "moderate"),
        TestModule("tests/test_remainder_theorems_multivariate_implicit.py", "expensive"),
    ),
    "differential": (
        TestModule("tests/test_dsolve_rsolve.py", "moderate"),
        TestModule("tests/test_integral_shadows_green.py", "moderate"),
        TestModule("tests/test_nonlinear_differential_lifting.py", "moderate"),
        TestModule("tests/test_nonlinear_differential_logexp.py", "expensive"),
    ),
    "discrete": (
        TestModule("tests/test_discrete_asymptotic_scales.py", "expensive"),
        TestModule("tests/test_birkhoff_trjitzinsky_tertiary.py", "stateful"),
        TestModule("tests/test_bt_metamorphic.py", "expensive"),
        TestModule("tests/test_recurrence_resonance_weight_sector.py", "moderate"),
    ),
    "probability-saddles": (
        TestModule("tests/test_advanced_saddles_and_sums.py", "moderate"),
        TestModule("tests/test_asymptotic_sum_methods.py", "moderate"),
        TestModule("tests/test_advanced_sum_extensions.py", "moderate"),
        TestModule("tests/test_probability_asymptotics.py", "moderate"),
        TestModule("tests/test_probability_bindings_and_contracts.py", "moderate"),
        TestModule("tests/test_stirling_pmf.py", "moderate"),
    ),
    "statistics": (
        TestModule("tests/test_statistical_hardening.py", "expensive"),
        TestModule("tests/test_statistical_transform_extensions.py", "moderate"),
        TestModule("tests/test_statistical_transforms_and_solve.py", "moderate"),
    ),
    "remainders-certificates": (
        TestModule("tests/test_certificate_reconstruction.py", "moderate"),
        TestModule("tests/test_negative_certification.py", "moderate"),
        TestModule("tests/test_obligations.py", "moderate"),
        TestModule("tests/test_remainder_operation_theorems.py", "moderate"),
        TestModule("tests/test_remainders.py", "moderate"),
    ),
    "invariants-properties": (
        TestModule("tests/test_asymptotic_contract_matrix.py", "expensive"),
        TestModule("tests/test_cross_api_invariants.py", "expensive"),
        TestModule("tests/test_independent_residual_oracles.py", "moderate"),
        TestModule("tests/test_metamorphic.py", "expensive"),
        TestModule("tests/test_power_expand_exact_properties.py", "expensive"),
        TestModule("tests/test_symbolic_robustness_regressions.py", "moderate"),
    ),
    "reference-docs": (
        TestModule("tests/test_documentation_examples.py", "moderate"),
        TestModule("tests/test_reference_cases.py", "expensive"),
        TestModule("tests/test_complexity_example.py", "moderate"),
    ),
    "performance-cache": (
        TestModule("tests/test_benchmark_smoke.py", "stateful"),
        TestModule("tests/test_canonicalization_and_cache_invariants.py", "stateful"),
        TestModule("tests/test_performance_profiles.py", "stateful"),
        TestModule("tests/test_symbolic_route_budgets.py", "stateful"),
    ),
    "artifact": (TestModule("tests/test_installed_wheel.py", "stateful"),),
}

COSTS = frozenset({"cheap", "moderate", "expensive", "stateful"})
