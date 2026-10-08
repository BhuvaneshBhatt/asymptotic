"""Deterministic pytest shard and symbolic-cost classification.

The test workflow imports this module through ``tools/run_test_shard.py``.
Cheap and moderate modules may share an interpreter within a shard. Expensive
and stateful modules run in fresh pytest subprocesses to prevent process-global
symbolic caches from coupling unrelated mathematical workloads.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path as _Path


@dataclass(frozen=True)
class TestModule:
    """A test module assigned to one test shard and one cost class."""

    path: str
    cost: str


SHARDS: dict[str, tuple[TestModule, ...]] = {
    "contracts": tuple(
        TestModule(path, "cheap")
        for path in (
            "tests/test_document_text_sanity.py",
            "tests/test_documentation_links.py",
            "tests/test_certified_ode_interchange_and_branch_invariants.py",
            "tests/test_expensive_symbolic_call_policy.py",
            "tests/test_generated_api_reference.py",
            "tests/test_instrumentation_event_registry.py",
            "tests/test_isolated_suite_runner.py",
            "tests/test_internal_helper_invariants.py",
            "tests/test_public_api_behavior.py",
            "tests/test_public_api_behavior_coverage.py",
            "tests/test_public_api_contract.py",
            "tests/test_public_assumptions_contract.py",
            "tests/test_publish_workflow.py",
            "tests/test_release_metadata.py",
            "tests/test_repository_coherence.py",
            "tests/test_root_import_boundaries.py",
            "tests/test_smoke.py",
            "tests/test_suite_layout.py",
            "tests/test_symbolic_instrumentation.py",
            "tests/test_symbolic_policy.py",
        )
    ),
    "core-algebra": tuple(
        TestModule(path, "moderate")
        for path in (
            "tests/test_asymptotic_algebra.py",
            "tests/test_as_element_protocol.py",
            "tests/test_foundations.py",
            "tests/test_frontier.py",
            "tests/test_perturbation_hierarchy.py",
            "tests/test_perturbation_dispatch.py",
            "tests/test_perturbation_problem.py",
            "tests/test_regular_perturbation.py",
            "tests/test_periodic_solvability.py",
        )
    ),
    "series-calculus": tuple(
        TestModule(path, "moderate")
        for path in (
            "tests/test_multiseries.py",
            "tests/test_nested.py",
            "tests/test_scale.py",
            "tests/limits/univariate/test_local_sqrt_scale.py",
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
        TestModule("tests/test_singular_implicits.py", "moderate"),
        TestModule("tests/test_asymptotic_optimization_and_roots.py", "moderate"),
    ),
    "relations-properties": (
        TestModule("tests/test_relations.py", "moderate"),
        TestModule("tests/test_relations_reference.py", "moderate"),
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
        TestModule(
            "tests/test_remainder_theorems_multivariate_implicit.py", "expensive"
        ),
    ),
    "differential": (
        TestModule("tests/test_dsolve_rsolve.py", "moderate"),
        TestModule("tests/test_lindstedt_poincare.py", "moderate"),
        TestModule("tests/test_multiple_scales.py", "moderate"),
        TestModule("tests/test_matched_asymptotics.py", "moderate"),
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
        TestModule("tests/test_sum_methods.py", "moderate"),
        TestModule("tests/test_advanced_sum_extensions.py", "moderate"),
        TestModule("tests/test_probability_asymptotics.py", "moderate"),
        TestModule("tests/test_multivariate_laplace.py", "moderate"),
        TestModule("tests/test_probability_bindings_and_contracts.py", "moderate"),
        TestModule("tests/test_stirling_pmf.py", "moderate"),
    ),
    "statistics": (
        TestModule("tests/test_statistical_boundaries.py", "expensive"),
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
    "extended": (
        TestModule("tests/test_advanced_metamorphic_contracts.py", "expensive"),
        TestModule("tests/test_correctness_boundaries.py", "moderate"),
        TestModule("tests/test_cylindrical_correlated_residuals.py", "moderate"),
        TestModule("tests/test_documentation_contract.py", "moderate"),
        TestModule("tests/test_example_api_contract.py", "moderate"),
        TestModule("tests/limits/shared/test_examples.py", "moderate"),
        TestModule("tests/test_expand_dispatch.py", "moderate"),
        TestModule("tests/test_first_three_residual_theorems.py", "moderate"),
        TestModule("tests/test_geometry_corpus.py", "moderate"),
        TestModule("tests/test_growth_comparison_global_integration.py", "moderate"),
        TestModule("tests/test_growth_comparison_public_api.py", "moderate"),
        TestModule("tests/test_growth_scale_corpus.py", "moderate"),
        TestModule("tests/test_growth_scale_recursive_algebra.py", "moderate"),
        TestModule("tests/test_growth_scale_symbolic_assumptions.py", "moderate"),
        TestModule("tests/test_hardening_context_registry.py", "moderate"),
        TestModule("tests/test_joint_vector_geometry.py", "moderate"),
        TestModule("tests/test_univariate_behavior_integration.py", "moderate"),
        TestModule("tests/test_univariate_behavior_corpus.py", "expensive"),
        TestModule("tests/test_univariate_frontier.py", "moderate"),
        TestModule("tests/limits/univariate/test_metamorphic.py", "expensive"),
        TestModule(
            "tests/limits/shared/test_capability_separation_examples.py", "moderate"
        ),
        TestModule("tests/limits/shared/test_metamorphic_contract.py", "expensive"),
        TestModule("tests/limits/reference/test_public_corpus.py", "moderate"),
        TestModule("tests/test_local_germ_algebra.py", "moderate"),
        TestModule("tests/test_local_germ_global_composition.py", "moderate"),
        TestModule("tests/test_local_strata_unification.py", "moderate"),
        TestModule("tests/test_mapped_strata.py", "moderate"),
        TestModule("tests/test_multivariate_adversarial_metamorphic.py", "expensive"),
        TestModule("tests/test_multivariate_analytic.py", "moderate"),
        TestModule("tests/test_multivariate_angular_optimization.py", "moderate"),
        TestModule("tests/test_multivariate_blowup_geometry.py", "moderate"),
        TestModule("tests/test_multivariate_branch_complex_geometry.py", "moderate"),
        TestModule("tests/test_multivariate_branch_divisor_geometry.py", "moderate"),
        TestModule("tests/test_multivariate_branch_monodromy.py", "moderate"),
        TestModule("tests/test_multivariate_branched_blowup_geometry.py", "moderate"),
        TestModule("tests/test_multivariate_capabilities.py", "moderate"),
        TestModule("tests/test_multivariate_capability_corpus.py", "moderate"),
        TestModule("tests/test_multivariate_capability_pairs.py", "moderate"),
        TestModule("tests/test_multivariate_cluster_coverage_invariant.py", "moderate"),
        TestModule("tests/test_multivariate_cluster_semantics.py", "moderate"),
        TestModule("tests/test_multivariate_common_chart_protocol.py", "moderate"),
        TestModule("tests/test_multivariate_complex_hankel_germs.py", "moderate"),
        TestModule("tests/test_multivariate_comprehensive_corpus.py", "expensive"),
        TestModule("tests/test_multivariate_conditional_limits.py", "moderate"),
        TestModule("tests/test_multivariate_conditional_pipeline.py", "moderate"),
        TestModule("tests/test_multivariate_coverage_certificates.py", "moderate"),
        TestModule("tests/test_multivariate_discontinuous_cluster_maps.py", "moderate"),
        TestModule("tests/test_multivariate_domain_cluster_geometry.py", "moderate"),
        TestModule("tests/test_multivariate_domain_evaluators.py", "moderate"),
        TestModule("tests/test_multivariate_exp_log_fan.py", "moderate"),
        TestModule("tests/test_multivariate_expansion.py", "moderate"),
        TestModule("tests/test_multivariate_extended_monodromy.py", "moderate"),
        TestModule("tests/test_multivariate_failure_regressions.py", "moderate"),
        TestModule("tests/test_multivariate_certificate_families.py", "moderate"),
        TestModule("tests/test_multivariate_geometry_extended.py", "moderate"),
        TestModule("tests/test_multivariate_geometry_unification.py", "moderate"),
        TestModule("tests/test_multivariate_growth_scale_comparability.py", "moderate"),
        TestModule(
            "tests/test_multivariate_inequality_tubular_geometry.py", "moderate"
        ),
        TestModule("tests/limits/parameterized/test_assumptions.py", "moderate"),
        TestModule(
            "tests/limits/multivariate/paths_clusters/test_advanced.py", "moderate"
        ),
        TestModule("tests/test_multivariate_metamorphic_generated.py", "expensive"),
        TestModule("tests/test_multivariate_parameter_cells.py", "moderate"),
        TestModule("tests/test_multivariate_parameter_cluster_sets.py", "moderate"),
        TestModule("tests/test_multivariate_parameterized_geometry.py", "moderate"),
        TestModule("tests/test_multivariate_projective_atlas.py", "moderate"),
        TestModule(
            "tests/test_multivariate_projective_cluster_decomposition.py", "moderate"
        ),
        TestModule("tests/test_multivariate_projective_parameter_cells.py", "moderate"),
        TestModule("tests/test_multivariate_proof_boundaries.py", "moderate"),
        TestModule("tests/test_multivariate_recursive_resolution.py", "moderate"),
        TestModule("tests/limits/reference/test_multivariate_corpus.py", "corpus"),
        TestModule("tests/test_multivariate_residual_regressions.py", "moderate"),
        TestModule(
            "tests/test_multivariate_semialgebraic_angular_image.py", "moderate"
        ),
        TestModule("tests/test_multivariate_signed_projective_clusters.py", "moderate"),
        TestModule("tests/test_multivariate_soundness.py", "moderate"),
        TestModule("tests/test_multivariate_special_function_germs.py", "moderate"),
        TestModule("tests/test_newton_cluster_geometry.py", "moderate"),
        TestModule("tests/test_piecewise_domain_accumulation.py", "moderate"),
        TestModule("tests/test_principal_branch_soundness.py", "moderate"),
        TestModule("tests/test_projective_infinite_dispatch.py", "moderate"),
        TestModule("tests/test_public_result.py", "moderate"),
        TestModule("tests/test_radial_reduction.py", "moderate"),
        TestModule("tests/test_reference_corpus_infrastructure.py", "expensive"),
        TestModule("tests/test_reference_normalization_capabilities.py", "expensive"),
        TestModule("tests/test_reference_vector_parser.py", "expensive"),
        TestModule("tests/limits/univariate/test_registry_closure.py", "moderate"),
        TestModule("tests/test_relation_certification_corpus.py", "moderate"),
        TestModule("tests/test_remainder_transforms.py", "moderate"),
        TestModule("tests/test_residual_cluster_theorems.py", "moderate"),
        TestModule("tests/test_dependency_private_api_regressions.py", "cheap"),
        TestModule("tests/limits/shared/test_public_dispatch.py", "moderate"),
        TestModule(
            "tests/limits/reference/test_sympy_github_regressions.py", "moderate"
        ),
        TestModule(
            "tests/limits/multivariate/paths_clusters/test_path_limits.py", "cheap"
        ),
        TestModule("tests/limits/multivariate/test_simultaneous.py", "moderate"),
        TestModule("tests/test_special_function_germs.py", "moderate"),
        TestModule("tests/test_stratified_mapping.py", "moderate"),
        TestModule("tests/test_stratum_evaluator.py", "moderate"),
        TestModule("tests/test_timeout_mechanism_certificates.py", "moderate"),
        TestModule("tests/test_unresolved_symbolic_certification.py", "moderate"),
        TestModule("tests/test_algebraic_curve_coverage.py", "moderate"),
        TestModule("tests/test_expansion_contract.py", "moderate"),
        TestModule("tests/test_local_order_proof_infrastructure.py", "moderate"),
        TestModule("tests/test_matched_boundary_layers.py", "moderate"),
        TestModule("tests/test_multidimensional_saddles.py", "moderate"),
        TestModule("tests/test_multivariate_coordinate_calculus.py", "moderate"),
        TestModule("tests/test_multivariate_matched.py", "moderate"),
        TestModule("tests/test_multivariate_matched_pde.py", "moderate"),
        TestModule("tests/test_multivariate_transseries.py", "moderate"),
        TestModule("tests/test_nscale_stratification.py", "moderate"),
        TestModule("tests/test_oscillatory_scales.py", "moderate"),
        TestModule("tests/test_parameter_uniformity.py", "moderate"),
        TestModule("tests/test_repository_integrity.py", "moderate"),
        TestModule("tests/test_sectorial_transseries.py", "moderate"),
        TestModule("tests/test_sectorial_transseries_corpus.py", "moderate"),
        TestModule("tests/test_singular_expansion.py", "moderate"),
        TestModule("tests/test_stokes_transseries.py", "moderate"),
        TestModule("tests/test_stratified_expansion.py", "moderate"),
        TestModule("tests/test_uniform_remainder.py", "moderate"),
    ),
}

COSTS = frozenset({"cheap", "moderate", "expensive", "stateful", "corpus"})

EXECUTION_BUDGET_SECONDS = {
    "cheap": 30,
    "moderate": 90,
    "expensive": 240,
    "stateful": 300,
    "corpus": 3600,
}

# Normalize the maintained shard manifest against the physical test tree.  Test
# modules are routinely renamed/moved as capability families are reorganized;
# stale paths must not make the release runner diverge from pytest collection.
# Explicit assignments and cost classes win.  Newly added modules default to a
# moderate catch-all until they receive a more specific cost classification.

_TESTS_DIR = _Path(__file__).resolve().parent
_CURRENT_TEST_MODULES = {
    f"tests/{path.relative_to(_TESTS_DIR).as_posix()}"
    for path in _TESTS_DIR.rglob("test_*.py")
}
_seen_paths: set[str] = set()
_normalized: dict[str, tuple[TestModule, ...]] = {}
for _shard, _modules in SHARDS.items():
    _kept = []
    for _module in _modules:
        if _module.path not in _CURRENT_TEST_MODULES or _module.path in _seen_paths:
            continue
        _kept.append(_module)
        _seen_paths.add(_module.path)
    _normalized[_shard] = tuple(_kept)

_missing = sorted(_CURRENT_TEST_MODULES - _seen_paths)
if _missing:
    _normalized["extended"] = _normalized.get("extended", ()) + tuple(
        TestModule(
            path,
            "corpus"
            if path
            in {
                "tests/limits/reference/test_multivariate_corpus.py",
                "tests/limits/reference/test_univariate_corpus.py",
            }
            else "moderate",
        )
        for path in _missing
    )
SHARDS = _normalized
