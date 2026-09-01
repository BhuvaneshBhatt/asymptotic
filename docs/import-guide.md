# Import guide

The root namespace contains **54** primary entry points. A root name is reserved for a common operation, common result type, or core representation. Specialized theorem and certificate builders, registries, detailed statistical transforms, adapters, and lower-level structural controls belong in their defining submodules.

There is intentionally **no root-level backward-compatibility shim**. Moved names are removed from `asymptotic` immediately and must be imported from their defining submodules.


## Common import locations

There is no compatibility alias at the root. Representative import locations are:

| Root-style import to avoid | Supported import |
|---|---|
| `from asymptotic import nonlinear_differential_transseries` | `from asymptotic.nonlinear_ode import nonlinear_differential_transseries` |
| `from asymptotic import certify_product_remainder` | `from asymptotic.remainder_theorems import certify_product_remainder` |
| `from asymptotic import certify_green_inverse_operator_remainder` | `from asymptotic.remainder_theorems import certify_green_inverse_operator_remainder` |
| `from asymptotic import implicit_singularity_profile` | `from asymptotic.implicit import implicit_singularity_profile` |
| `from asymptotic import multivariate_scaling_regimes` | `from asymptotic.multivariate import multivariate_scaling_regimes` |
| `from asymptotic import scaling_path` | `from asymptotic.multivariate import scaling_path` |
| `from asymptotic import algebraic_branches` | `from asymptotic.puiseux import algebraic_branches` |
| `from asymptotic import decompose_expression` | `from asymptotic.decomposition import decompose_expression` |
| `from asymptotic import asymptotic_differential_field` | `from asymptotic.asymptotic_field import asymptotic_differential_field` |
| `from asymptotic import asymptotic_cdf` | `from asymptotic.statistical_transforms import asymptotic_cdf` |

The complete tables below are the normative import locations for expert names.
Repository tests scan Python sources and documentation code fences so retired root
imports cannot silently reappear.

## Kept at the root

`AsymptoticAlgebra`, `AsymptoticContext`, `AsymptoticDSolveResult`, `AsymptoticElement`, `AsymptoticOptimizationResult`, `AsymptoticRSolveResult`, `AsymptoticRelationResult`, `AsymptoticRemainder`, `AsymptoticScale`, `AsymptoticSolveResult`, `AsymptoticSumResult`, `AsymptoticTruncation`, `GrowthComparison`, `Multiseries`, `NestedExpansion`, `RemainderKind`, `StatisticalAsymptoticResult`, `TransseriesExpansion`, `__version__`, `airy_uniform_saddle_asymptotic`, `asymptotic_argmax`, `asymptotic_argmin`, `asymptotic_big_o`, `asymptotic_dsolve`, `asymptotic_element`, `asymptotic_equivalent`, `asymptotic_expectation`, `asymptotic_integrate`, `asymptotic_little_o`, `asymptotic_maximize`, `asymptotic_minimize`, `asymptotic_probability`, `asymptotic_relation`, `asymptotic_root`, `asymptotic_rsolve`, `asymptotic_solve`, `asymptotic_sum`, `coalescing_saddle_asymptotic`, `compose_transseries`, `differentiate`, `discover_scale`, `dominant_balance_candidates`, `implicit_asymptotic`, `integrate`, `inverse_asymptotic`, `laplace_asymptotic_integral`, `mrv_decomposition`, `multiseries`, `multivariate_dominant_balance_candidates`, `multivariate_implicit_asymptotics`, `nested_expansion`, `puiseux_series`, `series_reversion`, `transseries_from_expression`.

## Moved: certificate/theorem machinery

| Name | Import from |
|---|---|
| `certified_logfactorial` | `asymptotic.stirling` |
| `certified_loggamma` | `asymptotic.stirling` |
| `certify_algebraic_substitution_remainder` | `asymptotic.remainder_theorems` |
| `certify_antiderivative_remainder` | `asymptotic.remainder_theorems` |
| `certify_differentiation_remainder` | `asymptotic.remainder_theorems` |
| `certify_finite_product_remainder` | `asymptotic.remainder_theorems` |
| `certify_finite_sum_remainder` | `asymptotic.remainder_theorems` |
| `certify_frechet_inverse_operator_remainder` | `asymptotic.remainder_theorems` |
| `certify_green_inverse_operator_remainder` | `asymptotic.remainder_theorems` |
| `certify_green_operator_data` | `asymptotic.ode_adapter` |
| `certify_inverse_remainder` | `asymptotic.remainder_theorems` |
| `certify_nonlinear_lifting_remainder` | `asymptotic.remainder_theorems` |
| `certify_product_remainder` | `asymptotic.remainder_theorems` |
| `certify_quotient_remainder` | `asymptotic.remainder_theorems` |
| `certify_reciprocal_remainder` | `asymptotic.remainder_theorems` |
| `certify_scaling_remainder` | `asymptotic.remainder_theorems` |
| `certify_unary_composition_remainder` | `asymptotic.remainder_theorems` |

## Moved: function/domain property machinery

| Name | Import from |
|---|---|
| `FunctionPropertyRegistry` | `asymptotic.function_properties.registry` |
| `PropertyDecision` | `asymptotic.function_properties.semantics` |
| `PropertyEnforcementError` | `asymptotic.function_properties.semantics` |
| `analytic_at` | `asymptotic.function_properties.query` |
| `branch_safe_substitution_decision` | `asymptotic.function_properties.query` |
| `domain_properties` | `asymptotic.function_properties.query` |
| `function_properties` | `asymptotic.function_properties.query` |
| `register_function_properties` | `asymptotic.function_properties.query` |
| `singularity_properties` | `asymptotic.function_properties.query` |

## Moved: specialized statistical transforms

| Name | Import from |
|---|---|
| `AsymptoticModeResult` | `asymptotic.statistical_transforms` |
| `LogProbabilityResult` | `asymptotic.statistical_transforms` |
| `StatisticalTransformResult` | `asymptotic.statistical_transforms` |
| `asymptotic_cdf` | `asymptotic.statistical_transforms` |
| `asymptotic_cgf` | `asymptotic.statistical_transforms` |
| `asymptotic_characteristic_function` | `asymptotic.statistical_transforms` |
| `asymptotic_covariance` | `asymptotic.statistical_transforms` |
| `asymptotic_cross_entropy` | `asymptotic.statistical_transforms` |
| `asymptotic_cumulant` | `asymptotic.statistical_transforms` |
| `asymptotic_cumulative_hazard` | `asymptotic.statistical_transforms` |
| `asymptotic_entropy` | `asymptotic.statistical_transforms` |
| `asymptotic_factorial_moment` | `asymptotic.statistical_transforms` |
| `asymptotic_hazard` | `asymptotic.statistical_transforms` |
| `asymptotic_kl_divergence` | `asymptotic.statistical_transforms` |
| `asymptotic_local_limit` | `asymptotic.statistical_transforms` |
| `asymptotic_log_probability` | `asymptotic.statistical_transforms` |
| `asymptotic_map` | `asymptotic.statistical_transforms` |
| `asymptotic_mgf` | `asymptotic.statistical_transforms` |
| `asymptotic_mode` | `asymptotic.statistical_transforms` |
| `asymptotic_moment` | `asymptotic.statistical_transforms` |
| `asymptotic_pgf` | `asymptotic.statistical_transforms` |
| `asymptotic_product` | `asymptotic.statistical_transforms` |
| `asymptotic_quantile` | `asymptotic.statistical_transforms` |
| `asymptotic_rate_function` | `asymptotic.statistical_transforms` |
| `asymptotic_survival` | `asymptotic.statistical_transforms` |
| `asymptotic_variance` | `asymptotic.statistical_transforms` |
| `normalize_positive_pmf` | `asymptotic.stirling` |

## Moved: specialized representations and adapters

| Name | Import from |
|---|---|
| `AsymptoticDifferentialField` | `asymptotic.asymptotic_field` |
| `AsymptoticFieldElementProtocol` | `asymptotic.algebra` |
| `AsymptoticStratification` | `asymptotic.stratification` |
| `ComplexBranchMetadata` | `asymptotic.complex_domain` |
| `ComplexSector` | `asymptotic.complex_domain` |
| `FormalODEAdapterError` | `asymptotic.ode_adapter` |
| `NestedForm` | `asymptotic.nested` |
| `asymptotic_differential_field` | `asymptotic.asymptotic_field` |
| `complex_germ_metadata` | `asymptotic.complex_domain` |
| `evaluate_parameter_strata` | `asymptotic.stratification` |
| `from_formal_ode_data` | `asymptotic.ode_adapter` |
| `merge_complex_germ_metadata` | `asymptotic.complex_domain` |

## Moved: low-level structural algorithms

| Name | Import from |
|---|---|
| `algebraic_branches` | `asymptotic.puiseux` |
| `asymptotic_integral` | `asymptotic.general_ops` |
| `decompose_expression` | `asymptotic.decomposition` |
| `inverse_logexp` | `asymptotic.general_ops` |
| `multivariate_scaling_regimes` | `asymptotic.multivariate` |
| `newton_polygon_candidates` | `asymptotic.puiseux` |
| `nonlinear_differential_dominant_balances` | `asymptotic.nonlinear_ode` |
| `nonlinear_differential_transseries` | `asymptotic.nonlinear_ode` |
| `periodic_bounds` | `asymptotic.periodic` |
| `periodic_decomposition` | `asymptotic.periodic` |
| `scaling_path` | `asymptotic.multivariate` |
| `transseries_dominant_balance_candidates` | `asymptotic.dominant` |

## Moved: other specialized helpers

| Name | Import from |
|---|---|
| `AsymptoticSolutionBranch` | `asymptotic.solve` |
| `LaplaceRemainderCertificate` | `asymptotic.probability` |
| `StirlingNormalization` | `asymptotic.stirling` |
| `asymptotic_equal` | `asymptotic.relations` |
| `asymptotic_greater` | `asymptotic.relations` |
| `asymptotic_greater_equal` | `asymptotic.relations` |
| `asymptotic_less` | `asymptotic.relations` |
| `asymptotic_less_equal` | `asymptotic.relations` |
| `asymptotic_same_order` | `asymptotic.relations` |
| `automatic_parameter_stratification` | `asymptotic.parameter_auto` |
| `implicit_singularity_profile` | `asymptotic.implicit` |
| `nested_form` | `asymptotic.nested` |

## Policy going forward

New root exports should satisfy at least one of these tests: they are a high-level workflow entry point, a core representation/result used across multiple workflows, or a ubiquitous relation/calculus operation. Otherwise they should remain submodule APIs.
