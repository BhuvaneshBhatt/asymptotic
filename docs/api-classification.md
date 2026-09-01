# API classification

The root namespace contains **54** intentionally selected primary entry points. Other names are classified as expert submodule APIs or internal implementation details so the boundary is explicit rather than accidental.

The rule is simple: ordinary users should start with the primary root API; algorithm developers may use expert submodule APIs; internal names are not stability contracts.

## Primary root API

Stable, commonly discoverable workflow entry points; import these from `asymptotic`.

| Name | Defining/import module |
|---|---|
| `AsymptoticAlgebra` | `asymptotic` |
| `AsymptoticContext` | `asymptotic` |
| `AsymptoticDSolveResult` | `asymptotic` |
| `AsymptoticElement` | `asymptotic` |
| `AsymptoticOptimizationResult` | `asymptotic` |
| `AsymptoticRSolveResult` | `asymptotic` |
| `AsymptoticRelationResult` | `asymptotic` |
| `AsymptoticRemainder` | `asymptotic` |
| `AsymptoticScale` | `asymptotic` |
| `AsymptoticSolveResult` | `asymptotic` |
| `AsymptoticSumResult` | `asymptotic` |
| `AsymptoticTruncation` | `asymptotic` |
| `GrowthComparison` | `asymptotic` |
| `Multiseries` | `asymptotic` |
| `NestedExpansion` | `asymptotic` |
| `RemainderKind` | `asymptotic` |
| `StatisticalAsymptoticResult` | `asymptotic` |
| `TransseriesExpansion` | `asymptotic` |
| `__version__` | `asymptotic` |
| `airy_uniform_saddle_asymptotic` | `asymptotic` |
| `asymptotic_argmax` | `asymptotic` |
| `asymptotic_argmin` | `asymptotic` |
| `asymptotic_big_o` | `asymptotic` |
| `asymptotic_dsolve` | `asymptotic` |
| `asymptotic_element` | `asymptotic` |
| `asymptotic_equivalent` | `asymptotic` |
| `asymptotic_expectation` | `asymptotic` |
| `asymptotic_integrate` | `asymptotic` |
| `asymptotic_little_o` | `asymptotic` |
| `asymptotic_maximize` | `asymptotic` |
| `asymptotic_minimize` | `asymptotic` |
| `asymptotic_probability` | `asymptotic` |
| `asymptotic_relation` | `asymptotic` |
| `asymptotic_root` | `asymptotic` |
| `asymptotic_rsolve` | `asymptotic` |
| `asymptotic_solve` | `asymptotic` |
| `asymptotic_sum` | `asymptotic` |
| `coalescing_saddle_asymptotic` | `asymptotic` |
| `compose_transseries` | `asymptotic` |
| `differentiate` | `asymptotic` |
| `discover_scale` | `asymptotic` |
| `dominant_balance_candidates` | `asymptotic` |
| `implicit_asymptotic` | `asymptotic` |
| `integrate` | `asymptotic` |
| `inverse_asymptotic` | `asymptotic` |
| `laplace_asymptotic_integral` | `asymptotic` |
| `mrv_decomposition` | `asymptotic` |
| `multiseries` | `asymptotic` |
| `multivariate_dominant_balance_candidates` | `asymptotic` |
| `multivariate_implicit_asymptotics` | `asymptotic` |
| `nested_expansion` | `asymptotic` |
| `puiseux_series` | `asymptotic` |
| `series_reversion` | `asymptotic` |
| `transseries_from_expression` | `asymptotic` |

Expert names are available from their defining submodules only. There is no lazy root alias: `asymptotic.NAME` raises `AttributeError` for a name outside the primary root API.

## Expert submodule API

Specialized/result/building-block API; import from its defining submodule.

| Name | Defining/import module |
|---|---|
| `AlgebraicBranch` | `asymptotic.puiseux` |
| `AnalyticCompositionFrontier` | `asymptotic.frontier` |
| `ArgumentDomain` | `asymptotic.function_properties` |
| `ArgumentSignature` | `asymptotic.function_properties` |
| `ArgumentSpec` | `asymptotic.function_properties` |
| `AssumptionProperties` | `asymptotic.function_properties` |
| `AsymptoticKnowledge` | `asymptotic.obligations` |
| `AsymptoticMonomial` | `asymptotic.monomial` |
| `AsymptoticObligation` | `asymptotic.obligations` |
| `BalanceTerm` | `asymptotic.dominant` |
| `BranchChoice` | `asymptotic.puiseux` |
| `CoefficientExpansionObligation` | `asymptotic.obligations` |
| `ComparabilityFactorObligation` | `asymptotic.obligations` |
| `CompositionLayer` | `asymptotic.decomposition` |
| `ContinuationStatus` | `asymptotic.sparse` |
| `DEFAULT_REGISTRY` | `asymptotic.function_properties` |
| `DepthLimitObligation` | `asymptotic.obligations` |
| `DifferentialBalanceTerm` | `asymptotic.nonlinear_ode` |
| `DifferentialTransseriesStep` | `asymptotic.nonlinear_ode` |
| `Discontinuity` | `asymptotic.function_properties` |
| `DomainEndpoint` | `asymptotic.function_properties` |
| `DomainInterval` | `asymptotic.function_properties` |
| `DomainProperties` | `asymptotic.function_properties` |
| `DominantBalanceBranch` | `asymptotic.dominant` |
| `DominantBalanceCandidate` | `asymptotic.dominant` |
| `DominantBalanceCertificate` | `asymptotic.dominant` |
| `ExpLogExtension` | `asymptotic.tower` |
| `ExpLogTower` | `asymptotic.tower` |
| `ExponentialDichotomyCertificate` | `asymptotic.remainder_theorems` |
| `ExponentialScaleObligation` | `asymptotic.obligations` |
| `FunctionProperties` | `asymptotic.function_properties` |
| `GlobalExtremum` | `asymptotic.function_properties` |
| `GreenMode` | `asymptotic.remainder_theorems` |
| `GreenOperatorCertificate` | `asymptotic.remainder_theorems` |
| `GrowthComparisonObligation` | `asymptotic.obligations` |
| `GrowthIdealDecision` | `asymptotic.asymptotic_field` |
| `ImplicitAsymptoticBranch` | `asymptotic.implicit` |
| `IntegralShadowExtension` | `asymptotic.asymptotic_field` |
| `IntegralShadowProjection` | `asymptotic.asymptotic_field` |
| `IntegrationConstantLocation` | `asymptotic.asymptotic_field` |
| `JointNewtonTerm` | `asymptotic.multivariate_implicit` |
| `LazySparseSeries` | `asymptotic.sparse` |
| `LazyTerm` | `asymptotic.sparse` |
| `LogExpInverseResult` | `asymptotic.general_ops` |
| `LogExpScale` | `asymptotic.exp_log_scale` |
| `LogarithmicScaleObligation` | `asymptotic.obligations` |
| `MRVClass` | `asymptotic.mrv` |
| `MRVDecomposition` | `asymptotic.mrv` |
| `MultiseriesTerm` | `asymptotic.multiseries` |
| `MultivariateDominantBalanceCandidate` | `asymptotic.multivariate` |
| `MultivariateImplicitBranch` | `asymptotic.multivariate_implicit` |
| `MultivariateImplicitRegime` | `asymptotic.multivariate_implicit` |
| `NestedLevel` | `asymptotic.nested` |
| `NewtonCandidate` | `asymptotic.puiseux` |
| `NewtonPolyhedronTerm` | `asymptotic.multivariate` |
| `NonAnalyticObligation` | `asymptotic.obligations` |
| `NonlinearDifferentialBalance` | `asymptotic.nonlinear_ode` |
| `NonlinearDifferentialTransseriesBranch` | `asymptotic.nonlinear_ode` |
| `ODETransseriesBlock` | `asymptotic.ode_adapter` |
| `ODETransseriesData` | `asymptotic.ode_adapter` |
| `ObligationKind` | `asymptotic.obligations` |
| `OscillationKind` | `asymptotic.periodic` |
| `OscillatoryFactor` | `asymptotic.periodic` |
| `ParameterStratum` | `asymptotic.stratification` |
| `PeriodicDecomposition` | `asymptotic.periodic` |
| `PropertyKnowledge` | `asymptotic.function_properties` |
| `PropertyProvenance` | `asymptotic.function_properties` |
| `PropertyRule` | `asymptotic.function_properties` |
| `PuiseuxSeries` | `asymptotic.puiseux` |
| `PuiseuxTerm` | `asymptotic.puiseux` |
| `RamificationModel` | `asymptotic.monomial` |
| `RealUnivariateProperties` | `asymptotic.function_properties` |
| `RecursiveLogExpMonomial` | `asymptotic.logexp_transseries` |
| `RemainderProvenance` | `asymptotic.remainder` |
| `RemainderTheoremCertificate` | `asymptotic.remainder_theorems` |
| `ReversionBranch` | `asymptotic.reversion` |
| `ScaleDiscovery` | `asymptotic.scale` |
| `ScaleElement` | `asymptotic.scale` |
| `ScalingPath` | `asymptotic.multivariate` |
| `ScalingRegime` | `asymptotic.multivariate` |
| `ShadowField` | `asymptotic.asymptotic_field` |
| `ShadowGhostDecomposition` | `asymptotic.asymptotic_field` |
| `SingularityLocus` | `asymptotic.function_properties` |
| `SingularityProperties` | `asymptotic.function_properties` |
| `SparseContinuation` | `asymptotic.sparse` |
| `SparseNodeState` | `asymptotic.sparse` |
| `SparseTerm` | `asymptotic.frontier` |
| `SparseTermStream` | `asymptotic.sparse` |
| `StructuralDecomposition` | `asymptotic.decomposition` |
| `TransseriesBalanceCandidate` | `asymptotic.dominant` |
| `TransseriesBalanceTerm` | `asymptotic.dominant` |
| `TransseriesTerm` | `asymptotic.transseries` |
| `TransseriesValuation` | `asymptotic.transseries` |
| `UnsupportedNodeObligation` | `asymptotic.obligations` |
| `WeightCone` | `asymptotic.multivariate` |
| `ZeroTestObligation` | `asymptotic.obligations` |
| `analytic_at_decision` | `asymptotic.function_properties` |
| `applicable_rule` | `asymptotic.function_properties` |
| `canonical_asymptotic_monomial` | `asymptotic.monomial` |
| `canonical_equal` | `asymptotic.canonical` |
| `canonical_expr` | `asymptotic.canonical` |
| `canonical_parameter_monomial` | `asymptotic.monomial` |
| `canonical_recursive_logexp_monomial` | `asymptotic.logexp_transseries` |
| `canonicalize_transcendentals` | `asymptotic.decomposition` |
| `ramification_index` | `asymptotic.monomial` |
| `compare_asymptotic_monomials` | `asymptotic.monomial` |
| `compare_log_exp_scales` | `asymptotic.exp_log_scale` |
| `compare_monomials` | `asymptotic.transseries` |
| `compose_analytic_terms` | `asymptotic.frontier` |
| `critical_parameter_expressions` | `asymptotic.parameter_auto` |
| `decide` | `asymptotic.function_properties` |
| `dependent_taylor_balance_terms` | `asymptotic.dominant` |
| `domain_contains_decision` | `asymptotic.function_properties` |
| `entails` | `asymptotic.function_properties` |
| `function_property_rules` | `asymptotic.function_properties` |
| `infinitesimal_ideal_decision` | `asymptotic.asymptotic_field` |
| `lift_dominant_balance_branches` | `asymptotic.dominant` |
| `lift_transseries_balance_branches` | `asymptotic.dominant` |
| `maximal_univariate_decomposition` | `asymptotic.decomposition` |
| `moderate_growth_decision` | `asymptotic.asymptotic_field` |
| `nested_branch_safe_substitution_decision` | `asymptotic.function_properties` |
| `nested_branch_safety_decisions` | `asymptotic.function_properties` |
| `newton_polyhedron_terms` | `asymptotic.multivariate` |
| `nonlinear_differential_balance_terms` | `asymptotic.nonlinear_ode` |
| `ordered_transseries_terms` | `asymptotic.transseries` |
| `parameter_symbols` | `asymptotic.parameter_auto` |
| `polynomial_balance_terms` | `asymptotic.dominant` |
| `rational_decomposition` | `asymptotic.decomposition` |
| `rational_valuation` | `asymptotic.dominant` |
| `recursive_logexp_scale` | `asymptotic.logexp_transseries` |
| `require_decision` | `asymptotic.function_properties` |
| `simplify_parameter_strata` | `asymptotic.stratification` |
| `specialize_expression` | `asymptotic.parameter_auto` |
| `stratify_parameter_cases` | `asymptotic.stratification` |
| `transseries_balance_terms` | `asymptotic.dominant` |
| `transseries_valuation` | `asymptotic.transseries` |
| `zero_nonzero_stratification` | `asymptotic.stratification` |

## Internal

Implementation/debugging detail; no stability promise.

| Name | Defining/import module |
|---|---|
| `canonical_key` | `asymptotic.canonical` |
| `characteristic_poly_cache_info` | `asymptotic.remainder_theorems` |
| `clear_characteristic_poly_cache` | `asymptotic.remainder_theorems` |
| `clear_weight_cone_cache` | `asymptotic.multivariate` |
| `weight_cone_cache_info` | `asymptotic.multivariate` |

