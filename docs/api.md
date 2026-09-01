# Primary API

These are the **136 names** intentionally exported from `asymptotic`. Expert records and low-level algorithms are documented separately in the API classification guide.

## Core representations

| Name | Contract |
|---|---|
| `AsymptoticAlgebra` | Coordinate-aware common algebra for heterogeneous asymptotic objects. |
| `AsymptoticContext` | Shared exact-asymptotic services. |
| `GrowthComparison` | Four-way result for asymptotic growth comparison: smaller, same order, larger, or unknown. |
| `AsymptoticScale` | A Shackell-style scale ordered from slowest to fastest vanishing. |
| `Multiseries` | Demand-driven recursive multiseries. |
| `NestedExpansion` | Resumable exact nested expansion. |
| `NestedForm` | Canonical structural description underlying a nested expansion. |
| `TransseriesExpansion` | Finite exact prefix of a generalized transseries branch. |
| `AsymptoticRemainder` | Certified or explicitly unknown remainder attached to a finite prefix. |
| `AsymptoticTruncation` | A finite prefix together with its explicit remainder semantics. |
| `RemainderKind` | Semantic strength of an asymptotic remainder statement. |
| `AsymptoticStratification` | Finite conditional family of asymptotic results. |
| `AsymptoticDifferentialField` | Finite computable realization of an asymptotic field with shadow maps. |
| `AsymptoticElement` | Uniform algebraic adapter over supported asymptotic representations. |
| `AsymptoticFieldElementProtocol` | Structural typing contract for common asymptotic-field operations. |
| `asymptotic_element` | Adapt a native representation to the common asymptotic-field protocol. |

## Expansion and structural analysis

| Name | Contract |
|---|---|
| `discover_scale` | Discover a dependency-driven exp-log scale. |
| `multiseries` | Create a lazy multiseries, discovering an asymptotic scale when omitted. |
| `nested_form` | Decompose an expression into a canonical nested asymptotic form. |
| `nested_expansion` | Build a resumable nested expansion, eagerly refining up to *depth* levels. |
| `decompose_expression` | Return canonical, compositional, and rationalized views of an expression. |
| `mrv_decomposition` | Compute explicit MRV comparability classes for an expression. |
| `transseries_from_expression` | Convert a finite additive exp/power/log expression to native terms. |
| `differentiate` | Differentiate an asymptotic object while preserving its abstraction. |
| `integrate` | Integrate an asymptotic object, allowing scale rediscovery when needed. |
| `asymptotic_integrate` | Compute a scale-aware finite asymptotic primitive. |
| `compose_transseries` | Compose a finite LE expression/transseries with a transseries argument. |

## Probability and expectation

| Name | Contract |
|---|---|
| `StatisticalAsymptoticResult` | Exact/formal/unknown statistical result with reduction and transformation provenance. |
| `StatisticalTransformResult` | High-level statistical transform with status, sources, conditions, and optional certified remainder. |
| `asymptotic_expectation` | Compute the expectation of an expression in random variables; exact joint expectations are attempted before one-variable density/PMF, moving-domain, Laplace, and lattice-sum fallbacks. |
| `asymptotic_probability` | Compute event probability asymptotics. `bindings=` maps ordinary symbols to random variables, `condition=` requests conditional probability, exact joint probability is attempted first, and structural density/PMF/Laplace fallbacks remain one-dimensional. |
| `asymptotic_log_probability` | Preserve exponential rate and logarithmic prefactor for small probabilities. |
| `asymptotic_local_limit` | Compute bounded-shift or `O(sqrt(n))` local mass/density asymptotics with lattice rounding retained. |
| `asymptotic_mode` / `asymptotic_map` | Return continuous saddle and lattice-corrected modal candidates. |
| `asymptotic_entropy`, `asymptotic_cross_entropy`, `asymptotic_kl_divergence` | Information-theoretic asymptotics. |
| `asymptotic_hazard`, `asymptotic_cumulative_hazard` | Hazard transforms derived from survival asymptotics. |
| `asymptotic_factorial_moment`, `asymptotic_pgf` | Discrete moment and generating-function transforms. |
| `laplace_asymptotic_integral` | Expand a one-dimensional nondegenerate interior-saddle or endpoint Laplace integral. |


## Asymptotic relations

| Name | Contract |
|---|---|
| `asymptotic_equivalent` | Ratio-1 equivalence, `f/g -> 1`. |
| `asymptotic_equal` / `asymptotic_same_order` | Theta equivalence: mutual constant-factor bounds. |
| `asymptotic_less` / `asymptotic_little_o` | Strictly smaller growth, little-o. |
| `asymptotic_less_equal` / `asymptotic_big_o` | Upper asymptotic bound, big-O. |
| `asymptotic_greater` | Strictly larger growth, little-omega. |
| `asymptotic_greater_equal` | Lower asymptotic bound, big-Omega. |
| `asymptotic_relation` | Structured tri-state relation result with directed evidence. |

All relation predicates accept `assumptions=`. An unresolved symbolic constant
factor therefore remains `None` without assumptions but can become a certified
decision when, for example, positivity/nonzeroness is explicitly known. Adding
assumptions is allowed to resolve `UNKNOWN`; removing assumptions must never
manufacture a stronger result.

## Algebraic, inverse, and implicit problems

| Name | Contract |
|---|---|
| `algebraic_branches` | Construct Puiseux branches with Newton-polygon leading balances. |
| `puiseux_series` | Construct a rational-exponent local series with explicit ramification. |
| `newton_polygon_candidates` | Compute leading Puiseux balances using the shared dominant-balance engine. |
| `series_reversion` | Revert a local (possibly Puiseux) series ``y=f(x)``. |
| `inverse_asymptotic` | Asymptotically invert ``y=f(x)`` at a finite point or infinity. |
| `inverse_logexp` | Invert a finite-height exp-log asymptotic function at infinity. |
| `asymptotic_root` | Find parameter-dependent roots using the asymptotic solver, with branch and limiting-root selection. |
| `asymptotic_minimize` / `asymptotic_maximize` | Optimize a univariate moving objective using complete stationary sets when available, asymptotic stationary branches, endpoint comparison, conservative certification, and floor/ceiling candidates on integer lattices. |
| `asymptotic_argmin` / `asymptotic_argmax` | Return only the asymptotic optimizer locations. |
| `dominant_balance_candidates` | Find exact Newton-style dominant balances. |
| `transseries_dominant_balance_candidates` | Find dominant balances in an expression-valued transseries domain. |
| `implicit_asymptotic` | Construct Puiseux or generalized-transseries implicit branches, automatically switching to Newton–Puiseux lifting for certified multiple roots. |
| `implicit_singularity_profile` | Diagnose local multiplicity, turning-point status, discriminant, and Newton scaling exponents. |

## Multivariate and differential problems

| Name | Contract |
|---|---|
| `scaling_path` | Construct a weighted one-parameter scaling path for multivariate limits. |
| `multivariate_scaling_regimes` | Discover admissible positive weight cones of the multivariate Newton diagram. |
| `multivariate_dominant_balance_candidates` | Compute dominant balances along an anisotropic multivariate scaling path. |
| `multivariate_implicit_asymptotics` | Discover and jointly lift multivariate implicit-system asymptotics. |
| `nonlinear_differential_dominant_balances` | Find power-law balances for a nonlinear differential polynomial. |
| `nonlinear_differential_transseries` | Recursively lift nonlinear differential dominant balances. |
| `asymptotic_dsolve` | High-level ODE dispatcher: use `odeanalysis` formal linear data when available, otherwise native nonlinear differential-transseries lifting. |
| `AsymptoticDSolveResult` | Structured ODE result retaining ordinary prefixes plus the richer formal/transseries branch objects. |
| `asymptotic_rsolve` | Solve scalar recurrences by exact solving when possible, otherwise by native discrete Newton analysis and factorial-scale Birkhoff--Trjitzinsky lifting for supported linear recurrences. |
| `AsymptoticRSolveResult` | Structured recurrence result with exact or native discrete-asymptotic provenance and resolved factorial-scale branches. |
| `automatic_parameter_stratification` | Automatically split on unresolved zero/nonzero structural coefficients. |
| `evaluate_parameter_strata` | Evaluate the same algorithm independently on certified parameter cases. |

## Periodic and function-property analysis

| Name | Contract |
|---|---|
| `periodic_decomposition` | Extract multiplicative periodic/oscillatory factors conservatively. |
| `periodic_bounds` | Return exact known bounds for a phase-periodic expression, if available. |
| `FunctionPropertyRegistry` | Mutable mapping from expression heads to property builders. |
| `PropertyDecision` | A tri-state mathematical decision together with auditable evidence. |
| `PropertyEnforcementError` | Raised when a required mathematical precondition is false or unresolved. |
| `function_properties` | Return reviewed mathematical properties for *expr*, or ``None`` if unknown. |
| `domain_properties` | Return the registered domain component for *expr*, when one is available. |
| `singularity_properties` | Return reviewed singularity and branch-locus information for *expr*. |
| `analytic_at` | Return a tri-state verdict for local analyticity after substituting *value*. |
| `branch_safe_substitution_decision` | Certify that substitution at ``value`` avoids registered branch obstructions. |
| `register_function_properties` | Register or replace a reviewed function-property builder in a registry. |

## Remainder certification

| Name | Contract |
|---|---|
| `certify_finite_sum_remainder` | Certify a finite sum, using a safe scale envelope when exact scale comparison is unavailable. |
| `certify_product_remainder` | Certify the exact binary product-error identity. |
| `certify_finite_product_remainder` | Certify a finite product by repeated binary product identities. |
| `certify_reciprocal_remainder` | Certify reciprocal stability under eventual nonvanishing and relative-smallness. |
| `certify_quotient_remainder` | Certify division through denominator reciprocal stability and product propagation. |
| `certify_algebraic_substitution_remainder` | Certify polynomial/rational substitution using exact algebraic perturbation identities. |
| `certify_differentiation_remainder` | Certify differentiation only when its regularity hypothesis is provable. |
| `certify_inverse_remainder` | Mean-value/Newton remainder theorem for an asymptotic inverse prefix. |
| `certify_unary_composition_remainder` | Local Lipschitz/Taylor propagation for nested exp-log composition. |
| `certify_nonlinear_lifting_remainder` | Simple-root implicit/Newton theorem for a lifted nonlinear branch. |
| `certify_frechet_inverse_operator_remainder` | Certify a scalar Fréchet inverse, first order or higher order. |
| `certify_green_inverse_operator_remainder` | Certify a higher-order scalar Green inverse using an exact dichotomy. |

## Field and ODE integration

| Name | Contract |
|---|---|
| `asymptotic_differential_field` | Construct an asymptotic differential field over the supplied scale generators. |
| `FormalODEAdapterError` | Raised when an ODE interchange object cannot be mapped safely. |
| `from_formal_ode_data` | Convert ``odeanalysis.FormalODEData`` to native asymptotic objects. |
| `certify_green_operator_data` | Certify an ``odeanalysis`` scalar-operator Green inverse. |

## Workflow links

The tables above are the curated root API. For mathematical selection guidance, see [Choosing an API by problem](workflows.md). For signatures and complete live docstrings generated from the installed source, see [Generated primary API reference](api-reference.md). Remainder/certification APIs are explained by failure mode in [Understanding `UNKNOWN`](unknown-results.md), while singular implicit, Newton-cone, Green/Frechet, and composition decisions are traced in [Worked algorithm traces](algorithm-traces.md).

## Operator matrix

| Problem | Primary operator | Exact route | Structural asymptotic route | Current strongest limitation |
|---|---|---|---|---|
| expression expansion | `multiseries` / `transseries_from_expression` | algebraic simplification | sparse exp-log/transseries evaluation | finite-height supported transseries domain |
| equation roots | `asymptotic_root` / `asymptotic_solve` | SymPy algebraic solving | implicit/Puiseux and MRV-Hardy Newton/Sturm | incomplete Hardy-field closure and complex-sector analysis |
| optimization | `asymptotic_minimize` / `asymptotic_maximize` | complete finite stationary sets | asymptotic stationary branches and lattice rounding | univariate objective/domain only |
| integral | `asymptotic_integrate` / `laplace_asymptotic_integral` | SymPy integration | scale-aware primitive/Laplace expansion | global certificates cover only selected real geometries |
| sum | `asymptotic_sum` | exact SymPy summation and finite geometric reduction | certified/formal termwise series, Abel transforms, creative telescoping into `asymptotic_rsolve`, Euler--Maclaurin, certified Mellin shifts, Gaussian Poisson summation, Riemann scaling, separable/fixed-box multidimensional sums, lattice saddles | coupled infinite multidimensional lattices, nonlinear oscillatory phases, and unrestricted creative telescoping remain conservative |
| expectation | `asymptotic_expectation` | exact joint SymPy expectation | one-dimensional density/PMF/Laplace/sum | structural multivariate fallback unsupported |
| probability | `asymptotic_probability` | exact joint SymPy probability | one-dimensional density/PMF/Laplace/sum | structural multivariate fallback unsupported |
| ODE | `asymptotic_dsolve` | formal linear `odeanalysis` data where applicable | nonlinear differential transseries | general variable-coefficient certification remains partial |
| recurrence | `asymptotic_rsolve` | exact recurrence solve, then discrete Newton edges | factorial/exponential/power lifting plus supported repeated-root secondary Newton phases | arbitrary repeated-secondary configurations, repeated tertiary roots, and further nested ramification remain conservative |
| growth relation | `asymptotic_relation` and convenience predicates | exact limit/comparison | deterministic ray falsification in several variables | finite rays never certify positive multivariate limits |


For dispatch behavior, see [Algorithm selection](algorithm-selection.md). For unresolved results, see [Understanding `UNKNOWN`](unknown-results.md).
