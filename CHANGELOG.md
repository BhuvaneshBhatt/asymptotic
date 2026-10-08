## 0.2.0

Version 0.2.0 substantially expands `asymptotic` from its original expansion-oriented functionality into a broader symbolic asymptotic-analysis system. Additions include limit analysis, classical perturbation methods, richer complex and multivariate geometry, expanded saddle/Laplace analysis.

### Limits

- Added symbolic limit analysis for univariate and simultaneous multivariate limits.
- Added one-sided and directed limits, explicit path limits, and limits at finite and infinite points.
- Added detection of non-existent limits using conflicting paths, cluster sets, oscillation, branch behavior, and geometric analysis.
- Added radial/angular, weighted-scaling, Newton, projective, blow-up, and semialgebraic methods for difficult multivariate limits.
- Added support for vector-valued limits and joint cluster geometry.
- Added parameter-dependent and conditional limits, including results whose validity depends on symbolic assumptions.
- Known issue: ~50 timeouts in the multivariate limit suite.

### Perturbation methods

- Added a common perturbation-problem and order-hierarchy framework for algebraic equations, algebraic systems, scalar ODEs, and coupled ODE systems.
- Added regular perturbation solving for algebraic problems and ODE initial- and boundary-value problems.
- Added Lindstedt–Poincaré analysis with automatic frequency corrections and periodic solvability conditions.
- Added multiple-scales analysis with symbolic fast/slow derivative calculus and slow-flow equations, including weak damping, forcing, near resonance, and coupled systems.
- Added matched-asymptotic support for endpoint boundary layers.
- Added custom perturbation gauges, propagated conditions, resumable/partially solved hierarchies, manual order injection, and residual reconstruction.

### Complex, branch, and sector-aware asymptotics

- Added richer ramification and sheet tracking for fractional-power and formal-series problems.
- Expanded sector and angular-domain metadata, sector lookup, dominance information, and formal Stokes geometry for ODE/transseries analysis.

### Multivariate asymptotics and geometry

- Expanded multivariate local analysis with projective charts, weighted blow-ups, Newton geometry, exceptional directions, and semialgebraic image/coverage certification.
- Added stronger handling of singular strata, tangent and exceptional curves, angular cluster sets, and higher-dimensional coercivity.
- Added parameter stratification for regions where asymptotic behavior changes, including transition loci, discriminants, exceptional sets, multiplicity changes, and branch changes.
- Improved analysis at multivariate infinity through reciprocal and projective coordinate transformations.

### Laplace, saddle, and statistical asymptotics

- Expanded multivariate Laplace and saddle-point analysis.
- Added higher-order Gaussian contractions and local Gaussian-form calculations.
- Added support for logarithmic integrals and normalized asymptotic ratios.
- Added analysis of competing modes and co-dominant saddles.
- Added product-domain and boundary-saddle geometry.
- Strengthened certification of moving-domain, degenerate, and coalescing-saddle calculations.
- Improved branch-safe normalization of factorial/Gamma probability models and associated remainder reasoning.

### Formal ODE and recurrence analysis

- Expanded interoperability w/ `odeanalysis` package for formal ODE and system asymptotics.
- Added preservation of ramification, sheets, angular domains, sector boundaries, dominance levels, and recurrence information.
- Improved reconstruction and independent validation of residual certificates received through the ODE interchange layer.
- Strengthened recurrence and discrete-asymptotic analysis for factorial, exponential, stretched-exponential, power, logarithmic, and fractional correction scales.

