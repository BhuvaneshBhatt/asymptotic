# Statistical transforms, local limits, and asymptotic solving

The high-level statistical APIs reuse the probability, expectation, summation, Stirling-normalization, inversion, and transseries machinery rather than defining a separate symbolic algebra.

## Moments and transforms

- `asymptotic_moment` reduces through exact expectation when possible and otherwise uses the density/PMF engine. Central moments, variance, and covariance evaluate centered observables with guard terms before the final truncation, so cancellation of large raw moments does not erase smaller requested scales. Exact centered results are preserved without truncation.
- `asymptotic_factorial_moment` exposes falling-factorial moments, with exact Binomial and Poisson fast paths.
- `asymptotic_mgf`, `asymptotic_cgf`, `asymptotic_cumulant`, `asymptotic_characteristic_function`, and `asymptotic_pgf` build transform asymptotics.
- `asymptotic_entropy`, `asymptotic_cross_entropy`, and `asymptotic_kl_divergence` reuse the expectation/log-PMF machinery; exact distribution identities are preferred when available. Cross-entropy and KL divergence require the reference support to be contained in the target support. A disproved inclusion gives exact infinity; an undecided inclusion is retained as the set-theoretic obligation `support(reference) - support(target) = EmptySet`.

## Probabilities, rates, hazards, and modes

- `asymptotic_cdf` and `asymptotic_survival` specialize one-sided probability problems, with `F(x)=P(X <= x)` and `S(x)=P(X > x)` for both continuous and discrete distributions. Their complementary boundary identities are regression-tested on integer and noninteger thresholds.
- `asymptotic_log_probability` separates an exponentially small probability into its large-deviation rate and logarithmic prefactor instead of constructing a tiny floating expression.
- `asymptotic_rate_function` extracts an exponential rate from an existing probability asymptotic.
- `asymptotic_hazard` and `asymptotic_cumulative_hazard` distinguish two standard boundary conventions explicitly. Discrete point hazard uses `P(X >= k)` in its denominator, while cumulative hazard is exactly `-log S(k)` for the package survival convention `S(k)=P(X > k)`.
- `asymptotic_mode` and `asymptotic_map` expose the continuous saddle together with exact lattice candidates and the retained rounding correction.

## Quantiles and atoms

`asymptotic_quantile` uses the generalized inverse

```text
Q(p) = inf { x : F(x) >= p }.
```

Exact distribution quantiles are used first. This is essential for discrete laws, where solving `F(x) = p` is generally wrong because the CDF has jumps and a requested probability need not be attained exactly. If an exact generalized inverse is unavailable for a lattice distribution, the function returns `UNKNOWN` rather than substituting equality inversion. Equality-based local inversion remains available for continuous distributions when a unique real CDF branch can be identified.

## Certification replay

A result with status `CERTIFIED` is expected to have replayable theorem evidence. Laplace, Euler--Maclaurin, optimization, lattice-tail, and certified local-limit paths are covered by a common replay contract in the test suite. Exact results require no theorem replay. A fabricated `CERTIFIED` result without evidence fails the contract, preventing status strings from silently replacing proof objects.

## Certified Binomial lattice asymptotics

Positive factorial/Gamma PMFs are first normalized with `normalize_positive_pmf`, which carries the positive-real Stieltjes log-Gamma remainder. For Binomial masses the normalized log PMF exposes the entropy/rate phase directly. Interior lattice saddles use the resulting Gaussian saddle. One-sided large-deviation endpoints are handled discretely: exact adjacent-mass ratios are expanded about the moving boundary and summed as geometric-polynomial series. The leading lattice factor is therefore `1/(1-rho)`, not the continuous Watson endpoint factor. Bounded `floor`/`ceiling` offsets are retained.

`asymptotic_local_limit` distinguishes two local regimes. Locations `np+O(1)` use the ordinary `1/n` local Stirling expansion. Locations `np+s*sqrt(n)` use a separate `1/sqrt(n)` expansion and retain the Gaussian factor and half-power corrections. A top-level `floor` or `ceiling` is split into a smooth moving center plus a bounded lattice correction before expansion. Pointwise Stirling certification is propagated together with the analytic truncation remainder; uniform claims over growing windows are not inferred unless separately proved.

## Asymptotic relations

The relation API has two naming layers. `asymptotic_equivalent(f,g,...)` means `f/g -> 1`. `asymptotic_equal(f,g,...)` is the coarser Theta relation: each magnitude is eventually bounded by a constant multiple of the other. `asymptotic_less` is little-o, `asymptotic_less_equal` is big-O, `asymptotic_greater` is little-omega, and `asymptotic_greater_equal` is big-Omega. The aliases `asymptotic_little_o`, `asymptotic_big_o`, and `asymptotic_same_order` remain available.

At a finite univariate real point, two-sided real relations are certified only when both directed germs agree. At finite multivariate points, deterministic coordinate and seeded rays are a falsification mechanism: a failing ray certifies `False`, but agreement on finitely many rays is returned as undecided rather than as a proof.

## Algebraic systems

`asymptotic_solve` accepts parameter-dependent algebraic equations together with inequalities. For transcendental coefficient fields it first tries the MRV/Hardy polynomial backend: coefficient valuations determine the lower Newton polygon, characteristic equations determine leading balances, and recursive translated lifting supplies corrections. Identically zero roots are extracted with multiplicity. For real-domain problems, an asymptotic Sturm sequence can independently certify the number and multiplicities of eventual real roots.

When the MRV backend does not apply, exact algebraic branches remain a useful fallback. Domain constraints, requested dependent-variable limits, explicit assumptions, and eventual inequalities are tested on exact branches before truncation. The local implicit/Puiseux backend remains available for singular branches and prescribed dependent-variable limits.
