# Certified Stirling normalization for positive PMFs

`certified_loggamma`, `certified_logfactorial`, and
`normalize_positive_pmf` provide a branch-safe bridge from factorial/Gamma
probability masses to exponential scale.

For positive real `z`, the implementation uses

\[
\log\Gamma(z)=(z-\tfrac12)\log z-z+\tfrac12\log(2\pi)
+\sum_{r=1}^{m-1}\frac{B_{2r}}{2r(2r-1)z^{2r-1}}+R_m
\]

together with the classical positive-real Stieltjes bound

\[
|R_m|\le
\frac{|B_{2m}|}{2m(2m-1)z^{2m-1}}.
\]

No forced logarithm or power expansion is used. Gamma arguments and other
multiplicative factors must be provably positive from SymPy assumptions or
explicit support assumptions.

This is designed for PMFs such as Poisson and binomial masses before they
enter `asymptotic_sum`'s lattice-saddle engine.

## Automatic lattice integration

The probability and local-limit layers now invoke this normalization automatically for supported positive factorial/Gamma PMFs. Binomial interior saddles are read from the normalized entropy phase. Endpoint large-deviation tails use exact adjacent-mass ratios and geometric domination, while `asymptotic_local_limit` has separate `np+O(1)` and `np+s*sqrt(n)` expansions. A rounded moving location is decomposed into its smooth center and a bounded floor/ceiling correction before the Stirling series is formed.
