# Asymptotic sums

`asymptotic_sum()` is the discrete analogue of the package's scale-aware
integration machinery. It returns an `AsymptoticSumResult` containing the
selected method, status, finite transseries when available, remainder evidence,
and the original symbolic sum.

The dispatcher uses mathematically distinct routes rather than one unrestricted
symbolic fallback:

1. **Exact summation** when SymPy closes the sum.
2. **Termwise parameter series** when the summation bounds are independent of
   the expansion parameter.
3. **Summation by parts (Abel transformation)** for products with a Gosper
   antidifference. Repeated transforms expose boundary asymptotics of rapidly
   decaying tails.
4. **Euler--Maclaurin** with its explicit error estimate retained as an
   `AsymptoticRemainder`.
5. **Creative telescoping / Zeilberger recurrence generation** for supported
   bivariate hypergeometric terms. A rational Gosper certificate is replayed
   before the resulting recurrence is sent to `asymptotic_rsolve()`.
6. **Poisson summation** for full Gaussian lattices, including a proved
   principal-cell linear phase and an exponentially small dual-lattice tail.
7. **Finite oscillatory geometric sums** for affine phases. The exact geometric
   reduction is retained rather than relabeling a truncated Taylor series exact.
8. **Euler--Maclaurin** with its explicit error estimate retained as an
   `AsymptoticRemainder`.
9. **Mellin poles** for supported infinite small-parameter sums. Contour shifts
   are `CERTIFIED` only when the fundamental strip, crossed poles, positive
   small parameter, and vertical Gamma decay are all proved.
10. **Scaled Riemann sums** using `k = n*x` and a bounded verified primitive.
11. **Discrete saddles** using the same Laplace/saddle machinery as continuous
   probability after the lattice-spacing factor is inserted.
12. **Multidimensional sums** for separable products and fixed finite boxes.

A route may be selected explicitly with `method="series"`,
`"summation-by-parts"`, `"zeilberger"`, `"poisson"`, `"oscillatory"`,
`"euler-maclaurin"`, `"mellin"`, `"riemann"`, or `"saddle"`. `method="auto"`
tries bounded structural routes and leaves unsupported cases `UNKNOWN`.

## Euler--Maclaurin tail

```python
import sympy as sp
from asymptotic import asymptotic_sum

n = sp.symbols("n", positive=True, integer=True)
k = sp.symbols("k", positive=True, integer=True)

result = asymptotic_sum(
    1/k**2,
    k,
    n,
    sp.oo,
    parameter=n,
    terms=3,
    method="euler-maclaurin",
)
```

The prefix is

```text
1/n + 1/(2*n**2) + 1/(6*n**3)
```

and the result carries the Euler--Maclaurin error estimate.

## Summation by parts

For an exponentially decaying tail,

```python
result = asymptotic_sum(
    sp.exp(-k)/k,
    k,
    n,
    sp.oo,
    parameter=n,
    terms=3,
    method="summation-by-parts",
)
```

repeated Abel transformations produce the first three boundary terms without
asking a general recurrence solver. The result is marked `FORMAL` because the
remaining transformed tail is not yet certified uniformly.

## Mellin example: a Bessel lattice sum

The classical small-parameter sum

```python
s = sp.symbols("s", positive=True)
result = asymptotic_sum(
    sp.besselk(0, s*k),
    k,
    1,
    sp.oo,
    parameter=s,
    point=0,
    terms=3,
    method="mellin",
)
```

uses

\[
\mathcal M[K_0(ks)](z)=2^{z-2}\Gamma(z/2)^2 k^{-z}
\]

and sums the Dirichlet factor to a zeta function. Its pole expansion begins

\[
\frac{\pi}{2s}+\frac12\log s+O(1),
\]

with lower-order coefficients represented symbolically by zeta derivatives
when SymPy does not simplify them further. For the supported Bessel lattice
case, the Mellin fundamental strip, crossed poles, positivity, and Gamma
vertical decay are replayed, so the contour shift is `CERTIFIED`; unsupported
Mellin geometries remain `FORMAL`.

## Scaled Riemann sums

For

```python
result = asymptotic_sum(
    (k/n)**2,
    k,
    1,
    n,
    parameter=n,
    method="riemann",
)
```

the leading scaled integral is `n/3`. Euler--Maclaurin should be preferred when
boundary corrections and a certified remainder are required.

## Discrete saddle

```python
result = asymptotic_sum(
    sp.exp(-n*(k/n)**2/2),
    k,
    -sp.oo,
    sp.oo,
    parameter=n,
    method="saddle",
)
```

returns `sqrt(2*pi*n)` for the lattice-Gaussian leading term. Distribution-
specific PMF normalization can feed a positive Stirling-scaled summand into the
same engine without taking unsafe logarithms of an arbitrary signed expression.

## Creative telescoping

For a supported bivariate hypergeometric summand, `method="zeilberger"` searches
within deterministic order/degree bounds for

\[
\sum_{j=0}^r p_j(n)F(n+j,k)=\Delta_k(R(n,k)F(n,k)).
\]

The polynomial coefficients and rational certificate are solved together. The
identity is replayed symbolically before summation, and the resulting recurrence
is delegated to `asymptotic_rsolve()`. The telescoping certificate proves the
recurrence identity; the final sum status still follows the recurrence solver,
so a formal Birkhoff--Trjitzinsky truncation is not upgraded merely because the
telescoper is exact. Search failure means only that no relation was found within
the configured bounded search, not that no telescoper exists.

## Poisson and oscillatory sums

For `x > 0`,

```python
x = sp.symbols("x", positive=True)
result = asymptotic_sum(
    sp.exp(-x*k**2), k, -sp.oo, sp.oo,
    parameter=x, point=0, method="poisson",
)
```

returns `sqrt(pi/x)` with a certified exponentially small dual-lattice
remainder. A linear phase is accepted only when it is proved to lie in the
principal Fourier cell used by the retained leading dual image.

`method="oscillatory"` reduces a finite affine-phase exponential sum exactly to
a geometric expression. The returned expression itself is exact; any finite
transseries view is derived from that exact object rather than being mislabeled
as an exact truncated Taylor polynomial.

## Multidimensional and uniform termwise sums

Tuple-valued variables and bounds support products that separate by summation
variable and coupled fixed finite boxes. Coupled infinite lattices remain
`UNKNOWN`. Termwise interchange is `CERTIFIED` only for replayable uniformity
classes: fixed parameter-independent finite integer lattices, and the supported
infinite geometric/binomial class with a proved convergent p-series majorant for
the actual Taylor remainder.
