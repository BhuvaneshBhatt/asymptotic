# Limit certificate dispatch

The scalar public solver normalizes the variables, target, known function
definitions and parameter assumptions, then tries certified routes on the original
normalized expression. Only an unresolved expression receives the bounded
recurrence pass. A rewrite retries the same certificate dispatch once, preserving
original-expression and identity evidence. General special-function analysis,
uniform regimes and projective/local geometry follow that retry. A successful
Stirling certificate never calls the recurrence simplifier.

For finite simultaneous limits, existing branch and complete-cluster certificates
likewise precede recurrence simplification and its one retry. This preserves the
atanh endpoint and vanishing-perturbation cluster routes.

The recurrence API remains independently callable inside
`asymptotic.recurrence_simplification`; the optional standalone simplifier does
not invoke the limit engine.

Reusable scalar certificates cover:

* finite Dirichlet projections with six exact terms and multiplied absolute tail
  bounds, preserving cancellations rather than replacing each zeta separately;
* finite-point Laurent pole subtraction for fixed Re(a)>0, declining amplified
  analytic remainders that do not provably vanish on every required approach side;
* harmonic tails of fixed real order below one, fixed-shift gamma ratios on a
  positive real growth chart, near-one powers with a small quadratic log error,
  and uniform bounded perturbations of positive quadratic radicals;
* signed upper incomplete-gamma cut germs for rational orders between zero and
  one, using exact monodromy and positivity of the jump integral;
* attained monomial-phase sequences and tangent pole-avoidance sequences, each
  recording the actual coordinate substitution and limit; commensurate periodic
  frequencies use a rational common base and retain both witnesses.

No interval bound proves nonexistence. Every new oscillatory DNE result carries
two distinct attained subsequence values. Unsupported approach domains, missing
function definitions, unresolved parameter strata and remaining nested-pole
problems stay explicit gaps. The certificates have fixed operation/power/depth
caps and no unbounded recurrence search; process-isolated audit deadlines remain
necessary because individual symbolic operations are not preemptively timed here.

Attainment checks decline undefined phase values, nested pole functions, moving
complex tails without a signed branch theorem, and variable powers whose bases
are not proved positive. Monotone arctangent and positive-base exponential bounds
allow outward squeeze certificates to finish before subsequence search.

## Additional recurrence families

The fallback collects same-argument, bounded integer shifts for:

* Hermite (both normalizations), Chebyshev T/U, Legendre and associated Legendre,
  Laguerre and associated Laguerre, Gegenbauer and Jacobi degrees;
* lower incomplete gamma, harmonic numbers, Lerch argument shifts, factorials,
  rising/falling factorial orders, subfactorials, and fixed-degree Bernoulli/Euler
  polynomial argument shifts;
* Struve H/L (including their inhomogeneous principal-power terms), parabolic
  cylinder D, Kummer U in either parameter, and Whittaker M/W in kappa;
* 0F1 denominator shifts, 1F1 in either parameter, and 2F1 in either upper or the
  lower parameter, including corresponding regularized functions;
* coupled Kelvin ber/bei and ker/kei order shifts for real order and positive real
  argument, normalized spherical-harmonic degree shifts, and Ferrers Legendre Q
  degree shifts on the proved real interval -1<x<1.

Polynomial degrees and spherical-harmonic indices retain their integer domain
contracts. Jacobi and contiguous-relation denominators must be proved nonzero.
Ordinary hypergeometric parameter poles remain guarded in the limit context;
regularized functions retain their entire-parameter definitions at those poles.
Lerch limit rewrites additionally require a proved |z|<1 germ and positive a.
Kelvin complex continuations and unspecified Legendre type conventions decline.

These are finite, same-argument identities, not an unrestricted port of the
attached catalogue's multi-parameter recurrence search. General mixed-axis
contiguity, arbitrary polynomial parameter shifts, half-integer Whittaker shifts
and Zernike recurrences remain explicit extensions. No identity changes the
source expression's branch or supplies a value at a removed pole.

Definitions and identities are checked against [DLMF 18.9](https://dlmf.nist.gov/18.9),
[11.4](https://dlmf.nist.gov/11.4), [8.8](https://dlmf.nist.gov/8.8),
[13.3](https://dlmf.nist.gov/13.3), [13.15](https://dlmf.nist.gov/13.15),
[12.8](https://dlmf.nist.gov/12.8), [15.5](https://dlmf.nist.gov/15.5),
[10.61](https://dlmf.nist.gov/10.61), [14.10](https://dlmf.nist.gov/14.10)
and [14.30](https://dlmf.nist.gov/14.30). Harmonic and Lerch argument shifts
follow directly by separating the first term of their defining series.

The complete first audit exposed unsafe generic results for cases 0206 (fractional
Legendre endpoint), 0247 (large-degree Laguerre), and 0309 (Struve cancellation).
After exact recurrence reduction, a dedicated germ prerequisite declines
these unsupported analyses before generic composition/native fallback. This also
protects registered special-function heads without a germ provider. The
original failed audit remains recorded; no reference or executable flag changes.

### Compact timeout perturbations

Three bounded scalar certificates avoid general analysis of large compositions.
The real logarithm/exponential route cancels only `log(exp(h))` with proved real
`h` on the positive punctured germ, then checks a degree-at-most-four polynomial
in `log(x)`. The gamma-root route retains the first fixed-shift correction and
the `O(1/z)` error after the positive fractional power; it accepts only an exact
additive cancellation of the growing polynomial `z`, never an amplified error.
The Lambert route uses the principal branch on both real infinite tails,
including its complex negative-real boundary value. An exact square-root scale
removes the large coordinate before a holomorphic removable-germ check. It
declines nonprincipal branches and nonanalytic outer germs. These routes remain
inside the original certified pass, before bounded recurrence fallback.

Sources: [DLMF 5.11.13](https://dlmf.nist.gov/5.11.E13) and
[DLMF 4.13.10](https://dlmf.nist.gov/4.13.E10). The outer isolated-process
watchdog remains authoritative for all symbolic operations.

### Local cancellation and fixed-argument tails

The bounded local-tail providers retain the original denominator obligations.
The Ci route checks a nonzero analytic denominator coefficient before exact
double-angle cancellation, then tracks each principal logarithmic-germ remainder
separately. The secant route accepts polynomial phases at an odd-pi pole,
including SymPy's exact pi-shifted presentation. Nonexistence records attained
`x=1/n` and `x=-1/n` subsequences; a nonzero local phase, its small magnitude,
and the rational prefactor's leading term prove eventual pole avoidance.

Fixed-argument upper gamma retains exact integer denominator shifts and uses
the superfactorial lower-gamma integral bound. Two shifted reciprocal gamma
factors are compared with a fixed-base affine exponential by Stirling's
logarithmic bound. Fixed finite parameters, all rational coefficients, leading
denominators, and a possible zero base must satisfy the relevant guards.
No finite leading ratio may erase an undefined lower-order coefficient.

The radical/trigonometric ratio uses a proved positive reciprocal-tail chart,
fourth-order phase remainders and nonzero cosine leading terms of order at most
two. Its prefactor must be regular, so discarded errors cannot be amplified.
Moving incomplete-gamma arguments, unproved square-root charts and unresolved
parameter strata are declined. Sources: [DLMF 6.6.6](https://dlmf.nist.gov/6.6.E6),
[8.11](https://dlmf.nist.gov/8.11) and [5.11](https://dlmf.nist.gov/5.11).

## Positive quotient bounds and trigonometric poles

Finite real simultaneous limits first try uniform monomial estimates for positive
sum and product denominators. A weighted AM–GM bound can combine two denominator
terms when neither one alone dominates a numerator term. Radical norms use power
inequalities, continuous positive units receive local lower bounds, and a proved
higher-order signed perturbation can be absorbed into the positive denominator.
Local exponential and logarithmic quotients retain their first-order bound without
requesting a general series. Logarithmic cancellation can also retain a quadratic
prefix with an explicit cubic error bound. Positive first-order elementary
denominator germs reuse the same local estimates. Every accepted vanishing quotient has a uniform
bound and an attained sequence in its original denominator domain. Original
pole bases are checked even when normalization removes a factor: a nonzero
center, a nonzero polynomial restriction or an explicit signed monomial bound
must prove eventual avoidance. Unsupported restrictions remain unresolved.

Real polynomial phases also support the quadratic germs of `1-cos(h)` and
`1-sqrt(cos(h))`. Their isolated phase zeros give pole avoidance on an attained
polynomial ray. A sine quotient with an infinite limit requires a uniform phase
sign. A changing sign remains unresolved by this certificate; a nonexistence
result needs independently attained conflicting limits.

These rules have fixed expression, degree, depth and monomial-count limits.
Exact finite numeric real centers reuse these estimates after a translation.
Approximate targets are declined by this certificate, so rounded subtraction
cannot erase a rational displacement and manufacture a pole.
Polynomial absolute values with nonzero centers have a fixed local sign. Sign
factors with real polynomial arguments have modulus at most one; their zeros
remain in the original domain. A certified inner zero can then be lifted through
sine, cosine, exponential or arctangent at zero.

An exponential multiplying a logarithmic cancellation uses an explicit unit
error: exp(g)=1+O(g) for a real polynomial g tending to zero. The logarithmic
quadratic prefix and cubic remainder remain separate. Positive reciprocal
exponentials over vanishing positive polynomial denominators have lower bound
1/denominator; both denominators are avoided by an attained positive diagonal.

Restricted domains, unresolved parameters and unsupported branch geometry are
passed to other certificates.

Finite real centers also admit attained reciprocal-phase certificates with continuous nonzero amplitudes. For polynomial phases over rational linear pole planes, bounded degree and grid searches construct a regular ray and a transverse polynomial curve with different limits. Analytic first-order sine, tangent, hyperbolic sine and arctangent germs, and real exponential endpoint limits, then certify nonexistence. Every original pole is checked on the attained sequences; unsuccessful searches remain unresolved.
