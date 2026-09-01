# Algorithm selection

The public functions choose algorithms from the mathematical structure of the
input rather than from expression size alone. This page describes those
choices so that a result's `method`, `status`, and possible `UNKNOWN` state are
predictable.

## General selection principles

A high-level operation follows three rules:

1. Prefer an exact symbolic reduction when it is cheap and structurally
   appropriate.
2. Otherwise choose the narrowest asymptotic method whose hypotheses match the
   problem.
3. Return `UNKNOWN` or raise `NotImplementedError` rather than silently using a
   method whose hypotheses cannot be established.

Certification is separate from construction. A formal expansion can be useful
without a proved remainder; `CERTIFIED` is reserved for results with replayable
proof evidence.

## Probability and expectation

`asymptotic_probability()` and `asymptotic_expectation()` first normalize random
variable bindings and conditioning. In `auto` mode they then try an exact SymPy
statistics reduction. Joint expressions can therefore succeed exactly even
while the structural fallback is one-dimensional.

For a single continuous random variable, the fallback reduces to a density
integral. Depending on the domain and phase, it can use direct integration,
endpoint Laplace analysis, or an interior saddle expansion. For a single
discrete or finite random variable, the fallback reduces to a PMF sum and can
use exact summation, lattice-tail analysis, saddle analysis, or
Euler--Maclaurin where applicable.

Method names such as `density`, `pmf`, `laplace`, `saddle`, and
`euler-maclaurin` constrain this routing. Incompatible requests fail explicitly;
for example, `laplace` is not accepted for a discrete probability space.

## Statistical transforms

Moments and cumulants are assembled from expectation or transform identities.
Variance and covariance use centered observables so leading raw moments do not
have to be subtracted after truncation. Derived results inherit certification
only when their source results and the transformation theorem justify it.

Discrete quantiles use the generalized inverse

\[
Q(p)=\inf\{x:F(x)\ge p\},
\]

rather than solving `F(x) = p`. Continuous exact CDFs may use ordinary inversion
when branch and support information determine the correct root.

## Algebraic and implicit solving

`asymptotic_solve()` first looks for a regular implicit branch. Singular roots
are analyzed by dominant balance, Puiseux scaling, parameter stratification,
and the package's Hardy/MRV machinery as needed. A branch is retained only when
its residual and balance data support the claimed asymptotic order.

Multivariate problems use Newton-polyhedron weight cones to discover candidate
scaling paths. Finite ray checks may disprove a relation but are not treated as
proof of a global multivariate relation.

## Integration and summation

`asymptotic_integrate()` uses exact integration when appropriate and otherwise
works through asymptotic term integration and remainder propagation.
`asymptotic_sum()` can use exact summation, termwise parameter expansion,
summation by parts, Euler--Maclaurin, Mellin-pole expansion, scaled Riemann
sums, discrete saddles, or distribution-specific lattice-tail methods. The
selected method is recorded on the structured result. Mellin, Riemann, and
Abel-transform prefixes remain formal when the required contour, uniformity,
or discarded-tail hypotheses have not been proved.

## Differential equations

`asymptotic_dsolve()` first tries the linear formal-data route for linear ODEs.
This route can expose Frobenius, exponential, ramified, monodromy, and Stokes
structure. Differential-polynomial nonlinear equations use recursive dominant
balance and transseries lifting.

Returned prefixes have a backend-independent residual contract through
`AsymptoticDSolveResult.residuals(equation)`. Formal completeness is not the same
as a certified remainder theorem.

## Recurrences

`asymptotic_rsolve()` first tries exact scalar recurrence solving. If that route
does not resolve a homogeneous linear recurrence with rational or polynomial
coefficients, the native route builds discrete Newton edges. Simple roots use
ordinary Birkhoff--Trjitzinsky lifting; repeated constant-coefficient roots use
exact polynomial Jordan chains; supported repeated variable-coefficient roots
use a secondary Newton polygon for stretched-exponential phases and ramified
correction lattices. Returned branches carry replayable normalized residual
orders, while the result object can substitute its reported expression through
`AsymptoticRSolveResult.residual(recurrence)`. Repeated tertiary roots and unsupported deeper Newton configurations,
further resonances, and connection constants remain conservative rather than
being guessed.

## Optimization

Univariate optimization distinguishes attained extrema from finite infima or
suprema approached at open or infinite boundaries. Candidate points come from
critical points, admissible boundaries, and one-sided/infinite boundary limits.
A limiting optimum can therefore be returned with an empty optimizer tuple.

## Relations and assumptions

Asymptotic relation decisions use the supplied endpoint, direction, assumptions,
and growth context. A missing sign or nonvanishing fact can leave a result
`UNKNOWN`; strengthening assumptions may refine the result, but cached decisions
are keyed by the full assumption context and cannot leak between incompatible
queries.

## Reading a result

When a structured result is returned, inspect these fields in order:

- `expression` or `solutions`: the computed finite approximation;
- `method`: the selected route;
- `status`: `EXACT`, `CERTIFIED`, `FORMAL`, or `UNKNOWN`;
- `conditions`: hypotheses still attached to the result;
- `remainder`: known truncation information;
- `certificate`: replayable evidence when certification is claimed.

See [Understanding `UNKNOWN`](unknown-results.md) for failure modes and ways to
supply the missing mathematical information.
