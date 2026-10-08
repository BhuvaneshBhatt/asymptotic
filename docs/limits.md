# Limits, domains and branches

Use a limit when you need the value approached by an expression throughout a specified local domain. A simultaneous limit is stronger than a collection of iterated limits. For example, `x*y/(x**2+y**2)` has axis limits zero but the diagonal approaches one half.

## Mathematical values and evidence

`limit(expr, variables, target)` requests a mathematical result. With `return_result=True`, the result records `status`, `value`, `variables`, `target`, `domain` and `evidence`. `PROVED` supplies a value, `DOES_NOT_EXIST` supplies a nonexistence argument, and `UNKNOWN` records an unresolved question. Proof-provider names are diagnostics; public correctness depends on the status, value and hypotheses.

## Real approaches

`one_sided_limit(expr, x, point, direction="+")` approaches from values greater than the target. `direction="-"` approaches from below. Symbol assumptions and an explicit `domain` must admit that approach. An incompatible typed-variable ray is rejected; an approach excluded by a relational assumption remains unresolved.

Parameters need separate conditions when their signs, poles or zero coefficients change the answer. Preserve parameter-dependent results or supply proved assumptions. Never substitute a convenient numerical parameter and treat that answer as unconditional.

## Complex rays and infinities

`complex_ray_limit(expr, z, point, ray=d)` uses the chart `z=point+d*t`, with real positive `t` tending to zero. The ray must be fixed, finite, nonzero and admissible for the variable's domain. This is a ray limit; it does not establish an unrestricted complex-plane limit.

The package supports elementary signed cuts, bounded rational poles and sign germs, selected error/integral functions, and fixed-order Bessel and elliptic cut germs. It declines unproved denominator zeros, unsupported sectors and parameter regimes. Rational poles can return `DirectionalInfinity(d)`, where the finite nonzero direction is normalized. Real positive and negative directions reduce to signed real infinity. Spherical infinity `zoo` has a different comparison contract.

## Explicit analytic contracts

`analytic_limit(f(x), x, p, analytic_functions=["f"], assumptions=Q.finite(p))` lets a caller declare an undefined function analytic at the relevant finite image point. The declaration is scoped to that call and does not imply continuity for any other function. The supported inner maps are the identity and a real cube root under real-point assumptions. The declaration cannot make an incompatible variable domain admissible.

## Discontinuities and oscillations

`SquareWave` is an odd period-one wave: one on `(0, 1/2)`, minus one on `(1/2, 1)`, and zero at integer and half-integer jumps. Periodicity extends this definition. `TriangleWave` and `NearestInteger` are specialist mathematical functions used by corresponding bounded certificates.

An interval enclosure of an oscillatory expression does not establish nonexistence. A certificate must attain different limiting values on admissible subsequences. Logarithmic phases use inverse exponential sequences; tangent phases use inverse arctangent branches. Every sequence must avoid the original denominator and its accumulating poles.

## Bounded tails and integral definitions

Affine `SquareWave` phases and `(-1)**floor(a*x+b)` admit interior attained sequences at real infinity. A vanishing rational or reciprocal-logarithmic multiplier proves convergence by the unit bound. A nonzero finite multiplier proves nonexistence by two distinct attained limits. Integer-only approach domains and nonlinear phases require separate certificates.

At a real punctured origin, `x**n*exp(-c/x**m)` tends to zero for fixed finite complex `n`, positive real `c`, and positive even integer `m` up to sixteen. The principal-power phase is bounded on each side; exponential decay dominates the algebraic modulus. An exponent that changes with `x`, a complex approach, or an unproved parameter denominator is outside this rule.

The cancellation

```python
from sympy import Symbol, erf, hyper, log, sqrt, pi, oo, Rational
from asymptotic import limit

y = Symbol("y", positive=True)
h = hyper((Rational(1, 2),) * 2, (Rational(3, 2),) * 2, -(y**2))
limit(erf(y) * log(y) - 2 * y * h / sqrt(pi), y, oo)
# -log(2) - EulerGamma/2
```

is evaluated as an absolutely convergent Gaussian logarithmic moment. Positive real rational substitutions and finite rational weights preserve this proof. The complex inverse-error-function branch at one needs a separate sector theorem.

`AppellF1`, `FresnelF`, and `FresnelG` are available from `asymptotic.special_functions`. `AppellF1` translates exactly to SymPy's `appellf1`; its real Euler integral proves finite continuity for `c>a>0` when both argument targets are strictly below one. The Fresnel auxiliary functions have exact rewrites in terms of `fresnelc` and `fresnels`, using [DLMF 7.5](https://dlmf.nist.gov/7.5). Their positive real rational argument tails converge to zero, with bounded rational multipliers. Complex argument shifts and growing multipliers need further asymptotic terms.

## Dispatch and responsiveness

1. Normalize variables, assumptions and exact function definitions.
2. Try bounded certified routes on the expression, including Stirling and uniform-regime recognition.
3. Apply bounded recurrence reduction if unresolved.
4. Retry relevant routes, identify scales and cancellation order, then use selective general analysis.

This ordering preserves inexpensive routes. Cancellation-sensitive expansions retain enough terms and remainder information to justify the final ratio. A certificate that cannot verify its conditions declines without inventing a result.

## Proof boundaries

General complex sectors, unrestricted variable order, and coupled oscillations with accumulating poles remain partial. Algebraic geometry and generic series work can be expensive. The process-isolated reference runner assigns a shared five-second budget to each original row, including its mapped applications. Longer diagnostics distinguish slow supported mathematics from missing capabilities and do not overwrite the primary timeout.

See [the capability matrix](capabilities.md), [certification](certification.md), [complex-branch example](../examples/branch_limits.py), and [testing](testing.md).

## Cancellation before general expansion

The solver combines bounded analytic jets before asking a general series engine to find a scale. Gamma and fixed-order polygamma expansions at argument one retain terms through order eight. A quotient is accepted only when its denominator has a nonzero leading coefficient and the retained remainder survives division. Real `acos` endpoint ratios require positive even-order deficits from one; this keeps the principal square-root branch consistent on both sides.

Incomplete elliptic functions with amplitudes tending to zero can be removed when each multiplier has a separately proved finite limit and the parameters stay fixed and finite. A diverging multiplier requires additional remainder terms and is declined by this rule.

Complex phase information can control a real limit even when it is bounded. The real part of a reciprocal logarithmic pole retains that phase before taking its limit. Conversely, a projective infinity proof may use a modulus lower bound without expanding an oscillatory phase. Argument steps are resolved from inside outward using nonzero leading real or imaginary components on each admitted side.

Fixed principal complex powers with exponentially decaying modulus are handled before general growth analysis. A logarithm of a power whose exponent tends to zero uses its exact principal-log identity once its imaginary part lies in the principal strip. An exponential reciprocal with an exponentially small denominator correction uses a modulus bound and proves denominator avoidance. Reciprocal trigonometric coefficients are excluded because bounded sine/cosine numerators do not establish pole avoidance.
