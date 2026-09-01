# Probability and expectation asymptotics

`asymptotic_expectation()` and `asymptotic_probability()` use a four-stage
strategy rather than treating probability objects as opaque special functions.


## Expectation expressions and probability spaces

The first argument of `asymptotic_expectation(expr, ...)` is always the
expression being averaged. For example, `asymptotic_expectation(X**2 + 3*X,
parameter=n)` computes the expectation of the entire expression, not merely of
`X`. Exact SymPy expectation is attempted before a random symbol is resolved,
so exact expressions involving several random variables can succeed.

The current structural density/PMF, moving-domain, and saddle fallbacks are
one-dimensional. The optional second argument is therefore a `RandomSymbol`
used to disambiguate that fallback; it is not the expression being averaged.
For explicit symbolic assignments, `bindings={x: X, y: Y}` substitutes
ordinary symbols by SymPy `RandomSymbol` objects before expectation reduction.
The binding value may also be a raw SymPy `Distribution`; the package creates a
single-variable probability space for the bound symbol.
For a SymPy joint random variable or `JointDistribution`, a tuple key such as
`bindings={(x, y): Z}` binds the ordinary symbols to the indexed components
`Z[0]`, `Z[1]`. Exact joint reduction is then attempted; an unresolved joint
problem remains `UNKNOWN` because the custom asymptotic fallback is still
one-dimensional.
`condition=pred` forms a conditional expectation using the same probability
space. Exact multivariate/conditional reductions may therefore succeed even
when the one-dimensional structural fallback is not applicable.

`asymptotic_probability(event, ...)` follows the same convention. The first
argument is always the event predicate. `bindings=` maps ordinary symbols to
SymPy random variables and `condition=` requests conditional probability:

```python
x, y = sp.symbols("x y")
X = Normal("X", 0, 1/sp.sqrt(n))
Y = Normal("Y", 0, 2/sp.sqrt(n))

joint = asymptotic_probability(
    x + y > 0,
    parameter=n,
    bindings={x: X, y: Y},
)

conditional = asymptotic_probability(
    x > 1,
    parameter=n,
    bindings={x: X},
    condition=x > 0,
)
```

Exact joint/conditional probability is attempted **before** resolving a single
random variable. Only the custom density/PMF/Laplace/sum fallback requires a
single distinguished random variable. This keeps multivariate exact cases
available without pretending that the package already has a general
multivariate saddle-probability engine.

## Strategy pipeline

1. **Exact reduction.** Try SymPy's exact expectation/probability machinery.
   If the result can be represented by the package's finite asymptotic algebra,
   retain the exact expression and expose its expansion through `.truncate()`.
   In `auto` mode, an exact special-function result that still hides the
   asymptotic scale falls through to the next stage.
2. **Density or PMF reduction.** Replace a continuous expectation/probability by
   its defining integral, or a discrete problem by its defining sum. Exact
   integral/sum evaluation is accepted when available. An unresolved discrete
   sum is returned with status `UNKNOWN`; the package does not pretend that an
   unevaluated sum is an asymptotic result.
3. **Moving-domain normalization.** For a continuous interval whose endpoint
   grows with the asymptotic parameter, discover an exact leading
   multiplicative scale and substitute `x = scale*y` when this makes all finite
   endpoints parameter-free.
4. **Laplace/saddle expansion.** Recognize `A(x,p) exp(-p phi(x))` at
   `p -> +oo`. The engine handles nondegenerate saddles, even-order degenerate
   saddles, monotone endpoints, stationary degenerate endpoints, multiple
   co-dominant minima, and a quartic coalescing-saddle transition normal form.

For real polynomial phases, `laplace_asymptotic_integral()` can now certify a
narrow global theorem class. The certificate enumerates stationary points,
checks the dominant minima and their local orders, proves polynomially coercive
infinite tails, and attaches an explicit asymptotic remainder. Such results are
`CERTIFIED`. Analytic cases outside that theorem class remain `FORMAL`, and
unsupported reductions remain `UNKNOWN`.

## Moving Gaussian tail

```python
import sympy as sp
from sympy.stats import Normal
from asymptotic import asymptotic_probability

n, a = sp.symbols("n a", positive=True)
X = Normal("X", 0, sp.sqrt(n))

result = asymptotic_probability(X > a*n, X, parameter=n, terms=3)
print(result.method)
print(result.truncate())
```

The moving boundary is normalized by `x = n*y`. The density becomes a
Laplace integral with phase `phi(y)=y**2/2` on `(a, oo)`, producing

```text
exp(-a**2*n/2)/(a*sqrt(2*pi*n))
    * (1 - 1/(a**2*n) + 3/(a**4*n**2))
```

for positive `a`.

## Interior saddle expectation

```python
Y = Normal("Y", 0, 1/sp.sqrt(n))
result = asymptotic_expectation(
    sp.exp(Y), Y, parameter=n, terms=3, method="laplace"
)
print(result.truncate())
```

The density is proportional to `exp(-n*x**2/2)`, with a nondegenerate interior
minimum of the phase at zero. Local Gaussian moment integration gives

```text
1 + 1/(2*n) + 1/(8*n**2)
```

which agrees with the expansion of the exact moment-generating function.

## Generic Laplace integral

The lower-level API can be used independently of probability distributions:

```python
from asymptotic import laplace_asymptotic_integral

x = sp.symbols("x", real=True)
result = laplace_asymptotic_integral(
    sp.exp(-n*x**2/2),
    x,
    (-sp.oo, sp.oo),
    parameter=n,
    terms=3,
)
```

It returns `sqrt(2*pi/n)` through the interior-saddle route.

## Degenerate and coalescing saddles

An even-order strict minimum of order `m` is expanded with the local scale
`p**(-1/m)` and generalized Gamma moments. For example,

```python
result = laplace_asymptotic_integral(
    sp.exp(-n*x**4), x, (-sp.oo, sp.oo), parameter=n
)
```

returns

```text
gamma(1/4)/(2*n**(1/4))
```

through the order-4 degenerate-saddle route.

`coalescing_saddle_asymptotic()` implements the symmetric quartic transition
normal form. For

```text
exp(-n*(x**4/4 + mu*x**2/2))
```

it uses `x=n**(-1/4)u` and the uniform transition parameter
`tau = mu*sqrt(n)`, retaining the canonical quartic profile integral. The
transition result is `FORMAL`; a uniform global remainder theorem for
bounded `tau` is outside the certified theorem.


### Airy uniform saddle workflow

`airy_uniform_saddle_asymptotic()` handles the complementary cubic turning-point
regime in which two simple oscillatory stationary points coalesce.  At a
coalescence `(x0, mu0)` the supported local hypotheses are

\[
\phi_x=\phi_{xx}=0,\qquad \phi_{xxx}\ne0,\qquad \phi_{x\mu}\ne0.
\]

After local cubic normalization, the transition variable is of size
`parameter**(2/3) * (mu-mu0)` and the integral has the characteristic
`parameter**(-1/3)` Airy scale.  The canonical example is exact at leading
order:

```python
import sympy as sp
from asymptotic import airy_uniform_saddle_asymptotic

n = sp.symbols("n", positive=True)
mu = sp.symbols("mu", real=True)
x = sp.symbols("x", real=True)
result = airy_uniform_saddle_asymptotic(
    sp.exp(sp.I*n*(x**3/3 + mu*x)),
    x,
    (-sp.oo, sp.oo),
    parameter=n,
    control_parameter=mu,
)
assert sp.simplify(
    result.expression - 2*sp.pi*n**(-sp.Rational(1, 3))*sp.airyai(mu*n**sp.Rational(2, 3))
) == 0
```

The returned status is **FORMAL**.  The routine proves the supported local cubic
normal form, but it does not yet certify the global contour deformation, absence
of competing remote saddles, or Stokes-sector dominance needed for a global
uniform remainder theorem.  `terms=1` is currently the only supported transport
order; Airy/Airy-prime correction coefficients remain future work.

## Discrete probability

Discrete PMF reductions now delegate unresolved integer sums to
`asymptotic_sum()`. Users can request `method="series"`,
`method="summation-by-parts"`, `method="zeilberger"`, `method="poisson"`, `method="oscillatory"`, `method="euler-maclaurin"`, `method="mellin"`,
`method="riemann"`, or
`method="saddle"` explicitly. Exact finite/point PMF sums are still preferred.
See [Asymptotic sums](asymptotic-sums.md).

## Current scope

The implementation supports one SymPy random variable at a time. Continuous
events are equality or one-sided inequality events that reduce to an interval
or point. The certified global Laplace theorem requires a real
polynomial phase and polynomial/constant local amplitude. Multidimensional
Laplace integrals, general complex steepest descent/Stokes analysis, and a
fully automatic Stirling normalizer for arbitrary PMFs remain outside the
certified core.
