# asymptotic

`asymptotic` is a symbolic asymptotics package built on SymPy. It combines lazy multiseries, nested/log-exp expansions, Puiseux and inverse series, dominant balance, multivariate scaling, nonlinear ODE lifting, parameter stratification, and theorem-oriented remainder certificates.

When an operation can be carried out formally but a mathematical hypothesis cannot be proved, the package keeps that uncertainty explicit instead of silently turning it into a certified claim.

## Install

```bash
python -m pip install asymptotic
```

Python 3.10+ is supported. `exprtest` is a required zero-equivalence backend. Optional integrations are available with:

```bash
python -m pip install "asymptotic[flint]"   # python-flint acceleration
python -m pip install "asymptotic[ode]"     # odeanalysis interchange
```

For development:

```bash
python -m pip install -e ".[test,lint,benchmark]"
pytest
pytest tests/test_benchmark_smoke.py
ruff check src tests benchmarks
```

## Start here

For ordinary asymptotic expansion, let the package discover a scale:

```python
import sympy as sp
from asymptotic import multiseries

x = sp.symbols("x", positive=True)
ms = multiseries(sp.exp(1/x + 1/sp.log(x)), x, terms=4)
print(ms.scale.exprs)
print(ms.terms(4))
```

For structural transseries representation:

```python
from asymptotic import transseries_from_expression

ts = transseries_from_expression(
    1/x + sp.exp(-x)/x,
    x,
    point=sp.oo,
    complete=True,
)
print(ts.truncate())
print(ts.remainder)
```

For inverse asymptotics and implicit problems:

```python
from asymptotic import implicit_asymptotic, series_reversion

y = sp.symbols("y")
inv = series_reversion(x + x**2, x, y, terms=5, branch=0)
print(inv.truncate())

branches = implicit_asymptotic(y**2 + x*y + x**3, y, x)
```

For nonlinear ODE lifting:

```python
from asymptotic.nonlinear_ode import nonlinear_differential_transseries

u = sp.Function("u")
branches = nonlinear_differential_transseries(
    sp.diff(u(x), x) - u(x)**2 + 1/x**2,
    u,
    x,
    point=sp.oo,
    terms=4,
)
```

For discrete asymptotics, the summation dispatcher combines exact summation,
creative telescoping, Euler-Maclaurin, certified Mellin shifts, Poisson
summation, oscillatory geometric sums, multidimensional factorization, and
rigorous termwise-uniformity checks:

```python
from asymptotic import asymptotic_sum

n = sp.symbols("n", nonnegative=True, integer=True)
k = sp.symbols("k", nonnegative=True, integer=True)

result = asymptotic_sum(
    sp.binomial(n, k), k, 0, sp.oo,
    parameter=n, method="zeilberger", terms=3,
)
assert sp.simplify(result.expression - 2**n) == 0
assert result.certificate.replay()  # exact telescoping identity
# Overall status follows asymptotic_rsolve(); a recurrence certificate alone
# does not certify the asymptotic solver's truncation.
```

For a small-parameter Gaussian lattice, `method="poisson"` retains an
exponentially small certified dual-lattice remainder. `method="mellin"` can
certify supported contour shifts when the strip, crossed poles, positivity, and vertical Gamma decay are all proved.


### Singular implicit roots and automatic blow-up

`implicit_singularity_profile()` figurew out the local dependent-root multiplicity, Jacobian, turning-point status, polynomial discriminant when cheaply available, and candidate Newton scaling exponents. `implicit_asymptotic()` uses the same diagnosis internally. Certified multiple roots are tagged as `newton-puiseux-blowup` and are lifted by the recursive Newton/Puiseux engine rather than being forced through a simple-root solve. Parameter stratification includes derivative/discriminant data at the translated dependent center, so nonzero-center multiplicity changes are split correctly.

```python
from asymptotic import implicit_asymptotic
from asymptotic.implicit import implicit_singularity_profile

profile = implicit_singularity_profile(y**2 - x, y, x)
assert profile.multiplicity == 2
assert profile.turning_point is True
assert profile.scaling_exponents == (sp.Rational(1, 2),)

branches = implicit_asymptotic(y**2 - x, y, x)
```

## Result semantics

The package has three levels of mathematical knowledge:

| Status | Meaning |
|---|---|
| Formal | An algebraic/asymptotic construction was produced, but no theorem-level remainder claim is implied. |
| Certified | Required hypotheses were proved and a replayable `O`/`o`-style certificate is available. |
| Unknown | The package could not prove a needed hypothesis and therefore refused to strengthen the result. |

`UNKNOWN` is intentional. In particular, branch safety, zero equivalence, growth comparisons, inverse-operator estimates, and parameter strata are not guessed when proof-oriented machinery is inconclusive.

A *replayable certificate* stores the concrete evidence behind a claim. To *replay* it means to verify that stored evidence again without rerunning the search that found it. For example, creative telescoping rechecks its telescoping identity, a remainder certificate rechecks its bound hypotheses, and a Green certificate rechecks the operator defect. Replay returns `True` when the retained evidence verifies, `False` when it fails, and `None` when the retained verifier cannot decide.

## Major capability areas

The package supports:

- lazy multiseries with automatically discovered logarithmic/exponential scales;
- canonical nested expansions and MRV/structural decomposition;
- finite-height log-exp transseries and structural transseries arithmetic;
- asymptotic sums via certified termwise expansion, summation by parts,
  creative telescoping into `asymptotic_rsolve()`, Euler--Maclaurin, certified
  Mellin shifts, Poisson summation, finite oscillatory sums, Riemann scaling,
  multidimensional factorization, and lattice saddles;
- Puiseux/algebraic branches, local series reversion, and inverse asymptotics;
- ordinary and generalized dominant balance;
- automatic multivariate weight-cone discovery and scaling paths;
- multivariate implicit asymptotics and canonical parameter stratification;
- recursive nonlinear differential transseries lifting;
- periodic/oscillatory decomposition;
- asymptotic differentiation and integration;
- asymptotic expectation/probability with exact, density/PMF, moving-domain, degenerate/coalescing saddle, and globally certifiable Laplace routes;
- reviewed function-property, singularity, domain, and branch-safety queries;
- asymptotic differential fields with shadows, ghosts, and integral-shadow extensions;
- replayable remainder certificates for finite arithmetic, reciprocal/quotient, algebraic/general composition, inverse problems, and Green/exponential-dichotomy estimates;
- optional `odeanalysis` interchange for formal ODE data and Green-operator descriptors.

See the [capability and limitations matrix](https://github.com/BhuvaneshBhatt/asymptotic/blob/main/docs/capabilities.md) for the exact boundary of each feature.

Maintainers can use the [PyPI release workflow](https://github.com/BhuvaneshBhatt/asymptotic/blob/main/docs/releasing.md), which validates the tagged source and built wheel before Trusted Publishing.

## One algebraic surface across representations

`AsymptoticAlgebra` provides a coordinate-aware common interface without imposing a single storage format. Existing representations keep their specialized algorithms while sharing coercion, arithmetic, comparison, calculus, composition, inversion, and remainder propagation:

```python
from asymptotic import AsymptoticAlgebra, asymptotic_element, multiseries

m = multiseries(sp.exp(1/x), x, terms=5)
u = asymptotic_element(m)
algebra = AsymptoticAlgebra(x, sp.oo, terms=4)

q = u.truncation(3)        # prefix + explicit remainder semantics
du = u.differentiate()      # keeps the native multiseries algorithm
r = u.reciprocal(terms=4)  # falls back to certified transseries algebra
c = u.compose(sp.exp(z), argument=z, terms=4)
```

The same algebra works for ``TransseriesExpansion``, ``Multiseries``, ``NestedExpansion``, ``PuiseuxSeries``, implicit branches, scale representatives, expressions attached to a shadow/differential field, and nonlinear-ODE transseries branches. Binary arithmetic and comparisons use one coordinate/coercion boundary rather than representation-specific conversion paths. Conversion is *not* mandatory: native operations are preferred, and a finite transseries view is used only where a representation lacks the requested algebraic operation.

The root namespace contains 54 primary entry points. Ordinary users should import these from `asymptotic`; specialized result records and algorithm-building blocks remain in their defining submodules, and cache/debug helpers are internal.

- [Primary API guide](https://github.com/BhuvaneshBhatt/asymptotic/blob/main/docs/api.md)
- [API classification](https://github.com/BhuvaneshBhatt/asymptotic/blob/main/docs/api-classification.md)

## Certified PMF normalization

Positive factorial/Gamma PMFs can be converted to branch-safe exponential scale with certified Stieltjes remainder bounds; see `docs/stirling-pmf.md`.


## Statistical transforms and algebraic systems

Higher-level moments, transforms, CDF/survival/quantiles, rate functions, positive products, and parameter-dependent algebraic systems are documented in the [statistical transforms and algebraic systems guide](https://github.com/BhuvaneshBhatt/asymptotic/blob/main/docs/statistical-transforms.md).

## Documentation

- [Introduction to asymptotics](https://github.com/BhuvaneshBhatt/asymptotic/blob/main/docs/introduction-to-asymptotics.md) — concepts and theory, computational techniques, and applications across mathematics, statistics, algorithms, numerical analysis, special functions, and physics.
- [Choosing an API](https://github.com/BhuvaneshBhatt/asymptotic/blob/main/docs/workflows.md) — problem-oriented decision tree.
- [User guide](https://github.com/BhuvaneshBhatt/asymptotic/blob/main/docs/user-guide.md) — workflows and examples.
- [Capabilities and limitations](https://github.com/BhuvaneshBhatt/asymptotic/blob/main/docs/capabilities.md) — what is formal, certified, partial, or unsupported.
- [Understanding UNKNOWN](https://github.com/BhuvaneshBhatt/asymptotic/blob/main/docs/unknown-results.md) — why certification can remain inconclusive.
- [Worked algorithm traces](https://github.com/BhuvaneshBhatt/asymptotic/blob/main/docs/algorithm-traces.md) — Newton/Puiseux, weight-cone, composition, and Green decisions.
- [Primary API](https://github.com/BhuvaneshBhatt/asymptotic/blob/main/docs/api.md) — curated root-level contracts.
- [Generated API reference](https://github.com/BhuvaneshBhatt/asymptotic/blob/main/docs/api-reference.md) — live signatures and public docstrings.
- [API classification](https://github.com/BhuvaneshBhatt/asymptotic/blob/main/docs/api-classification.md) — primary vs expert vs internal.
- [Import guide](https://github.com/BhuvaneshBhatt/asymptotic/blob/main/docs/import-guide.md) — primary and expert import locations.
- [Architecture](https://github.com/BhuvaneshBhatt/asymptotic/blob/main/docs/architecture.md) — data flow and module responsibilities.
- [Function properties](https://github.com/BhuvaneshBhatt/asymptotic/blob/main/docs/function-properties.md) — reviewed branch-aware registry semantics and coverage.
- [Testing strategy](https://github.com/BhuvaneshBhatt/asymptotic/blob/main/docs/testing.md) — behavioral, metamorphic, negative, and reference-corpus testing.
- [Benchmark suite](https://github.com/BhuvaneshBhatt/asymptotic/blob/main/docs/benchmarks.md) — micro, scaling, stateful, and instrumentation benchmarks.
- [Probability and expectation asymptotics](https://github.com/BhuvaneshBhatt/asymptotic/blob/main/docs/probability-asymptotics.md) — exact reduction, moving domains, degenerate/coalescing saddles, and global Laplace certificates.
- [Asymptotic sums](https://github.com/BhuvaneshBhatt/asymptotic/blob/main/docs/asymptotic-sums.md) — creative telescoping, Euler-Maclaurin, Mellin/Poisson methods, uniform termwise expansion, multidimensional sums, and discrete saddles.
- [Computational complexity](https://github.com/BhuvaneshBhatt/asymptotic/blob/main/docs/computational-complexity.md)
- [Power simplification and branch semantics](https://github.com/BhuvaneshBhatt/asymptotic/blob/main/docs/power-simplification.md)
- [Mathematical scope and non-goals](https://github.com/BhuvaneshBhatt/asymptotic/blob/main/docs/scope.md) — explicit boundaries of the mathematical model.
- [Algorithm selection](https://github.com/BhuvaneshBhatt/asymptotic/blob/main/docs/algorithm-selection.md) — how high-level APIs choose exact, structural, saddle, implicit, ODE, recurrence, and optimization routes.
- [Discrete asymptotics](https://github.com/BhuvaneshBhatt/asymptotic/blob/main/docs/discrete-asymptotics.md) — factorial scales, discrete Newton polygons, and recurrence lifting.
- [Understanding `UNKNOWN`](https://github.com/BhuvaneshBhatt/asymptotic/blob/main/docs/unknown-results.md) — common unresolved hypotheses, unsupported cases, and how to inspect evidence.


## Executable examples

Complete examples are tested in CI:

- [ordinary expansion](https://github.com/BhuvaneshBhatt/asymptotic/blob/main/examples/ordinary_expansion.py)
- [common algebra](https://github.com/BhuvaneshBhatt/asymptotic/blob/main/examples/common_algebra.py)
- [singular implicit asymptotics](https://github.com/BhuvaneshBhatt/asymptotic/blob/main/examples/singular_implicit.py)
- [certified Green/Frechet estimate](https://github.com/BhuvaneshBhatt/asymptotic/blob/main/examples/certified_green.py)
- [probability and expectation asymptotics](https://github.com/BhuvaneshBhatt/asymptotic/blob/main/examples/probability_asymptotics.py)
- [degenerate/coalescing saddles and asymptotic sums](https://github.com/BhuvaneshBhatt/asymptotic/blob/main/examples/advanced_saddles_and_sums.py)

## Significant Limitations

The implementation is principally a **directed real asymptotics** engine, at least for now. It records explicit complex-sector/branch metadata and can preserve formal Stokes ray/sector geometry supplied by optional `odeanalysis`, but it does not provide general sectorial certification, Stokes connection theory/resurgence, or a fully closed transseries differential field. Green/Frechet certification covers first-order, constant-coefficient higher-order, and a conservative asymptotically constant `L=L0+E(x)` higher-order class; general variable coefficients remain outside the certified domain. Function-domain and branch reasoning is registry-driven (registry needs to be improved) rather than a complete special-function theorem prover. Internal symbolic fallbacks are bounded by policy, while explicitly user-requested general antiderivatives may still inherit SymPy cost.

Those limitations are described operation-by-operation in [docs/capabilities.md](https://github.com/BhuvaneshBhatt/asymptotic/blob/main/docs/capabilities.md).

## License

The project is licensed under the GNU General Public License version 3 (GPL-3.0-only). Contributions are welcomed, whether they are bug reports, suggestions, or code implementations.
