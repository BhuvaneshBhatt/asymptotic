# User guide

This guide is organized by the question you are trying to answer. For mathematical background, begin with the [introduction to asymptotics](introduction-to-asymptotics.md). For precise support boundaries, read the [capability matrix](capabilities.md); for signatures, read the [API guide](api.md).

## 1. Choose the representation

`asymptotic` has several related representations because different asymptotic problems require different algebraic structure.

- Use `multiseries()` when you want a lazy expansion over one or more small scale generators.
- Use `nested_form()` / `nested_expansion()` when the main issue is nested powers, exponentials, or logarithms.
- Use `transseries_from_expression()` when you need structural monomials, arithmetic, composition, or rigorous remainder objects.
- Use `algebraic_branches()` / `puiseux_series()` for algebraic equations and fractional powers.
- Use `series_reversion()` / `inverse_asymptotic()` for local or asymptotic inverses.
- Use dominant-balance, implicit, multivariate, or nonlinear-ODE APIs when the unknown function itself must be solved asymptotically.

These layers share `AsymptoticContext` for expensive zero, sign, limit, and growth decisions.

## 2. Lazy multiseries

```python
import sympy as sp
from asymptotic import AsymptoticScale, multiseries

x = sp.symbols("x", positive=True)
scale = AsymptoticScale.from_exprs(x, [1/sp.log(x), 1/x])
ms = multiseries(sp.exp(1/x + 1/sp.log(x)), x, scale=scale, terms=5)

leading = ms.terms(3)
next_level = ms.coefficient_series(0).terms(3)
```

If `scale` is omitted, `discover_scale()` is used. Expansion is demand-driven: asking for more terms refines the same sparse object rather than recomputing from scratch.

Recoverable structural failures are represented as obligations and may extend the scale dynamically. Terminal unsupported nodes remain explicit rather than being treated as proved expansions.

## 3. Structural and nested analysis

```python
from asymptotic import mrv_decomposition
from asymptotic.decomposition import decompose_expression
from asymptotic.nested import nested_form

expr = sp.exp(x**2) * (1 + 1/sp.log(x))
structure = decompose_expression(expr, x)
mrv = mrv_decomposition(expr, x)
form = nested_form(expr, x)
```

`decompose_expression()` separates canonical transcendental structure, maximal unary composition, and rationalized substitutions. `mrv_decomposition()` identifies most-rapidly-varying structure. `nested_form()` then peels finite limits, eventual sign, exponential/logarithmic depth, power, and a lower-level remainder.

## 4. Transseries and arithmetic

```python
from asymptotic import compose_transseries, transseries_from_expression

z = sp.symbols("z")
inner = transseries_from_expression(1/x, x, complete=True)
out = compose_transseries(sp.exp(z), inner, argument=z, terms=5)
```

The structural transseries layer keeps coefficients, monomials, valuation, ramification, and remainder information separate. Exact finite expressions can remain exact; finite truncation produces an explicit next-scale remainder when one is available.

## 5. Algebraic branches and Puiseux series

```python
from asymptotic import puiseux_series
from asymptotic.puiseux import algebraic_branches, newton_polygon_candidates

y = sp.symbols("y")
poly = y**2 - x

candidates = newton_polygon_candidates(poly, y, x)
branches = algebraic_branches(poly, y, x)
series = puiseux_series(poly, y, x, terms=5)
```

Newton-polygon candidates determine possible leading exponents and coefficients. Ramification is represented explicitly, so fractional powers are not flattened into ordinary Taylor series accidentally.

## 6. Series reversion and inverse asymptotics

```python
from asymptotic import inverse_asymptotic, series_reversion

t = sp.symbols("t")
local = series_reversion(x + x**2, x, t, terms=6, branch=0)
print(local.truncate())

at_infinity = inverse_asymptotic(x + 1/x, x, t, point=sp.oo, terms=5)
```

Reversion tracks branch decisions produced by nested principal-branch checks. Multiple algebraic leading inverses are returned as separate branches unless a branch index is selected.

## 7. Dominant balance and implicit equations

```python
from asymptotic import dominant_balance_candidates, implicit_asymptotic

candidates = dominant_balance_candidates(y**2 + x*y + x**3, y, x)
branches = implicit_asymptotic(y**2 + x*y + x**3, y, x, terms=5)
```

Dominant-balance candidates include replayable certificates for their weighted-valuation decisions. Parameter-dependent leading coefficients can trigger automatic strata rather than silently assuming a generic nonzero case. Generated polynomial conditions are canonicalized: repeated factors and nonzero rational scalars are removed, small equality systems use a deterministic Groebner basis, and contradictions/nonzero conditions are reduced modulo that equality ideal.

For generalized log-exp balances, use `transseries_dominant_balance_candidates()`.

For singular centers, inspect the multiplicity/scaling diagnosis directly:

```python
from asymptotic.implicit import implicit_singularity_profile

profile = implicit_singularity_profile(y**2 - x, y, x)
print(profile.multiplicity)       # 2
print(profile.turning_point)      # True
print(profile.scaling_exponents)  # (1/2,)
```

`implicit_asymptotic()` performs this diagnosis automatically. A certified multiple root uses the recursive Newton–Puiseux/scaling path and each returned branch records `method="newton-puiseux-blowup"`. The structural parameter probe is evaluated after translating `dependent_limit`, so multiplicity changes at nonzero centers are stratified correctly.

## 8. Multivariate scaling and implicit asymptotics

```python
from asymptotic import multivariate_implicit_asymptotics
from asymptotic.multivariate import multivariate_scaling_regimes, scaling_path

z = sp.symbols("z", positive=True)
path = scaling_path((x, z), (1, 2))
regimes = multivariate_scaling_regimes(y**2 + x*y + z**3, y, (x, z))
branches = multivariate_implicit_asymptotics(
    (y**2 + x*y + z**3,),
    (y,),
    (x, z),
)
```

Automatic weight-cone discovery partitions positive scaling-weight space into chambers and walls where the active Newton face changes. Parameter strata are applied before geometric discovery when coefficients can vanish.

## 9. Nonlinear differential transseries

```python
from asymptotic.nonlinear_ode import nonlinear_differential_transseries

u = sp.Function("u")
equation = sp.diff(u(x), x) - u(x)**2 + 1/x**2
branches = nonlinear_differential_transseries(equation, u, x, terms=5)
```

The solver combines differential dominant balance, recursive correction lifting, logarithmic/exponential descendants, parameter strata, and Frechet linearization. A formal branch may exist even when a rigorous inverse-operator estimate is unavailable; inspect its certificate/remainder information rather than assuming convergence.

## 10. Function properties and branch safety

```python
from asymptotic.function_properties import (
    analytic_at,
    branch_safe_substitution_decision,
    domain_properties,
    function_properties,
    singularity_properties,
)

props = function_properties(sp.log(x))
domain = domain_properties(sp.log(x))
singularities = singularity_properties(sp.log(x))
safe = branch_safe_substitution_decision(sp.log(x), x, 2)
```

The registry contains reviewed facts rather than ad-hoc assumptions. Queries are tri-state where appropriate. Register application-specific function facts with `FunctionPropertyRegistry` and `register_function_properties()`. See [function properties and branch knowledge](function-properties.md) for the reviewed coverage and branch model.

## 11. Formal vs certified remainders

`AsymptoticRemainder` and `AsymptoticTruncation` distinguish exact tails, `O`, `o`, and unknown remainder information. Operation-specific theorem functions can upgrade a formal result only when their hypotheses are proved.

```python
from asymptotic.remainder_theorems import certify_differentiation_remainder

cert = certify_differentiation_remainder(...)
if cert.certified:
    assert cert.replay()
```

The same pattern is used for finite sums/products, reciprocal and quotient propagation, algebraic substitution, general unary composition, inversion, nonlinear lifting, Frechet inverses, and Green operators. Polynomial/rational substitution uses exact finite perturbation identities; general composition searches for the first nonzero Taylor derivative, so stationary points can still yield certified higher-order scales. Higher-order Green certification supports both exact constant coefficients and a conservative asymptotically constant case `L=L0+E(x)` at `+/-oo`. In the latter case, normalized coefficients must converge to a finite hyperbolic constant operator, the limiting Green particular is substituted back into the full operator, the resulting defect must be `o(R)`, and a strict exponential-rate gap must control the homogeneous modes. `GreenOperatorCertificate.replay_asymptotic()` rechecks those asymptotic conditions; `exact_right_inverse` remains false unless the full variable-coefficient identity is exact.

Certification code uses a shared bounded symbolic policy. Polynomial/linear/rational exact methods are tried before general SymPy algorithms, and proof-critical failures remain inconclusive rather than launching open-ended symbolic searches.

## 12. Asymptotic differential fields, shadows, and ghosts

```python
from asymptotic.asymptotic_field import asymptotic_differential_field

field = asymptotic_differential_field(x, (1/x, sp.exp(-x)))
```

The field layer models moderate growth, infinitesimal ideals, shadow projection, ghosts, integral-shadow extensions, and integration-constant placement. It implements a useful subset of Shackell-style shadow machinery, not a complete asymptotic differential-field decision procedure.

## 13. ODE interchange

`from_formal_ode_data()` consumes the structural schema exported by optional `odeanalysis` without creating an import cycle. `certify_green_operator_data()` validates and replays a constant-coefficient operator descriptor before using it for a Green/dichotomy theorem.

## 14. Understanding UNKNOWN

An inconclusive result is usually one of four things:

1. a zero/equality claim could not be proved;
2. a branch/domain condition could not be established;
3. a growth/sign/limit comparison remained undecidable;
4. a theorem-specific regularity or nondegeneracy hypothesis was not certified.

You can often resolve it by strengthening symbol assumptions, supplying a parameter stratum, selecting a branch, registering reviewed function properties, or using a simpler equivalent expression. Do not reinterpret UNKNOWN as false.

## 15. Working uniformly across representations

Use `asymptotic_element()` when an algorithm should accept more than one asymptotic representation:

```python
from asymptotic import asymptotic_element, multiseries, nested_expansion

m = asymptotic_element(multiseries(sp.exp(1/x), x, terms=5))
n = asymptotic_element(nested_expansion(sp.log(x) + 1/x, x))

m.truncation(3)
m.differentiate()
m.integrate(terms=4)
m.reciprocal(terms=4)
m.compose(sp.exp(z), argument=z, terms=4)
m.compare(n)
```

The wrapper retains the original object as `.native`. Operations use its specialized implementation when available; otherwise they convert a finite prefix to the generalized transseries algebra. This means a multiseries derivative remains a multiseries inside the wrapper, while cross-representation multiplication becomes a transseries result with the existing product-remainder theorem attached.

### Truncation versus representation exactness

`element.remainder` and `element.truncation(n).remainder` answer different questions. A lazy `Multiseries` or `NestedExpansion` retains its exact source expression, so the representation itself has no approximation error. A finite prefix can have a tail. For multiseries the first omitted active-scale term certifies that tail as big-O; for nested forms, structural refinement depth has no universal additive-term interpretation, so additive truncation is performed through the transseries view.

### Scale and shadow-field elements

A discovered scale can expose a representative directly:

```python
scale = discover_scale(1/x + sp.exp(-x), x)
slow = scale.element(0)
fast = scale.element(1)
slow.compare(fast)
```

Likewise, expressions attached to a shadow or differential field can join the same algebra without discarding the field object:

```python
field = asymptotic_differential_field(x, (1/x,))
u = field.element(1 + 1/x)
u.reciprocal(terms=4)
```

Shadow projection itself remains a `ShadowField` operation; the common element protocol is for algebra on the resulting/attached expressions, not a replacement for the shadow-field theorem machinery.


## Common algebra object

For workflows mixing representations repeatedly, construct one coordinate-aware algebra instead of adapting each pair manually:

```python
from asymptotic import AsymptoticAlgebra

A = AsymptoticAlgebra(x, sp.oo, terms=6)
u = A.element(multiseries(sp.exp(1/x), x, terms=8))
v = A.element(nested_expansion(1 + 1/x, x, depth=1))
w = A.multiply(u, v)
q = A.reciprocal(w, terms=5)
relation = A.compare(q, 1)
```

The algebra rejects mismatched endpoints/variables rather than silently converting them. Native unary algorithms are preferred; heterogeneous arithmetic uses the finite certified transseries normal form. Incomplete implicit branches carry an `UNKNOWN` solution remainder through this conversion instead of being treated as exact finite series.

## Finding the right entry point

If you know the mathematical problem but not the representation, use the [problem-oriented API decision guide](workflows.md). It separates expansion, inversion, implicit, multivariate, differential, and certification workflows without requiring knowledge of source modules.

## Diagnosing inconclusive results

An `UNKNOWN` remainder or property decision is intentionally auditable. See [Understanding `UNKNOWN`](unknown-results.md) for the common hypotheses behind reciprocal, composition, inverse, implicit, and Green/Frechet inconclusive results.

## Executable examples

The repository's `examples/` directory contains complete scripts for ordinary multiseries truncation, common-algebra interoperability, singular implicit Newton--Puiseux switching, and asymptotically constant Green certification. The test suite imports and executes these scripts, so the examples are kept synchronized with the public API.


For dispatch behavior, see [Algorithm selection](algorithm-selection.md). For unresolved results, see [Understanding `UNKNOWN`](unknown-results.md).
