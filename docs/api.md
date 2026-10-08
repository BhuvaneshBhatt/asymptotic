# Primary API guide

Use the root `asymptotic` namespace for the **66 primary entry points** that cover ordinary expansion, comparison, summation, implicit problems, perturbation methods, probability asymptotics, and common result types.

## Why use the primary API?

The primary API is the shortest path from a mathematical problem to a mathematical result. It keeps common workflows discoverable without exposing theorem-building machinery, internal symbolic policy, or specialized certificate constructors in the root namespace.

Typical starting points are:

| Goal | Root entry point |
|---|---|
| Take a full-domain or directed limit | `limit`, `one_sided_limit`, `complex_ray_limit`, `complex_limit` |
| Find an exact finite cluster set | `cluster_set` |
| Use an explicit analytic contract | `analytic_limit` |
| Expand an expression on a discovered scale | `multiseries` |
| Expand near a point | `local_series`, `series` |
| Compare asymptotic growth | `relation` and the relation predicates |
| Analyze a parameter-dependent sum | `sum` |
| Solve an implicit or inverse problem | `implicit`, `inverse`; specialist `asymptotic.reversion.series_reversion` |
| Analyze an integral | `integrate`, `mellin` |
| Work with probability or expectation | `probability`, `expectation` |
| Solve a perturbation problem | `regular_perturbation`, `lindstedt_poincare` |

For exact signatures and public docstrings, use the [generated API reference](api-reference.md). For the complete distinction between primary, expert, and internal names, use [API classification](api-classification.md). For support boundaries and certification levels, use the [capability matrix](capabilities.md).

## Mathematical values and evidence

Root-level calculation functions return mathematical values by default. Use
`return_result=True` to retain the representation, conditions, error information
and certificates. Specialist submodule functions continue to return their native
records for theorem construction and asymptotic algebra.

| Workflow | Default mathematical result |
|---|---|
| `sum`, `product`, `expectation`, `probability`, `rsolve`, `mellin` | Finite expression, with any validity condition retained |
| `series`, `multiseries`, `puiseux_series`, `local_series` | Finite expression |
| `solve` | Tuple of solution dictionaries, with branch conditions retained |
| `dsolve` | Tuple of solution expressions |
| `implicit` | Tuple of branch expressions when no parameter stratification is needed |
| `inverse`, selected `root` | Branch expression or collection of branches |
| `minimize`, `maximize` | Optimum value, retaining its validity condition |
| `regular_perturbation` | Ordered approximation tuples for completed branches |
| `lindstedt_poincare` | Physical-time approximation for a completed hierarchy |
| `hyperasymptotic_series` | Approximation including terminant levels |
| `compose`, `differentiate`, `integrate`, `leading_term` | Mathematical expression |
| `relation`, `big_o`, `little_o`, `equivalent` | Boolean or conditional decision; unresolved decisions remain explicit |

A finite prefix alone does not claim a certified error bound. Request the record
when that distinction matters. Unproved formal calculations, unknown solver results and partially solved
hierarchies keep their records, so a partial answer cannot look complete.
Parameter conditions are mathematical conditions, not metadata to discard.

`as_element` and `discover_scale` construct algebra and scale objects;
`explain` returns an explanation. `stratified_series` retains its distinct scale
regimes and their coverage evidence. `nested_series` is a lazy exact decomposition
whose terminal residual is part of its mathematical structure; it has no
independent finite truncation. These operations retain their structured outputs.
A parameter-stratified `implicit` result also retains its separate cells.

```python
import sympy as sp
from asymptotic import sum

n, k = sp.symbols("n k", positive=True, integer=True)
prefix = sum(k**-2, k, n, sp.oo, parameter=n, terms=3, method="euler-maclaurin")
assert prefix == 1 / n + 1 / (2 * n**2) + 1 / (6 * n**3)
record = sum(
    k**-2,
    k,
    n,
    sp.oo,
    parameter=n,
    terms=3,
    method="euler-maclaurin",
    return_result=True,
)
assert record.certificate.replay() is True
assert sp.limit(n**5 * record.remainder.scale, n, sp.oo) == sp.Rational(1, 30)
```

The three-term prefix has an error of order `n**-5`. The retained bound combines
the estimate for the full Euler–Maclaurin approximation with the difference
between that approximation and the returned prefix. Further representation
operations, such as certified truncation, should use the detailed record.

## Certified multivariate expansions

`multivariate_expansion(expr, variables, target, order=..., weights=...)`
constructs a weighted blow-up expansion `x_i-a_i=r**w_i*u_i` and returns a
`MultivariateAsymptoticExpansion`.  `status=CERTIFIED` means the reported
remainder has a uniform compact-angular bound; `FORMAL` is available only when
explicitly requested, and unsupported certification remains `UNKNOWN`.

`UniformRemainderCertificate` records the power, constant, chart radius, and
certification provider.  Semialgebraic optimization is preferred; the optional
certified `symbopt` route is a fallback for general optimization.

### Certified multivariate expansion atlases

`multivariate_atlas(expr, variables, target, order=..., weights=...)`
builds all certified candidate Newton/weighted blow-up families. Each family is split into closed max-coordinate angular sectors. `MultivariateAsymptoticAtlas.coverage_certificate` records whether a complete certified sector family covers a punctured neighborhood. Restricted-domain coverage is not claimed yet.

## Import guide

The root namespace contains **66** primary entry points for common workflows, core representations, and common result types. Specialized theorem builders, certificate records, registries, adapters, and lower-level structural controls live in their defining submodules.

## Common workflows

Use root imports for operations that form the normal user-facing workflow:

```python
from asymptotic import multiseries, sum, implicit
```

## Specialized APIs

Import specialized objects from the module that owns their mathematical role. Representative locations include:

```python
from asymptotic.dominant import DominantBalanceCertificate
from asymptotic.remainder_theorems import GreenOperatorCertificate
from asymptotic.monomial import AsymptoticMonomial
from asymptotic.obligations import AsymptoticKnowledge
```

This boundary keeps ordinary imports compact while making expert ownership explicit. See [API classification](api-classification.md) for the complete split and [generated API reference](api-reference.md) for signatures and docstrings.

## Automatic stratified power expansions

`stratified_series(expr, variables, target=..., order=...)` discovers the
finite power-scale regimes that materially change an elementary finite-target
expansion.  For two competing scales it distinguishes the two strict
small-ratio cells and a comparable-scale cell represented by an exact scaled
coordinate.  Symbolic power denominators are split only on critical exponent
comparisons that change the dominant monomial.  `StratifiedExpansion` carries
an explicit coverage certificate; successful branches never imply complete
coverage when another cell requires a non-power scale.
