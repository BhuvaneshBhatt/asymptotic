# Migrating public API calls

The root namespace exposes mathematical workflows under short names. Python 3.11
or newer and SymPy 1.14 or newer are required. Existing code using the former
`asymptotic_` prefixes needs updated imports and calls.

| Former root name | Current root name |
| --- | --- |
| `asymptotic_element` | `as_element` |
| `implicit_asymptotic` | `implicit` |
| `nested_expansion` | `nested_series` |
| `asymptotic_solve`, `asymptotic_root` | `solve`, `root` |
| `asymptotic_dsolve`, `asymptotic_rsolve` | `dsolve`, `rsolve` |
| `asymptotic_sum` | `sum` |
| `asymptotic_minimize`, `asymptotic_maximize` | `minimize`, `maximize` |
| `asymptotic_argmin`, `asymptotic_argmax` | `argmin`, `argmax` |
| `asymptotic_expectation`, `asymptotic_probability` | `expectation`, `probability` |
| `asymptotic_big_o`, `asymptotic_little_o`, `asymptotic_equivalent` | `big_o`, `little_o`, `equivalent` |
| `asymptotic_relation` | `relation` |
| `inverse_asymptotic` | `inverse` |
| `AsymptoticScale` | `Scale` |
| `AsymptoticRemainder`, `AsymptoticTruncation` | `Remainder`, `Truncation` |
| `AsymptoticSolveResult`, `AsymptoticDSolveResult`, `AsymptoticRSolveResult` | `SolveResult`, `DSolveResult`, `RSolveResult` |
| `AsymptoticOptimizationResult`, `AsymptoticSumResult`, `StatisticalAsymptoticResult` | `OptimizationResult`, `SumResult`, `StatisticalResult` |

Specialist APIs remain in their defining submodules. Consult the
[API guide](https://github.com/BhuvaneshBhatt/asymptotic/blob/main/docs/api.md)
for signatures and result types; a rename alone does not imply identical
argument or return conventions.

## Limits and evidence

The default `limit` call returns a mathematical value. Request `return_result=True`
when you need the status, assumptions and proof evidence. Nonexistence and
unsupported analysis are distinct outcomes: an unresolved computation is not a
proof of nonexistence. A nonexistence certificate must retain attained paths or
subsequences, together with their domain prerequisites.

```python
import sympy as sp
from asymptotic import limit

x, y = sp.symbols("x y", real=True)
value = limit(x + y, (x, y), (0, 0))  # 0
result = limit(x / y, (x, y), (0, 0), return_result=True)
assert result.status.name == "DOES_NOT_EXIST"
```

A direction at complex infinity carries more information than spherical
infinity. Use `complex_ray_limit` for a specified ray and `complex_limit` for a
whole-plane approach. `DirectionalInfinity` retains the phase; `zoo` records
unbounded modulus without prescribing a phase. Reference comparisons may erase
phase only when the reference explicitly specifies the extended-complex codomain.

Install optional integrations with `asymptotic[flint]`, `asymptotic[ode]` or
`asymptotic[optimization]`. The supported accelerated arithmetic backend requires
`python-flint>=0.9.0`.

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

