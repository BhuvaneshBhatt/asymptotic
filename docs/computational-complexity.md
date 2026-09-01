# Computational complexity

Symbolic asymptotic algorithms rarely have one useful global Big-O bound: runtime can depend on polynomial degree, support size, nesting depth, branch count, parameter strata, and on the cost of backend algebraic operations. The benchmark suite therefore distinguishes **structural operation counts** from empirical wall-clock scaling.

## Exact example: finite transseries multiplication

Consider two finite transseries with `n` and `m` retained nonconstant terms. The multiplication kernel forms each pair of retained terms before canonical collection. The candidate-generation stage therefore performs exactly

\[
nm
\]

term products. Its structural complexity is

\[
\Theta(nm),
\]

and for equal-size inputs it is \(\Theta(n^2)\).

The executable example uses package instrumentation rather than timing to verify this count:

```python
from examples.computational_complexity import multiplication_pair_count

for n in (2, 4, 8):
    retained, products = multiplication_pair_count(n)
    assert retained == n
    assert products == n**2
```

This isolates the multiplication kernel's candidate count. It is **not** a claim that total symbolic runtime is always exactly quadratic: canonicalization, coefficient simplification, remainder certification, and comparison of generated monomials have their own costs and can depend on the expressions being multiplied.

## Empirical scaling

`benchmarks/benchmark_scaling.py` complements structural counts with timed scaling cases for multiseries expansion, reversion, implicit asymptotics, and finite transseries multiplication. These benchmarks vary mathematical complexity such as requested term count rather than padding expressions with irrelevant syntax.

For performance regression testing, structural counters are preferable whenever an exact count is available. Wall-clock measurements remain useful for operations whose dominant cost is delegated to symbolic algebra and therefore cannot be represented by one package-level operation count.

## What to measure for other algorithms

Useful independent variables include polynomial degree, Newton support size, number of parameters, number of generated strata, nesting depth, branch multiplicity, requested correction depth, and retained transseries terms. New benchmarks should record both elapsed time and an algorithmic counter when the implementation has a meaningful discrete work unit.
