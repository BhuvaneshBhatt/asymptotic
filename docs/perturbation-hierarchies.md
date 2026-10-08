# Perturbation hierarchies

`asymptotic` represents perturbation problems independently of the method that
will eventually solve them.  The common hierarchy layer expands equations,
systems, differential equations, and their conditions in one ordered family of
small gauges.

```python
import sympy as sp
from asymptotic.perturbation_hierarchy import perturbation_hierarchy

eps = sp.Symbol("eps", positive=True)
u = sp.Symbol("u")

hierarchy = perturbation_hierarchy(u**2 - 1 - eps, u, eps, order=2)
for level in hierarchy.orders:
    print(level.gauge, level.equations)
```

The default gauges are `1, eps, ..., eps**order`.  An explicit gauge sequence
may instead use fractional powers or other ordered asymptotic scales:

```python
gauges = (1, sp.sqrt(eps), eps)
hierarchy = perturbation_hierarchy(u**2 - eps, u, eps, gauges=gauges)
```

General gauges are extracted recursively in asymptotic order.  The first gauge
must be one, and each later gauge must be asymptotically smaller than the one
before it whenever that comparison can be decided.

## Differential equations and conditions

Applied undefined functions are expanded as function families, not merely at
one syntactic argument.  Consequently an expansion of `y(x)` also propagates
to `y(0)`, `y(a)`, derivatives of `y`, and the same function in boundary or
normalization conditions.

```python
x = sp.Symbol("x", positive=True)
y = sp.Function("y")

hierarchy = perturbation_hierarchy(
    sp.diff(y(x), x) + y(x) + eps * y(x) ** 2,
    y(x),
    eps,
    order=1,
    conditions=sp.Eq(y(0), 1 + eps),
)
```

The hierarchy also retains an `assumptions` expression for method-specific
solvers and later solvability checks.

Each `PerturbationOrder` stores its coefficient equations, coefficient
conditions, unknown coefficient functions, optional supplied solution, and a
separate tuple of solvability conditions.  The latter is part of
the common representation even though construction of resonance conditions is
handled by higher perturbation methods.

## Manual order solutions

Hierarchy objects are immutable.  Supplying an order solution returns a new
hierarchy:

```python
u0, u1, u2 = hierarchy.coefficient_unknowns[0]
solved = hierarchy.with_solution(0, 1)
solved = solved.with_solution(1, {u1: sp.Rational(1, 2)})
solved = solved.with_solution(2, -sp.Rational(1, 8))
```

`approximation()` reconstructs the truncated dependent variables, while
`residual()` and `condition_residual()` substitute that approximation into the
original problem.  Unsolved coefficient functions remain symbolic, so a user
may inspect, partially solve, or externally solve any order without losing the
rest of the hierarchy.

The hierarchy layer does not choose a perturbation method or call
a general equation solver.  Regular perturbation, strained-time methods,
multiple scales, and matched expansions can therefore share the same order
representation, condition propagation, reconstruction, and residual logic.

## Regular perturbation solving

`regular_perturbation` applies a solving policy to the same hierarchy objects.
It recursively solves each coefficient problem, preserving algebraic branches
and retaining a partial hierarchy when a later order is unresolved.

```python
import sympy as sp
from asymptotic import regular_perturbation

eps = sp.Symbol("eps", positive=True)
u = sp.Symbol("u")
result = regular_perturbation(u**2 - 1 - eps, u, eps, order=2, return_result=True)
result.approximations
# ((-1 - eps/2 + eps**2/8,), (1 + eps/2 - eps**2/8,))
```

The same entry point handles ordinary differential equations. Conditions are
expanded order by order and are used to determine integration constants.

```python
x = sp.Symbol("x")
y = sp.Function("y")
result = regular_perturbation(
    sp.diff(y(x), x) + y(x) + eps * y(x) ** 2,
    y(x),
    eps,
    order=1,
    conditions=sp.Eq(y(0), 1 + eps),
    return_result=True,
)
result.approximations
# ((exp(-x) + eps*exp(-2*x),),)
```

Each `RegularPerturbationBranch` exposes the solved hierarchy, reconstructed
approximation, original-equation residual, condition residual, and an exact
`verified` check for all solved coefficient equations. A branch that cannot be
continued records `unresolved_order` and keeps its earlier solved orders.

## Periodic solvability and Lindstedt–Poincaré expansions

Periodic perturbation orders use Fredholm orthogonality instead of accepting
secular terms.  `asymptotic.resonance.periodic_solvability_conditions`
projects an order forcing against supplied adjoint null modes over one period
and returns explicit `SolvabilityCondition` objects.  The projection layer is
method-independent and is also the solvability foundation for multiple scales.

`lindstedt_poincare` applies that machinery to an autonomous, undamped scalar
oscillator by introducing a fast variable `tau = Omega(eps)*t`.  The base
frequency is inferred from the unperturbed constant-coefficient oscillator when
possible, and later frequency corrections are determined by periodic
orthogonality before each profile correction is solved.

```python
import sympy as sp
from asymptotic import lindstedt_poincare

eps = sp.Symbol("eps", positive=True)
t = sp.Symbol("t", real=True)
y = sp.Function("y")

result = lindstedt_poincare(
    sp.diff(y(t), t, 2) + y(t) + eps * y(t) ** 3,
    y(t),
    eps,
    order=2,
    conditions=(
        sp.Eq(y(0), 1),
        sp.Eq(sp.diff(y(t), t).subs(t, 0), 0),
    ),
    return_result=True,
)
result.frequency
# 1 + 3*eps/8 - 21*eps**2/256
```

The solved hierarchy retains every Fredholm condition at the order that
created it.  If a frequency correction is not uniquely determined, or a
profile equation cannot be solved within the bounded symbolic policy, the
result records an unresolved order instead of admitting secular growth.
