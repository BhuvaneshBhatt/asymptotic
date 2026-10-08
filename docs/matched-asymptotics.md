# Matched asymptotic expansions

`matched_asymptotic_expansion` constructs uniformly valid endpoint-layer approximations for singularly perturbed scalar boundary-value problems. The automatic solver recognizes linear second-order equations whose reduced problem loses one differential order, decides which endpoint condition must be restored by an inner layer, discovers the stretched-coordinate exponent from a local Newton-style balance, and matches the resulting inner and outer perturbation hierarchies.

For

```python
import sympy as sp
from asymptotic.matched import matched_asymptotic_expansion

x = sp.Symbol("x", positive=True)
eps = sp.Symbol("eps", positive=True)
y = sp.Function("y")

result = matched_asymptotic_expansion(
    eps * sp.diff(y(x), x, 2) + sp.diff(y(x), x),
    y(x),
    eps,
    conditions=(sp.Eq(y(0), 0), sp.Eq(y(1), 1)),
)
branch = result.branches[0]
```

the reduced equation is first order. The right boundary value determines the outer solution `1`; the left condition is restored on the inferred scale `x/eps`. The inner solution is `1 - exp(-X)` and the composite approximation is

$$
1-e^{-x/\epsilon}.
$$

Scale discovery is not restricted to integer powers. For `eps*y'' + x*y' = 0`, the local balance gives a layer width $\sqrt{\epsilon}$. When higher matching orders are requested, the inner hierarchy automatically uses the corresponding ramified gauge lattice, such as `1, sqrt(eps), eps`.

The automatic solver requires a scalar linear second-order problem with first-order reduced equation and two endpoint Dirichlet conditions. Nonlinear problems, higher-order reductions, derivative boundary data, and multiple/interior layers remain explicit unsupported structures instead of being assigned heuristic solutions. The lower-level hierarchy and dominant-balance APIs can still be used to analyze such problems manually.
