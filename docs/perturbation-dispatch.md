# Unified perturbation workflow

`perturbation_expansion` provides one entry point for regular perturbation, Lindstedt–Poincaré expansions, multiple scales, and matched asymptotic expansions. It diagnoses the problem before solving it and keeps method choice explicit whenever two inequivalent asymptotic descriptions are equally appropriate.

```python
import sympy as sp
from asymptotic.perturbation_dispatch import perturbation_expansion

x = sp.Symbol("x", real=True)
eps = sp.Symbol("eps", positive=True)
y = sp.Function("y")

result = perturbation_expansion(
    sp.diff(y(x), x) + y(x) + eps * y(x) ** 2,
    y(x),
    eps,
    order=1,
    conditions=sp.Eq(y(0), 1),
)
assert result.selected_method == "regular"
```

## Ranked applicability

`perturbation_diagnostics` returns structural assessments for all implemented methods. A lower rank is preferred. Automatic dispatch runs only when the best rank belongs to one method.

For a weak nonlinear oscillator such as

$$
y''+y+\epsilon y^3=0,
$$

both Lindstedt–Poincaré and multiple scales are natural long-time descriptions. They answer different questions, so automatic dispatch returns an ambiguity instead of selecting either one. The caller can then request `method="lindstedt"` or `method="multiple-scales"` explicitly. Regular perturbation may still be structurally possible but receives a lower preference because secular growth can make it nonuniform on long time intervals.

For a singular boundary-value problem such as

$$
\epsilon y''+y'=0,
\qquad y(0)=0,\quad y(1)=1,
$$

the reduced problem loses differential order. Regular perturbation is therefore marked inapplicable and matched asymptotics is selected automatically.

## Manual continuation

`resume_perturbation` supplies or replaces one order in a regular or Lindstedt perturbation result. Later stored solutions are discarded before continuation so changing an earlier branch or profile cannot leave stale coefficients behind. Regular perturbation recursively resolves later branch equations; Lindstedt continuation also re-derives later Fredholm solvability conditions and frequency corrections.

```python
from asymptotic.perturbation_dispatch import resume_perturbation

solved = perturbation_expansion(
    sp.Symbol("u") ** 2 - 1 - eps,
    sp.Symbol("u"),
    eps,
    method="regular",
    order=2,
)
```

The common `PerturbationHierarchy.with_solution(...)` API remains available for direct manipulation. `resume_perturbation` preserves the method-specific policy for regular and Lindstedt results. Multiple-scales and matched-asymptotic results retain their dedicated transformed-problem workflows instead of being reinterpreted as ordinary regular hierarchies.

## Formal residuals versus rigorous error bounds

`certify_perturbation` makes a strict distinction between two kinds of evidence.

A **formal residual certificate** replays the solved hierarchy or composite residual checks. It proves that the coefficient equations and supplied conditions vanish through the solved perturbation orders. This alone is not an error theorem.

A **rigorous remainder bound** is attached only when an already certified `Remainder` for the same perturbation parameter at zero is supplied. Unknown remainders are rejected instead of upgraded from formal cancellation.

```python
from asymptotic import Remainder
from asymptotic.perturbation_dispatch import certify_perturbation

formal = certify_perturbation(result)
assert formal.formal.verified
assert not formal.has_rigorous_bound

bound = Remainder.big_o(eps**2, eps, 0)
rigorous = certify_perturbation(result, rigorous_remainder=bound)
assert rigorous.has_rigorous_bound
```

This separation lets theorem-backed remainder machinery compose with perturbation methods without claiming that every formally consistent expansion automatically has a uniform asymptotic error estimate.
