# Multivariate limits

Use `limit` when all coordinates approach their targets together. The solver returns a proof-aware result when `return_result=True`, distinguishing a certified value, certified nonexistence, and an unresolved problem.

```python
import sympy as sp
from asymptotic.limits import LimitStatus, limit

x, y = sp.symbols("x y", real=True)
result = limit(
    x * y / sp.sqrt(x**2 + y**2),
    (x, y),
    (0, 0),
    return_result=True,
)
assert result.status is LimitStatus.PROVED
assert result.value == 0
```

## How the solver approaches a problem

Cheap exact certificates run before geometric machinery: regular composition, variable-support reduction, uniform majorants, radial-order bounds, and simple nonexistence witnesses. Harder problems can use radial/angular coordinates, projective charts, local strata, correlated directional images, or semialgebraic reasoning. A method is accepted only when its hypotheses and coverage obligations are certified.

The solver may return `UNKNOWN`. That means the available methods did not establish a complete proof; it is not evidence that the limit exists or fails to exist.

## Geometry guides

- [Geometry](multivariate/geometry.md): local germs, blow-ups, angular images, strata, and chart coverage.
- [Cluster sets](multivariate/cluster-sets.md): cluster images, correlated directions, and coverage certificates.
- [Domains and parameters](multivariate/domains-and-parameters.md): restricted approaches and parameter-dependent answers.
- [Branches and resolution](multivariate/branches-and-resolution.md): branch-sensitive geometry and recursive resolution.
- [Methods](multivariate/methods.md): analytic, valuation, exp/log, and special-function methods.

See [Certification](certification.md) for result semantics and [Capabilities and limitations](capabilities.md) for the public support matrix.

### Exceptional curves and mixed infinite targets

For modest bivariate rational germs, the negative-witness search first extracts exceptional projective directions from leading homogeneous numerator and denominator forms. It can then lift a low-degree denominator divisor to an algebraic curve and probe transverse perturbations before invoking Newton resolution. This catches higher-order cancellation geometry that straight rays miss.

Product ends at simultaneous infinities use coercive growth/decay certificates, including nonnegative polynomial scales and positive-definite quadratic forms. Mixed finite/infinite targets use separable product-end certificates when available and otherwise retain joint semantics through reciprocal coordinates; they are not interpreted as iterated limits.
