# Domains and parameters

This guide collects the related mathematical mechanisms used by the multivariate limit engine. Public entry points are described in [Multivariate limits](../multivariate-limits.md).

## Domain-relative cluster geometry

`domain=` is geometric input, not a post-filter. Semialgebraic domains are
punctured at the target and decomposed into exact connected components using
`semialg`. Only components whose closure contains the target are retained.
The resulting `LocalDomainGeometry` carries the same coverage-certificate
discipline as blow-up and branch atlases.

This represents wedges, cusps, algebraic boundaries, deleted varieties and
multivariate one-sided approaches uniformly. Complex branch atlases intersect
their upper/lower branch sectors with these accumulating domain components.

`Piecewise` branch conditions can therefore be treated as domains, while
Heaviside and modular/fractional discontinuities use the cluster-map layer.

## Exhaustive parameters and unified cluster semantics

## Exhaustive parameter strata

Once a parameter-aware proof mechanism discovers a regime, the parameter bridge
covers the entire complementary parameter space explicitly. Strict
inequalities are refined into the equality boundary and opposite open region.
Equality boundaries are exactly specialized and sent through the ordinary limit
engine; unresolved open regimes remain explicit UNKNOWN strata.

For the regression
`x**16*y**22/(x**2+y**2)**p` the structured result contains:

- `p < 19`: PROVED, value 0;
- `p = 19`: DOES_NOT_EXIST, certified after exact specialization;
- `p > 19`: UNKNOWN.

Value mode remains `ConditionalExpression(0, p < 19)`; it does not fabricate a
value on DNE or unknown strata.

## Unified cluster semantics

`cluster_limit_semantics` is the common logical boundary for complete
cluster-set proofs:

- certified singleton cluster set -> PROVED limit;
- certified non-singleton cluster set -> DOES_NOT_EXIST;
- partial or uncertified cluster set -> UNKNOWN.

`limit` uses this as its final proof layer via
`newton_fan_cluster_set`. `ClusterSetResult.limit_semantics` exposes the same
interpretation directly. Sampled/path values cannot become a proof through this
layer unless the originating cluster result is certified complete.

## Parameter-conditioned limits

Certified free-parameter conditions are first-class mathematical
limit results.

- `limit(..., return_result=False)` returns an ordinary expression
  when unconditional and `ConditionalExpression(value, condition)` when only a
  certified parameter regime is known.
- `return_result=True` preserves `LimitStatus`, evidence, `conditional_value`,
  and `condition`.
- `stratification_expression` converts finite certified parameter strata to an
  ordinary value, `ConditionalExpression`, or `Piecewise`.
- No value is fabricated for unresolved strata.

Canonical regression:
`x**16*y**22/(x**2+y**2)**p -> ConditionalExpression(0, p < 19)`.

## Parameter-cell geometry

Parameter-dependent angular and projective topology share the
`parameter_truth_cells` engine. It enumerates exact truth signatures of
semialgebraic discriminants and removes a cell only after real satisfiability
proves it empty.

The angular optimizer uses these cells when symbolic stationary points enter
or leave the simplex. Multiple real parameters are supported when stationary
roots have been solved and their membership conditions are semialgebraic.

Projective geometry uses the same layer. The homogeneous Möbius family is
partitioned by its determinant: a nonzero determinant is a projective
automorphism with full real image, while the zero determinant collapses to a
constant image. This is the first symbolic projective topology family; higher
degree rational families can add their pole/resultant/critical discriminants
to the same cell engine without changing cluster semantics.

## Parameterized angular optimization

Multivariate homogeneous limits reduce critical radial strata to optimization of an angular form on the unit sphere. The parameterized angular optimizer provides an exact algebraic interface for that step.

Current certified families are general real bivariate quadratic forms, including cross terms and symbolic coefficients, and even homogeneous bivariate polynomial forms whose critical points are uniformly certified in the simplex reduction `z = x**2`, `y**2 = 1-z`.

For a quadratic form `a*x**2 + b*x*y + c*y**2`, the optimizer returns the exact eigenvalue interval with discriminant `(a-c)**2 + b**2`. Parameter stratification splits the repeated-eigenvalue locus from the distinct-eigenvalue region, so exceptional scalar limits are not hidden inside a generic DNE result.

For even homogeneous forms, exact endpoint and interior critical values are used. If a symbolic critical point can cross the simplex boundary and its membership cannot be proved uniformly, the optimizer returns an uncertified result instead of dropping that critical point.

The parameter-cluster layer consumes these angular results. Critical radial strata therefore obtain finite parameter-dependent cluster intervals or singleton sets from the same general optimizer; supercritical signed geometry uses the same extrema to classify one-sided or two-sided unbounded cluster sets.

## Atlas dependency robustness

Unrestricted weighted blow-up atlas coverage is a geometric consequence of the max-coordinate sector construction and no longer imports semialg merely to restate that fact. Restricted-domain closure checks still use semialg. This prevents a missing optional runtime dependency of semialg from turning an unrestricted exact atlas into an uncertified result.

## Parameterized local geometry

`parameterized_geometry_cells` partitions parameter space where local
geometric topology may change. Predicates include coefficient-vanishing
events that change Newton support and, in RP^1, denominator discriminants,
critical-point discriminants, and numerator/denominator resultants.

Each exact parameter truth cell owns a projective atlas, valuation rays and a
composed coverage certificate. This moves parameter dependence from
individual limit recognizers into the geometric atlas layer.

## Piecewise and domain-relative accumulation

A multivariate Piecewise limit is a limit of local branch germs, not an
evaluation of the branch selected at the target.

For branch `i`, the effective approach domain is the ambient domain intersected
with its condition and with the negation of every earlier condition. This
preserves Piecewise priority. The target point itself is deleted before local
accumulation is considered.

A branch contributes only when its effective punctured domain accumulates at
the target. Nonaccumulating branches are ignored. If accumulation cannot be
certified, the Piecewise limit remains unknown. Every accumulating branch is
then evaluated relative to its own effective domain. Equal certified branch
limits prove the Piecewise limit; conflicting certified branch limits or a
certified nonconvergent branch prove nonexistence.

Domain-relative path witnesses are permitted only when substitution proves
that the entire punctured path germ lies in the approach domain.
