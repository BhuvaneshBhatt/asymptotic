# Certification

`asymptotic` separates symbolic construction from mathematical certification.

## Limit status

For simultaneous limits:

- `PROVED` means the limit exists and the returned value is certified.
- `DOES_NOT_EXIST` means nonexistence is certified.
- `UNKNOWN` means neither conclusion was proved by the supported methods within their symbolic budgets.

`UNKNOWN` is not a numerical confidence score and is not equivalent to DNE.

## Formal and certified asymptotics

Expansion-producing APIs can construct useful formal objects even when a remainder theorem is unavailable. A theorem-level claim requires the hypotheses attached to the relevant certificate to be proved.

## Evidence and replay

Evidence records explain how a result was obtained. Replayable certificates retain enough exact information to verify the stored claim without repeating the search that discovered it. This is useful for debugging, regression tests, and downstream code that needs a stronger contract than a printed expression.

## What certification does not infer

The package does not infer a multivariate limit from finitely many agreeing paths, assume unresolved symbolic parameters are nonzero, treat a formal expansion as a remainder theorem, or replace correlated directional variables with independent scalar ranges.

## Testing the contract

Public tests should assert status, value, conditions, and exact cluster information. Tests for a specific theorem may additionally assert the evidence provider. Performance tests should measure expensive-operation entry and decline cost separately from mathematical correctness.

## Parameter-conditioned result pipeline

Parameter dependence is represented structurally instead of by a
certificate-specific branch in the limit front end.

## Contract

Proof mechanisms discover certified parameter regimes and feed
`AsymptoticStratification`. `mathematical_result` is the sole value-mode
projection:

- unconditional certified result -> ordinary SymPy value;
- one certified proper regime -> `ConditionalExpression(value, condition)`;
- several certified regimes -> `Piecewise`;
- uncovered parameter space -> the result remains conditional on the union of
  certified regimes;
- unknown, DNE, or incomplete strata do not acquire invented values.

`return_result=True` returns the first-class stratification so conditions,
individual structured results, evidence, completeness and provenance remain
inspectable.

The public `limit` implementation has no dependency on the
weighted-order certificate. The current weighted-order proof is merely one
producer adapted by `parameter_conditions`; future proof mechanisms enter
through the same bridge without changing value mode.

## Soundness boundaries

The soundness corpus covers principal-cut side limits, independent monodromy generators,
branch-point radial versus winding behavior, Piecewise boundary intersections,
nonaccumulating lower-dimensional pieces, disconnected approach domains,
strict versus weak inequalities, singular domain boundaries, denominator
strata, cancellation across deleted varieties, signed infinity, and finite
versus infinite cluster sets.

Two invariants were strengthened while making these cases executable:

1. A direct top-level `arg`, `log`, or nonintegral power whose argument lands
   on the negative-real principal cut is not continuous by substitution.  If
   the imaginary part crosses the cut transversely, exact upper/lower boundary
   germs are compared.  Distinct germs prove nonexistence.  Tangential contact
   is left unresolved unless another supported theorem applies.
2. An exact conflicting admissible path is a negative certificate and takes
   precedence over positive fast-path certificates.  Relative-domain and
   Piecewise geometry also retains ambient coordinates that can provide
   punctured accumulation even when the branch value does not depend on them.

The current bounded capability-matrix run over all 212 curated cases reports
81 PROVED, 69 DNE, 62 UNKNOWN, and zero wrong-result certifications. UNKNOWN
continues to represent missing capability instead of failure.
