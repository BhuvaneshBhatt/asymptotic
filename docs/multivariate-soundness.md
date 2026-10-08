# Multivariate limit soundness

The multivariate limit engine uses a common coverage/certification architecture. The guarantees below describe the mathematical contract.

- Reference execution is resumable and shardable. Each case is written
  incrementally and the isolated runner supports 60 shards, a two-second
  symbolic alarm, hard worker isolation, resume, and aggregation.
- Reference 0966 has a dedicated regression and an exact radial reduction; it
  proves `pi/2` instead of treating an `AccumBounds` object as a scalar
  ray witness.
- Algebraic approaches can be normalized branch-by-branch in a common Puiseux
  parameter with the relative domain pulled back exactly.
- Symbolic Newton weights have an assumption-conditioned certification API;
  generated exp/log comparison cells are no longer marked COMPLETE merely
  because candidate cells were generated.
- Branched blow-up atlases retain a branch-point accumulation obligation and
  expose independent monodromy generators instead of conflating branch sides.
- Integer-order Bessel-Y germs and shifted Gamma-pole residues are available
  through the local-germ layer.
- Exact n-dimensional direction maps have a unit-sphere joint-cluster fast
  path, avoiding the former 3-D direction-map timeout.
- The public `limit` boundary converts unsupported symbolic
  geometry exceptions to evidence-bearing UNKNOWN results.
- `ValuationCone` has been removed. Growth metadata wraps the common
  geometric `Valuation` when weights are certified; unresolved symbolic
  weights remain explicitly uncertified until partitioned.
- Higher-order/tangent algebraic path witnesses and a Newton certification
  guard prevent finite weight atlases from certifying across missed
  cancellation branches.

## Reference validation

The 1,168-case reference corpus is an exhaustive release/soundness gate. Cases run in isolated processes through the resumable 60-shard coordinator. A release claim requires a strict aggregate with all indices present, no conflicting duplicates, and no `WRONG`, `ERROR`, or `TIMEOUT` outcomes. `UNKNOWN` and `KNOWN_GAP` are retained in the report for explicit proof-boundary review.

Reference 0966 has a dedicated exact radial-reduction regression proving `pi/2`; it is not treated as an `AccumBounds` scalar witness. See `tests/reference_cases/docs/reference-corpus-execution.md` for the durable execution procedure.
