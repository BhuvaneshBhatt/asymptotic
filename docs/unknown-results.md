# Understanding `UNKNOWN`

`UNKNOWN` is a mathematical result state, not an exception. It means the package
cannot justify a stronger answer under the current assumptions, branch data,
algorithmic coverage, or bounded symbolic policy.

The distinction is intentional:

- `EXACT` means no asymptotic approximation remains in the returned expression.
- `CERTIFIED` means a finite asymptotic result is accompanied by replayable proof
  evidence for the claimed remainder or decision.
- `FORMAL` means a structurally valid asymptotic construction was produced but
  the required remainder theorem was not established.
- `UNKNOWN` means the package cannot safely settle the requested conclusion.

## Start with the evidence

Structured results may expose `conditions`, `remainder`, `certificate`,
`limitation`, or property-decision provenance. Inspect the first unresolved
hypothesis rather than immediately increasing the requested number of terms.
More terms do not resolve a missing sign, branch, support, or nondegeneracy fact.

## Common causes

| Area | Typical unresolved fact | Useful response |
|---|---|---|
| quotient or inverse | denominator not proved eventually nonzero | add sign/nonzero assumptions or choose a different local branch |
| composition | analyticity or branch safety unresolved | inspect function-property and branch-safety decisions |
| relation | growth comparison depends on an unknown parameter sign | add the relevant assumption or split parameter cases |
| implicit equation | multiplicity or dominant scale unresolved | inspect singularity profile, dominant balances, and parameter strata |
| multivariate relation | tested paths do not prove all scaling paths | use weight-cone analysis or retain `UNKNOWN` |
| probability | exact joint reduction failed and no one-dimensional fallback applies | reduce dimension, provide a supported distributional form, or use an exact external derivation |
| discrete quantile | generalized inverse cannot be determined exactly | provide a distribution/parameter regime with decidable CDF ordering |
| KL/cross entropy | support containment undecided | establish `supp(P) subseteq supp(Q)` explicitly |
| Laplace/saddle | global dominance, curvature, or tail coercivity unresolved | supply domain/sign assumptions or accept a formal local expansion |
| ODE | formal branch exists but no remainder theorem applies | inspect residuals and branch metadata; status remains `FORMAL` |
| recurrence | exact and native Newton hierarchies do not resolve a complete basis | inspect primary/secondary Newton roots; supported repeated secondary roots descend through one tertiary stretched-exponential level; repeated tertiary roots or unsupported Newton configurations remain unresolved |
| optimization | sign/order of candidates cannot be established | strengthen real-domain assumptions or stratify parameters |

## Examples

### Eventual nonvanishing

An exact expression need not be safely invertible. `sin(x)` is exact, but as
`x -> +oo` it has infinitely many zeros. A reciprocal theorem therefore cannot
claim an eventually nonzero denominator without additional structure.

### Parameter-dependent comparison

For a real parameter `a`, the relation between `a*x` and `x` depends on `a`.
An assumption such as `a > 0`, `a = 0`, or `a < 0` can change the answer. The
cache stores these contexts separately; evaluating one assumption regime first
must not influence another.

### Multivariate paths

Agreement on coordinate axes or several linear rays is not proof of a relation
on every path to a multivariate limit. Nonlinear scalings can expose a different
dominant balance. The package therefore uses path checks as counterexample tools
unless a stronger weight-cone argument is available.

### Discrete quantiles

A jump CDF often has no solution to `F(x) = p`. Quantiles are generalized
inverses, so failure of ordinary equation solving is not evidence that a
quantile is absent. If the lattice ordering needed for the generalized inverse
cannot be established, `UNKNOWN` is preferable to choosing an arbitrary root.

### Formal ODE solutions

A formal basis can be structurally complete while lacking a theorem that bounds
the omitted tail. `AsymptoticDSolveResult.residuals()` lets callers check the
returned prefixes directly, but residual improvement alone does not upgrade the
status to `CERTIFIED`.

## Why symbolic searches are bounded

Internal proof paths deliberately use bounded symbolic policies. An unresolved
proof is preferable to an unpredictable global `solve`, `limit`, assumptions,
integration, or simplification search. This keeps failure modes reproducible and
prevents a difficult symbolic subproblem from turning an otherwise local
asymptotic computation into an unbounded calculation.

## When `UNKNOWN` indicates missing implementation

Some `UNKNOWN` states identify mathematics outside the implemented coverage,
rather than missing assumptions. Examples include general multidimensional
probability saddles, unresolved resonant recurrence hierarchies,
and full sectorial complex/Stokes certification. These cases should remain
explicitly unsupported until a method with clear hypotheses and residual or
certificate contracts exists.

See [Algorithm selection](algorithm-selection.md) for the route chosen before an
`UNKNOWN` result is produced.
