# Capabilities and limitations

This matrix is the normative high-level description of what `asymptotic` attempts to support. “Formal” means an expansion/construction may be produced without a theorem-level remainder guarantee. “Certified” means the package can prove the stated result under the hypotheses it checks.

| Area | Current support | Certification | Main limitation |
|---|---|---|---|
| Univariate scale discovery | Powers, logs, exponentials, iterated finite-height log-exp scales, MRV guidance, and net Gamma/factorial-family normalization | Growth comparisons are conservative and factorial normalization is positive-real/domain-aware | Not a complete Hardy-field comparability decision procedure |
| Lazy multiseries | Multiple ordered small scales, sparse demand-driven coefficients, dynamic scale obligations | Formal/exact coefficient algebra | Unsupported analytic heads may fall back or terminate with an obligation |
| Nested forms | Finite limits, sign, power/log/exp depth, resumable refinement | Structural decisions use exact/tri-state services | Primarily directed real limits; complicated oscillatory/complex nesting is partial |
| Common asymptotic algebra | Coordinate-aware coercion across transseries, multiseries, nested, Puiseux/implicit, scale, shadow-field, and ODE-generated elements | Cross-representation arithmetic reuses certified transseries remainder rules | Finite transseries is still the interoperability normal form; this is not a complete Hahn/transseries field |
| Composition | Analytic/meromorphic composition, exact polynomial/rational substitution, and nested log-exp cases | Finite-order Taylor theorem finds the first nonzero derivative; algebraic substitution uses exact polynomial identities and certified quotient stability | General branch-sensitive complex composition remains partial |
| Differentiation | Symbolic/transseries differentiation | Operation-specific remainder theorem | Regularity must be proved to upgrade UNKNOWN |
| Integration | Power/log transitions, exponential integration-by-parts, field/integral-shadow construction | Certification paths are bounded | User-facing general symbolic integration may still inherit SymPy cost |
| Puiseux/algebraic branches | Newton polygon, rational ramification, multiple leading branches, multiplicity/turning-point profiling, automatic singular blow-up | Formal algebraic branch structure with explicit singularity diagnostics | General algebraic-geometry resolution of singularities is out of scope |
| Series reversion | Rational leading exponents, multiple branches, finite/translated points | Branch decisions retained | General global inverse continuation and complex sheet tracking are partial |
| Inverse asymptotics | Local and infinity transformations, log-exp inverse iteration | Inverse remainder theorem where hypotheses prove | General transseries inversion is not closed for all representations |
| Dominant balance | Polynomial/rational and generalized transseries-valued balances | Replayable dominant-balance certificates | Candidate discovery is not a universal differential-algebra decision method |
| Parameter strata | Automatic zero/nonzero coefficient strata, square-free principal-radical normalization, bounded Groebner equality-ideal normalization, Boolean coalescing | Conditions/provenance are explicit and ordering deterministic | General multivariate radical-ideal decomposition and arbitrary semialgebraic canonicalization remain out of scope |
| Multivariate scaling | Weighted paths, automatic Newton weight cones/chambers/walls | Balance replay within discovered regimes | General tropical/Newton-polyhedron geometry and nonpositive weight domains are partial |
| Implicit equations/systems | Simple-root lifting plus automatic Newton–Puiseux/scaling handoff for certified multiple roots; joint multivariate Newton regimes | Branch method/multiplicity diagnostics are explicit; remainder information when hypotheses prove | Higher-dimensional singular Jacobian resolution and general blow-up trees remain partial |
| Nonlinear ODEs | Differential balance, recursive corrections, log/exp descendants, Frechet linearization | First-order, constant-coefficient, and asymptotically constant ``L=L0+E`` Green/Frechet estimates | General variable coefficients without a finite hyperbolic limit remain unsupported |
| Probability/expectation asymptotics | Exact joint expectation/probability, explicit symbol bindings, conditioning, one-variable density/PMF reduction, moving domains, Laplace and lattice-saddle/local-limit routes | Exact routes retain exact provenance; positive PMF Stirling and selected Binomial lattice tails/local limits carry explicit certificates | Custom structural fallback remains one-dimensional; general multivariate probability/Laplace geometry is not implemented |
| Periodic/oscillatory factors | Period detection, finite bounds, zero-crossing awareness | Conservative boundedness facts | Quasiperiodic and general oscillatory stationary-phase analysis is not implemented |
| Function properties | Reviewed domains, singularities, branch cuts, analyticity, extrema/ranges for registered heads | Tri-state, provenance-carrying decisions | Registry coverage is finite; not a complete special-function domain engine |
| Zero equivalence | `exprtest` first, conservative SymPy fallback | Certified by default; probable mode opt-in | Hard identities can remain UNKNOWN |
| Shadows/ghosts | Moderate growth, infinitesimal ideals, shadow projection, integral-shadow extensions | Structural field decisions | Partial implementation of Shackell-style asymptotic domains, not full closure |
| Remainder objects | EXACT, O, o, UNKNOWN-style theorem state, replay metadata | Replayable finite sums, exact scaling/negation, products, reciprocal/quotient, differentiation, bounded antiderivative propagation, algebraic/general composition, inversion, nonlinear lifting, and Green/Frechet theorems | A formal expansion is not automatically a certified asymptotic expansion |
| ODE interchange | Structural data import plus replayed constant-coefficient operator descriptor | Green descriptor validated before use | Requires optional `odeanalysis`; no shared mutable runtime dependency |
| High-level ODE solve | `asymptotic_dsolve` dispatches linear formal data or nonlinear differential-polynomial lifting | Preserves formal-data completeness and nonlinear residual information | Unsupported non-polynomial nonlinear equations and general global connection problems remain outside scope |
| Recurrence solve | `asymptotic_rsolve` prefers exact solutions and otherwise constructs discrete Newton edges plus native particular solutions for rational/polynomial-coefficient linear recurrences | Factorial/exponential/power scales, simple-root lifts, exact constant-coefficient repeated-root chains, supported stretched-exponential secondary Newton lifts, first-order rational/hypergeometric forcing, and simple logarithmic resonance | Higher-order resonant forcing, arbitrary repeated-secondary configurations, repeated tertiary roots, deeper nested ramification, and connection constants remain partial |
| Complex asymptotics | Explicit `ComplexSector`/`ComplexBranchMetadata`, branch-cut/Stokes-ray provenance, and optional `odeanalysis` sector/sheet interchange | Metadata is conflict-checked and propagated through transseries operations; ODE Stokes geometry retains cover/local/original angles | No general sectorial asymptotic certification, connection matrices, resurgence, or Borel summation |

## Endpoint model

The strongest coverage is for one-sided real asymptotics at `0`, finite translated real points, and `+/-oo`. The local coordinate is treated with a directed sign convention. Algorithms that depend on branch cuts or eventual signs should not be assumed sectorially valid in the complex plane.

## Exactness and proof boundaries

The package intentionally separates calculation from proof. For example, it may compute a formal nonlinear correction but leave the Frechet inverse theorem inconclusive; or it may construct an inverse branch while retaining an unresolved branch-safety decision. This is preferable to silently asserting an `O`/`o` statement from an unproved hypothesis.

## Performance boundary

Internal symbolic work is routed through a bounded policy layer. Cheap rational/polynomial/linear methods run first; general `solve`, `limit`, assumptions/SAT, simplification, and integration fallbacks are invoked only when a caller explicitly permits them and the expression is below a configured complexity budget. Proof-critical primitive construction never launches an unrestricted integrator. Expensive zero tests are memoized per `AsymptoticContext`; the package disables `exprtest`'s additional process-global cache because the context already supplies local reuse.

The deliberate exceptions are explicitly user-requested general-antiderivative operations such as `Multiseries.integrate()` and `NestedExpansion.integrate()`. Those may still inherit SymPy's cost on difficult inputs.

### Asymptotically constant Green/Frechet boundary

For a scalar higher-order operator, the variable-coefficient theorem requires an infinite endpoint and monic normalized coefficients converging to finite constants. The limiting characteristic polynomial must be hyperbolic. The limiting Green particular is then replayed in the full operator; its defect must be `o(R)`, and the selected correction must have a strict exponential-rate gap from stable limiting modes. This is a conservative tail theorem based on roughness of exponential dichotomies, not a general variable-coefficient Green-function solver.

## Near-term extensions

Related extensions outside the supported scope include broader reviewed function-property coverage, asymptotic Green quadrature when the limiting forcing primitive is not elementary, more complete log-exp/transseries closure, stronger branch tracking, and full sectorial complex asymptotics with Stokes phenomena.

## Operation-level certification matrix

This table is intended to answer whether a finite result can be trusted as a theorem-level asymptotic statement.

| Operation | Supported inputs | Certification level | Main hypotheses | Typical reason for `UNKNOWN` |
|---|---|---|---|---|
| finite sum | compatible certified remainders | certified | common variable/point | an input remainder is already unknown |
| exact scaling/negation | one certified remainder and variable-independent exact factor | certified | exact finite-prefix scaling | input remainder is unknown |
| product | finite prefixes + certified remainders | certified | compatible coordinates | an input remainder is unknown |
| reciprocal | one prefix + certified remainder | certified | eventual nonvanishing and relative error `R/a -> 0` | zeros cannot be excluded or error is not relatively small |
| quotient | numerator/denominator approximations | certified | reciprocal hypotheses for denominator | denominator stability unresolved |
| algebraic substitution | polynomial/rational outer function | certified | denominator nonvanishing for rational case | substitution hits/unresolved pole |
| analytic composition | finite Taylor jet | conditional | analyticity/branch safety and stable first nonzero derivative | cut/singularity or next-term control unresolved |
| differentiation | certified remainder | conditional | derivative control/regularity of remainder scale | regularity cannot be proved |
| asymptotic integration | supported scale transitions | formal/conditional; certified propagation when an exact stored error and scale have directly checkable bounded primitives | primitive rule and scale conditions | abstract O/o alone does not control an indefinite primitive |
| inverse | local/infinity inverse prefix | conditional | derivative nonzero and Newton stability | degeneracy or branch behavior unresolved |
| implicit lifting | simple or Newton--Puiseux local branch | formal/conditional | dominant balance and branch residual replay | multiplicity/scale cannot be resolved |
| Green/Frechet | scalar first order, constant or asymptotically constant higher order | certified on theorem domain | hyperbolic limiting operator, convergent perturbation, controlled modes | center spectrum, nonconvergent coefficients, or uncontrolled mode |
| probability/expectation | exact joint SymPy expressions/events plus one-variable structural fallback | exact/certified/formal | exact distribution reduction, certified Stirling/lattice theorems, or local Laplace geometry | unsupported multivariate fallback, unresolved event/domain, competing saddle, or missing global tail proof |
| parameter stratification | bounded polynomial/Boolean conditions | explicit conditional family | case conditions can be normalized and evaluated | algebraic condition exceeds bounded canonicalization policy |

For diagnostic steps after an unknown result, see [Understanding `UNKNOWN`](unknown-results.md).
