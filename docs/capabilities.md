# Capabilities and limitations

This matrix is the normative high-level description of what `asymptotic` attempts to support. “Formal” means an expansion/construction may be produced without a theorem-level remainder guarantee. “Certified” means the package can prove the stated result under the hypotheses it checks.

| Area | Supported inputs | Certification | Main limitation |
|---|---|---|---|
| Univariate scale discovery | Powers, logs, exponentials, iterated finite-height log-exp scales, MRV guidance, and net Gamma/factorial-family normalization | Growth comparisons are proof-bounded and factorial normalization is positive-real/domain-aware | Not a complete Hardy-field comparability decision procedure |
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
| Perturbation workflows | Structured algebraic/ODE scalar and system problems, canonical value/derivative/limit conditions, explainable dispatch, inspectable hierarchies, symbolic residual-order validation, and numerical observed-order diagnostics | Symbolic validation records exact residual orders; numerical checks are explicitly supporting evidence, not proofs | Automatic dispatch declines tied methods; singular-layer/profile generation and general nonlinear systems remain partial |
| Nonlinear ODEs | Differential balance, recursive corrections, log/exp descendants, Frechet linearization | First-order, constant-coefficient, and asymptotically constant ``L=L0+E`` Green/Frechet estimates | General variable coefficients without a finite hyperbolic limit remain unsupported |
| Probability/expectation asymptotics | Exact joint expectation/probability, explicit symbol bindings, conditioning, one-variable density/PMF reduction, moving domains, lattice-saddle/local-limit routes, and generic multivariate Laplace geometry | Exact routes retain exact provenance; positive PMF Stirling, selected Binomial lattice tails/local limits, and supported multivariate Laplace geometries carry explicit certificates | Curved constraints, degenerate multivariate saddles, and unrestricted non-polynomial global dominance remain formal or unsupported |
| Periodic/oscillatory factors | Period detection, finite bounds, zero-crossing awareness | Proof-bounded boundedness facts | Quasiperiodic and general oscillatory stationary-phase analysis is not implemented |
| Certified scalar tails | Fixed complex powers times flat even-order real exponentials; affine square waves and floor parity; Gaussian log-moment cancellation; positive-real Fresnel auxiliary tails; Appell F1 continuity | Modulus domination, interior attained subsequences, and convergent integral definitions with checked domain conditions | Nonlinear/discrete phase charts, complex special-function sectors, growing weights, and parameter poles require additional proofs |
| Function properties | Reviewed domains, singularities, branch cuts, analyticity, extrema/ranges for registered heads | Tri-state, provenance-carrying decisions | Registry coverage is finite; not a complete special-function domain engine |
| Zero equivalence | Bounded rational-polynomial identities, `exprtest`, and proof-bounded SymPy fallback | Certified by default; probable mode opt-in | Hard identities can remain UNKNOWN |
| Joint modular phases | Mod(P,2) with sin(P), cos(sqrt(2)*P), folded cosine and finite-tail Min/Max algebra; affine, quadratic, logarithmic and one cubic inverse chart | Exact least-hit subsequences, rational independence and eventual denominator avoidance | Growing multipliers, oscillatory denominators, additional domain restrictions and integer sampling need separate proofs; no complete cluster-set claim |
| Gaussian sides and vertical integral tails | Real/purely imaginary square-root Gaussian poles; fixed-order exponential integrals on affine vertical rays | Principal square-root sides and a checked sector remainder | General complex coefficients, variable order and possibly zero slopes remain unresolved |
| Shadows/ghosts | Moderate growth, infinitesimal ideals, shadow projection, integral-shadow extensions | Structural field decisions | Partial implementation of Shackell-style asymptotic domains, not full closure |
| Remainder objects | EXACT, O, o, UNKNOWN-style theorem state, replay metadata | Replayable finite sums, exact scaling/negation, products, reciprocal/quotient, differentiation, bounded antiderivative propagation, algebraic/general composition, inversion, nonlinear lifting, and Green/Frechet theorems | A formal expansion is not automatically a certified asymptotic expansion |
| ODE interchange | Formal blocks, verified coefficient recurrences, independently checked residual certificates, ramified sector domains/dominance, plus replayed constant-coefficient operator descriptors | Producer recurrence replay and independent residual replay; Green descriptor validated before use | Requires optional `odeanalysis>=0.1.0`; no shared mutable runtime dependency |
| High-level ODE solve | `dsolve` dispatches linear formal data or nonlinear differential-polynomial lifting | Preserves formal-data completeness and nonlinear residual information | Unsupported non-polynomial nonlinear equations and general global connection problems remain outside scope |
| Recurrence solve | `rsolve` prefers exact solutions and otherwise constructs discrete Newton edges plus native particular solutions for rational/polynomial-coefficient linear recurrences | Factorial/exponential/power scales, simple-root lifts, exact constant-coefficient repeated-root chains, supported stretched-exponential secondary Newton lifts, first-order rational/hypergeometric forcing, and simple logarithmic resonance | Higher-order resonant forcing, arbitrary repeated-secondary configurations, repeated tertiary roots, deeper nested ramification, and connection constants remain partial |
| Complex asymptotics | Explicit `ComplexSector`/`ComplexBranchMetadata`, branch-cut/Stokes-ray provenance, and optional `odeanalysis` sector/sheet interchange | Metadata is conflict-checked and propagated through transseries operations; ODE Stokes geometry retains cover/local/original angles | No general sectorial asymptotic certification, connection matrices, resurgence, or Borel summation |
| Whole-plane complex limits | Quotients of regular analytic germs at finite points; bounded positive rational powers of polynomials at spherical infinity | Exact Taylor valuations, uniform modulus growth or full two-coordinate proofs; branch nonexistence uses attained ray sequences | General angular behavior, unresolved parameter denominators and unregistered branch germs remain unresolved |
| Exact finite cluster sets | Fixed imaginary powers, common affine trigonometric phases, and registered tail families | Containment plus attained sequences, with late-tail domain and pole avoidance | Continuous sampling only; accumulating-pole and general coupled-phase families remain partial |
| Scaled angular corners | Fixed entire scalar coefficients multiplying radial logarithmic or rational corners | Coefficient-zero cells and two distinct finite attained limits on the other cells | General coefficient poles and other angular shapes remain unresolved |
| Uniform real norm bounds | Positive sums of coordinate powers, absolute norms, Max norms, fixed logarithmic powers and flat exponential tails | Uniform zero bounds, including shifted finite numeric centers and exact represented floating exponents | Restricted domains, unresolved parameters and noncoercive denominator geometry require other certificates |
| Removable trigonometric kernels | Sine/tangent divided differences, sinc/csc factors and low-order sine/cosine cancellation | Exact half-angle identities or holomorphic Taylor extensions with denominator-safe attained rays | Cofactors must be continuous or have an independent uniform norm bound |
| Attained periodic approaches | Affine period-one waves at real coordinate infinities; fixed rational hyperbolic reciprocal poles in two real coordinates | Opposite open-half-period values, or a real zero limit and a nonzero complex-period value; original poles are avoided | Restricted domains, parameter-dependent phases and other periodic pole geometries remain unresolved |
| Signed inverse-trigonometric poles | Atan of a locally positive-denominator real pole | Uniform signed infinity of the inner expression and the corresponding signed pi/2 endpoint | Mixed-sign poles and unsupported domain restrictions remain unresolved |
| Exact multivariate source names | Registered real roots, Log10, real/imaginary parts, conjugation, argument and selected entire/special functions | Definitions are translated before existing certificates run | Missing user definitions, unsettled endpoint conventions and explicit execution exclusions remain visible |
| Displaced rational poles | Fixed finite real or complex pole displacement | Rational continuity on the nonzero parameter cell; two attained pole-free rays at zero displacement | Other displaced singularities need their own parameter cells |
| Analytic parameter corners | Trigonometric, hyperbolic, exponential and error functions of `x*y**2/(x**(2*q)+a*y**2)`, for real finite `a` and `q=1..4` | Uniform convergence for `a>0`; distinct attained subsequences for `a<=0`, with tangent pole avoidance | Restricted domains and other rational corner shapes need separate certificates |
| Conditional growth parameters | Real power and exponential growth rates, and source conditions on fixed parameters | Certified cells are retained as Piecewise/conditional values with unresolved complements | A reference-condition pass does not cover missing parameter cells |

See [limit contracts](limit-contracts.md) for examples and the precise finite-cluster convention.

## Endpoint model

Strong coverage is for one-sided real asymptotics at `0`, finite translated real points, and `+/-oo`. The local coordinate is treated with a directed sign convention. Algorithms that depend on branch cuts or eventual signs should not be assumed sectorially valid in the complex plane.

## Exactness and proof boundaries

The package separates calculation from proof. For example, it may compute a formal nonlinear correction but leave the Frechet inverse theorem inconclusive; or it may construct an inverse branch while retaining an unresolved branch-safety decision. The result remains uncertified when an `O`/`o` hypothesis cannot be proved.

## Performance boundary

Internal symbolic work is routed through a bounded policy layer. Cheap rational/polynomial/linear methods run first; general `solve`, `limit`, assumptions/SAT, simplification, and integration fallbacks are invoked only when a caller explicitly permits them and the expression is below a configured complexity budget. Proof-critical primitive construction never launches an unrestricted integrator. Expensive zero tests are memoized per `AsymptoticContext`; `exprtest` proof-cache reuse is also enabled for repeated identities across contexts.

The exceptions are user-requested general-antiderivative operations such as `Multiseries.integrate()` and `NestedExpansion.integrate()`. Those may still inherit SymPy's cost on difficult inputs.

### Asymptotically constant Green/Frechet boundary

For a scalar higher-order operator, the variable-coefficient theorem requires an infinite endpoint and monic normalized coefficients converging to finite constants. The limiting characteristic polynomial must be hyperbolic. The limiting Green particular is then replayed in the full operator; its defect must be `o(R)`, and the selected correction must have a strict exponential-rate gap from stable limiting modes. This is a proof-bounded tail theorem based on roughness of exponential dichotomies, not a general variable-coefficient Green-function solver.

## Scope beyond the supported contract

Broader function-property coverage, more complete log-exp/transseries closure, stronger branch tracking, and general sectorial complex asymptotics with Stokes phenomena are outside the supported contract.

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
| implicit lifting | simple or Newton–Puiseux local branch | formal/conditional | dominant balance and branch residual replay | multiplicity/scale cannot be resolved |
| Green/Frechet | scalar first order, constant or asymptotically constant higher order | certified on theorem domain | hyperbolic limiting operator, convergent perturbation, controlled modes | center spectrum, nonconvergent coefficients, or uncontrolled mode |
| probability/expectation | exact joint SymPy expressions/events, one-variable structural fallback, and generic product-domain multivariate Laplace analysis | exact/certified/formal | exact distribution reduction, certified Stirling/lattice theorems, or replayable local/global Laplace geometry | unresolved event/domain, curved/degenerate multivariate constraints, or missing global dominance proof |
| parameter stratification | bounded polynomial/Boolean conditions | explicit conditional family | case conditions can be normalized and evaluated | algebraic condition exceeds bounded canonicalization policy |

For diagnostic steps after an unknown result, see [Understanding `UNKNOWN`](certification.md).

## Multivariate limits: certification boundary

The multivariate solver is a proof-oriented local-geometry engine, not a
general-purpose promise to decide every symbolic limit.  Its strongest path
uses exact rational/analytic germs, Newton support, common valuations,
weighted/projective blow-ups, semialgebraic angular images, local-domain
components, Puiseux-normalized singular curves, and explicit coverage
certificates.

| Multivariate feature | Supported route | What may still produce `UNKNOWN` |
|---|---|---|
| Rational/analytic finite germs | Taylor/valuation and exact cancellation certificates | unresolved symbolic coefficients or non-analytic heads |
| Signed real poles | Continuous nonzero real numerator over a certified nonnegative denominator tending to zero, with an attained pole-free ray | Restricted domains, sign-indefinite denominators, and unresolved ray witnesses |
| Positive denominator sums | Polynomial numerators over positive monomial sums and their positive rational powers; uniform termwise vanishing bounds | Exact rational coefficients and numerator order at most twelve; other orders or denominator signs may remain unresolved |
| Weighted positive quotients | Exact finite shifts, local absolute-value normalization, positive sums/products, radical norms, bounded polynomial sign factors, elementary compositions and logarithmic cancellation | Weighted AM–GM and explicit unit/remainder bounds prove uniform vanishing; original pole bases are checked on an attained approach | Bounded monomial/degree search; approximate targets, unresolved parameters, restricted domains and unsupported original pole restrictions remain unresolved |
| Positive exponential quotient poles | A positive reciprocal polynomial phase over a vanishing nonnegative polynomial denominator | The exponential is at least one and the reciprocal denominator diverges uniformly; an attained diagonal avoids both original zeros | Requires certified polynomial positivity; signed phases and unsupported domains remain unresolved |
| Real trigonometric pole germs | Polynomial real phases through `1-cos(h)` and `1-sqrt(cos(h))`, including fixed sine powers and exponential poles | Quadratic phase germs with attained pole-free rays; signed sine infinities additionally require a uniform phase sign | Changing phase signs do not prove nonexistence; complex phases and other denominators require separate branch proofs |
| Principal-root quotient cancellation | Exact polynomial division in coordinate square roots, with an attained denominator-domain sequence | Exact rational coefficients and root-polynomial degree at most twelve; residual poles, nested roots, or missing domain accumulation proof |
| Local nonexistence witnesses | Bounded elementary meromorphic rays and curves; reciprocal-power sine/cosine phases with nonzero continuous amplitudes | Failed searches remain unresolved; accumulating poles need separate avoidance proofs |
| Removable scalar germs | Elementary holomorphic quotients composed with continuous zero-valued inner germs, including principal-power complex approaches | Unproved inner continuity or original-domain accumulation |
| Newton/weighted scaling | positive certified valuation rays and weighted atlases | symbolic weight order not implied by assumptions |
| Singular algebraic approaches | local algebraic/Puiseux branch normalization and pulled domains | implicit branches not normalizable within the bounded algebraic route |
| Semialgebraic domains | local components, closure/accumulation tests, transformed chart domains | expensive high-dimensional quantifier elimination |
| Projective directions | affine/projective charts and exact bivariate projective images | higher-dimensional projective topology requiring general elimination |
| Branch-sensitive germs | principal cuts, pulled branch divisors, sector splitting, winding state | nested/tangential branch geometry whose coverage cannot be proved |
| Joint vector clusters | correlated common-domain images; exact sphere fast paths | general high-dimensional nonlinear angular images |
| Parameter-dependent geometry | parameter cells and assumption-aware symbolic weights | undecidable sign/order conditions or cell explosion |
| Exp/log relative growth | proof-bounded GrowthScale proofs and candidate valuation fans | no exact exhaustiveness proof for a symbolic fan |
| Special-function germs | reviewed local rules including Bessel and Gamma families | parameter-dependent order/branch cases outside reviewed rules |

### Limitations and non-goals

`UNKNOWN` is intentional whenever the implementation has evidence for useful
candidate charts or paths but lacks a proof that they exhaust the approach
domain.  In particular:

- finitely many agreeing paths never constitute a proof of a multivariate
  limit;
- automatic exp/log fans are candidates until their union is proved
  exhaustive;
- arbitrary higher-dimensional semialgebraic image/quantifier-elimination
  problems can exceed the package's bounded symbolic budget;
- general complex-sector asymptotics and complete monodromy/analytic
  continuation are outside the supported contract;
- symbolic parameters can force a parameter-space partition; if exact
  positivity/order/nonvanishing conditions cannot be certified, the result is
  proof-bounded instead of guessed;
- special-function support is registry-driven instead of a complete theory
  for arbitrary parameter-dependent special functions;
- public limit APIs should return `UNKNOWN` with evidence for unsupported
  symbolic geometry instead of leak an internal symbolic exception.

See [capabilities.md](capabilities.md) for package-wide non-goals and
[architecture.md](architecture.md) for the certification pipeline.

## Scope and non-goals

`asymptotic` is principally a directed-real symbolic asymptotics engine. Its preferred behavior is proof-bounded: produce a formal finite object when possible, certify it when hypotheses are proved, and otherwise preserve `UNKNOWN`.

## In scope

- finite power/log/exp and nested asymptotic scales;
- lazy multiseries and finite generalized transseries prefixes;
- Puiseux branches, local reversion, and singular implicit blow-ups;
- multivariate weighted scaling regimes and automatic Newton weight-cone discovery in public balance APIs;
- nonlinear differential dominant balance and recursive corrections;
- operation-specific `O/o` remainder theorems;
- constant and asymptotically constant scalar Green/Frechet estimates;
- finite asymptotic differential-field/shadow constructions;
- reviewed real-domain, singularity, and branch facts for a finite function registry;
- explicit complex-sector/branch metadata and optional ODE Stokes-sector interchange (metadata, not general sectorial certification).

## Outside the supported scope

- general sectorial complex asymptotics;
- general Stokes connection/continuation theory, resurgence, alien calculus, or Borel summation (optional `odeanalysis` can supply formal Stokes ray/sector geometry);
- a complete Hahn/log-exp transseries field with arbitrary well-based supports;
- a complete Hardy-field zero/sign/comparability decision procedure;
- general resolution of higher-dimensional singular implicit systems;
- arbitrary variable-coefficient Green operators without a hyperbolic asymptotic limit;
- complete branch-sheet/monodromy tracking for all special functions;
- guaranteed termination of explicitly user-requested general SymPy antiderivatives.

## Formal, certified, unknown

A **formal** result records a finite asymptotic construction. A **certified** result additionally carries a proved remainder theorem or replayable other symbolic certificate. **Unknown** means a necessary hypothesis could not be proved; it must not be interpreted as false.

## Choosing a supported limit family

| Family | Checked setting | Evidence | Limitation | Example |
|---|---|---|---|---|
| Simultaneous algebraic | Real variables and admissible local domains | Exact geometry or uniform bounds | Difficult parameter geometry may be unresolved | [Simultaneous limits](../examples/certified_limits.py) |
| Analytic cancellations | Gamma/polygamma at argument one, bounded elementary jets and real acos endpoints | Nonzero denominator order and retained Taylor remainder; positive even-order endpoint deficits | Order at most eight, numeric fixed coefficients and checked branch centers | [Limit semantics](limits.md) |
| Local complex phases | Selected logarithmic real poles, argument steps and inverse-sech projective poles | Principal phase signs, attained side germs and modulus lower bounds | General nested complex branches and arbitrary domains remain unresolved | [Limit semantics](limits.md) |
| Harmonic and gamma tails | Positive real tail; convergent harmonic order or fixed gamma shifts | Tail bounds, exact ratios and retained cancellation remainders | Variable parameters need uniform hypotheses | [Tail limits](../examples/special_function_limits.py) |
| Complex ray poles | Fixed finite nonzero direction; bounded rational degree/order | Laurent leading coefficient and eventual denominator avoidance | Does not prove an unrestricted complex-plane limit | [Ray branches](../examples/branch_limits.py) |
| Bessel and elliptic cuts | Selected numeric fixed orders and finite branch centers | Signed continuation and nonzero-denominator checks | Unproved zeros, sectors and variable order are declined | [Limit semantics](limits.md) |
| Oscillatory nonexistence | Recognized inverse phase maps with admissible approaches | Attained subsequences with different limits | Enclosures alone do not prove nonexistence | [Limits](limits.md) |
| Periodic discontinuities | Real affine arguments and verified local sides | Open-period subsequences and exact jump values | More complicated phase geometry needs a separate certificate | [Limits](limits.md) |

A function appearing in a supported family does not make every composite involving it supported. An unsupported result records the unproved condition or missing theorem; a timeout records a performance failure under the chosen budget.

## Executable workflows

| Workflow | Example | Proof boundary |
|---|---|---|
| Simultaneous limits and attained path conflicts | [Limit examples](../examples/certified_limits.py) | The approach domain and denominator avoidance matter. |
| Fixed complex rays and cut values | [Branch examples](../examples/branch_limits.py) | One ray does not certify every complex direction. |
| Harmonic tails and fixed-shift gamma ratios | [Special-function examples](../examples/special_function_limits.py) | Orders and shifts satisfy the stated fixed-parameter hypotheses. |
| Prefixes and omitted-term scales | [Expansion example](../examples/ordinary_expansion.py) | Formal coefficients and proved remainders are distinct outputs. |
| Cross-representation multiplication | [Algebra example](../examples/common_algebra.py) | The variable and approach regime must agree. |

Scorer real tails, local elliptic amplitude/nome germs, Owen endpoint poles and circular Piecewise rays also have bounded certificates. Their supported arguments, remainder guards and branch restrictions are described in [Limit contracts](limit-contracts.md).

The positive-denominator bounds also recognize signed absolute norms on their
original defined domain and exponential denominators through `exp(u) - 1 >= u`
for nonnegative `u`. A displayed sine–tangent quotient can be replaced by its
exact holomorphic removable extension before requesting a general series.

Finite real parameter families with a monomial denominator can use attained
signed monomial paths. Each path is nonzero in every coordinate, so the
parameter-independent denominator avoids poles. Divergent path values require
a leading coefficient whose sign holds for every admitted parameter value.
A failed search leaves the problem unresolved and does not prove existence.

### Local compositions and attained approaches

Real signed positive powers are continuous through a zero of a continuous
argument. The solver verifies the complete outer expression in an independent
real endpoint coordinate before composing this rule. Rational inner arguments
with a uniform radial bound also compose with analytic scalar heads near zero.
These rules preserve the original defined domain and reject restrictive
assumptions or unsupported branch charts.

For nonexistence, bounded elementary ray germs and binomial pole curves supply
explicit attained sequences. Every denominator must be nonzero eventually.
Opposite rays through an odd linear exponential pole separate decay from
signed growth. Agreement on the tested rays never proves existence; a failed
witness search leaves the result unresolved. Cancellation without a certified
next term also remains unresolved.

### Real discontinuities and local pole bounds

The reference vocabulary normalizes signed fractional parts, unit steps and
real-root charts from their exact definitions. Signed fractional parts use
truncation toward zero and apply componentwise to complex arguments. Unit
steps take the value one at zero. These definitions do not grant continuity
at jumps. Rational pole phases use attained integer subsequences away from
jumps; real step conflicts use attained approaches on opposite sides.

For plain real variables and fixed numeric centers, uniform bounds admit
positive continuous factors in polynomial norm estimates, inverse logarithmic
rates, bounded decaying exponentials and logarithmic divided differences.
Signed polynomial poles and real reciprocal inverse-function charts retain
original denominator holes and principal branches. The rules use bounded
expression size, polynomial degree and expansion width. Restricted domains,
unproved parameter signs and unrecognized branch charts remain unresolved.
