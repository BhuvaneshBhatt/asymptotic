# Testing

The test suite separates mathematical behavior from implementation routing. This lets the solver gain a faster or stronger theorem without turning a correct result into a failing public-contract test.

## 1. Public mathematical contract

Table-driven public corpora record the expression, variables, target, domain/assumptions, expected status, and certified value or DNE outcome. `UNKNOWN` is a valid expected outcome when it describes a documented proof boundary.

Correctness tests check the mathematical conclusion, independently of which provider or valid witness is chosen. For nonexistence, replay the returned approaches: check convergence to the target, membership in the original domain, eventual pole avoidance and the distinct attained values. Theorem-family tests can pin a particular construction when that construction is the subject of the test.

## 2. Theorem-family tests

Each proof mechanism has focused tests for its hypotheses and conclusion. Pair a successful example with nearby cases where a prerequisite fails. These tests may assert provider/evidence details because dispatch is part of what they verify.

## 3. Soundness and metamorphic tests

Useful transformations include coordinate permutation, positive coordinate rescaling, algebraically equivalent forms, introduction of an unused coordinate, and domain-preserving translations. A transformation should preserve the mathematical result when its hypotheses say it should.

Parameter tests should include exceptional cells. A theorem valid for `a != 0` must not be promoted to an unconditional result when `a = 0` behaves differently.

### Limit and series relations

The focused transformation tests use exact arithmetic and bounded, reproducible Hypothesis generators. Each limit relation checks a proved status and a value, or distinct attained values for nonexistence. Baseline anchors prevent two equally wrong computations from satisfying the relation.

| Area | Transformation | Required condition and checked result |
|---|---|---|
| Finite limits | x=a*t*(1+c*t**2) | a>0, c>=0; a local inverse preserves both real sides and the limit |
| Finite limits | a*f+b+r, with r=x*sin(1/x) times a constant | f has a finite limit and r tends to zero; the result is a*L+b |
| Tail limits | x=1/t | x approaches positive infinity as t approaches zero from the positive side |
| Reciprocal limits | f maps to 1/f | The original limit is nonzero; the new limit is its reciprocal |
| One-sided limits | x=-a*t | a>0; positive and negative approach sides exchange, including signed infinities |
| Oscillatory nonexistence | Affine coordinate and nonzero affine output changes | The tested modular/trigonometric grammar retains two distinct attained limits |
| Analytic series | Add, multiply and take a nonzero reciprocal | Compare coefficients modulo t**N, including cancellation; a reciprocal requires a nonzero constant coefficient |
| Analytic series | Substitute t+a*t**2 into the truncated outer series | The chart vanishes to first order, so terms below t**N are preserved |
| Analytic series | Differentiate | A jet modulo t**N gives a derivative modulo t**(N-1); analytic germs supply the derivative remainder control |
| Sparse series | Increase the term budget | Restrict both expansions to the same exponent cutoff before comparing |
| Puiseux series | h=t**q | Positive coordinates preserve the real root; exponents multiply by q and coefficients agree |
| Series and limits | Divide the Taylor residual by its first omitted power | The resulting limit equals the first omitted coefficient |

The limit examples include sine, logarithmic and radical germs, rational tails, signed poles and coupled modular oscillations. The series examples exercise the native multiseries backend with its generic series fallback disabled. Addition and multiplication are compared using polynomial coefficient arithmetic, not another invocation of the series engine.

A term budget counts nonzero terms; it does not specify a power cutoff. For example, three terms of 1+t**3+t**7 contain the t**7 term, while the jet modulo t**3 is just 1. Differentiation tests use analytic functions: differentiating an unspecified O(t**N) remainder does not generally justify an O(t**(N-1)) remainder.

The tests include failed-hypothesis examples. Squaring the input of abs(x)/x covers only its positive side and changes a two-sided nonexistence result into 1. Squaring the output also identifies the two distinct values. Neither transformation preserves nonexistence without the relevant inverse or injectivity condition.

The coupled modular generator uses odd positive coordinate slopes and offsets retained by the period-two grammar. Even scaling can normalize Mod(2*t,2) to 2*Mod(t,1); that equivalent expression needs additional recognition before the current joint-phase certificate can handle it. Some integer shifts similarly separate the cosine phase from the normalized modular phase. These are coverage boundaries, not counterexamples to the mathematical invariance. General complex-sector and variable-order series need their own metamorphic generators and branch contracts.

Run these focused checks with:

```bash
pytest tests/limits/univariate/test_limit_transformations.py tests/test_series_transformations.py
```

## 4. Performance tests

Prefer deterministic operation budgets over fragile wall-clock thresholds. Track whether expensive CAD/QE or general simplification was entered, how many candidate certificates were attempted, and how cheaply a fast path declines. Wall-clock benchmarks remain useful for P50/P95/P99 comparisons on fixed corpora but should not be the only CI gate.

## 5. Documentation examples

Introductory examples are executable tests. When a public API changes, update the example and its test together so README/docs snippets do not drift from actual behavior.

## Commands

```bash
pytest
ruff check src tests examples
ruff format --check src tests examples
```

For focused multivariate work:

```bash
pytest tests/limits/reference/test_public_corpus.py
pytest tests/limits/shared/test_metamorphic_contract.py
pytest tests/limits/multivariate/paths_clusters/test_advanced.py
```

## Example/API contract

Files directly under `examples/` are public-API examples. They import documented names from the package root and are checked by `tests/test_example_api_contract.py`. Examples that demonstrate lower-level extension or instrumentation APIs live under `examples/expert/` so they cannot be mistaken for the primary interface.

Reference-corpus specifications and execution notes live in `tests/reference_cases/docs/`. Benchmark methodology and performance reports live in `benchmarks/docs/`. Keeping these assets outside `docs/` prevents validation history from obscuring the user guides.

## Multivariate test organization

Multivariate tests are organized by mathematical capability. The capability groups include parameterized angular optimization, parameter stratification, cluster-set semantics, projective geometry, complex germs, atlas coverage, asymptotic relations, and proof-boundary/soundness tests.

Regression inputs preserve a mathematical counterexample or corpus identifier. Group them with the capability they exercise.

## Release-grade structural gates

Before building a release artifact, run the syntax/collection and package-boundary gates in addition to the ordinary suite:

```bash
python -m compileall src tests examples benchmarks
pytest --collect-only
pytest tests/test_root_import_boundaries.py
pytest tests/test_instrumentation_event_registry.py
pytest tests/test_symbolic_route_budgets.py
```

The canonical Birkhoff–Trjitzinsky and Airy route-budget tests require zero unrestricted `solve`, `rsolve`, `limit`, and `integrate` fallbacks. This is a deterministic structural contract, not a wall-clock benchmark.

The exhaustive multivariate reference corpus is process-isolated and resumable. Run all 60 shards, then aggregate with `--strict`; a strict aggregate fails on missing/conflicting rows or `WRONG`, `ERROR`, and `TIMEOUT` outcomes. `UNKNOWN` and `KNOWN_GAP` remain visible for explicit mathematical triage rather than being rewritten as passes. See `tests/reference_cases/docs/reference-corpus-execution.md`.

Installed-wheel behavior is a separate release contract. Build the wheel, set `ASYMPTOTIC_WHEEL` to the resulting wheel path, and run `pytest tests/test_installed_wheel.py`; source-tree imports are not a substitute for this check.

## Sequential process-isolated complete suite

For long symbolic validation runs, `tools/run_isolated_suite.py` provides a
single-coordinator execution mode that never runs pytest children concurrently.
The default launches each test module in a fresh interpreter, records its log,
and atomically checkpoints progress after every child:

```bash
python tools/run_isolated_suite.py
```

An interrupted run resumes from `.isolated-suite/state.json`. Use
`--no-resume` to start again. A module that exceeds `--timeout` is automatically
replayed one collected test at a time, each in another fresh process. This
keeps the remaining module tests observable and identifies the individual
pathological case. For maximum isolation from the outset, use:

```bash
python tools/run_isolated_suite.py --isolation test --timeout 300
```

Selection arguments may be passed after `--`, for example
`python tools/run_isolated_suite.py -- tests/test_remainders.py`. The runner
disables third-party pytest plugin autoload and ignores inherited
`PYTEST_ADDOPTS` inside children so local plugins or shell configuration cannot
change release semantics. Logs and resumable state are diagnostic
artifacts and are not package source.

## Limit test layout

Limit tests are grouped first by semantic arity and then by proof/input family:

- `tests/limits/univariate/`: one-variable limit behavior and integration.
- `tests/limits/multivariate/`: simultaneous limits, with `algebraic/`, `analytic/`, `domains/`, `paths_clusters/`, and `special_functions/` subfamilies.
- `tests/limits/parameterized/`: assumptions, conditional limits, and parameter strata.
- `tests/limits/shared/`: public dispatch and cross-family contracts.
- `tests/limits/reference/`: imported/public reference corpora and regressions.
- `tests/limits/performance/`: planner, architecture, and performance contracts.

Tests should be placed by the mathematical behavior they specify rather than by the current provider that happens to prove the result. Provider-specific timing/dispatch assertions belong in `performance/`.

## User-facing contracts and documentation

`tests/test_documentation.py` runs the README Python example, executes the introductory scripts with budgets, checks local Markdown links and verifies absolute README repository links. `test_limit_invariants.py` checks affine local charts, positive ray rescaling, positive harmonic tails and even-root domain rejection independently of provider names.

Symbolic metrics verify cache sharing and dispatch costs without replacing library functions during a test. Builtin local-expansion providers form an immutable table; custom registration and privately injected boundary evaluators have direct contract tests. Correctness tests compare mathematical values and conditions; dedicated route tests inspect evidence and counters.

Useful additions are independent numerical branch samples, attained-sequence pole checks, parameter-boundary pairs and executable multiscale/remainder tutorials. Add a counterexample next to a successful theorem input so tests establish where the theorem must decline.

Property tests use a deterministic symbolic profile with 32 examples and no per-example wall-clock deadline. Module/process budgets and dedicated metrics tests cover performance; property tests check mathematical invariants without depending on machine load. Individual tests can set a smaller example count for expensive families.

## Reproducing the univariate audit

Run each reference row in an isolated child with a five-second parsing, solver and comparison budget and a twelve-second process budget:

```bash
python tools/audit_univariate_corpus.py --budget 5 --wall-budget 12 --jobs 4 --output audit/corpus-run.json
python tools/build_univariate_audit_report.py
python tools/classify_capability_gaps.py
python tools/prioritize_missing_theorems.py
```

The raw report records source and corpus hashes, exact IDs, elapsed times and observed outcomes. The reviewed report applies the independent mathematical analyses in `audit/reference-reviews.json`. Agreement with a reference is an observed pass; it does not independently prove every reference. Explicitly non-executable rows stay visible and are not sent to the solver. Capability gaps, reference defects and budget failures remain separate classifications.

Use longer isolated diagnostic runs to determine whether a timeout eventually produces a valid answer or reaches an unsupported branch. A successful diagnostic does not convert a five-second budget failure into a pass.

To collect longer diagnostic profiles for every recorded timeout, run:

```bash
python tools/profile_timeout_rows.py --budget 30 --wall-budget 40 --jobs 2
```

Each child saves a profile, a stack snapshot and its observed result under `audit/timeout-profiles/`. Profiling adds overhead; use the ordinary isolated audit to assess the five-second target. The diagnostic budget distinguishes eventual supported answers from unresolved capability paths.

## Specialized elimination and expansion checks

`tests/test_specialized_rules.py` checks definite and indefinite polynomial germs, weighted perturbations, attained zeros, exact rational coefficient residuals, cancellation, finite and infinite charts, denominator avoidance and curved domains requiring elimination. Performance comparisons use fresh processes against a frozen source tree, alternating before/after order and recording cold and warm evaluations. Result normalization is outside the measured interval. Successful rules and declined recognition must both be measured before extending dispatch.
