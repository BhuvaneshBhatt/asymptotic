# Exprtools migration: implementation and validation

Completed in the requested commit order: baseline, extraction, dependency
separation, then the simplification framework and independent refinement.

## Delivered code

- `exprtest` repository: independently built `exprtools/` distribution plus
  `exprtest` compatibility distribution; 36 predicate implementation modules
  match the original text exactly except one package-name docstring reference.
- `exprtools.recurrence`: native SymPy defaults, per-call family providers,
  generic transformation evidence, optional instrumentation callback. It imports
  no asymptotic modules. All existing recurrence safeguards are retained.
- `asymptotic`: explicit adapters retain original custom heads, limit evidence,
  event counters and recurrence API. Its zero oracle imports exprtools.
- New `MathematicalContext`, immutable `TransformationRegistry`, bounded
  dispatcher, conservative definedness checks, and independent `refine` rules.
  No core delegation to SymPy refine. Branch/domain counterexamples, bounded
  scopes, Piecewise ordering, invalid assumptions, coefficient expansion caps,
  certified-only custom proposals and resource limits have regression checks.

API and mathematical policies are documented in the exprtools README. The
pointwise dispatcher and lower-level meromorphic recurrence API intentionally
have different domain policies; structured results retain the original input.
Assumption reasoning and definedness checking are conservative and incomplete.
Time budgets are cooperative, not hard preemption of individual CAS/plugin calls.

## Tests and coverage

Primary environment: Python 3.12.14, SymPy 1.14.0, mpmath 1.3.0,
python-flint 0.9.0, pytest 9.1.1, macOS arm64.

| Check | Result |
| --- | --- |
| Stage 1 original exprtest suite | 321 passed, 10 optional SciPy skips |
| Stage 1 recurrence/integration selection | 39 passed |
| Stage 2 unchanged predicate suite | 321 passed, 10 skipped |
| Stage 2 unchanged recurrence/integration selection | 39 passed |
| Stage 2 added alias/pickle checks | 3 passed |
| Stage 3 unchanged recurrence/integration selection | 39 passed |
| Stage 3 added independence/adapter checks | 5 passed |
| Final predicate, compatibility, framework suite | **373 passed, 10 skipped** |
| Final recurrence/context/instrumentation integration selection | **53 passed** |
| New framework checks included in final suite | **49 passed** |

Final full exprtest/exprtools coverage: 4691/5764
statements (81.38%),
1987/2804 branches
(70.86%), combined
77.94%. This scope now includes the extracted
recurrence engine and new framework, so it is not directly comparable to the
predicate-only baseline. The separate recurrence/adapter integration run has
84.14% combined coverage. Original predicate algorithms
were not changed.

The recurrence integration selection adds adapter evidence checks, hardening
context/registry, polynomial zero context, symbolic instrumentation, event
registry, and transseries structural-zero tests to the original four modules.
The full asymptotic suite and its large reference corpus were not run locally.
The optional SciPy compatibility suite was not run.

### Minimum-dependency comparison

Python 3.10.14, SymPy 1.12, mpmath 1.3.0, pytest 7.0.0, no python-flint:

- Untouched source: **68 failed, 250 passed, 2 skipped, 11 deselected**.
- Migrated source: **68 failed, 302 passed, 2 skipped, 11 deselected**.
- Failure identities match exactly; all 52 added framework/compatibility checks
  passed. `compatibility-audit.json` records the comparison.

These legacy failures predate extraction and are retained to preserve the
existing algorithms. SymPy 1.14.0 is the validated primary environment. The
minimum-version suite is **not green** and no broad legacy-version correctness
claim is made. The new context supplies exact interval intersection fallback
for affine inequalities when SymPy 1.12 lacks the LRA option.

## Benchmarks

Separate uninstrumented runs, same workloads and environment; times in seconds.
A single before/after sample measures overhead, not a statistically established
speed change. Workloads were warmed identically as described in baseline notes.

| Workload | N | Baseline total | Final total | Baseline median (µs) | Final median (µs) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Zero oracle, result cache disabled | 5000 | 0.595673 | 0.560171 | 32.92 | 32.67 |
| Zero oracle, repeated | 5000 | 0.098175 | 0.098092 | 16.83 | 16.92 |
| Recurrence adapter | 600 | 0.394888 | 0.404708 | 582.15 | 610.83 |

Verdict counts and all six recurrence outputs match the baseline exactly.
Cold zero verdicts: 2332 true, 2668 false; repeated: 2280 true, 2720 false.
The separate `framework-benchmarks.json` measures seven new guarded/refusal
workloads, 100 repetitions each, without claiming a preexisting baseline.

## Reproduction and installation

From sibling `exprtest/` and `asymptotic/` repositories, install the extracted
package, compatibility package and consumer together:

```sh
python -m pip install -e './exprtest/exprtools[test,flint]' -e ./exprtest -e './asymptotic[test]'
(cd exprtest && python -m pytest --cov=exprtools --cov=exprtest --cov-branch)
(cd asymptotic && python -m pytest tests/limits/univariate/test_recurrence_simplification.py tests/limits/univariate/test_recurrence_families.py tests/test_recurrence_resonance_weight_sector.py tests/test_exprtest_integration.py tests/test_exprtools_adapter.py tests/test_hardening_context_registry.py tests/test_polynomial_zero_context.py tests/test_symbolic_instrumentation.py tests/test_instrumentation_event_registry.py tests/test_transseries_structural_zero_bypass.py)
cp exprtest/docs/exprtools-migration/baseline/benchmark_migration.py ./benchmark_migration.py
python benchmark_migration.py measurements.json
python -m build exprtest/exprtools --outdir dist
python -m build exprtest --outdir dist
python -m build asymptotic --outdir dist
python -m pip install dist/*.whl
```

Publish exprtools before releasing exprtest 0.2.0 and asymptotic 0.2.2. The
asymptotic CI installs the extracted package from a pinned implementation commit
until its first release. No package publication or merging is performed here.
Raw coverage, JUnit results, benchmarks and compatibility audit accompany this
report. Wheels and source distributions are supplied as downloadable artifacts.

## Additional repository contract audit

A supplementary selection of release/workflow, metadata, import-boundary,
suite-layout and repository-coherence checks was compared with untouched
asymptotic source: baseline **9 failed, 18 passed**; final **8 failed, 20 passed**
(the final run also includes the new adapter test). All eight remaining failing
identities are pre-existing. Synchronizing the 0.2.2 runtime/project version and
its version test resolves one baseline failure. No new failures were introduced.
These legacy checks mean the repository's full CI is not claimed green.
`asymptotic-contract-audit.json` identifies the failures. Asymptotic CI pins the
new exprtools implementation while its distribution awaits initial publication.

The new germ policy also refuses generic/rational cancellation through unresolved
fixed parameter poles; only the recurrence provider may use its guarded germ
proofs. Affine implication checks reject compact nonlinear structures before
constructing polynomials, preserving the coefficient expansion safeguards.
