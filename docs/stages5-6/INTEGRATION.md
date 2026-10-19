# Stage 6: downstream integration

A complete GitHub code-search inventory for `exprtest` under BhuvaneshBhatt
returned 64 matches (incomplete_results=false), identifying four additional
consumers: semialg, funcprops, odeanalysis and probstats. They were inspected and
cloned before edits. No other dependent repository was identified by this scan;
it does not claim to find private or dynamically generated third-party imports.

All default runtime oracles now use exprtools. Asymptotic's recurrence adapter
retains its signature, function conventions, LimitEvidence objects and event
counters. Active profiling tools now report exprtools. Legacy exprtest remains
a separate compatibility distribution sharing the new package's implementation,
classes and caches. Odeanalysis retains an optional legacy fallback, and
probstats retains existing injection keywords and their explicit legacy metadata.

## Validation

| Consumer | Baseline selected tests | Final selected tests |
| --- | ---: | ---: |
| semialg | 41 | 43 |
| funcprops | 23 | 26 |
| odeanalysis | 11 | 14 |
| probstats | 48 | 57 |

The final consumer total is 140 passing tests. An additional asymptotic
recurrence/context/adapter/cache suite passed 41 tests. The full expression suite
passed 404 tests after installing SciPy (the earlier stage-5 run had 394 passes
and 10 optional skips). Expression line coverage is 80.44%; the new structural-bound module has 98.13% line
coverage. All raw measurements are in the delivered coverage JSON. Changed
Python files pass each project's lint configuration. Wheels and source archives
for all seven distributions build and pass twine metadata checks.

The delivered `verify_installation.py` runs outside the source trees, verifies
installed-wheel provenance and tests the whole stack with exprtest absent and
legacy import actively blocked, followed by a separate compatibility-wheel run.
See individual consumer guides for exact suite file selections and baseline SHAs.
No full asymptotic reference corpus or complete consumer suites were run.
Historical stage-4 minimum-version and unrelated repository-contract failures
remain documented and were not silently treated as passes.

## Measurements and release order

Targeted stage-5 improvements and their proof are in `ALGORITHMS.md` and
`STAGE5-VALIDATION.md`. The final mixed zero/recurrence benchmark also runs after
integration and retains its outputs; timing results are observations on this
host, not CI thresholds. The recurrence workload comprises 600 calls over six
cases including a refused excessive shift.

Publish exprtools before new downstream releases, then semialg/funcprops before
their consumers. CI uses an immutable source pin pending the first exprtools
release. These changes are delivered for review; no merge or package publication
was performed. Downstream package versions are left for their release process.

See `CI-AUDIT.md` for the no-flint baseline comparison and CI follow-up.
