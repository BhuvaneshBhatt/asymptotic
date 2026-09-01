# Testing strategy

The suite is layered by the type of regression it is intended to catch.

## Behavioral root-API contracts

Every root export is named in at least one behavioral test outside the import/docstring contract. `test_public_api_behavior_coverage.py` enforces that rule automatically. Result objects are tested through invariants such as certificate replay, residual reconstruction, branch count/multiplicity, or remainder semantics rather than formatting.

## Property and metamorphic tests

Hypothesis tests cover commutative argument ordering, algebraically equivalent input forms, dummy-symbol renaming, positive coordinate rescaling, increasing truncation budgets, transseries differentiation, and reversion round trips. These tests are deterministic (`derandomize=True`) and do not impose fragile per-example deadlines.

The public operator layer additionally checks cross-API identities and relation
lattice laws: probability agrees with expectation of an indicator on exact
cases; relation aliases agree with the `little_o`/`big_o`/
`same_order` aliases; requesting one more asymptotic term preserves the already
computed prefix; and adding valid assumptions may resolve an unknown relation
without changing a certified result under the same mathematical context.

## Negative certification

`test_negative_certification.py` deliberately violates one theorem hypothesis at a time. Expected outcomes are `UNKNOWN`, not crashes and not false certification. Cases include oscillatory denominator zeros, a rational pole, singular composition, center-spectrum Green operators, nonconvergent coefficients, a degenerate inverse derivative, and nonlinear lifting with zero linearization.

## Reference corpus

`tests/reference_cases/` is a durable mathematical corpus. Every case declares one expected capability status: `CERTIFIED`, `FORMAL`, or `UNKNOWN`. The status describes the strength expected from the package; the executable check verifies the corresponding mathematical outcome.

## Documentation examples

The scripts in `examples/` are imported and executed by `test_documentation_examples.py`. Documentation therefore links to executable examples rather than maintaining untested copies of larger workflows.

## Dependency/backend matrix

Release testing should cover Python 3.10--3.13, required `exprtest`, optional `python-flint`, and optional `odeanalysis` where available. Backend-specific tests should assert mathematical contracts, not implementation-dependent expression formatting.

## Performance regressions

Performance is tested structurally as well as temporally. `asymptotic.instrumentation.symbolic_metrics()` records bounded-policy calls and expensive general fallbacks without patching SymPy at runtime. The PR smoke test asserts generous degradation and fallback ceilings; scheduled benchmarks retain detailed timing and memory observations.

Long-lived symbolic tests also exercise repeated discrete lifting in one interpreter. Internal Newton/BT coordinates are module-private reusable `Dummy` objects: they remain collision-safe but do not create a fresh family of process-global SymPy cache keys for every recurrence. The test harness still clears SymPy's global cache between modules to keep unrelated symbolic workloads order-independent; correctness must not rely on that clearing.

`test_instrumentation_event_registry.py` also parses the source and requires
every literal `record_symbolic_event(...)` name to have a declared
`SymbolicMetrics` counter. This prevents instrumentation-only failures when a
new symbolic route is added but its counter declaration is forgotten.

## Semantic contract matrix

Primary algorithms should cover the Cartesian product that is relevant to
their domain: finite/translated/infinite endpoint, exact/certified/formal/
unknown status, sufficient/insufficient assumptions, and branch/domain choice.
The suite does not require every meaningless combination, but a new root API is
expected to include at least one success, one genuinely asymptotic case, and
one conservative failure/unknown case.

## Release and namespace-boundary gates

Narrow unit batches can miss repository-boundary failures: repository examples, benchmarks, or documentation
can retain imports that no longer exist at the package root.  Release CI should
therefore run these cheap gates before expensive mathematical tests:

```text
python -m compileall src tests examples benchmarks
pytest --collect-only
pytest tests/test_root_import_boundaries.py
```

`test_root_import_boundaries.py` parses repository Python and executable-looking
Python fences in `README.md`/`docs/`; every `from asymptotic import NAME` must be
a member of the curated primary `__all__`.  Expert names must be imported from
their defining submodules.

The release artifact job should additionally set `ASYMPTOTIC_WHEEL` to the built
wheel and run `tests/test_installed_wheel.py`.  That test installs the wheel into
an isolated target, verifies the hard root boundary, and smoke-tests the Airy
entry point from the installed artifact rather than from the source checkout.

## Performance-shape tests

Wall-clock thresholds are deliberately coarse.  The primary regression contract
is structural: representative cold, warm, and cache-polluted runs must stay
within bounded general-SymPy fallback counts.  Deep BT and Airy regressions also
assert zero unrestricted solve/rsolve/limit/integrate fallbacks for their canonical
reference cases.
