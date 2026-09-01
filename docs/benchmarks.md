# Benchmark suite

Benchmarks are separate from ordinary unit tests. They answer three different questions.

## Operation benchmarks

The `benchmark_*.py` files under `benchmarks/` use `pytest-benchmark` for representative multiseries, nested/transseries, reversion, implicit, multivariate, nonlinear-ODE, parameter-stratification, remainder, and Green/Frechet operations.

Run them with:

```bash
python -m pip install -e ".[benchmark]"
pytest benchmarks/benchmark_*.py --benchmark-only
```

## Scaling benchmarks

`benchmark_scaling.py` varies mathematical complexity (term count) rather than merely textual expression size. New scaling dimensions should be added when an algorithm has a meaningful degree, parameter-count, support-size, nesting-depth, multiplicity, or correction-depth parameter.

For an exact structural example, finite transseries multiplication forms `n*m` raw retained-term pairs before collection, so its candidate-generation stage is `Theta(n*m)` and `Theta(n^2)` for equal-size inputs. See [Computational complexity](https://github.com/BhuvaneshBhatt/asymptotic/blob/main/docs/computational-complexity.md) and the executable `examples/computational_complexity.py`.

## Stateful mixed workload

The mixed benchmark repeatedly runs heterogeneous algorithms in one interpreter. Its primary regression statistic is

`median(last two cycles) / median(first two cycles)`.

This catches process-state degradation that isolated microbenchmarks miss. It also records peak memory and symbolic-policy counters.

```bash
PYTHONPATH=src python -m benchmarks.run_benchmarks --cycles 6 --json benchmark.json
```

Absolute time is observational unless CI hardware is controlled. Prefer degradation ratios and structural ceilings such as general solver/limit/integrator counts.

## Instrumentation

```python
from asymptotic.instrumentation import symbolic_metrics

with symbolic_metrics() as metrics:
    result = some_asymptotic_operation()

assert metrics.general_solve_calls == 0
```

Instrumentation is opt-in, context-local, and does not patch SymPy. Normal code pays only the inactive context check at instrumented policy boundaries.

## CI tiers

- **Every PR:** unit/property tests, Ruff, generated-doc check, and `tests/test_benchmark_smoke.py`.
- **Scheduled:** the complete `pytest-benchmark` suite and stateful JSON report.
- **Release:** scheduled suite plus the supported Python/backend matrix and installed-wheel smoke tests.

The smoke benchmark intentionally uses generous thresholds. Its job is to catch catastrophic regressions, not declare a 10% host-to-host timing change a failure.

The metrics snapshot also includes zero-oracle calls, parameter strata created, Newton cones generated, and theorem certificates that end `UNKNOWN`. These counters help distinguish a real algorithmic change from host timing noise.

## Probability and expectation benchmarks

`benchmark_probability.py` measures a moving Gaussian tail and a concentrating
Gaussian expectation. The stateful mixed workload includes both, and symbolic
metrics record exact statistical reductions, density/PMF reductions, moving
domain transforms, interior saddles, and endpoint Laplace expansions.


The stateful mixed workload also exercises an order-4 degenerate saddle and a
scaled Gaussian lattice saddle. Instrumentation records degenerate-saddle,
global-certificate, Euler--Maclaurin, and discrete-saddle route counts so a
a refactor cannot silently replace these structural algorithms with a
general symbolic fallback.

## Cold, warm, and mixed profiles

Fast CI profiles exercise representative reversion/implicit workloads in three
states: after explicit cache clearing, immediately repeated with warm caches, and
after unrelated multiseries/transseries work has populated process state.  The
portable assertions are symbolic fallback ceilings; the timing ratio is only a
loose catastrophe guard.  The substantially deeper repeated-secondary BT case
is protected separately by route-budget and residual-replay tests so routine CI
does not pay for three copies of a deliberately expensive recurrence lift.
