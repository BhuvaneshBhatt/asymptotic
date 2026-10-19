# Stage 5 validation

Python 3.12.14, SymPy 1.14, python-flint 0.9.0 on this macOS host.

- Full expression suite: **394 passed, 10 optional skips**.
- Focused Python 3.10.14 / SymPy 1.12 without flint: **25 passed, 6 skips**.
- Downstream recurrence/context/instrumentation guard suite: **37 passed**.
- Coverage: 80.44% (4471/5558 lines).

## Targeted median speedups

| Case | Direct cold | Direct warm | Public API |
| --- | ---: | ---: | ---: |
| quadratic_nested_zero | 68.3x | 121.0x | 65.60x |
| dependent_radicals_zero | 67.2x | 128.2x | 90.63x |
| rational_nested_zero | 73.4x | 130.7x | 101.68x |
| nearby_nonzero | 152.8x | 307.5x | 0.98x |

Each pair has identical certified verdicts. Public nonzero already has an early cheap proof and is essentially unchanged. Times are host-specific observations, with no CI wall-clock threshold.

The existing 5,000-expression portfolio also retained identical verdict counts: cold 0.5630s → 0.4857s, cached repeat 0.0993s → 0.0979s. A single run cannot establish a small general performance improvement.

Stage-4 minimum-version and unrelated asymptotic contract failures remain recorded in `docs/exprtools-migration/final`; this milestone does not claim the full asymptotic repository suite was run.

See `ALGORITHMS.md` for reproduction, proof, restrictions and rejected prototype.
