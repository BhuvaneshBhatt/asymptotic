# Exprtools migration baseline — Stage 1

Measured before any implementation change on 2026-10-18.

- exprtest commit: `068d37e9249f43a0a5233060e4411a4645313d5e`
- asymptotic commit: `82b55363fdffa29a6d899c9b9a2a24cfd5393563`
- Python 3.12.14, SymPy 1.14.0, python-flint 0.9.0, macOS arm64.
- exprtest complete suite: **321 passed, 10 skipped**, 29.93 s.
  The optional SciPy compatibility backend is absent.
- recurrence/integration selection: **39 passed**, 31.80 s.
  Selection: `tests/limits/univariate/test_recurrence_simplification.py`,
  `tests/limits/univariate/test_recurrence_families.py`,
  `tests/test_recurrence_resonance_weight_sector.py`, `tests/test_exprtest_integration.py`.
- Coverage: exprtest 3538/4248 statements (83.29%), 1528/2078 branches
  (73.53%), combined 80.08%; recurrence modules 465/552 statements (84.24%),
  312/382 branches (81.68%), combined 83.19%.

Coverage adds overhead; benchmark runs were separate, after tests completed.
The zero workload is the upstream 5000-expression benchmark: cold disables the
oracle result cache, repeated uses 125 expressions repeated 40 times. Recurrence
uses six workloads repeated 100 times (gamma, polygamma, Bessel, upper gamma,
Hurwitz zeta, and large-shift refusal), with a one-second cooperative budget.
Caches inside SymPy are warmed for recurrence. This is one machine/run, not a
statistical performance guarantee. Output expressions and verdict counts are
recorded for behavior comparison.

Reproduce with Python 3.12 in an environment containing the two source checkouts,
pytest, pytest-cov, and python-flint. Run the selections above with `--cov-branch`
and `--cov=exprtest` or both `--cov=asymptotic.recurrence_simplification` and
`--cov=asymptotic._recurrence_families`. The benchmark script expects sibling
`exprtest/` and `asymptotic/` checkouts; install both editable and run
`python benchmark_migration.py measurements.json` from their parent directory.

Raw benchmark measurements and machine/dependency metadata: `benchmarks.json`.
Raw coverage and JUnit test counts are retained in this directory.
