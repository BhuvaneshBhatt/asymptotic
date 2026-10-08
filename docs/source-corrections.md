# Recovered source contracts

Both archived Stirling expressions contain `sqrt(2*pi)`. The corpus now uses that explicit multiplication in 0248 and 3456, and retains the original exported expressions beside the corrections. Their expected value remains positive infinity. Both rows are executable again; the five-second performance checks remain separate from source correctness.

The setup preceding the archived tests for 3533 and 3534 gives this real function:

```python
def f(x):
    return x if x > 0 else x + 1
```

The corresponding symbolic expression is `Piecewise((x, x > 0), (x + 1, True))`. Case 3533 approaches zero from below and has limit 1. Case 3534 approaches zero from above and has limit 0. At zero the function value is 1, while its two-sided limit does not exist.

The function setup remains in force through both the default and nonanalytic test pairs, then is cleared. The exported unevaluated outcomes were historical implementation checks, not mathematical limits. Recovering the local setup justifies these numerical expectations without defining every unspecified `f` in the package.

| Case | Archive member | Evidence |
|---|---|---|
| 0248 | `Limit/Bugs/Bugs-200000.mt` | Test block at line 181; square-root constant at line 182 |
| 3456 | `Limit/UnivariateLimit.mt` | Test block at line 575; square-root constant at line 576 |
| 3533 | `Limit/Bugs/Bugs-10000.mt` | Setup at lines 345–346; left-hand tests at lines 350 and 382 |
| 3534 | `Limit/Bugs/Bugs-10000.mt` | Same setup; right-hand tests at lines 358 and 390 |

The extracted source members were checked byte for byte against the original archive. The JSON report retains the archive and member fingerprints, source test IDs, original contracts and corrected rows.

## Verification results

3533 and 3534 pass with values 1 and 0. The corrected rows 0248 and 3456 pass with positive infinity, certified by retained Stirling remainders and the compatible differentiated expansion. The source translation and asymptotic proofs are separate obligations. See limit-contracts.md.

All 3,536 rows were reclassified after these corrections. Final counts: {"bad_reference_or_missing_assumptions": 68, "pass": 3331, "solver_wrong_result": 0, "timeout_performance_failure": 8, "unsupported_capability": 129}.
