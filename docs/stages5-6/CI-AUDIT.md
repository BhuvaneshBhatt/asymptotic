# CI follow-up audit

The primary validation environment passes all 404 expression tests with
python-flint and SciPy. GitHub CI also tests configurations with neither backend
and old dependency versions. A full green matrix is not claimed.

- Core and coverage jobs without flint report seven failed tests (probable
  numeric witnesses and two closed-expression regressions). A fresh local
  Python 3.12.14 / SymPy 1.14 environment with flint absent reproduced exactly
  those seven failure identities at stage-4 commit
  `ccc7b78744cc76551da742de80f8b55d8e01b31b` and at the final implementation:
  baseline **7 failed, 363 passed, 2 skipped, 11 deselected**; final
  **7 failed, 378 passed, 2 skipped, 17 deselected**. No new failing identities.
  See `noflint-comparison.json` and the two raw logs.
- Minimum SymPy/Flint jobs still fail in their older dependency matrix.
  The inspected minimum-flint job reports **60 failed, 333 passed, 11
  deselected** under SymPy 1.12/python-flint 0.5. Those exact CI failures have
  not been independently replayed against the original baseline in that
  environment. The earlier stage-4 report separately records its minimum-SymPy
  baseline/final comparison; this job is not presented as passing.
- Funcprops' existing Python 3.10 CI job fails because its project metadata
  requires Python >=3.11. The integration does not change that requirement.
- Asymptotic's first static job found one formatting issue in the newly added
  adapter regression. This was corrected and its two adapter checks pass.
  The initial all-tests aggregate was skipped/failed as a consequence of static
  failure; it is not evidence that its full test corpus was executed.
- Current-flint, structural-performance, benchmark, SciPy and package jobs
  succeeded in the inspected exprtest runs. Semialg integration, funcprops
  registry/package checks and probstats quality checks also succeeded at the
  snapshot. Other jobs were still running; no continuing monitoring is implied.

Raw GitHub status snapshots and inspected job logs are included in the task
artifact. Their states reflect retrieval time. The PR descriptions deliberately
report local/focused validation scope rather than claiming a full green CI run.
