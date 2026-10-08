# Preparing a release

A release should install independently of the repository and preserve the distinction between mathematical proof, nonexistence and unresolved capability. The required dependencies include SymPy 1.14 or later, exprtest, funcprops and semialg. semialg requires python-flint 0.9.0 or later; the backend is part of the dependency closure.

## Check the source

Run Ruff lint and formatting, compile and collect the test tree, and execute the maintained shards on Python 3.11 through 3.14. The ordinary and publish workflows must include every shard, including the extended modules. Run the optional ODE and optimization integrations with their declared dependencies. The ODE extra uses the published distribution requirement `odeanalysis>=0.1.0`; the adapter accepts its registered native record type and explicit schema-1 interchange objects.

Keep reference runs separate from ordinary tests. The univariate audit processes all 3,536 rows with fresh child processes, five-second solver/comparison budgets and twelve-second wall budgets. Reports must retain unresolved cases, raw disagreements and recorded mathematical reviews. The multivariate release gate separately checks all 1,318 rows and rejects wrong results, errors, timeouts, missing rows, duplicate records and invalid case identities. Unknown outcomes remain visible capability gaps. The univariate pytest table still compares raw expectations: a reviewed audit label does not make a contradictory expectation pass. Resolve its source assumptions and infinity conventions and repair supported timeouts before declaring that gate green.

A green source test run does not certify a wheel: the installed-artifact test is skipped until `ASYMPTOTIC_WHEEL` points to the actual build.

## Build and verify artifacts

```bash
python -m pip install -e ".[test,lint,release]"
ruff check .
ruff format --check .
python -m build
python -m twine check --strict dist/*
ASYMPTOTIC_WHEEL="$(find dist -name '*.whl' -print -quit)" pytest tests/test_installed_wheel.py
```

The source archive includes the tests, reference data, examples, documentation and verification tools. Build the wheel from that archive as well as from the checkout. Confirm that both wheels contain the same package files, license and runtime metadata. The wheel should contain the package and its distribution metadata, without tests, caches or local audit logs.

Install the wheel in clean environments on the supported Python versions and operating systems, then run the smoke checks without source-tree imports. Validate the minimum dependency stack and the current stack. A local test that shares already installed dependencies does not replace clean dependency resolution in CI.

## Review before tagging

1. Review any wrong-result reports and prove or repair every discrepancy in the supported release contract. Preserve unsupported mathematics as explicit unknowns.
2. Confirm the documented assumptions, branches, output types and practical budgets match the validated artifact. Remaining corpus defects do not become solver guarantees.
3. Check optional backend compatibility and wheel availability on Linux, macOS and Windows. Confirm documentation examples work from an installed package.
4. Confirm that package and runtime metadata agree, choose an unused release number, and summarize API changes and proof boundaries in the release notes. The current API has renamed and relocated specialist interfaces; users need an import migration note.
5. Confirm the PyPI trusted publisher matches the repository, `publish.yml` and `pypi` environment. Configure the environment's release approval and protect the release tag.

See the [PyPI trusted-publisher documentation](https://docs.pypi.org/trusted-publishers/adding-a-publisher/) for the publisher settings. The tag workflow checks that the tag matches package metadata, runs source and corpus gates, builds distributions, validates the installed wheel across the Python matrix, then publishes. It must publish the same validated artifacts.

Capability gaps alone do not block a bounded release with clear limitations. Unreviewed wrong results, a failing supported contract, broken installation or a failed release gate do.
