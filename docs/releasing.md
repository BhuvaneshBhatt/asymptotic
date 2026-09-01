# Releasing to PyPI

PyPI publication is handled by `.github/workflows/publish.yml` using PyPI
Trusted Publishing. No long-lived PyPI API token is stored in the repository.

A release tag must exactly match the version in `pyproject.toml`, using the form `vX.Y.Z`. The workflow then reruns the static checks and every release shard across
Python 3.10–3.13, builds both the wheel and source distribution, runs `twine check`,
installs/tests the built wheel through the installed-wheel contract, and only
then exposes the build artifacts to the publish job.

The GitHub repository must have an environment named `pypi`, and the PyPI
project must trust that GitHub environment/repository for the workflow. The
publish job receives only `id-token: write` plus read-only repository access.
If any validation, build, metadata, or installed-wheel step fails, publication
does not run.

Normal pushes and pull requests continue to use `tests.yml`, including the
full Python 3.10–3.13 compatibility matrix. The tag workflow deliberately
repeats the supported-version release matrix so the exact tagged source is
validated inside the same workflow that creates and publishes its distributions.
