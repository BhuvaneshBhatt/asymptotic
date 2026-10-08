# Documentation

`asymptotic` provides proof-aware symbolic asymptotics. Start with the user guides below; implementation and validation material is kept separately so the conceptual documentation stays navigable.

## Start here

- [Introduction to asymptotics](introduction-to-asymptotics.md): concepts, techniques, and applications before choosing a solver API.
- [Getting started](getting-started.md): installation, first computations, and result objects.
- [Expansions and error terms](expansions.md): representations, truncation and remainder propagation.
- [Limits](limits.md): univariate and joint multivariate limits, domains, and DNE.
- [Multivariate limits](multivariate-limits.md): the multivariate workflow and links to geometry topics.
- [Certification](certification.md): `PROVED`, `DOES_NOT_EXIST`, `UNKNOWN`, and evidence.
- [Capabilities and limitations](capabilities.md): the supported mathematical surface and known boundaries.
- [User guide](user-guide.md): broader asymptotic workflows.
- [Unified perturbation workflow](perturbation-dispatch.md): automatic method selection, validation, and evidence.

- [Documentation and example checks](documentation.md)
- [Executable examples](../examples/README.md)

## Mathematical guides

- [Asymptotic sums](asymptotic-sums.md)
- [Discrete asymptotics](discrete-asymptotics.md)
- [Probability asymptotics](probability-asymptotics.md)
- [Statistical transforms](statistical-transforms.md)
- [Matched asymptotics](matched-asymptotics.md)
- [Multiple scales](multiple-scales.md)
- [Perturbation hierarchies](perturbation-hierarchies.md)
- [Growth comparison](growth-comparison.md)
- [Function properties](function-properties.md)

## Multivariate geometry

The detailed multivariate material is grouped by mathematical role instead of implementation history:

- [Geometry](multivariate/geometry.md): blow-ups, angular images, strata, and local charts.
- [Cluster sets](multivariate/cluster-sets.md): directional and correlated images, coverage, and complex clusters.
- [Domains and parameters](multivariate/domains-and-parameters.md): relative domains, parameter cells, and conditioned limits.
- [Branches and resolution](multivariate/branches-and-resolution.md): branch geometry, monodromy, and recursive resolution.
- [Methods](multivariate/methods.md): analytic, valuation, exp/log, and special-function limit methods.
- [Special-function expansion certificates](special-function-expansions.md): sectors, parameter hypotheses, branches, and remainder bounds.

## Specialist guides

- [Algorithm selection](algorithm-selection.md): how high-level APIs choose among exact, structural, multivariate, recurrence, ODE, and saddle routes.
- [API classification](api-classification.md): primary, expert, and internal interfaces.
- [Function properties](function-properties.md): branch-aware property and domain reasoning.
- [Probability asymptotics](probability-asymptotics.md): expectation, probability, and Laplace workflows.
- [Asymptotic sums](asymptotic-sums.md): Euler–Maclaurin, recurrence, Mellin/Poisson, and discrete saddle methods.
- [Discrete asymptotics](discrete-asymptotics.md): factorial scales, discrete Newton geometry, and recurrence lifting.
- [Power simplification and branch semantics](power-simplification.md): branch-sensitive power transformations and their proof obligations.

## Reference and contributor material

- [API](api.md) and [API reference](api-reference.md)
- [Architecture](architecture.md)
- [Testing](testing.md)
- [Releasing](releasing.md)

Reference-corpus specifications live under `tests/reference_cases/docs/`; benchmark methodology and reports live under `benchmarks/docs/`. They are validation assets, not user guides.

- [Asymptotic regime selection](asymptotic-regime-selection.md)
