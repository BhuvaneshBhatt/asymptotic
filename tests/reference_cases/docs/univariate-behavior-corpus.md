# Univariate behavior corpus

The univariate behavior corpus provides executable mathematical cases for the
main one-variable limit families supported by the package. Cases are grouped by
mathematical structure, not by solver implementation.

## Core cases

The core corpus contains 120 cases spanning log-exp normalization, generalized
series, singular compositions, algebraic and Newton–Puiseux series, exact
oscillatory cluster intervals, special-function behavior at infinity, and
assumption-sensitive sign and zero cases.

## Boundary cases

A separate 40-case corpus exercises harder forms in the same families, including
multi-frequency oscillation, special functions at infinity, singular generalized
series, and implicit systems. These cases prevent broad capability claims from
being inferred from only elementary examples.

## Acceptance criteria

A corpus run is sound when no case produces an incorrect mathematical result.
Unsupported cases remain explicit as missing capability, and metamorphic tests
check invariance under transformations such as positive rescaling and target
translation. Release gates combine these corpora with theorem, soundness, and
repository-coherence tests.
