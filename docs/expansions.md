# Expansions and error terms

An asymptotic expansion answers a more detailed question than a limit: which terms matter, and how large is the error after stopping? Declare the approach point and symbol assumptions before choosing a representation. For positive `x` tending to infinity, `1/x` is small; near zero, `x` is small. A power of `1/log(x)` decays more slowly than a power of `1/x`, while `exp(-x)` decays faster.

## Choose a representation

Use `local_series` for a finite local germ and order term. Use `multiseries` for several ordered small scales and sparse coefficient calculations. Use `nested_series` when nested logarithms and exponentials are the useful structure. The common asymptotic algebra supports arithmetic across these representations when their variables and regimes agree.

## Read a truncation

The [ordinary expansion example](../examples/ordinary_expansion.py) expands `exp(1/x)` and keeps `1 + 1/x + 1/(2*x**2)`. Its remainder records the exact difference and an `O(1/x**3)` scale. `remainder.check()` replays the applicable asymptotic claim. The scale is an asymptotic order statement; it is not a numerical error bar valid at every finite input.

A coefficient calculation can succeed while a branch, parameter or remainder hypothesis remains unresolved. Inspect the result's certification status and obligations before using it as a theorem. See [certification](certification.md) for the proof protocol and [capabilities](capabilities.md) for its limits.

## Propagate an approximation

The [common algebra example](../examples/common_algebra.py) multiplies the expansion by `1 + 1/x`. It checks the resulting coefficients against the exact product, retains the exact residual, and verifies the propagated remainder. This illustrates why a displayed prefix should travel with its error information through subsequent calculations.

Composition and differentiation have additional proof requirements. A small remainder need not have a small derivative, and a substitution crossing a branch cut may invalidate the germ. A formal result remains useful for exploration, but a certified result requires the hypotheses of the corresponding operation.

## Bound the work

Request the terms needed for the question. High cancellation orders, many independent scales and general special-function sectors can require substantially more work. The [testing guide](testing.md) explains how correctness and responsiveness are checked separately.

## Regular rational jets

`generalized_series` computes regular rational germs directly before general symbolic expansion. For `P(z)/Q(z)` with rational coefficients and `Q(0)!=0`, coefficient comparison in `Q*A=P` gives a triangular exact recurrence. The nonzero constant denominator supplies an analytic remainder `O(z**N)`, including when cancellation removes every retained coefficient. Degrees and requested order are capped at sixty-four, with a structural operation budget. Singular denominators, symbolic coefficient parameters and nonrational germs use the existing expansion route. Finite targets and both real infinity charts preserve the original variable when reconstructing the result.
