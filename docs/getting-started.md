# Getting started

An asymptotic statement describes what happens as a variable approaches a point or grows without bound. A leading term describes the main scale. A remainder describes the error left after truncation. Those statements depend on an approach domain, and complex powers additionally depend on a branch.

## Define the mathematical domain first

Use `sympy.Symbol("x", real=True)` for a real variable. Use `positive=True` when the variable must remain positive. These are hypotheses, not hints to make a difficult expression simplify. For a restriction involving several variables, pass an explicit `domain` to a limit calculation.

The finite-point real limit is two-sided unless the variable's assumptions or a one-sided call restrict its approach. A simultaneous limit in `(x, y)` must hold on every admissible local approach. Checking axes or a few numerical paths cannot establish that.

## Select a result mode

`limit`, `one_sided_limit`, `complex_ray_limit` and `analytic_limit` return mathematical values by default. Their `return_result=True` mode exposes status, evidence, variables, target and domain. Expansion and solver workflows retain structured objects when scales, branches or remainders are part of the answer.

Inspect `UNKNOWN` before performing further arithmetic. It may indicate an unproved sign, missing parameter conditions, an unsupported special-function regime or a computation that cannot finish within a practical budget.

## Follow a small example

Run `python examples/certified_limits.py` from a source checkout. It compares an existing simultaneous limit with a path-dependent expression and a signed one-sided pole. Continue with `examples/branch_limits.py` to see why a fixed ray matters at a branch cut.

## Expand in a named regime

For positive `x` tending to infinity, powers of `1/x`, `1/log(x)` and exponentials may have very different sizes. Use `multiseries` when several such scales matter. Use `local_series` for a finite local prefix with its order term. A formal coefficient calculation does not automatically prove a uniform remainder.

## Navigate the documentation

- [API selection](api.md) groups operations by the question they answer.
- [Limits](limits.md) explains domain and branch semantics.
- [Capabilities](capabilities.md) states the proof boundary for each area.
- [Certification](certification.md) explains evidence and remainder replay.
- [Examples](../examples/README.md) provides a sequence of executable workflows.
