# Recursive growth-scale algebra

Growth scales are normalized recursive trees for positive constants, variables,
symbolic powers, products, exp/log layers, and general unary composition.
Products are flattened and positive constants are separated from scale-bearing
factors.

Comparisons return `GrowthComparisonProof` with a named proof rule and the
assumptions used. Structural rules certify symbolic power ordering,
symbolic inverse-log ordering, flat exponential versus positive powers,
positive powers versus inverse-log powers, nested negative exponentials, common
real powers, and invariance under positive constant factors. Missing sign or
ordering assumptions remain `UNKNOWN`.

Exact ratio limits remain a restricted fallback proof kernel; they are not
the representation or the sole comparison mechanism. The multivariate limit
entry point accepts an `assumptions=` condition and refines expressions and
domains before dispatch.

Method-specific comparison expectations are executable independently of the global limit solver, so scale-order proofs can be tested without depending on downstream multivariate resolution.
