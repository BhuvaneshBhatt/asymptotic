# Symbolic execution policy

Internal asymptotic search and certification should not launch unbounded general
SymPy algorithms accidentally. The symbolic policy routes recurrence solving,
general algebraic solving, limits, and ordinary assumption queries through
`_symbolic_policy.py`, where expression-complexity budgets and instrumentation
apply.

The policy permits three direct call sites outside the central policy layer:

- `_power_simplify.py`: one assumption query used by the lower-level
  branch-safe power simplifier. Keeping it below `_symbolic_policy.py` avoids a
  dependency cycle; the helper already performs its own bounded structural
  checks.
- `multiseries.py`: explicit antiderivative operations requested by the user.
  These are computational operations, not hidden certification/search
  fallbacks; inability to integrate remains visible in the returned object.
- `nested.py`: the corresponding explicit nested-form antiderivative operation.

No direct `sp.rsolve`, `sp.solve`, or `sp.limit` remains elsewhere in
`src/asymptotic`. Any direct call to these generic engines must
be either routed through `_symbolic_policy.py` or documented here with a specific architectural justification.
