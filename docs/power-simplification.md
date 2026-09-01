# Power simplification and branch semantics

SymPy's `powsimp(..., force=True)`, `expand_power_base(..., force=True)`, and
`expand_log(..., force=True)` are PowerExpand-like operations: they apply
identities without requiring the assumptions that make those identities valid
on principal complex branches.

For example, with unconstrained `a` and `b`,

```python
sqrt(a)*sqrt(b) - sqrt(a*b)
```

must not be normalized to zero.  In particular, a remainder certificate must
never use a forced power simplification on its exact error.

The package therefore distinguishes two semantics:

* **analytic expressions and proof obligations** use conservative power
  simplification (`force=False`);
* **formal monomial coordinates** may use PowerExpand-like rules when those
  rules are part of the representation's algebra rather than a claim about a
  principal analytic branch.

The positive local uniformizer used by ramified expansions does not justify
forcing identities for unrelated symbolic parameters.  SymPy can already use
the uniformizer's positivity with `force=False`.  Thus `log(a*b*t**2)` may
safely expose `2*log(t)` while keeping `log(a*b)` intact unless further
assumptions justify splitting it.

The mixed nonlinear-ODE and dominant-balance layers require special care:
their local scale factors are formal, while coefficients and residuals remain
analytic expressions.  The implementation therefore uses a three-way helper
policy: `analytic_powsimp()` for ordinary expressions, `formal_powsimp()` only
for expressions known in their entirety to be formal monomials, and
`mixed_powsimp(coefficient, monomial)` when a formal scale is multiplied by an
analytic coefficient.  The mixed helper canonicalizes the monomial first and
then recombines it conservatively, so a coefficient such as
`sqrt(a)*sqrt(b)` is never silently changed to `sqrt(a*b)`.

This boundary is enforced package-wide by source-policy regressions: no module
outside `_power_simplify.py` may invoke `powsimp(..., force=True)` or
`expand_power_base(..., force=True)` directly.
