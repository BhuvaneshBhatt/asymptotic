# Representation-aware root operations

The root API uses representation-aware dispatch only when two call forms denote
the same mathematical operation and the package can preserve their semantics.
It does not overload names merely to reduce the number of functions.

## `integrate`

`integrate(representation, ...)` applies the common asymptotic calculus protocol
and preserves the native representation when possible. `integrate(expr,
variable, point=..., ...)` constructs a scale-aware finite asymptotic primitive
from a symbolic expression. These are two input forms of the same asymptotic
integration operation, so they share one public function.

The former `asymptotic_integrate` and `asymptotic_integral` spellings are not
extra public aliases.

## `differentiate`

`differentiate(representation, order=...)` remains representation calculus.
A proposed `differentiate(expr, variable)` overload was rejected: for a raw
SymPy expression that operation is ordinary exact symbolic differentiation and
adds no asymptotic analysis beyond `sympy.diff`. Users can differentiate first
and call `series`, or call `differentiate` on an asymptotic representation.

## `series`

`series(expr, variable, ...)` is the analysis dispatcher that constructs an
asymptotic representation. Existing representations already carry their scale,
point, remainder, and certification semantics; passing them back through
`series` would be conversion or refinement rather than the same operation.
Representation-specific refinement stays on the representation APIs.

## `solve` and `root`

`solve` computes the branch structure of an asymptotic algebraic system.
`root` is a deliberate one-variable convenience operation over that solver: it
adds prescribed-limit and branch-selection semantics and can return one branch
expression directly. They are related but do not dispatch on representation,
so merging them would make return behavior depend on incidental argument shape.

## `sum`

`sum(summand, variable, lower, upper, parameter=..., ...)` analyzes a
parameter-dependent discrete sum. There is no corresponding operation on an
existing asymptotic representation: adding or accumulating series objects is
ordinary asymptotic algebra. No representation overload is added.

## Rule

A root operation receives a representation overload only when both forms are
the same mathematical task, share compatible certification semantics, and can
be distinguished without guessing from expression structure.

## Representation algebra audit

The same rule was applied to the representation algebra itself.

`inverse` now accepts either `inverse(representation, ...)` or
`inverse(expr, variable, ...)`. Both forms solve the same functional-inversion
problem and use the representation's native context when one is already
available.

`compose(outer, representation, ...)`, `truncate(representation, terms)`, and
`leading_term(representation)` are generic root operations. Composition uses
the common asymptotic algebra and accepts either an outer symbolic expression
with `argument=...` or a callable such as `sympy.exp`. Truncation and leading
term extraction are representation queries with stable meanings across the
representations that implement them; unsupported representations raise
`TypeError` rather than being guessed or converted implicitly.

`valuation` and `normalize` remain representation methods. Their current
meanings depend on the storage/order model of a particular representation, so
a root dispatcher would suggest more uniform semantics than the package can
currently guarantee.

`exp` and `log` also remain representation methods. They are not implemented
uniformly across the representation family, and root-level names would be too
easily confused with ordinary symbolic `exp` and `log`. They can be revisited
if the common algebra later provides certified implementations for every
principal representation.

## Composition

`compose(outer, inner)` is the generic composition operation.  The first argument is the outer function or outer series; the second is the inner asymptotic representation, matching `(f \circ g)(x) = f(g(x))`:

```python
compose(sp.airyai, s)  # Ai(s)
compose(lambda z: sp.besselj(nu, z), s)  # J_nu(s)
compose(outer_series, inner)  # outer_series(inner)
```

Composition dispatches in this order:

1. exact transseries algebra for sums, products, powers, `exp`, and `log`;
2. registered local/special-function expansion providers;
3. certified uniform/turning-point providers exposed through `local_series`;
4. finite-center analytic Taylor composition.

Provider remainder orders and certificates are retained in the resulting
transseries metadata.  Series-on-series composition additionally requires the
inner germ to tend to the expansion point of the outer series and propagates
the outer series remainder after substitution.

A provider expansion must still be representable by the transseries algebra.
For example, a fixed-order Bessel expansion at positive infinity contains an
oscillatory trigonometric phase; until oscillatory monomials are part of the
common representation, that composition is rejected instead of storing an
opaque term with incorrect ordering semantics.
