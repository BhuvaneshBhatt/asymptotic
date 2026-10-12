# Order-targeted Laurent–logarithmic multiseries

The public `multiseries(..., order=0)` operation requests **all algebraic
powers at or below degree zero**. Algebraic order is inclusive: `order=1`
retains powers through `t**1`; `order=-1` retains negative powers through
`t**-1`. Every finite polynomial in `log(t)` at an included degree is retained.
`t` is the positive, right-hand uniformizer `x - point` (or `1/x` at +∞).

This solves a distinct task from the older demand-driven `terms=6` API, which
returns the first six surviving terms of a possibly nested multiscale
expansion. The default `terms` behavior is unchanged. Do not combine a
nondefault `terms` count or an explicit `scale` with `order`.

```python
import sympy as sp
from asymptotic import multiseries

e = sp.Symbol("e", positive=True)
expr = e**-2 + sp.log(e)/e + 3*sp.log(e)**2 + 7 + e*sp.log(e)
r = multiseries(expr, e, point=0, order=0, return_result=True)
assert r.is_complete_through(0)
assert sp.simplify(r.truncate() - (expr - e*sp.log(e))) == 0
```

## Completeness and mathematical strength

`return_result=True` yields an `OrderedMultiseries` from
`asymptotic.order_expansion`. Its `status` is either `"complete"` (the requested
formal coefficient through the order is exposed) or `"unknown"`. The
`remainder` field contains an explicit omitted expression and/or a SymPy
`Order(...)` term. `is_complete_through(n)` is true only for requested
orders `n` no larger than the completed order. `certified` is always false:
this API does not independently prove remainder bounds for arbitrary symbolic
special functions. An unsupported case returns an unknown record; its
`truncate()` raises `NotImplementedError` to prevent false conclusions.

## Bounded expansion

The implementation first recognizes exact finite Laurent–logarithmic
polynomials. Remaining local analytic and special-function expressions are
sent through a **single** SymPy expansion at the exact requested order,
including for dilogarithmic radical cutoffs. This deliberately avoids the
term-count fallback's repeated expansion-order doubling. The routine rejects
large expressions and out-of-range orders; a best-effort POSIX main-thread
timer bounds an individual expansion if no other alarm is active. On other
platforms/threads the order and operation limits apply, but no hard wall-clock
limit is guaranteed. A budget failure produces `status="unknown"`.

`order=0` is useful for Hadamard finite-part workflows because it obtains
negative Laurent powers, logarithmic divergences and the algebraic constant
in a single expansion. It does **not** itself extract a finite part: the
consumer must separate logarithmic and negative-power divergent terms under
the selected cutoff prescription.
