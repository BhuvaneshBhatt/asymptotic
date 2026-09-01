# Discrete asymptotic scales and recurrence lifting

Discrete asymptotics studies sequences whose natural comparison scales are not
limited to ordinary powers of the index. A linear recurrence can have solutions
combining factorial growth, exponentials, stretched exponentials, powers,
logarithms, and fractional correction lattices. For a linear recurrence

\[
\sum_{j=0}^{r} a_j(n)y_{n+j}+f(n)=0,
\]

with rational or polynomial coefficient functions, the native recurrence layer
uses scales of the form

\[
\Gamma(n+1)^\kappa\lambda^n e^{\Phi(n)}n^\theta
(\log n)^\beta
\left(1+\sum_{m\ge1}c_m n^{-m/D}\right).
\]

The factors have separate roles:

- `kappa` records factorial or inverse-factorial growth;
- `lambda` records an ordinary exponential scale;
- `Phi(n)` records a stretched-exponential phase such as `c*sqrt(n)`;
- `theta` and `beta` record power and logarithmic prefactors;
- `D` is the ramification denominator of the correction lattice.

The native implementation is algorithmic: named sequences are useful tests, but
factorial and stretched-exponential factors are derived from recurrence
geometry rather than looked up from a table.

## Normalizing the recurrence

`linear_recurrence_data()` converts an equation to normalized scalar linear
recurrence data with integer shifts. Rational coefficient denominators are
cleared, the smallest shift is translated to zero, and the inhomogeneous
forcing is retained in the same normalization. Every nonzero shift coefficient
is stored once as exact polynomial metadata:

- normalized shift;
- exact polynomial expression;
- `Poly` object;
- degree;
- leading coefficient.

Newton construction, coefficient lifting, and residual replay reuse that
metadata. They do not repeatedly reconstruct polynomial objects from the same
expressions.


## Inhomogeneous and resonant lifting

For nonzero forcing, the native solver first seeks a particular solution by
writing the forcing on a hypergeometric/rational shift scale and lifting a
Laurent multiplier. This covers first-order rational recurrences and simple
linear hypergeometric forcing without invoking a general recurrence solver.

If the reduced first-order multiplier operator is resonant, a bounded ansatz

\[
n^p\left(c_{\log}\log n+c_0+c_1n^{-1}+\cdots\right)
\]

is solved coefficientwise. Thus harmonic resonance such as
`a(n+1)-a(n)=1/n` produces the expected logarithmic particular expansion.
Higher-order/nested resonances remain conservative and may return unresolved.

## Primary discrete Newton polygon

Write

\[
d_j=\deg a_j.
\]

If a solution contains `Gamma(n+1)**kappa`, a shift by `j` contributes
asymptotically `n**(kappa*j)`. The effective Newton height is therefore

\[
h_j(\kappa)=d_j+\kappa j.
\]

A primary Newton edge occurs when the maximum height is attained by at least two
shifts. Its slope determines `kappa`. On its active shifts the leading
coefficients form the characteristic polynomial

\[
P_E(\lambda)=\sum_{j\in E}\operatorname{lc}(a_j)\lambda^j.
\]

The same construction discovers:

- ordinary exponential scales when `kappa = 0`;
- factorial growth when `kappa > 0`;
- inverse-factorial decay when `kappa < 0`.

### Worked factorial example

Consider

\[
y_{n+1}-ny_n=0.
\]

The coefficient degrees are `d_0=1` and `d_1=0`. Balancing heights gives

\[
1+0\kappa=0+1\kappa,
\]

so `kappa=1`. The edge characteristic polynomial is `lambda - 1`, hence
`lambda=1`. The next coefficient balance yields `theta=-1`, so the scale is

\[
\Gamma(n+1)n^{-1}=\Gamma(n)=(n-1)!.
\]

No factorial-specific lookup rule is required.

## Simple-root Birkhoff--Trjitzinsky lifting

For a simple nonzero characteristic root, the solver substitutes

\[
y_n\sim\Gamma(n+1)^\kappa\lambda^n n^\theta
\left(1+\frac{c_1}{n}+\frac{c_2}{n^2}+\cdots\right).
\]

Shifted Gamma ratios are reduced structurally to finite products before local
expansion. Once previous coefficients are substituted, a nonresonant
Birkhoff--Trjitzinsky coefficient equation must be linear in the next unknown.
The implementation therefore extracts the degree-one polynomial coefficient and
solves it directly; it does not invoke a general symbolic equation solver in
this hot path. A nonlinear or degenerate coefficient equation is treated as a
signal that a more singular Newton level is required.

## Repeated constant-coefficient roots

A repeated characteristic root does not always require ramified asymptotics.
For a constant-coefficient recurrence, a root `lambda` of multiplicity `m`
has the exact Jordan chain

\[
\lambda^n,\quad n\lambda^n,\quad\ldots,\quad n^{m-1}\lambda^n.
\]

For example,

\[
y_{n+2}-2y_{n+1}+y_n=0
\]

has `(lambda-1)**2`, so the native backend returns the exact modes `1` and `n`.
These branches have exact residual order `oo`.

## Secondary Newton polygon and stretched exponentials

Variable-coefficient repeated roots can require a second asymptotic scale. Let
`t=1/n` and let `u` denote the leading logarithmic shift of an unknown phase:

\[
\frac{e^{\Phi(n+j)}}{e^{\Phi(n)}}\sim e^{ju}.
\]

After primary Newton normalization, expand the recurrence as monomials
`t**a*u**b`. If the characteristic root has multiplicity `m`, the unperturbed
edge begins with `u**m`. A secondary Newton balance compares

\[
t^a u^b \quad\text{with}\quad u^m.
\]

A balanced monomial gives

\[
q=\frac{a}{m-b},\qquad \alpha=1-q.
\]

Writing `u ~ v*t**q` produces a secondary edge polynomial in `v`. If a simple
nonzero root is found, `v=c*alpha` and therefore

\[
\Phi(n)=c n^\alpha.
\]

This is the source of stretched exponentials such as `exp(c*sqrt(n))`.

### Worked ramified example

For

\[
ny_{n+2}-2ny_{n+1}+(n-1)y_n=0,
\]

the primary characteristic polynomial is `(lambda-1)**2`. Secondary expansion
contains the balance

\[
u^2-t=0.
\]

Thus `q=1/2`, `alpha=1/2`, and `v=+/-1`. Since `v=c*alpha`, the two phases are

\[
\Phi_\pm(n)=\pm2\sqrt n.
\]

The native lift then works on the half-power lattice `n**(-1/2)` and obtains

\[
y_n^{\pm}\sim
n^{-1/4}e^{\pm2\sqrt n}
\left(
1\mp\frac{65}{48}n^{-1/2}
+\frac{5737}{4608}n^{-1}+\cdots
\right).
\]

The correction lattice is represented by `branch.lattice_step = 1/2` rather
than flattening the branch into an ordinary inverse-power series.

## Residual order and replay

Every finite native branch records a measured `residual_order`. The recurrence
is divided by the branch's primary Newton scale and expanded in the branch's
local coordinate

\[
s=n^{-1/D}.
\]

If the first nonzero normalized residual term is `s**k`, then

\[
\text{residual_order}=k/D.
\]

Exact branches use `oo`. `branch.replay_residual(data)` reconstructs the
normalized shift ratios and measures the order again. It therefore checks the
returned branch rather than trusting stored metadata.

For ramified branches, phase ratios are expanded from the exact monomial phase
`c*n**alpha`; shifted Gamma factors continue to use finite products. The replay
path and the lifting path therefore agree on the discrete scale algebra.

## Interaction with `asymptotic_rsolve()`

`asymptotic_rsolve(..., method="auto")` prefers an exact recurrence solution
when one is available. `method="native"` directly requests the discrete Newton
backend. Native branches do not infer connection constants from finite initial
conditions; if the exact route cannot resolve those constants, the native route
does not pretend that a local asymptotic basis determines them.

A native result is `FORMAL`. Its `branches` carry the resolved fundamental
scales and replayable finite residual information. `limitation` is `None` only
when the number of resolved branches equals the normalized recurrence order.

## Supported recurrence geometry

| Recurrence feature | Native behavior |
| --- | --- |
| Linear homogeneous scalar recurrence | Supported |
| Integer shifts | Supported |
| Rational coefficient functions | Supported after denominator clearing |
| Polynomial coefficient functions | Supported |
| Primary factorial/exponential Newton scales | Supported |
| Simple characteristic roots | Inverse-power Birkhoff--Trjitzinsky lift |
| Repeated constant-coefficient roots | Exact polynomial Jordan chain |
| Repeated variable-coefficient roots | Secondary Newton analysis |
| Simple stretched-exponential secondary phases | Supported |
| Reciprocal-integer fractional correction lattices | Supported |
| Measured residual replay | Supported |
| Initial-condition connection constants | Exact route only |
| Inhomogeneous native recurrence lifting | Unsupported |
| Noninteger shifts | Unsupported |
| Nonlinear recurrences | Unsupported |
| Selected repeated secondary roots | One tertiary stretched-exponential Newton descent with residual replay |
| Repeated tertiary roots / deeper nested Newton degeneracy | Conservatively unresolved |
| Multiple nested stretched-exponential levels | Conservatively unresolved |
| General logarithmic resonance in repeated roots | Conservatively unresolved |

## Relation to MRV and Hardy analysis

Gamma and factorial factors are explicit MRV candidates only when the active
asymptotic context justifies their Stirling scale. The Gamma argument must be
proved eventually positive and tend to `+oo`. Pole-crossing, bounded, or
unsupported complex-sector arguments fall back to generic conservative growth
comparison.

MRV classes form a partial order. `most_rapid` is populated only if a class is
proved larger than every distinct competitor. An `UNKNOWN` comparison therefore
cannot turn traversal order into a false maximal class.

## Failure is informative

A missing native branch is not automatically a bug. It can mean that the
primary edge is degenerate, the secondary phase polynomial has unresolved or
repeated roots, the required correction lattice is outside the supported
reciprocal-integer form, or a further logarithmic/stretched-exponential Newton
level is necessary. In those cases the solver returns an incomplete formal
basis or reports the recurrence as unsupported rather than inventing a scale.

For route selection, see [Algorithm selection](algorithm-selection.md). For the
meaning of unresolved results, see [Understanding `UNKNOWN`](unknown-results.md).

## Repeated secondary roots and tertiary descent

A repeated root of the secondary characteristic polynomial is not automatically a
logarithmic resonance.  The first uncancelled recurrence residual can instead
force another stretched-exponential scale.  The current BT implementation
therefore performs one additional Newton descent before ordinary power/log
transport.

Consider the fourth-order operator

\[
n^2(E-1)^4-2n(E-1)^2+1.
\]

For the primary root `lambda = 1`, the first stretched-exponential scale is
`exp(v*sqrt(n))`.  The secondary characteristic polynomial is

\[
(v^2-1)^2,
\]

so the secondary roots `v=1` and `v=-1` are both repeated.  Rather than treating
that multiplicity as an ordinary power transport equation, the implementation
measures the first surviving residual valuation and constructs a tertiary
Newton edge.  This produces the four phase branches

\[
2\sqrt n \pm 2\sqrt2\,n^{1/4},
\qquad
-2\sqrt n \pm 2\sqrt2\,i\,n^{1/4}.
\]

For this example all four branches have the power prefactor `n**(3/8)` and a
`1/4` correction lattice.  `replay_residual()` independently reapplies the
normalized recurrence and verifies the measured residual order.

```python
import sympy as sp
from asymptotic.discrete_scale import (
    birkhoff_trjitzinsky_branches,
    linear_recurrence_data,
)

n = sp.symbols("n", positive=True, integer=True)
a = sp.Function("a")
recurrence = (
    n**2 * (a(n + 4) - 4*a(n + 3) + 6*a(n + 2) - 4*a(n + 1) + a(n))
    - 2*n * (a(n + 2) - 2*a(n + 1) + a(n))
    + a(n)
)
data = linear_recurrence_data(recurrence, a(n), n)
branches = birkhoff_trjitzinsky_branches(data, terms=1)
assert len(branches) == 4
assert all(branch.replay_residual(data) for branch in branches)
```

This is deliberately a bounded hierarchy: if the tertiary characteristic root
is itself repeated, or if the Newton data fall outside the implemented finite
power lattice, the solver remains conservative rather than guessing the next
scale.
