# Asymptotic regime selection

Uniform asymptotic formulas are selected only after their scaling hypotheses are
proved. `select_asymptotic_regime` distinguishes five regimes:

| Regime | Required proof | Route |
| --- | --- | --- |
| fixed parameter, large argument | order independent of the limit variable | fixed-order provider |
| large order, scaled argument | `argument/order` independent of the large parameter | Olver/Debye uniform provider |
| turning point | `(argument-order)/order**(1/3)` independent of the large parameter | Bessel Airy transition |
| coalescing saddles | saddle analyzer supplies certified separation scaling | CFU Airy reduction |
| Stokes boundary | Stokes analyzer supplies certified singulant/distance scaling | terminant engine |

If none of these conditions is proved, the dispatcher returns `UNKNOWN`; it
does not fall back to a formula whose uniformity assumptions are unproved.

## Integration with limits

`local_series(expr, n, oo)` first asks the regime dispatcher whether a
uniform large-parameter theorem applies. Composite expressions replace all
compatible special-function atoms simultaneously and propagate their remainder
scales through the composite expression. Exact Hankel connection identities are
canonicalized before expansion so cancellation occurs before valuation.

`limit()` invokes this route before reciprocal/projective normalization at
infinity. Thus

`n**(1/3) * besselj(n, n + a*n**(1/3))`

is evaluated from the transition theorem, not from a fixed-order large-argument
series. The transition formula is uniform for fixed complex `a` in the standard
large-order sector. DLMF §10.19(iii) is the convention used by the implementation.

## Convention audit

`tools/audit_uniform_asymptotics.py` checks Debye and Airy coefficient
normalizations, turning-point coefficient values, the terminant normalization,
negative-axis branch convention, A/B/C/D indexing, documentation presence, and
production exception style. The specialist corpus in
`tests/data/uniform_asymptotic_reference_cases.json` contains numerical,
coefficient, regime-boundary, connection, recurrence, Wronskian, conjugation,
Stokes, optimal-truncation, ODE-residual, and zero cases.

## Proved asymptotic equivalence

The dispatcher proves limits such as `order/n -> 1`, `argument/n -> z`, and
`(argument-order)/n**(1/3) -> a`. Lower-order perturbations remain inside the
selected uniform expansion instead of causing a fallback to a fixed-order
formula.

## Composite uniform algebra

Compatible special-function atoms are placed on a common Bessel/Airy basis.
For polynomial composites, the remainder is propagated by expanding
`F(prefix + delta) - F(prefix)` exactly, including products of remainder
terms. Non-polynomial composites retain a first-order remainder estimate and
are not marked as algebraically certified.

Symbolic derivatives that SymPy rewrites through neighboring-order Bessel
recurrences are supported because orders asymptotic to the large parameter are
recognized by the same regime proof.

## Automatic saddle and Stokes geometry

`analyze_saddle_geometry` solves polynomial stationary-point equations through
degree four and proves coalescence from the logarithmic order of saddle
separation. Its `RegimeGeometry` can select the CFU Airy route.

`analyze_stokes_geometry` works from an `ExponentialScale`. It distinguishes
Stokes and anti-Stokes phase limits and verifies square-root angular boundary
scaling. `stokes_transition` retains the exact terminant multiplier and supplies
the leading erfc smoothing only after that geometry has been verified.
