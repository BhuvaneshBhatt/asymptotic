# Uniform, turning-point, and Stokes expansions

The fixed-parameter provider catalog in `special_expansions.py` is not used for
large-order transition problems. `uniform_special_expansions.py` carries a
separate theorem contract for large parameters and beyond-all-orders data.

## Bessel large order

For positive real `nu` and `0 < z <= 1`, `bessel_j_large_order(nu, z)` and
`bessel_y_large_order(nu, z)` use the Olver coordinate

$$
\frac{2}{3}\zeta^{3/2}
 = \log\frac{1+\sqrt{1-z^2}}{z}-\sqrt{1-z^2}.
$$

The leading approximants are

$$
J_\nu(\nu z)\sim
\left(\frac{4\zeta}{1-z^2}\right)^{1/4}
\frac{\operatorname{Ai}(\nu^{2/3}\zeta)}{\nu^{1/3}},
$$

and the corresponding `Bi` expression with a minus sign for `Y`. The removable
turning-point limit at `z=1` is represented explicitly. `hankel_large_order`
uses the rotated-Airy form.

The full coefficient pairs are generated from the Debye recurrence. `debye_u`
implements the differential-integral recurrence for `U_k`; `debye_v` derives
`V_k`. `olver_coefficient` then constructs `A_k`, `B_k`, `C_k`, and `D_k`.
At `z=1` it switches to the analytic `q=sqrt(1-z^2)` coordinate so the
removable singularities are evaluated before cancellation destroys numerical
accuracy. The derivative APIs use the `C_k,D_k` hierarchy.

Certification currently covers positive real large order and the real
turning-point interval. The remainder field is the scale of the first omitted
Olver pair, not an interval-arithmetic error enclosure.

`bessel_j_transition(nu, a)` handles the local scaling
`nu + a*nu**(1/3)` and remains finite through the turning point.

## Stokes geometry

`StokesCertificate` records the singulant, Stokes rays, anti-Stokes rays,
smoothing variable, multiplier convention, and verification state. For Airy
Ai the principal Stokes set is represented by the rays `arg(z)=+-2*pi/3`; the
anti-Stokes rays are recorded separately.

`airy_stokes_expansion` produces the ordinary Poincare prefix and its Stokes
metadata. Its exponentially improved mode now uses the exact rescaled
terminant re-expansion. For `Ai`, the Poincare sum is truncated at
`floor(2*abs(xi))`; the retained terminant terms are followed by the certified
asymptotic residual scale `exp(-2*abs(xi))*xi**(-m)` in
`abs(arg(z)) <= 2*pi/3`. The terminant itself carries the smooth Stokes
transition; an error-function profile is no longer substituted for it.

## Terminants

`terminant(p, z)` and `rescaled_terminant(p, z)` implement the incomplete-gamma
representations used in exponentially improved Airy and Bessel expansions.
They are exact symbolic building blocks used directly by
`airy_ai_exponentially_improved` and `hankel_exponentially_improved`.
`ExponentialRemainderCertificate` records the optimal truncation index,
re-expansion depth, theorem sector, and residual asymptotic scale.

## Certification boundary

A formula is not certified merely because its symbolic continuation exists.
The current certified large-order Bessel route covers the real simple-turning-
point regime, including higher Olver coefficients and derivatives. The
terminant layer certifies the first exponentially improved Airy and fixed-order
Hankel re-expansions in their implemented theorem sectors. Complex Olver-domain
membership, hyperasymptotic levels beyond the first terminant re-expansion, and
uniform large-order modified-Bessel expansions remain separate work.

## Complex Olver continuation

The Bessel turning-point coordinate is continued from the positive real axis
to the plane cut along the negative real axis. For real `z > 1`, the
oscillatory continuation is evaluated explicitly so that `zeta < 0`; the
fractional powers in `A_k`, `B_k`, `C_k`, and `D_k` use the corresponding
Olver phase substitutions. `ComplexDomainCertificate` records the cut,
connected component, sector margin, continuation convention, and conjugation
symmetry. Points on the negative real axis are never certified by a
principal-value shortcut.

## Modified Bessel large order

`modified_bessel_i_large_order` and `modified_bessel_k_large_order` implement
the Debye expansions for `I_nu(nu*z)` and `K_nu(nu*z)`. Their derivative
variants use the same `V_k` recurrence. The principal quantities are

$$
p=(1+z^2)^{-1/2},\qquad
\eta=(1+z^2)^{1/2}+\log\frac{z}{1+(1+z^2)^{1/2}}.
$$

Certification uses continuation from the positive axis and the implemented
closed sector inside `abs(arg(z)) < pi/2`; the turning points `z=+-i` and
branch boundaries are excluded.

## Remainder bounds

`RemainderBound` distinguishes three independent statements:

- the asymptotic scale of the remainder;
- a theorem bound `C_N * Phi_N`;
- an actual numerical enclosure.

The uniform Olver and modified-Debye routes currently certify the first two
with a symbolic positive theorem constant when its computable variation bound
has not been constructed. `enclosure_certified` remains false in
that case. Existing terminant certificates continue to carry their
beyond-all-orders remainder theorem independently.

## Total-variation error control

The uniform-expansion remainder contract now uses explicit Liouville--Green
variation data. `VariationCertificate` records the path, endpoints, integrand
family, and a proved majorant. For Debye polynomials the path is represented in
`p=(1+z^2)^(-1/2)`: after parametrizing the straight segment from `0` to `p`,
the derivative polynomial is integrated coefficientwise. This gives a finite,
computable upper bound for its total variation.

`OlverErrorControl` combines the first error-control variation and the omitted
coefficient variation into an exponential majorant. Consequently the positive
real modified-Bessel routes now carry a finite `RemainderBound` with
`enclosure_certified=True`; there is no free `C_N` symbol. Complex
paths retain the same computable variation representation but are not upgraded
to numerical enclosures unless the path hypotheses are proved.

For Airy turning-point Bessel/Hankel expansions, `TotalVariation` represents
the exact family/path variation functional in the certificate. It replaces an
unspecified theorem constant while preserving the distinction between a
represented variation functional and an evaluated numerical enclosure.

## Generic terminants and hyperasymptotics

`ExponentialScale` represents a singulant, adjacent saddle, orientation, and
its Stokes and anti-Stokes curves. `terminant_reexpand` performs the common
least-term -> terminant re-expansion step. `hyperasymptotic_series` recursively
applies that operation to a bounded number of levels (three by default), with
each `TerminantLevel` carrying its own singulant, multiplier, truncation index,
and residual exponential scale. Airy and fixed-order Hankel terminants use the
same principal terminant implementation.

## Large-order zeros

`bessel_zero_large_order(nu, m)` inverts the Olver coordinate at the m-th Airy
zero. The derivative form uses Airy-prime zeros. The first uniform corrections
use `B_0` for function zeros and `C_0` for derivative zeros, making the zero
layer an independent check on the A/B/C/D coefficient conventions.

## Other turning-point families

`TurningPointFamily` and `TurningPointAdapter` separate transformed ODE
geometry from special-function normalization. The first adapters cover the
parabolic-cylinder Airy transition and leading large-parameter Whittaker and
associated-Legendre approximations. Their geometry, branch, turning points,
approximating family, and remainder scale are represented through the same
objects instead of separate family-specific result types.
