# Worked algorithm traces

These traces describe the mathematical decisions that lead to a result. They are organized around inputs, candidates, hypotheses, and certificate verification rather than source modules.

## Newton--Puiseux turning point

Consider `F(x,y) = y^2 - x` near `(0,0)`.

1. Translate the requested independent and dependent limits to local zero coordinates. Here no translation is needed.
2. Compute `F_y(0,0)=0`, so the ordinary simple-root implicit theorem is not applicable.
3. The first nonzero derivative in `y` is second order, giving multiplicity two. `F_x(0,0)=-1`, so this is a certified turning point.
4. Balance monomials `y^2` and `x`: `2 p = 1`, hence `p=1/2`.
5. Substitute `y ~ c x^(1/2)`: the characteristic equation is `c^2-1=0`, yielding `c=±1`.
6. Lift each branch recursively and replay the residual in the original implicit equation.

If multiplicity cannot be established under the active assumptions, the package records it as unresolved rather than silently selecting a generic simple-root branch.

## Parameter-dependent multiplicity

For `(y-1)^2 + a(y-1) - x = 0` near `y=1`, translation gives `u^2 + a u - x`. The coefficient of the linear dependent term is `a`, so automatic stratification separates `a=0` and `a!=0`. The first stratum has multiplicity two and uses Newton--Puiseux scaling; the second has multiplicity one and remains on the ordinary implicit branch.

## Newton weight cones

For `y^2 - x - z^2 = 0`, assign weights `w_x,w_z>0` and dependent exponent `p`. Each support monomial induces a linear weighted valuation. Candidate regimes occur where at least two valuations tie for the minimum. The implementation constructs and canonicalizes these linear inequalities, separates chambers/walls, then replays the tied dominant balance inside each discovered regime.

## Green/Frechet certification

For `L = D^2 + x^-1 D - 1` at `+oo`:

1. Normalize the leading coefficient to one.
2. Prove normalized coefficient limits exist, obtaining `L0=D^2-1`.
3. Factor the limiting characteristic polynomial. Roots `-1,+1` give a hyperbolic exponential dichotomy.
4. Construct the Green particular for `L0` using a bounded exact primitive.
5. Substitute that candidate into the full operator `L`.
6. Prove the full defect is `o(R)` relative to the forcing and that the selected mode is separated from uncontrolled homogeneous modes.
7. Store the limiting coefficients, perturbations, dichotomy, defect, and replayable hypotheses in the certificate.

A root on the imaginary axis or a coefficient without a finite limit blocks certification.

## Composition remainder at a stationary point

For `F(z)=cos(z)` and an input perturbation `R=O(x^-1)` around `z=0`, the first derivative vanishes. The theorem searches the Taylor jet rather than stopping: `F''(0)=-1` is the first nonzero derivative, so the propagated error scale is `O(x^-2)`. The next Taylor contribution is checked to be asymptotically smaller before certification is issued.
