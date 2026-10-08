# Special-function expansion certificates

`local_series` keeps special-function formulas behind the same finite-prefix
contract used by other local limit proofs. A primitive provider records the
finite prefix, a real-path remainder order, and an `ExpansionCertificate`.
The certificate separates four facts that should not be inferred from the
formula alone:

- the complex argument sector in which the expansion is uniform on closed
  subsectors;
- parameter hypotheses, such as fixed finite Bessel order;
- the branch convention for fractional powers;
- an explicit asymptotic remainder bound.

A provider may expose a useful symbolic expansion while marking its hypotheses
unverified. Such an expansion is available for inspection, but
`adaptive_path_limit` will not use it as a proof certificate.

## Implemented families

| Family | Expansion regime | Certified sector / restriction |
|---|---|---|
| Bessel `J_nu`, `Y_nu` | fixed order, large argument | principal sector `-pi < arg(z) < pi`, away from the boundary |
| Modified Bessel `I_nu` | fixed order, large argument | `|arg(z)| < pi/2`, away from Stokes boundaries |
| Modified Bessel `K_nu` | fixed order, large argument | `|arg(z)| < 3*pi/2`, away from the boundary |
| Hankel `H1_nu`, `H2_nu` | fixed order, large argument | inherited fixed-order Bessel sector |
| Airy `Ai`, `Ai'` | large argument | single-exponential principal expansion away from `arg(z)=+-pi` |
| Airy `Bi`, `Bi'` | large argument | dominant single-exponential sector `|arg(z)| < pi/3` |
| Struve `H_nu`, modified Struve `L_nu` | finite argument near zero | principal power branch; convergent defining series |
| Struve `H_nu` | fixed order, large argument | Bessel `Y_nu` plus algebraic Struve expansion, `|arg(z)| < pi` |
| Modified Struve `L_nu` | fixed order, large argument | Bessel `I_nu` plus algebraic Struve expansion, `|arg(z)| < pi/2` |
| `erf`, Fresnel C/S, `Si`, `Ci` | selected large-real-argument regimes | provider-specific real or complex sectors |

The Bessel coefficients are generated symbolically from the standard fixed-order
Hankel coefficient polynomial. Airy coefficients are generated from their gamma
ratio formula, including the distinct derivative coefficients. Struve large-
argument expansions reuse the Bessel providers and add the independently
bounded algebraic remainder, so branch and parameter conditions remain visible.

## Deliberate boundaries

The catalog does not certify a formula across a Stokes line merely because the
same symbolic expression can be evaluated there. It also does not treat a
symbolic parameter as finite unless that hypothesis is known. Uniform
large-order Bessel/Airy expansions, turning-point expansions, and exponentially
improved Stokes smoothing require different theorem data and should be added as
separate providers instead of weakening the fixed-order certificates.

The separate [uniform and Stokes layer](uniform-special-function-expansions.md) covers large-order turning points and beyond-all-orders metadata.
