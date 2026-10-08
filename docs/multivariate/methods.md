# Multivariate limit methods

This guide collects the related mathematical mechanisms used by the multivariate limit engine. Public entry points are described in [Multivariate limits](../multivariate-limits.md).

## Automatic exp-log valuation fans

The multivariate engine extracts variable, power, logarithmic, exponential,
and nested exp-log growth atoms from an expression. Cross-variable atoms
generate symbolic log-magnitude comparability cells and valuation cones.
Rational positive specializations feed directly into the common weighted
blow-up atlas. Unsupported comparisons remain symbolic/UNKNOWN; candidate-fan
coverage means the regimes were represented, not that every comparison limit
was independently proved.

## Growth-scale comparability

`GrowthScale` is the proof-facing representation for one-variable Hardy/exp-log
scales inside a multivariate growth cell. `prove_growth_comparison` compares
absolute-value ratios exactly with symbolic limits. It certifies LESS,
EQUIVALENT, or GREATER only when the ratio limit proves the relation; otherwise
it returns UNKNOWN. `certify_exp_log_fan` uses these proofs to discharge
same-scale orderings while preserving independent-variable comparability cells.
This proves examples such as `exp(-1/x) << x**N << Abs(log(x))**(-M)` for
concrete positive N and M and recursively handles nested exp/log expressions
whenever SymPy can certify the exact ratio limit.

## Candidate and analytic certificates

The solver has proof-producing fast certificates for candidate-centered rational, analytic-equivalence, divided-difference, high-dimensional coercivity, radical, and oscillatory-DNE families.

## Candidate-centered rational certification

`candidate_rational_limit_certificate()` shifts the germ to the target and
works with the exact rational numerator and denominator.  It builds cheap local
coercivity models from either an exact positive/negative-definite quadratic
initial form (Sylvester criterion) or a weighted pure-even-power core.  For a
finite candidate `L`, it certifies that `numerator - L*denominator` has strictly
higher weighted order than the denominator lower bound.  This handles nonzero
candidates as well as zero limits.

For noncoercive linear/quadratic denominator varieties, the certificate builds
exact polynomial paths on and transverse to the leading zero variety and proves
DNE from two distinct exact path limits.  A separate fixed-sign argument handles
signed poles over simple semidefinite denominator varieties.  The signed-pole
proof uses the numerator's certified *radial lower-bound exponent*, rather than
its lowest monomial degree, so anisotropic numerators such as `x**2 + y**10`
are not incorrectly promoted to a pole.

## Certified analytic-equivalence rewriting

`analytic_equivalence_rewrite_certificate()` structurally certifies local real
analyticity, computes a tensor-product Taylor model, and retains an explicit
uniform remainder order.  For a positive weight vector `w`, omitting all powers
above coordinate order `N` gives a remainder of weighted order at least
`(N + 1) min(w)`.  A Taylor replacement is used only when that remainder is
strictly higher order than the certified denominator model.  Candidate
subtraction is then checked on the polynomial jets exactly.

This covers Taylor-equivalent functions inside sums instead of only exact
multiplicative sinc-like factors.  The existing removable-germ normalization was
also relaxed from a transcendental-atom-count-only progress test to a
lexicographic `(transcendental atoms, operation count)` measure, allowing safe
reductions such as `sin(h)**2/h` without permitting recursive growth.

## Generalized divided differences

`generalized_divided_difference_certificate()` recognizes analytic differences
with nonlinear or multivariate inner arguments.  First-order cases use the
analytic divided-difference theorem; `cos`/`cosh` at zero additionally use their
analytic dependence on the squared argument.  Numerator and denominator are
reduced to a common certified inner increment before taking the constant ratio.

Examples certified directly include

- `(x-y)/(sin(x)-sin(y)) -> 1`,
- `(x**2-y**2)/(cos(x)-cos(y)) -> -2`, and
- the three-variable sine/tangent cases whose inner differences both reduce to
  `z - x**2 - y**2`.

## Cheap high-dimensional coercivity

The weighted pure-even-core model avoids expensive geometry when a denominator
already supplies a direct anisotropic lower bound.  In particular,

`x**6/(ell**2 + t**2 + w**6 + x**2 + y**2 + z**2)`

is certified as zero with weights `(3, 3, 3, 1, 3, 3)` and denominator weighted
order `6`, instead of reaching the expensive fallback stack.

## Radical rationalization with domain tracking

`radical_rationalization_certificate()` performs exact conjugate
rationalization for `sqrt(A)-sqrt(B)` numerator/denominator differences.  It
tracks principal-real constraints `A >= 0`, `B >= 0`, the original deleted
denominator variety, and any caller-supplied relative domain.  A small exact
ray search then proves that the approach domain actually accumulates at the
target.  This prevents the rationalized continuous extension from being used
with an invalid path lying entirely on the original zero denominator.

The imported square-root quotients at `(0,0)` and `(4,3)` return `0` and
`1/4` respectively, including the explicit `x>0, y>0` relative-domain case.

## Oscillatory and analytic DNE certificates

`oscillatory_phase_sequence_certificate()` recognizes
`sin(1/Max(Abs(...)))` and the corresponding cosine form.  It constructs two
explicit reciprocal-phase sequences tending to the target whose values are
exactly `+1` and `-1`; the main limit pipeline treats any fast certificate
whose method ends in `_dne` as a DNE proof.

The same Taylor-remainder infrastructure also supplies
`analytic_jet_path_conflict_certificate()`.  It proves DNE either with a
perturbed path along a low-degree denominator-zero direction or with two
weighted rays when a coercive denominator has a nonconstant leading quotient.
This closes the two remaining analytic DNE cases in the frozen failure set.

## Validation contract

These methods are covered by focused theorem tests, paired prerequisite-failure tests, public mathematical corpora, and the exhaustive process-isolated reference corpus. Release status is determined by the current test and corpus gates described in `../testing.md`; this guide does not preserve historical pass counts or environment-specific validation snapshots.

## Multivariate certification

The multivariate certification pipeline handles unresolved corpus families in dependency order.

1. **Parameter-aware weighted orders.** Symbolic exponents at a vanishing base are no longer accepted by direct continuous substitution. The weighted-order certificate records the sufficient condition (for example `p < 19`) instead of proving an unconditional zero.
2. **Generalized logarithmic/product orders.** Positive algebraic order dominates logarithmic growth for polynomial monomials, absolute-value powers, and fractional powers such as `x**2*sqrt(Abs(y))*log(x**2+y**2)`.
3. **Angular/radial clusters.** A small exact ray atlas proves DNE only from two distinct finite cluster values. Radial signed poles run first, avoiding false DNE from different representations of the same infinite limit.
4. **Discontinuity and one-sided branches.** `atanh` endpoint approach from its real interior branch is certified, and `frac` of a rational pole gets explicit distinct phase subsequences.
5. **Singular polylogarithmic special-function germs.** `Y_1(z)=-2/(pi*z)+O(z*log|z|)` is propagated through removable products; a negative even reciprocal pole can then dominate the finite/polylog remainder.

## Multivariate limit families

Resolved frozen regressions:
- weak-sign / inverse-Sertöz: 6/6 terminate with exact signed-pole or DNE certificates;
- parameter stratification: relational specialization is fixed, and the direction-dependent parameter case has an exhaustive `a=1` proved leaf and `a!=1` DNE leaf;
- meromorphic difference rules: 2/2 use analytic divided-difference certificates;
- weighted Newton fallback: the DNE case reaches Newton-curve path conflict and the zero case reaches uniform angular optimization without being blocked by eager QE;
- isolated singleton regressions: `meromorphic_loja_01`, `analytic_taylor_02`, `sign_pole_pos`, `growth_log_02`, and `nonisolated_den_01` all terminate with the intended exact status/value in isolated reruns.

## Relative-growth valuations

Explicit relative-growth assumptions such as
`log(abs(y))/log(abs(x)) = alpha` or `y = x**alpha` generate valuation cones
rather than living in a separate assumptions subsystem. Polynomial/rational
orders use the induced weights, while exp/log nodes retain a generalized
order layer for later transseries refinement. Unsupported relations remain
UNKNOWN.

## Special-function local germs

The local-germ corpus covers varying-order
Bessel J/Y, erf/erfi, Gamma poles, inverse-trigonometric branch points/cuts,
and compositions whose inner germ lands on a special-function singularity.

The local germ registry contains first-order erf and erfi germs and the
simple Gamma pole, alongside the existing varying-order Bessel J germ.
Principal inverse-trigonometric cut detection handles direct top-level cases for
direct top-level asin/acos outside their real principal interval. Branch
endpoints themselves are not incorrectly treated as jumps: for example
acos(z) is continuous in value at z=-1 even though it is not analytic there.

The corpus distinguishes a singularity from a DNE result. A radial Gamma pole
may have a certified extended-real limit, while signed approaches to the pole
can prove nonexistence.
