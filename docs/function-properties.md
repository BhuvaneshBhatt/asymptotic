# Function properties and branch knowledge

The property registry is intentionally reviewed and conservative. It has explicit entries for `exp`, `sin`, `cos`, `sinh`, `cosh`, `tanh`, `log`, principal `sqrt`, `asin`, `acos`, `atan`, `gamma`, `erf`, `erfc`, `erfi`, `airyai`, and `airybi`. Unknown heads remain tri-state rather than borrowing unverified assumptions from generic simplification.

## Highest-return coverage additions

1. **Elementary branch families:** `asinh`, `acosh`, `atanh`, arbitrary rational powers, `Abs`, `sign`, and simple `Piecewise`. These directly improve composition, inversion, and real-domain reasoning.
2. **Gamma family:** `loggamma`, `digamma`/`polygamma`, beta, and reciprocal gamma. Pole lattices and real domains are structured and comparatively easy to review.
3. **Lambert W:** represent the branch index explicitly, with the branch point at `-1/e`, the principal/`-1` real branches, and the standard logarithmic cuts. This is especially valuable for asymptotic inversion.
4. **Classical ODE special functions:** Bessel J/Y/I/K, Hankel, Airy derivatives, and elementary parameter restrictions. These unlock branch/domain checks around differential-equation transformations.
5. **Integral special functions:** `Ei`, `E1`, `Si`, `Ci`, `Shi`, `Chi`, and error-function inverses, where cuts and endpoint behavior matter directly to asymptotic integration.

## Model improvements

The next model revision should distinguish **function head** from **branch instance**. A branch-aware property record should carry a branch/sheet identifier, branch-point loci, cut loci, jump or monodromy data when known, and parameter conditions under which those facts apply. Rational `Pow` should be normalized into this same branch model instead of treating only exponent `1/2` as a special `sqrt` head.

Properties should also become compositional. For `F(g(x))`, the branch engine should pull each singular/cut locus of `F` back through `g`, then combine that with the property decisions for `g`. The existing nested branch-safety walk is a good starting point, but it should return the pulled-back locus and the path/side information rather than only a Boolean decision.

For real asymptotics, add reviewed endpoint records: one-sided limits, monotonicity intervals, sign intervals, zeros/poles, and local invertibility. These facts can discharge nonvanishing and derivative-stability hypotheses without asking generic assumptions/limit engines.

## Testing standard

Every registry entry should have table-driven tests for real/complex domain membership, ordinary analytic points, every registered pole/branch point, points on each branch cut, points immediately off each side of a cut when the distinction matters, parameter-degenerate cases, and at least one nested pullback through a nontrivial inner function. Branch-family entries such as Lambert W and Bessel functions should test multiple branch/order parameters explicitly.

The registry should remain conservative: incomplete knowledge is represented by `None`/`UNKNOWN`, not by an optimistic default.
