# Stage 5: evaluated algorithms and soundness

Stage 5 builds on stage 4 (`ccc7b78744cc76551da742de80f8b55d8e01b31b`).
The attached `rred.txt` was inspected before implementation: `radicalbase` and
`radinbase` (lines 207–357), `LiYapRec`/`LiYapBound` (967–1135),
`BFMSSRec`/bound (1298–1448), and `AlgebraicZT` (1664–1808).
The attachment is not redistributed; this is an independent implementation of
mathematical ideas, without copying its C source or floating point bounds.

## Evaluation and choice

The existing package already has minimal polynomials, shared number fields,
sparse towers, radical denesting and several polynomial separation bounds.
Duplicating those mechanisms would not add an independently useful path.
A factor-free square-class grouping prototype was also evaluated: its warm
median on two small radical sums was 0.81 and 0.87 times the speed of the
existing cached squarefree implementation. It was rejected. Its discovery
measurements are retained in `results/prototype-results.json`; they compare
prototype operations, not complete equivalent public APIs.

The adopted path computes a conservative structural norm bound with integer
arithmetic, then compares a rigorous Acb enclosure with that bound. Shared
radical DAG nodes contribute their index once. No factorization, minimal
polynomial, or primitive element is constructed. Unsupported inputs fall back
to the existing methods. General simplification can use the registered
`algebraic` transformation to reduce a certified closed radical identity to zero.

## Proof and branches

Inductively write each supported value as B/C, with B an algebraic integer and
C a positive integer. Store integer ceilings u,l such that every conjugate of B
has absolute value at most 2^u and C <= 2^l. Rationals initialize these bounds.
For addition use `(max(u1+l2,u2+l1)+1,l1+l2)`; for multiplication use
`(u1+u2,l1+l2)`. For a qth root, its numerator is a root of
`X^q - B*C^(q-1)`, so use `(ceil((u+(q-1)*l)/q),l)`. Raising to an integer
power p multiplies both bounds by p. The positive denominator preserves the
principal complex branch. Distinct radical generators give an upper bound D
on the field degree; dependencies can only make this bound more conservative.

If B is nonzero, its nonzero integer norm has absolute value at least one.
Bounding the other conjugates proves
`|B/C| >= 2^-((D-1)*u+l)`. This bound is CONDITIONAL on nonzero; it does not
itself prove nonzeroness. `StructuralSeparationBound` is deliberately separate
from a certificate. An enclosure excluding zero proves nonzero. An entire
complex absolute enclosure strictly below the conditional bound proves zero.
A midpoint, small radius, or approximate equality is never a certificate.

Supported syntax is bounded closed arithmetic in rationals, I, sums, products,
and nonnegative rational powers. Symbols, floats, nonfinite values, functions,
root objects, negative exponents, and oversized inputs are refused. Refusing
reciprocals avoids erasing possible zero denominators. Complex powers keep their
principal branches; enclosures crossing a cut may yield UNKNOWN. Precision,
node count, degree, exponent, integer size and bound size all have explicit caps.
Without python-flint the new certificate path returns UNKNOWN. Context precision
is restored on success, refusal, and exceptions.

The norm argument and constructive separation approach can be compared with
[Burnikel et al., A Separation Bound for Real Algebraic Expressions](https://graphics.stanford.edu/~sfunke/Papers/ESA01/sepbound01.pdf)
and [Li and Yap, A New Constructive Root Bound for Algebraic Expressions](https://cs.nyu.edu/exact/doc/rootBd_abs.pdf).
The implementation uses the conservative bound proved above, rather than
claiming the full tighter BFMSS or Li–Yap algorithms.

## Reproduction and checks

Run `python benchmarks/benchmark_structural_algebraic.py --output results.json`
from an environment with the editable exprtools package and python-flint.
The script retrieves the stage-4 sources from Git and compares them with the
candidate in one interpreter; it checks identical verdicts. Cold mode clears
package caches for each observation (7 samples); warm mode uses 60; public API
mode disables the result cache (20). These are targeted microbenchmarks, not a
claim about every symbolic expression. Raw results include medians and p95.

Regression tests check conditional bounds, shared generators, refusal budgets,
no factorization/minimal-polynomial calls, dependent nested roots, tiny nonzero
perturbations, complex branches, interval strictness, precision restoration,
optional-backend behavior, cache clearing, dispatcher integration, and a seeded
24-expression comparison with exact minimal polynomials.
