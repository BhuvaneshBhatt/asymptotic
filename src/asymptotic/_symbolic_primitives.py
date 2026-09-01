"""Small deterministic symbolic primitives used by certification paths.

The helpers in this module intentionally avoid open-ended symbolic algorithms.
Certification code should return an unresolved result when these recognizers do
not apply rather than delegating to a potentially unbounded general integrator.
"""

from __future__ import annotations

import sympy as sp


def exact_elementary_primitive(expr: sp.Expr, variable: sp.Symbol) -> sp.Expr | None:
    """Return a verified primitive for common asymptotic elementary terms.

    Supported forms are deliberately conservative: finite sums of supported
    terms, constant multiples of powers/logarithms, and exact exponential
    derivatives ``c*q'(x)*exp(q(x))``.  Every candidate is replayed by
    differentiation before it is returned.
    """

    expr = sp.expand_mul(sp.sympify(expr))
    if expr == 0:
        return sp.S.Zero
    if expr.is_Add:
        pieces = []
        for term in expr.args:
            primitive = exact_elementary_primitive(term, variable)
            if primitive is None:
                return None
            pieces.append(primitive)
        candidate = sp.Add(*pieces)
        return candidate if sp.simplify(sp.diff(candidate, variable) - expr) == 0 else None

    powers = expr.as_powers_dict()
    exponent = sp.sympify(powers.get(variable, 0))
    base = variable**exponent
    coefficient = sp.simplify(expr / base)
    if variable not in coefficient.free_symbols:
        if exponent == -1:
            candidate = coefficient * sp.log(variable)
        elif variable not in exponent.free_symbols:
            candidate = coefficient * variable ** (exponent + 1) / (exponent + 1)
        else:
            candidate = None
        if candidate is not None and sp.simplify(sp.diff(candidate, variable) - expr) == 0:
            return sp.simplify(candidate)

    exponential_factors = [factor for factor in sp.Mul.make_args(expr) if factor.func is sp.exp]
    if len(exponential_factors) == 1:
        exponential = exponential_factors[0]
        phase = exponential.args[0]
        phase_derivative = sp.diff(phase, variable)
        if phase_derivative != 0:
            ratio = sp.simplify(expr / (phase_derivative * exponential))
            if variable not in ratio.free_symbols:
                candidate = sp.simplify(ratio * exponential)
                if sp.simplify(sp.diff(candidate, variable) - expr) == 0:
                    return candidate
    return None


def certification_primitive(expr: sp.Expr, variable: sp.Symbol) -> sp.Expr | None:
    """Return a bounded exact primitive suitable for theorem certification.

    No call to :func:`sympy.integrate` is made.  A miss is intentionally
    represented by ``None`` so callers can report an UNKNOWN hypothesis.
    """

    return exact_elementary_primitive(sp.simplify(expr), variable)
