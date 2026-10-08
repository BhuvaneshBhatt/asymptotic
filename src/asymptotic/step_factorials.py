"""Finite step factorials and their principal gamma continuation."""

import sympy as sp


class StepFactorialPower(sp.Function):
    """Step falling factorial with explicit regular and pole strata.

    Integer orders use finite products. For noninteger order and nonzero step,
    the principal analytic extension is gamma(1+x/h) divided by
    (1/h)**n*gamma(1+x/h-n); certificates must check its regularity.
    """

    nargs = 3

    @classmethod
    def eval(cls, x, n, h):
        if n.is_Integer and abs(n) <= 128:
            if n >= 0:
                return sp.Mul(*(x - j * h for j in range(int(n))))
            if x.is_zero is False or not x.free_symbols:
                return 1 / sp.Mul(*(x + j * h for j in range(1, int(-n) + 1)))
            return None
