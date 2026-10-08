"""Exact discontinuous functions with explicit real endpoint conventions."""

import sympy as sp


class SignedFractionalPart(sp.Function):
    """Fractional part after truncation toward zero.

    For real x this is sign(x)*(abs(x)-floor(abs(x))). For complex x
    the same operation applies separately to the real and imaginary parts.
    Its real magnitude is less than one and it equals x on (-1,1).
    """

    nargs = 1

    @classmethod
    def eval(cls, argument):
        if argument.is_number and argument.is_real is True:
            return sp.sign(argument) * sp.frac(sp.Abs(argument))
        if argument.is_number and argument.is_finite is True:
            return cls(sp.re(argument)) + sp.I * cls(sp.im(argument))

    def _eval_is_real(self):
        return self.args[0].is_real

    def _eval_rewrite_as_frac(self, argument, **kwargs):
        if argument.is_real is True:
            return sp.sign(argument) * sp.frac(sp.Abs(argument))
        return sp.sign(sp.re(argument)) * sp.frac(
            sp.Abs(sp.re(argument))
        ) + sp.I * sp.sign(sp.im(argument)) * sp.frac(sp.Abs(sp.im(argument)))


def unit_step(*arguments):
    """Return the real step product, with value one at every zero argument."""
    return sp.Mul(*(sp.Heaviside(argument, 1) for argument in arguments))
