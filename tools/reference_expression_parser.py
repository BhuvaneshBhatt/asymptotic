"""Parser helpers for scalar and vector-valued frozen reference expressions."""

from __future__ import annotations

import re

import sympy as sp


class _BroadcastVector:
    _op_priority = 10000

    def __init__(self, *components):
        self.components = tuple(map(sp.sympify, components))

    def _binary(self, other, op):
        if isinstance(other, _BroadcastVector):
            if len(other.components) != len(self.components):
                raise ValueError("vector dimensions differ")
            return _BroadcastVector(
                *(
                    op(a, b)
                    for a, b in zip(self.components, other.components, strict=True)
                )
            )
        return _BroadcastVector(*(op(a, other) for a in self.components))

    def __add__(self, o):
        return self._binary(o, lambda a, b: a + b)

    def __radd__(self, o):
        return self._binary(o, lambda a, b: b + a)

    def __sub__(self, o):
        return self._binary(o, lambda a, b: a - b)

    def __rsub__(self, o):
        return self._binary(o, lambda a, b: b - a)

    def __mul__(self, o):
        return self._binary(o, lambda a, b: a * b)

    def __rmul__(self, o):
        return self._binary(o, lambda a, b: b * a)

    def __truediv__(self, o):
        return self._binary(o, lambda a, b: a / b)

    def __neg__(self):
        return _BroadcastVector(*(-a for a in self.components))

    def _sympy_(self):
        """Coerce vector literals to SymPy tuples inside relational constructors."""
        return sp.Tuple(*self.components)

    def as_tuple(self):
        return sp.Tuple(*self.components)


def parse_reference_expression(text, locals):
    """Parse corpus syntax, broadcasting scalar arithmetic over ``(x, y)`` vectors."""
    if "(x, y)" not in text:
        return sp.sympify(text, locals=locals)
    env = dict(locals)
    env["_V"] = _BroadcastVector
    rewritten = re.sub(r"(?<![A-Za-z0-9_])(\(x, y\))", r"_V(x, y)", text)
    value = sp.sympify(rewritten, locals=env)
    return value.as_tuple() if isinstance(value, _BroadcastVector) else value
