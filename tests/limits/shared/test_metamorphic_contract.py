"""Metamorphic checks for public simultaneous-limit semantics."""

import sympy as sp

from asymptotic.limits import LimitStatus, limit

x, y, z = sp.symbols("x y z", real=True)


def _result(expr, variables, target, *, domain=True):
    return limit(expr, variables, target, domain=domain, return_result=True)


def test_coordinate_permutation():
    expr = x * y / sp.sqrt(x**2 + y**2)
    original = _result(expr, (x, y), (0, 0))
    swapped = _result(expr.xreplace({x: y, y: x}), (y, x), (0, 0))
    assert original.status is swapped.status is LimitStatus.PROVED
    assert original.value == swapped.value == 0


def test_positive_rescaling():
    expr = x * y / sp.sqrt(x**2 + y**2)
    original = _result(expr, (x, y), (0, 0))
    scaled = _result(expr.xreplace({x: 2 * x, y: 3 * y}), (x, y), (0, 0))
    assert original.status is scaled.status is LimitStatus.PROVED
    assert original.value == scaled.value == 0


def test_unused_coordinate_lift():
    expr = sp.atan(1 / sp.Abs(x))
    one_dimensional = _result(expr, (x,), (0,))
    lifted = _result(expr, (x, z), (0, 0))
    assert one_dimensional.status is lifted.status is LimitStatus.PROVED
    assert one_dimensional.value == lifted.value == sp.pi / 2


def test_domain_blocks_cylindrical_drop():
    expr = sp.atan(1 / sp.Abs(x))
    result = _result(expr, (x, z), (0, 0), domain=sp.Ge(z, x**2))
    assert z in result.reduced_variables or result.reduced_variables == (x, z)
