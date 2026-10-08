"""Capability-separation examples derived from the univariate behavior corpus.

These are package regressions, not claims about every current computer algebra system
kernel.  They exercise semantics/algorithms present here for which the audited
reference implementation has no corresponding complete route.
"""

import pytest
import sympy as sp

from asymptotic.limits import LimitStatus, limit

x, y = sp.symbols("x y", real=True)

UNIVARIATE = [
    (sp.sign(x), x > 0, 1),
    (sp.Abs(x) / x, x > 0, 1),
    (sp.Abs(x) / x, x < 0, -1),
    (x / sp.Abs(x), x < 0, -1),
    (sp.sign(x**3), x > 0, 1),
    (sp.Abs(x**3) / x**3, x < 0, -1),
    (sp.floor(x), x > 0, 0),
    (sp.ceiling(x), x < 0, 0),
    (sp.Heaviside(x), x > 0, 1),
    (sp.Piecewise((1, x > 0), (-1, True)), x > 0, 1),
]

MULTIVARIATE = [
    (x**2 * y / (x**2 + y**4), 0),
    (x**3 * y / (x**6 + y**2), "dne"),
    (sp.exp(x**2 * y**2 / (x**2 + y**2)), 1),
    (sp.sin(x**2 + y**2) / (x**2 + y**2), 1),
    (sp.erf(x * y) / (x * y), 2 / sp.sqrt(sp.pi)),
    (sp.erf(sp.sin(x * y)) / (x * y), 2 / sp.sqrt(sp.pi)),
    (x * y / sp.sqrt(x**2 + y**2), 0),
    ((x**2 + y**2) ** 2 / sp.sin(x**2 + y**2), 0),
    (-sp.pi * x * sp.bessely(1, x) * (sp.cosh(y) - y**2 / sp.Integer(16)) / 2, 1),
    (sp.cot((x**2 + y**2) / 4 + sp.pi / 4) * sp.atan(1 / (x**2 + y**2)), sp.pi / 2),
]


@pytest.mark.parametrize("expr,domain,expected", UNIVARIATE)
def test_univariate_domain_capability_separation(expr, domain, expected):
    r = limit(expr, x, 0, domain=domain, return_result=True)
    assert r.status is LimitStatus.PROVED
    assert sp.simplify(r.value - expected) == 0


@pytest.mark.parametrize("expr,expected", MULTIVARIATE)
def test_multivariate_capability_separation(expr, expected):
    r = limit(expr, (x, y), (0, 0), return_result=True)
    if expected == "dne":
        assert r.status is LimitStatus.DOES_NOT_EXIST
    else:
        assert r.status is LimitStatus.PROVED
        assert sp.simplify(r.value - expected) == 0


def test_bessely_singular_germ_precedes_unsound_direct_path_limit():
    """Integer-order Y1 germ is certified without trusting SymPy's t*Y1(t) path limit."""
    expr = -sp.pi * x * sp.cosh(y) * sp.bessely(1, x) / 2 - y**2 / sp.Integer(16) - 1
    r = limit(expr, (x, y), (0, 0), return_result=True)
    assert r.status is LimitStatus.PROVED
    assert r.value == 0
