from __future__ import annotations

import sympy as sp

from asymptotic import (
    AsymptoticRelationResult,
    asymptotic_big_o,
    asymptotic_equivalent,
    asymptotic_little_o,
    asymptotic_relation,
)
from asymptotic.relations import (
    asymptotic_equal,
    asymptotic_greater,
    asymptotic_greater_equal,
    asymptotic_less,
    asymptotic_less_equal,
    asymptotic_same_order,
)


def test_named_asymptotic_relations_match_order_notation():
    x = sp.symbols("x", positive=True)
    assert asymptotic_less(x, x**2, x, sp.oo) is True
    assert asymptotic_little_o(x, x**2, x, sp.oo) is True
    assert asymptotic_less_equal(x, x**2, x, sp.oo) is True
    assert asymptotic_big_o(x, x**2, x, sp.oo) is True
    assert asymptotic_greater(x**2, x, x, sp.oo) is True
    assert asymptotic_greater_equal(x**2, x, x, sp.oo) is True


def test_equal_is_theta_while_equivalent_requires_ratio_one():
    x = sp.symbols("x", positive=True)
    assert asymptotic_equal(2 * x, x, x, sp.oo) is True
    assert asymptotic_same_order(2 * x, x, x, sp.oo) is True
    assert asymptotic_equivalent(2 * x, x, x, sp.oo) is False
    assert asymptotic_equivalent(x + 1, x, x, sp.oo) is True


def test_multivariate_rays_only_certify_counterexamples():
    x, y = sp.symbols("x y", real=True)
    result = asymptotic_relation(x**2 + y**4, x**2 + y**2, (x, y), (0, 0), relation="equal")
    assert isinstance(result, AsymptoticRelationResult)
    assert result.value is False
    assert result.certified is True
