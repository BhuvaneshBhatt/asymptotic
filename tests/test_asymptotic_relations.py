from __future__ import annotations

import importlib

import sympy as sp

from asymptotic import (
    big_o,
    equivalent,
    little_o,
    relation,
)
from asymptotic.relations import (
    RelationResult,
    equal,
    greater,
    greater_equal,
    less,
    less_equal,
    same_order,
)

relations = importlib.import_module("asymptotic.relations")


def test_named_relations_match_order_notation():
    x = sp.symbols("x", positive=True)
    assert less(x, x**2, x, sp.oo) is True
    assert little_o(x, x**2, x, sp.oo) is True
    assert less_equal(x, x**2, x, sp.oo) is True
    assert big_o(x, x**2, x, sp.oo) is True
    assert greater(x**2, x, x, sp.oo) is True
    assert greater_equal(x**2, x, x, sp.oo) is True


def test_equal_is_theta_while_equivalent_requires_ratio_one():
    x = sp.symbols("x", positive=True)
    assert equal(2 * x, x, x, sp.oo) is True
    assert same_order(2 * x, x, x, sp.oo) is True
    assert equivalent(2 * x, x, x, sp.oo) is False
    assert equivalent(x + 1, x, x, sp.oo) is True


def test_multivariate_rays_only_certify_counterexamples():
    x, y = sp.symbols("x y", real=True)
    result = relation(
        x**2 + y**4, x**2 + y**2, (x, y), (0, 0), relation="equal", return_result=True
    )
    assert isinstance(result, RelationResult)
    assert result.value is False
    assert result.certified is True


def test_six_predicates_share_one_univariate_fact_bundle():
    x = sp.symbols("x", positive=True)
    relations._univariate_relation_facts.cache_clear()
    relations._ratio_properties.cache_clear()
    from asymptotic.instrumentation import symbolic_metrics

    with symbolic_metrics() as metrics:
        predicates = (
            relations.less,
            relations.greater,
            relations.less_equal,
            relations.greater_equal,
            relations.equal,
            relations.equivalent,
        )
        values = tuple(predicate(x, x**2, x, sp.oo) for predicate in predicates)
    assert values == (True, False, True, False, False, False)
    assert metrics.growth_comparisons == 1
