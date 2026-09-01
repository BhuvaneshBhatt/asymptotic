import sympy as sp

from asymptotic import (
    GrowthComparison,
    discover_scale,
)
from asymptotic.context import AsymptoticContext
from asymptotic.tower import ExpLogTower


def test_exp_log_tower_is_dependency_ordered():
    x = sp.symbols("x", positive=True)
    expr = sp.log(sp.log(x)) + sp.exp(x)
    tower = ExpLogTower.from_expr(expr, x)
    generators = [item.generator for item in tower.extensions]
    assert sp.log(x) in generators
    assert sp.log(sp.log(x)) in generators
    assert sp.exp(x) in generators
    assert generators.index(sp.log(x)) < generators.index(sp.log(sp.log(x)))


def test_discover_scale_deduplicates_same_exponential_comparability_class():
    x = sp.symbols("x", positive=True)
    expr = sp.exp(-x) + sp.exp(-2 * x) + 1 / x
    scale = discover_scale(expr, x)
    ctx = AsymptoticContext(x)
    exp_class_count = 0
    for elem in scale.exprs:
        relation, _ = ctx.compare_log_growth(elem, sp.exp(-x))
        if relation is GrowthComparison.SAME_ORDER:
            exp_class_count += 1
    assert exp_class_count == 1


def test_scale_discovery_uses_tower_depth_instead_of_fixed_log_limit():
    from asymptotic.scale import ScaleDiscovery

    x = sp.symbols("x", positive=True)
    expr = x
    for _ in range(10):
        expr = sp.log(expr)

    discovery = ScaleDiscovery(expr, x)
    candidates = discovery.candidates()
    assert len(discovery.tower.extensions) == 10
    assert any(sp.simplify(c - 1 / expr) == 0 for c in candidates)


def test_scale_discovery_records_comparison_obligations():
    from asymptotic.obligations import ObligationKind
    from asymptotic.scale import ScaleDiscovery

    x = sp.symbols("x", positive=True)
    expr = sp.log(sp.log(x)) + sp.exp(-x)
    discovery = ScaleDiscovery(expr, x)
    scale = discovery.discover()

    assert len(scale) >= 3
    assert discovery.obligation_history
    assert all(
        item.kind is ObligationKind.GROWTH_COMPARISON for item in discovery.obligation_history
    )
