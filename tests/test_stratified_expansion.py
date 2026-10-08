import sympy as sp

from asymptotic.stratified_expansion import StratifiedExpansion, stratified_expand


def test_discovers_three_scale_cells_for_rational_expression():
    x, e = sp.symbols("x e", positive=True)
    result = stratified_expand(x / (x + e), (x, e), order=3)
    assert isinstance(result, StratifiedExpansion)
    assert result.certified
    assert len(result.branches) == 3
    assert sp.simplify(result.branches[0].approximation - (x / e - x**2 / e**2)) == 0
    assert (
        sp.simplify(result.branches[2].approximation - (1 - e / x + e**2 / x**2)) == 0
    )
    assert result.branches[1].method == "comparable_scaled_coordinate"


def test_no_false_single_expansion_without_regime():
    x, e = sp.symbols("x e", positive=True)
    result = stratified_expand(1 / (x + e), (x, e), order=2)
    assert result.certified and len(result.branches) == 3
    assert all(b.certified for b in result.branches)


def test_symbolic_exponent_stratification_is_only_critical_surface():
    x = sp.symbols("x", positive=True)
    a, b, c = sp.symbols("a b c", real=True)
    result = stratified_expand(x**a / (x**b + x**c), x, order=2)
    assert result.certified
    assert len(result.branches) == 3
    conditions = {sp.sstr(q) for q in result.coverage_certificate.conditions}
    assert conditions == {sp.sstr(b < c), sp.sstr(sp.Eq(b, c)), sp.sstr(b > c)}


def test_unsupported_geometry_declines_with_obligation():
    x, y, z = sp.symbols("x y z", positive=True)
    result = stratified_expand((x + y) * (1 + z), (x, y, z), order=2)
    assert not result.certified
    assert result.obligations


def test_non_power_cell_is_completed_by_logexp():
    x, e = sp.symbols("x e", positive=True)
    result = stratified_expand(sp.exp(x / e), (x, e), order=3)
    assert result.certified
    assert result.branches
    assert not result.obligations
    assert result.coverage_certificate.certified


def test_parameter_equal_surface_is_exact_not_formal():
    x = sp.symbols("x", positive=True)
    a, b, c = sp.symbols("a b c", real=True)
    result = stratified_expand(x**a / (x**b + x**c), x, order=3)
    equal = next(q for q in result.branches if q.regime.condition == sp.Eq(b, c))
    assert equal.remainder_scale == 0
    assert sp.simplify(equal.approximation - x ** (a - b) / 2) == 0
