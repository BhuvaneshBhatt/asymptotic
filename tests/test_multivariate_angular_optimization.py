import sympy as sp

from asymptotic.angular_optimization import (
    parameterized_angular_range,
    parameterized_angular_strata,
)
from asymptotic.limits import LimitStatus, limit


def test_general_symbolic_quadratic_cross_term_uses_exact_eigenvalues():
    x, y, a, b, c = sp.symbols("x y a b c", real=True)
    result = parameterized_angular_range(a * x**2 + b * x * y + c * y**2, (x, y))
    disc = (a - c) ** 2 + b**2
    assert result.certified
    assert sp.simplify(result.minimum - (a + c - sp.sqrt(disc)) / 2) == 0
    assert sp.simplify(result.maximum - (a + c + sp.sqrt(disc)) / 2) == 0
    assert result.parameter_discriminant == disc


def test_parameterized_quartic_extrema_include_critical_value():
    x, y, q = sp.symbols("x y q", real=True)
    result = parameterized_angular_range(x**4 + q * x**2 * y**2 + y**4, (x, y))
    assert result.certified
    middle = sp.Rational(1, 2) + q / 4
    assert middle in result.critical_values
    assert result.minimum == sp.Min(1, middle)
    assert result.maximum == sp.Max(1, middle)


def test_critical_parameter_cluster_uses_general_cross_term_optimizer():
    x, y, b, p = sp.symbols("x y b p", real=True)
    expr = (x**2 + b * x * y + y**2) / (x**2 + y**2) ** p
    result = limit(expr, (x, y), (0, 0), return_result=True)
    critical = [
        s
        for s in result.strata
        if sp.simplify(s.condition.subs(p, 1)) is not sp.S.false
    ]
    assert critical
    at_b_zero = [
        s for s in critical if sp.simplify(s.condition.subs({p: 1, b: 0})) is sp.S.true
    ]
    assert any(
        s.result.status is LimitStatus.PROVED and s.result.value == 1 for s in at_b_zero
    )
    generic = [s for s in critical if s.result.status is LimitStatus.DOES_NOT_EXIST]
    assert generic


def test_quadratic_discriminant_stratifies_exceptional_singleton():
    x, y, b = sp.symbols("x y b", real=True)
    strata = parameterized_angular_strata(x**2 + b * x * y + y**2, (x, y))
    assert len(strata) == 2
    singleton = [s for s in strata if s.cluster_set == sp.FiniteSet(1)]
    assert len(singleton) == 1
    assert sp.simplify(singleton[0].condition.subs(b, 0)) is sp.S.true


def test_symbolic_critical_point_membership_parameter_cells():
    x, y, q = sp.symbols("x y q", real=True)
    strata = parameterized_angular_strata(x**4 + q * x**2 * y**2, (x, y))
    assert len(strata) == 2
    inside = [s for s in strata if s.condition.subs(q, -1) is sp.S.true]
    outside = [s for s in strata if s.condition.subs(q, 1) is sp.S.true]
    assert len(inside) == len(outside) == 1
    assert inside[0].provider == "even_simplex_parameter_cell_qe"
    assert inside[0].minimum.has(q)
    assert outside[0].minimum == 0
    assert outside[0].maximum == 1


def test_parameter_cluster_pipeline_uses_membership_cells():
    x, y, p, q = sp.symbols("x y p q", real=True)
    expr = (x**4 + q * x**2 * y**2) / (x**2 + y**2) ** p
    result = limit(expr, (x, y), (0, 0), return_result=True)
    critical = [
        s
        for s in result.strata
        if sp.simplify(s.condition.subs({p: 2, q: -1})) is sp.S.true
    ]
    assert critical
    assert any(s.result.status is LimitStatus.DOES_NOT_EXIST for s in critical)
