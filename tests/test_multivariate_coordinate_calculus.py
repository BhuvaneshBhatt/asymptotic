import sympy as sp

from asymptotic.multivariate_coordinate_calculus import (
    certified_local_inverse,
    certified_multivariate_compose,
    check_atlas_consistency,
    correlated_path,
    generalized_scale_cone,
    newton_dominance_fan,
    transport_chart,
)


def test_newton_dominance_fan():
    x, y = sp.symbols("x y")
    fan = newton_dominance_fan([x**2 + x * y + y**3], [x, y])
    assert fan.coverage_certified
    assert len(fan.cones) >= 1


def test_correlated_path():
    x, y = sp.symbols("x y", positive=True)
    result = correlated_path(x**2 + y, [x, y], [1, 2])
    assert result.certified
    assert result.leading_power == 2
    assert result.leading_coefficient == 2
    assert result.limit == 0


def test_path_dependence():
    x, y = sp.symbols("x y", positive=True)
    expression = x * y / (x**2 + y**2)
    diagonal = correlated_path(expression, [x, y], [1, 1], coefficients=[1, 1])
    curved = correlated_path(expression, [x, y], [1, 2])
    assert diagonal.limit == sp.Rational(1, 2)
    assert curved.limit == 0


def test_scale_cone():
    x, y = sp.symbols("x y", positive=True)
    cone = generalized_scale_cone([x, y, x * y], [x, y], [1, 2])
    assert cone.certified
    assert cone.valuations == (1, 2, 3)
    assert cone.order == ((0, 1, -1), (0, 2, -1), (1, 2, -1))


def test_composition():
    x, y, u, v = sp.symbols("x y u v")
    certificate = certified_multivariate_compose(x + y, [x, y], [u, u * v])
    assert certificate.certified
    assert sp.expand(certificate.composed) == u + u * v
    assert certificate.obligations == ()


def test_inverse_and_singular_map():
    x, y = sp.symbols("x y")
    inverse = certified_local_inverse([x + y, y], [x, y])
    singular = certified_local_inverse([x**2, y], [x, y])
    assert inverse.certified
    assert inverse.jacobian_determinant == 1
    assert inverse.obligations == ()
    assert singular.certified is False
    assert singular.inverse is None
    assert singular.jacobian_determinant == 2 * x
    assert singular.obligations == ("nonzero Jacobian at target",)


def test_chart_round_trip():
    x, y, u, v = sp.symbols("x y u v")
    result = transport_chart(
        x + y,
        [x, y],
        [u, v],
        [u, u * v],
        inverse_transition=[x, y / x],
    )
    assert result.certified
    assert sp.expand(result.transported) == u + u * v
    assert result.obligations == ()


def test_atlas_consistency():
    x, y, u, v = sp.symbols("x y u v")
    transitions = {(0, 1): ((x, y), (u, v), (u, u * v))}
    consistent = check_atlas_consistency([x + y, u + u * v], transitions)
    inconsistent = check_atlas_consistency([x + y, u + u * v + 1], transitions)
    assert consistent.certified and consistent.consistent is True
    assert consistent.discrepancies == (0,)
    assert inconsistent.certified and inconsistent.consistent is False
    assert inconsistent.discrepancies == (-1,)
