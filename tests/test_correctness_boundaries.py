import sympy as sp

from asymptotic._multivariate_analytic import special_function_germ_contract
from asymptotic.algebraic_series import implicit_map_series
from asymptotic.complex_cluster_geometry import independent_monodromy_generators
from asymptotic.generalized_series import generalized_series
from asymptotic.gruntz_normal_form import gruntz_normal_form
from asymptotic.multivariate_limits_advanced import _semialgebraic_limiting_image
from asymptotic.puiseux import normalize_algebraic_approaches
from asymptotic.relative_growth import (
    certify_exp_log_fan,
    certify_symbolic_weights,
    specialize_parameter_cell,
)


def test_implicit_map_requires_proved_nonzero_jacobian():
    x, y, a = sp.symbols("x y a", real=True)
    assert implicit_map_series((a * y - x,), (y,), x) is None
    assert (
        implicit_map_series((a * y - x,), (y,), x, assumptions=sp.Ne(a, 0)) is not None
    )


def test_parameter_cell_equalities_specialize_expression_and_weights():
    a, b, x = sp.symbols("a b x", real=True)
    e, w, _subs = specialize_parameter_cell(
        a * x + b, (a, b), sp.And(sp.Eq(a, b), sp.Eq(b, 2))
    )
    assert sp.simplify(e - 2 * x - 2) == 0 and w == (sp.Integer(2), sp.Integer(2))


def test_symbolic_weights_partition_order_boundary():
    a = sp.symbols("a", real=True)
    cells, cov = certify_symbolic_weights((1, a), assumptions=sp.Gt(a, 0))
    assert cov.certified and cells
    assert any(sp.ask(sp.Q.positive(a - 1), c.condition) is True for c in cells)
    assert any(sp.ask(sp.Q.zero(a - 1), c.condition) is True for c in cells)
    assert any(sp.ask(sp.Q.negative(a - 1), c.condition) is True for c in cells)


def test_cross_variable_exp_log_fan_has_exhaustive_trichotomy():
    x, y = sp.symbols("x y", positive=True)
    fan, _ = certify_exp_log_fan(
        sp.exp(-1 / x) + sp.exp(-1 / y), (x, y), domain=sp.And(x > 0, y > 0)
    )
    assert fan.coverage.certified and len(fan.cells) >= 3


def test_generalized_series_carries_real_order_certificate():
    x = sp.symbols("x")
    s = generalized_series(sp.sin(x), x, terms=6)
    assert s is not None and s.certified and s.order_bound == 6


def test_cusp_normalization_uses_exact_ramified_roots():
    x, y = sp.symbols("x y", real=True)
    a = normalize_algebraic_approaches(y**2 - x**3, y, x, domain=sp.Ge(y, 0), terms=6)
    assert a and all(q.exact for q in a) and all(q.coverage.certified for q in a)
    assert all(sp.simplify((y**2 - x**3).subs(dict(q.substitutions))) == 0 for q in a)


def test_parameter_dependent_bessel_contract_requires_parameter_cell():
    z, nu = sp.symbols("z nu", positive=True)
    c = special_function_germ_contract(sp.besselj(nu, z), sp.Gt(nu, 0))
    assert c is not None and c.leading_power == nu


def test_gruntz_normal_form_records_exact_transform_and_nested_scale():
    x = sp.symbols("x", positive=True)
    n = gruntz_normal_form(sp.exp(sp.exp(x)) / x, x)
    assert n.exact_identity and n.transformed.has(sp.exp(sp.exp(1 / n.local_variable)))


def test_log_monodromy_has_explicit_additive_action():
    z = sp.symbols("z")
    gs, _cov = independent_monodromy_generators(sp.log(z), (z,), (0,))
    if gs:
        assert any(g.action.additive_shift.has(sp.pi) for g in gs)


def test_high_dimensional_angular_image_stereographic_chart():
    u = sp.symbols("u0:3", real=True)
    image, condition, provider = _semialgebraic_limiting_image(
        (u[0] * u[1], u[2] ** 2), u, sp.S.true
    )
    assert (
        image is not None
        and condition is sp.S.true
        and provider == "stereographic_sphere_image"
    )
