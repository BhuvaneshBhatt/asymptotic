import sympy as sp

from asymptotic.complex_cluster_geometry import branched_blowup_atlas
from asymptotic.limits import LimitStatus, limit
from asymptotic.local_germ_algebra import function_germ, shifted_gamma_germ
from asymptotic.multivariate_limits_advanced import (
    AdvancedLimitStatus,
    joint_cluster_geometry,
)
from asymptotic.puiseux import normalize_algebraic_approaches
from asymptotic.relative_growth import (
    automatic_exp_log_valuation_fan,
    certify_symbolic_weights,
)


def test_reference_0966_regression():
    x, y = sp.symbols("x y", real=True)
    e = sp.cot((x * x + y * y) / 4 + sp.pi / 4) * sp.atan(1 / (x * x + y * y))
    r = limit(e, (x, y), (0, 0), return_result=True)
    assert r.status is LimitStatus.PROVED and sp.simplify(r.value - sp.pi / 2) == 0


def test_puiseux_normalizes_cusp_and_domain():
    x, y = sp.symbols("x y", real=True)
    a = normalize_algebraic_approaches(y**2 - x**3, y, x, domain=sp.Ge(y, 0), terms=6)
    assert a and all(q.ramification_index >= 1 for q in a)


def test_symbolic_weight_certification_is_assumption_conditioned():
    a = sp.symbols("a", positive=True)
    cells, cov = certify_symbolic_weights((1, a), assumptions=sp.Gt(a, 0))
    assert cov.certified and cells[0].certified


def test_generated_exp_log_fan_does_not_overclaim_completeness():
    x, y = sp.symbols("x y", positive=True)
    fan = automatic_exp_log_valuation_fan(sp.exp(-1 / x) + sp.exp(-1 / y), (x, y))
    assert not fan.coverage.certified


def test_branched_atlas_keeps_branch_point_obligation():
    x, y = sp.symbols("x y", real=True)
    charts, cov = branched_blowup_atlas(sp.log(x + sp.I * y), (x, y), (0, 0))
    assert charts and cov.certified
    assert all(chart.coverage.certified for chart in charts)


def test_special_function_germs_bessely_and_shifted_gamma():
    r = sp.symbols("r", positive=True)
    assert function_germ(sp.bessely, 1, radial_variable=r) is not None
    assert shifted_gamma_germ(-2, radial_variable=r).coefficient == sp.Rational(1, 2)


def test_3d_direction_joint_cluster_fast_path():
    x, y, z = sp.symbols("x y z", real=True)
    rho = sp.sqrt(x * x + y * y + z * z)
    r = joint_cluster_geometry((x / rho, y / rho, z / rho), (x, y, z), (0, 0, 0))
    assert (
        r.status is AdvancedLimitStatus.CERTIFIED
        and r.provider == "joint_cluster_unit_sphere"
    )


def test_undefined_geometry_is_unknown():
    x, y = sp.symbols("x y", real=True)
    result = limit(sp.Function("undefined")(x, y), (x, y), (0, 0), return_result=True)
    assert result.status is LimitStatus.UNKNOWN
    assert result.value is None


def test_atan_positive_radial_blowup_path_value():
    import sympy as sp

    from asymptotic.limits import LimitStatus, limit

    x, y = sp.symbols("x y", real=True)
    expr = sp.atan((sp.Abs(x) + sp.Abs(y)) / (x**2 + y**2))
    result = limit(expr, (x, y), (0, 0), return_result=True)
    assert result.status is LimitStatus.PROVED
    assert result.value == sp.pi / 2
