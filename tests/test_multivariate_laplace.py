from __future__ import annotations

import sympy as sp

from asymptotic.multivariate_laplace import (
    LaplaceRatioResult,
    LocalGaussianForm,
    MultivariateLaplaceResult,
    laplace_ratio_asymptotic,
    local_gaussian_form,
    log_laplace_asymptotic,
    multivariate_laplace_integral,
    stationary_point_geometry,
)


def test_certified_quadratic_multivariate_laplace_integral():
    x, y = sp.symbols("x y", real=True)
    n = sp.symbols("n", positive=True)
    phase = (x - 1) ** 2 / 2 + (y - 2) ** 2

    result = multivariate_laplace_integral(1, phase, (x, y), n, terms=2)

    assert isinstance(result, MultivariateLaplaceResult)
    assert result.status == "CERTIFIED"
    assert result.points == ((1, 2),)
    assert sp.simplify(result.expression - sp.sqrt(2) * sp.pi / n) == 0
    assert result.certificate.certified
    assert result.certificate.replay() is True


def test_higher_order_multivariate_gaussian_moment_contraction():
    x, y, lam = sp.symbols("x y lam", real=True)
    n = sp.symbols("n", positive=True)
    phase = (x**2 + y**2) / 2 + lam * x**4 / 24

    result = multivariate_laplace_integral(
        1, phase, (x, y), n, terms=2, stationary_points=((0, 0),)
    )

    expected = 2 * sp.pi / n * (1 - lam / (8 * n))
    assert result.status == "FORMAL"
    assert sp.simplify(result.expression - expected) == 0


def test_ratio_expansion_recovers_gaussian_second_moment():
    x, y = sp.symbols("x y", real=True)
    n = sp.symbols("n", positive=True)
    phase = (x**2 + y**2) / 2

    result = laplace_ratio_asymptotic(x**2, 1, phase, (x, y), n, terms=2)

    assert isinstance(result, LaplaceRatioResult)
    assert result.status == "CERTIFIED"
    assert sp.simplify(result.expression - 1 / n) == 0


def test_log_laplace_uses_stable_log_domain_expansion():
    x, y = sp.symbols("x y", real=True)
    n = sp.symbols("n", positive=True)
    phase = (x**2 + y**2) / 2

    result = log_laplace_asymptotic(1, phase, (x, y), n, terms=2)

    assert result.status == "CERTIFIED"
    assert sp.simplify(result.expression - (sp.log(2 * sp.pi) - sp.log(n))) == 0


def test_local_gaussian_form_exposes_normalized_density():
    x, y = sp.symbols("x y", real=True)
    n = sp.symbols("n", positive=True)
    phase = (x - 1) ** 2 / 2 + (y - 2) ** 2

    result = local_gaussian_form(1, phase, (x, y), n, terms=2)

    assert isinstance(result, LocalGaussianForm)
    assert result.point == (1, 2)
    assert result.covariance_scale == sp.diag(1 / n, sp.Rational(1, 2) / n)
    z0, z1 = result.local_variables
    expected = sp.sqrt(2) * sp.exp(-(z0**2) / 2 - z1**2) / (2 * sp.pi)
    assert sp.simplify(result.density_expansion - expected) == 0


def test_equal_competing_multivariate_saddles_are_summed():
    x, y = sp.symbols("x y", real=True)
    n = sp.symbols("n", positive=True)
    phase = (x**2 - 1) ** 2 + y**2 / 2

    result = multivariate_laplace_integral(1, phase, (x, y), n, terms=1)

    assert result.method == "multivariate-laplace-co-dominant"
    assert result.points == ((-1, 0), (1, 0))
    assert result.certificate.competing
    assert result.status == "CERTIFIED"
    assert sp.simplify(result.expression - sp.sqrt(2) * sp.pi / n) == 0


def test_axis_aligned_boundary_saddle_tangent_scaling():
    x, y = sp.symbols("x y", real=True)
    n = sp.symbols("n", positive=True)
    phase = x + (y - 1) ** 2 / 2
    domain = (sp.Interval(0, sp.oo), sp.Interval(-sp.oo, sp.oo))

    result = multivariate_laplace_integral(1, phase, (x, y), n, domain=domain, terms=2)

    assert result.method == "multivariate-laplace-boundary"
    assert result.points == ((0, 1),)
    assert result.status == "CERTIFIED"
    assert (
        sp.simplify(result.expression - sp.sqrt(2 * sp.pi) / n ** sp.Rational(3, 2))
        == 0
    )


def test_geometry_reports_boundary_activity_gaussian_claim():
    x, y = sp.symbols("x y", real=True)
    phase = x + y**2
    domain = (sp.Interval(0, sp.oo), sp.Interval(-sp.oo, sp.oo))

    geometry = stationary_point_geometry(phase, (x, y), domain=domain)
    boundary = next(item for item in geometry if item.point == (0, 0))

    assert boundary.kind == "boundary"
    assert boundary.active_boundaries == (0,)
    assert not boundary.nondegenerate_minimum
