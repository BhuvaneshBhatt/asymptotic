import sympy as sp

from asymptotic.multivariate_expansion import (
    ExpansionStatus,
    multivariate_atlas,
    multivariate_expansion,
)


def test_polynomial_expansion_has_zero_uniform_remainder():
    x, y = sp.symbols("x y", real=True)
    result = multivariate_expansion(x + y**2, (x, y), (0, 0), order=2)
    assert result.status is ExpansionStatus.CERTIFIED
    assert result.certified
    assert result.remainder_certificate.constant == 0


def test_rational_expansion_certifies_uniform_remainder():
    x, y = sp.symbols("x y", real=True)
    result = multivariate_expansion(1 / (1 + x**2 + y**2), (x, y), (0, 0), order=2)
    assert result.status is ExpansionStatus.CERTIFIED
    assert result.remainder_certificate is not None
    assert result.remainder_certificate.order == 3


def test_explicit_weighted_chart_is_recorded():
    x, y = sp.symbols("x y", real=True)
    result = multivariate_expansion(x + y**2, (x, y), (0, 0), order=2, weights=(2, 1))
    assert result.weights == (2, 1)
    assert result.status is ExpansionStatus.CERTIFIED


def test_transcendental_formal_is_not_mislabeled_certified():
    x, y = sp.symbols("x y", real=True)
    result = multivariate_expansion(
        sp.sin(x**2 + y**2), (x, y), (0, 0), order=2, return_formal=True
    )
    assert result.status in (ExpansionStatus.FORMAL, ExpansionStatus.UNKNOWN)
    assert not result.certified


def test_certified_atlas_has_finite_sector_cover():
    x, y = sp.symbols("x y", real=True)
    atlas = multivariate_atlas(x + y**2, (x, y), (0, 0), order=2)
    assert atlas.certified
    assert atlas.coverage_certificate.certified
    assert len(atlas.charts) >= 2
    for weight in atlas.weights:
        family = [c for c in atlas.charts if c.expansion.weights == weight]
        assert {c.dominant_coordinate for c in family} == {0, 1}


def test_newton_atlas_retains_distinct_weight_families_when_certified():
    x, y = sp.symbols("x y", real=True)
    expr = x**2 + y**4
    atlas = multivariate_atlas(expr, (x, y), (0, 0), order=4)
    assert atlas.certified
    assert (1, 1) in atlas.weights
    assert (2, 1) in atlas.weights


def test_restricted_domain_atlas_certifies_relative_coverage():
    x, y = sp.symbols("x y", real=True)
    atlas = multivariate_atlas(x + y, (x, y), (0, 0), domain=sp.Ge(x, 0))
    assert atlas.certified
    assert atlas.coverage_certificate.certified


def test_big_o_is_certified_atlas_wide():
    from asymptotic.multivariate_expansion import multivariate_big_o

    x, y = sp.symbols("x y", real=True)
    result = multivariate_big_o(x**2 + y**2, x**2 + y**2, (x, y), (0, 0))
    assert result.certified
    assert result.constant is not None


def test_little_o_is_certified_atlas_wide():
    from asymptotic.multivariate_expansion import multivariate_little_o

    x, y = sp.symbols("x y", real=True)
    result = multivariate_little_o((x**2 + y**2) ** 2, x**2 + y**2, (x, y), (0, 0))
    assert result.certified


def test_asymptotic_equivalence_is_certified_atlas_wide():
    from asymptotic.multivariate_expansion import multivariate_equivalent

    x, y = sp.symbols("x y", real=True)
    g = x**2 + y**2
    result = multivariate_equivalent(g + g**2, g, (x, y), (0, 0))
    assert result.certified


def test_relative_domain_without_local_approach_is_not_certified():
    x, y = sp.symbols("x y", real=True)
    atlas = multivariate_atlas(x + y, (x, y), (0, 0), domain=sp.Gt(x**2 + y**2, 1))
    assert not atlas.coverage_certificate.certified


def test_sector_simplex_certifies_even_rational_angular_big_o():
    from asymptotic.multivariate_expansion import multivariate_big_o

    x, y = sp.symbols("x y", real=True)
    r2 = x**2 + y**2
    result = multivariate_big_o(x**2 * y**2, r2, (x, y), (0, 0))
    assert result.certified
    assert {"sector_simplex_semialg", "angular_coefficient_norm"} & set(
        result.certificate_backends
    )


def test_sector_simplex_certifies_quartic_angular_big_o():
    from asymptotic.multivariate_expansion import multivariate_big_o

    x, y = sp.symbols("x y", real=True)
    r2 = x**2 + y**2
    result = multivariate_big_o(x**4 + 5 * x**2 * y**2 + y**4, r2**2, (x, y), (0, 0))
    assert result.certified
    assert {"sector_simplex_semialg", "angular_coefficient_norm"} & set(
        result.certificate_backends
    )


def test_non_even_angular_form_uses_exact_sphere_coefficient_bound():
    from asymptotic.multivariate_expansion import multivariate_big_o

    x, y = sp.symbols("x y", real=True)
    r2 = x**2 + y**2
    result = multivariate_big_o(x**2 + 3 * x * y + 2 * y**2, r2, (x, y), (0, 0))
    assert result.certified
    assert "angular_coefficient_norm" in result.certificate_backends


def test_multi_newton_weighted_decay():
    from asymptotic.multivariate_expansion import multivariate_little_o

    x, y = sp.symbols("x y", real=True)
    result = multivariate_little_o(x**4 + y**8, x**2 + y**4, (x, y), (0, 0))
    assert result.certified
    assert result.atlas is not None
    assert (2, 1) in result.atlas.weights


def test_algebraic_radial_relations_receive_radical_remainders():
    from asymptotic.multivariate_expansion import (
        multivariate_equivalent,
        multivariate_little_o,
    )

    x, y = sp.symbols("x y", real=True)
    r2 = x**2 + y**2
    equivalent = multivariate_equivalent(sp.sqrt(1 + r2), 1, (x, y), (0, 0))
    decay = multivariate_little_o(sp.sqrt(1 + r2) - 1, sp.sqrt(r2), (x, y), (0, 0))
    assert equivalent.certified
    assert decay.certified
    assert "positive_radical_denominator" in decay.certificate_backends


def test_radial_transcendental_compositions_use_taylor_certificates():
    from asymptotic.multivariate_expansion import multivariate_equivalent

    x, y = sp.symbols("x y", real=True)
    r2 = x**2 + y**2
    exponential = multivariate_equivalent(sp.exp(r2), 1, (x, y), (0, 0))
    sinc = multivariate_equivalent(sp.sin(r2), r2, (x, y), (0, 0))
    assert exponential.certified
    assert sinc.certified
    assert "univariate_taylor_theorem" in exponential.certificate_backends
    assert "alternating_sine_remainder" in sinc.certificate_backends


def test_three_variable_newton_fan_cad_timeout():
    from asymptotic.multivariate_expansion import multivariate_little_o

    x, y, z = sp.symbols("x y z", real=True)
    result = multivariate_little_o(
        x**4 + y**8 + z**12, x**2 + y**4 + z**6, (x, y, z), (0, 0, 0)
    )
    assert result.certified


def test_three_variable_newton_fan_angular_bound_uses_sector_geometry():
    from asymptotic.multivariate_expansion import multivariate_big_o

    x, y, z = sp.symbols("x y z", real=True)
    result = multivariate_big_o(
        x * y**2 * z**3, x**2 + y**4 + z**6, (x, y, z), (0, 0, 0)
    )
    assert result.certified
    assert "sector_dominant_coordinate_bound" in result.certificate_backends


def test_nonradial_transcendental_composition_relations():
    from asymptotic.multivariate_expansion import (
        multivariate_equivalent,
        multivariate_little_o,
    )

    x, y, z = sp.symbols("x y z", real=True)
    assert multivariate_equivalent(
        sp.exp(x * y + y * z + z * x), 1, (x, y, z), (0, 0, 0)
    ).certified
    assert multivariate_little_o(
        sp.sin(x * y * z), x**2 + y**2 + z**2, (x, y, z), (0, 0, 0)
    ).certified
