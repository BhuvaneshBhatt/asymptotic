import sympy as sp

from asymptotic.multivariate_limits_advanced import (
    AdvancedLimitStatus,
    bounded_factor_envelope_limit,
    certify_local_germ,
    compose_local_germ,
    multivariate_cluster_set,
    newton_fan_limit,
)


def test_newton_fan_limit_three_variable_decay():
    x, y, z = sp.symbols("x y z", real=True)
    f = (x**4 + y**8 + z**12) / (x**2 + y**4 + z**6)
    r = newton_fan_limit(f, (x, y, z), (0, 0, 0))
    assert r.status is AdvancedLimitStatus.CERTIFIED
    assert r.value == 0
    assert (6, 3, 2) in r.weights


def test_local_germ_composition_log1p():
    x, y = sp.symbols("x y", real=True)
    h = x**2 * y**2 / (x**2 + y**2)
    g = certify_local_germ(h, (x, y), (0, 0))
    assert g.certified and g.limit == 0
    c = compose_local_germ(lambda z: sp.log(1 + z), h, (x, y), (0, 0))
    assert c.certified and c.limit == 0


def test_cluster_interval_weighted_homogeneous():
    x, y = sp.symbols("x y", real=True)
    c = multivariate_cluster_set(x**2 / (x**2 + y**2), (x, y), (0, 0))
    assert c.certified
    assert c.cluster_set == sp.Interval(0, 1)
    assert c.liminf == 0 and c.limsup == 1


def test_oscillatory_vanishing_envelope():
    x, y = sp.symbols("x y", real=True)
    f = (x**2 + y**2) * sp.sin(1 / (x**2 + y**2))
    r = bounded_factor_envelope_limit(f, (x, y), (0, 0))
    assert r.certified and r.value == 0


def test_extended_limit_mixed_finite_infinite():
    from asymptotic.multivariate_limits_advanced import extended_limit

    x, y = sp.symbols("x y", real=True)
    assert extended_limit(x + 1 / y, (x, y), (0, sp.oo)) == 0


def test_relative_limit_public_semantics():
    from asymptotic.multivariate_limits_advanced import relative_limit

    x, y = sp.symbols("x y", real=True)
    r = relative_limit(x / y, (x, y), (0, 0), domain=sp.Eq(y, x), return_result=True)
    assert r.certified and r.value == 1


def test_vector_germ_composition_outer_singularity():
    from asymptotic.multivariate_limits_advanced import compose_vector_local_germ

    x, y = sp.symbols("x y", real=True)
    u, v = sp.symbols("u v", real=True)
    r = compose_vector_local_germ(
        u**2 / (u**2 + v**2), (x, 2 * x), (x, y), (0, 0), outer_variables=(u, v)
    )
    assert r.certified and r.limit == sp.Rational(1, 5)


def test_complex_multivariate_limit_real_imaginary_reduction():
    from asymptotic.multivariate_limits_advanced import complex_multivariate_limit

    z, w = sp.symbols("z w")
    r = complex_multivariate_limit(
        z * w, (z, w), (1 + sp.I, 2 - sp.I), return_result=True
    )
    assert r.certified and sp.simplify(r.value - (3 + sp.I)) == 0


def test_projective_chart_feeds_newton_and_cluster_analysis():
    from asymptotic.multivariate_limits_advanced import (
        extended_multivariate_cluster_set,
        extended_newton_fan_limit,
    )

    x, y = sp.symbols("x y", real=True)
    n = extended_newton_fan_limit(1 / x**2 + y**2, (x, y), (sp.oo, 0))
    assert n.certified and n.value == 0
    c = extended_multivariate_cluster_set(x**2 / (x**2 + y**2), (x, y), (sp.oo, sp.oo))
    assert c.certified and c.cluster_set == sp.Interval(0, 1)


def test_vector_germ_carries_and_uses_image_constraint():
    from asymptotic.multivariate_limits_advanced import (
        certify_vector_local_germ,
        compose_vector_local_germ,
    )

    x, y = sp.symbols("x y", real=True)
    u, v = sp.symbols("u v", real=True)
    g = certify_vector_local_germ((x, 2 * x), (x, y), (0, 0))
    assert g.certified
    assert g.image_provider.startswith("semialgebraic_local_image:")
    assert g.image_constraint != sp.S.true
    r = compose_vector_local_germ(
        u**2 / (u**2 + v**2), (x, 2 * x), (x, y), (0, 0), outer_variables=(u, v)
    )
    assert r.certified and r.limit == sp.Rational(1, 5)
    assert r.provider == "vector_image_constraint_composition"


def test_complex_holomorphic_certificate_avoids_realification():
    from asymptotic.multivariate_limits_advanced import complex_multivariate_limit

    z, w = sp.symbols("z w")
    r = complex_multivariate_limit(
        sp.exp(z * w) + z**2, (z, w), (1 + sp.I, 2 - sp.I), return_result=True
    )
    assert r.certified
    assert r.provider == "complex_holomorphic_substitution"


def test_complex_meromorphic_removable_certificate():
    from asymptotic.multivariate_limits_advanced import complex_multivariate_limit

    z = sp.symbols("z")
    r = complex_multivariate_limit((z**2 - 1) / (z - 1), z, 1, return_result=True)
    assert r.certified and r.value == 2
    assert r.provider in {
        "complex_holomorphic_substitution",
        "complex_meromorphic_removable_singularity",
    }


def test_vector_germ_semialgebraic_image_keeps_projected_inequality():
    from asymptotic.multivariate_limits_advanced import certify_vector_local_germ

    x, y = sp.symbols("x y", real=True)
    g = certify_vector_local_germ((x, x**2 + y**2), (x, y), (0, 0))
    assert g.certified
    assert g.image_provider.startswith("semialgebraic_local_image:")
    u, v = g.image_variables
    # The exact image is v >= u**2.
    assert sp.simplify(g.image_constraint.subs({u: 0, v: -1})) is sp.S.false
    assert sp.simplify(g.image_constraint.subs({u: 1, v: 1})) is sp.S.true


def test_vector_germ_semialgebraic_image_propagates_source_domain():
    from asymptotic.multivariate_limits_advanced import certify_vector_local_germ

    x = sp.symbols("x", real=True)
    g = certify_vector_local_germ((x, x**2), x, 0, domain=sp.Ge(x, 0))
    assert g.certified
    assert g.image_provider.startswith("semialgebraic_local_image:")
    u, v = g.image_variables
    assert sp.simplify(g.image_constraint.subs({u: -1, v: 1})) is sp.S.false
    assert sp.simplify(g.image_constraint.subs({u: 1, v: 1})) is sp.S.true


def test_vector_germ_rational_image_qe():
    from asymptotic.multivariate_limits_advanced import certify_vector_local_germ

    x = sp.symbols("x", real=True)
    g = certify_vector_local_germ((x / (1 + x**2), x**2), x, 0)
    assert g.certified
    assert g.image_provider.startswith("semialgebraic_local_image:")
    u, v = g.image_variables
    assert sp.simplify(g.image_constraint.subs({u: 0, v: 1})) is sp.S.false


def test_vector_image_constraint_certifies_branch_sensitive_radical():
    from asymptotic.multivariate_limits_advanced import compose_vector_local_germ

    x = sp.symbols("x", real=True)
    u, v = sp.symbols("u v", real=True)
    r = compose_vector_local_germ(
        sp.sqrt(v) / u,
        (x, x**2),
        x,
        0,
        outer_variables=(u, v),
        domain=sp.Ge(x, 0),
    )
    assert r.certified and r.limit == 1
    assert r.provider == "vector_image_constraint_composition"


def test_vector_image_constraint_certifies_logarithmic_singularity():
    from asymptotic.multivariate_limits_advanced import compose_vector_local_germ

    x = sp.symbols("x", real=True)
    u, v = sp.symbols("u v", real=True)
    r = compose_vector_local_germ(
        sp.log(u) / v,
        (1 + x**2, x**2),
        x,
        0,
        outer_variables=(u, v),
    )
    assert r.certified and r.limit == 1
    assert r.provider == "vector_image_constraint_composition"


def test_vector_image_constraint_certifies_piecewise_branch():
    from asymptotic.multivariate_limits_advanced import compose_vector_local_germ

    x = sp.symbols("x", real=True)
    u, v = sp.symbols("u v", real=True)
    outer = sp.Piecewise((v / u, u > 0), (sp.S.Zero, True))
    r = compose_vector_local_germ(
        outer,
        (x, x**2),
        x,
        0,
        outer_variables=(u, v),
        domain=sp.Ge(x, 0),
    )
    assert r.certified and r.limit == 0
    assert r.provider == "vector_image_constraint_composition"


def test_vector_local_image_uses_algebraic_function_graph():
    from asymptotic.multivariate_limits_advanced import certify_vector_local_germ

    x, y = sp.symbols("x y", real=True)
    g = certify_vector_local_germ((x, sp.sqrt(x**2 + y**2)), (x, y), (0, 0))
    assert g.certified
    assert g.image_provider.startswith("semialgebraic_local_image:")
    u, v = g.image_variables
    assert sp.simplify(g.image_constraint.subs({u: 0, v: -1})) is sp.S.false
    assert g.image_strata


def test_vector_local_image_discards_domain_stratum():
    from asymptotic.multivariate_limits_advanced import certify_vector_local_germ

    x = sp.symbols("x", real=True)
    domain = sp.Or(sp.Abs(x) < sp.Rational(1, 2), x > 10)
    g = certify_vector_local_germ((x,), x, 0, domain=domain)
    assert g.certified
    assert g.image_provider.startswith("semialgebraic_local_image:")
    assert g.image_strata
    z = g.image_variables[0]
    assert sp.simplify(g.image_constraint.subs({z: 11})) is sp.S.false


def test_vector_local_image_records_fixed_local_radius():
    from asymptotic.multivariate_limits_advanced import certify_vector_local_germ

    x = sp.symbols("x", real=True)
    g = certify_vector_local_germ((x, x**2), x, 0, domain=x < 2)
    assert g.local_radius == 1
    assert all(stratum.accumulating for stratum in g.image_strata)


def test_projection_planner_eliminates_explicit_before_qe():
    from asymptotic.multivariate_limits_advanced import _planned_image_projection

    x, y, u, v = sp.symbols("x y u v", real=True)
    source = sp.And(sp.Eq(u, x), sp.Eq(v, x + y), sp.Ge(y, 0))
    planned = _planned_image_projection(source, (u, v), (x, y))
    assert planned is not None
    projected, provider = planned
    assert provider in {
        "branch_local_qe",
        "equality_elimination",
        "reduced_monolithic_qe",
    }
    assert not (projected.free_symbols & {x, y})
    assert sp.simplify(projected.subs({u: 1, v: 0})) is sp.S.false


def test_vector_limit_and_singleton_joint_cluster_are_first_class():
    from asymptotic.multivariate_limits_advanced import (
        vector_cluster_set,
        vector_limit,
    )

    x, y = sp.symbols("x y", real=True)
    limit = vector_limit((x + y, x**2 + y**2), (x, y), (0, 0), return_result=True)
    assert limit.certified and limit.value == (0, 0)
    cluster = vector_cluster_set((x + y, x**2 + y**2), (x, y), (0, 0))
    assert cluster.certified
    assert cluster.cluster_set == sp.FiniteSet(sp.Tuple(0, 0))


def test_vector_cluster_preserves_joint_interval_product():
    from asymptotic.multivariate_limits_advanced import vector_cluster_set

    x, y = sp.symbols("x y", real=True)
    r2 = x**2 + y**2
    result = vector_cluster_set((x**2 / r2, y**2 / r2), (x, y), (0, 0))
    assert result.certified
    text = str(result.cluster_set)
    assert ("Eq" in text and ">= 0" in text) or (
        "ImageSet" in text and "Interval(0, 1)" in text
    )


def test_newton_fan_cluster_set_unions_all_certified_weight_regimes():
    from asymptotic.multivariate_limits_advanced import newton_fan_cluster_set

    x, y = sp.symbols("x y", real=True)
    expr = x**2 / (x**2 + y**2)
    result = newton_fan_cluster_set(expr, (x, y), (0, 0))
    assert result.certified
    assert result.cluster_set == sp.Interval(0, 1)
    assert result.liminf == 0 and result.limsup == 1


def test_vector_newton_cluster_preserves_joint_line_segment_geometry():
    from asymptotic.multivariate_limits_advanced import vector_cluster_set

    x, y = sp.symbols("x y", real=True)
    r2 = x**2 + y**2
    result = vector_cluster_set((x**2 / r2, y**2 / r2), (x, y), (0, 0))
    assert result.certified
    text = str(result.cluster_set)
    assert ("Eq" in text and "1" in text) or ("ImageSet" in text and "1 -" in text)


def test_oscillatory_scalar_cluster_interval():
    from asymptotic.multivariate_limits_advanced import oscillatory_cluster_set

    x, y = sp.symbols("x y", real=True)
    result = oscillatory_cluster_set(sp.sin(1 / (x**2 + y**2)), (x, y), (0, 0))
    assert result.certified
    assert result.cluster_set == sp.Interval(-1, 1)


def test_oscillatory_vector_cluster_circle():
    from asymptotic.multivariate_limits_advanced import oscillatory_vector_cluster_set

    x, y = sp.symbols("x y", real=True)
    phase = 1 / (x**2 + y**2)
    result = oscillatory_vector_cluster_set(
        (sp.cos(phase), sp.sin(phase)), (x, y), (0, 0)
    )
    assert result.certified
    assert "Eq" in str(result.cluster_set)


def test_projective_cluster_atlas_unions_signed_ends():
    from asymptotic.multivariate_limits_advanced import projective_cluster_atlas

    x = sp.symbols("x", real=True)
    result = projective_cluster_atlas(x / sp.sqrt(1 + x**2), x, ((sp.oo,), (-sp.oo,)))
    assert result.certified
    assert result.cluster_set == sp.FiniteSet(-1, 1)


def test_complex_meromorphic_geometry_reports_zero_and_pole_orders():
    from asymptotic.multivariate_limits_advanced import complex_meromorphic_geometry

    z = sp.symbols("z")
    zero = complex_meromorphic_geometry(z**3 / z, z, 0)
    assert zero.certified and zero.classification == "holomorphic" and zero.value == 0
    pole = complex_meromorphic_geometry(1 / z**2, z, 0)
    assert pole.certified and pole.classification == "pole"
    assert pole.denominator_order == 2


def test_oscillatory_cluster_scales_by_convergent_amplitude():
    from asymptotic.multivariate_limits_advanced import oscillatory_cluster_set

    x, y = sp.symbols("x y", real=True)
    phase = 1 / (x**2 + y**2)
    result = oscillatory_cluster_set((2 + x) * sp.sin(phase), (x, y), (0, 0))
    assert result.certified
    assert result.cluster_set == sp.Interval(-2, 2)


def test_complex_multivariate_normal_crossing_valuation():
    from asymptotic.multivariate_limits_advanced import complex_meromorphic_geometry

    z, w = sp.symbols("z w")
    result = complex_meromorphic_geometry(z**2 * w / (1 + z + w), (z, w), (0, 0))
    assert result.certified
    assert result.classification in {"holomorphic", "normal_crossing_zero"}
    assert result.value == 0


def test_joint_cluster_geometry_preserves_algebraic_curve():
    from asymptotic.multivariate_limits_advanced import joint_cluster_geometry

    x, y = sp.symbols("x y", real=True)
    r2 = x**2 + y**2
    u = x**2 / r2
    result = joint_cluster_geometry((u, u**2), (x, y), (0, 0))
    assert result.certified
    text = str(result.cluster_set)
    assert "ImageSet" in text


def test_joint_cluster_geometry_restricted_domain_piece():
    from asymptotic.multivariate_limits_advanced import joint_cluster_geometry

    x, y = sp.symbols("x y", real=True)
    r2 = x**2 + y**2
    result = joint_cluster_geometry(
        (x**2 / r2, y**2 / r2), (x, y), (0, 0), domain=sp.Ge(x, 0)
    )
    assert result.certified
    assert result.strata
    assert any(s.source_domain != sp.S.true for s in result.strata)


def test_joint_cluster_geometry_oscillatory_fiber_circle():
    from asymptotic.multivariate_limits_advanced import joint_cluster_geometry

    x, y = sp.symbols("x y", real=True)
    phase = 1 / (x**2 + y**2)
    result = joint_cluster_geometry((sp.cos(phase), sp.sin(phase)), (x, y), (0, 0))
    assert result.certified
    assert "ConditionSet" in str(result.cluster_set)


def test_oscillatory_certificate_does_not_replacement_cancellation():
    x, y = sp.symbols("x y", real=True)
    f = sp.sin(1 / x) - sp.cos(1 / y)
    r = bounded_factor_envelope_limit(f, (x, y), (0, 0))
    assert not r.certified


def test_bounded_factor_envelope_handles_sum_termwise():
    x, y = sp.symbols("x y", real=True)
    f = x * sp.sin(1 / x) + y**2 * sp.cos(1 / y)
    r = bounded_factor_envelope_limit(f, (x, y), (0, 0))
    assert r.certified and r.value == 0


def test_piecewise_common_vanishing_envelope():
    from asymptotic.limits import limit

    x, y = sp.symbols("x y", real=True)
    f = sp.Piecewise((x * sp.sin(1 / x), x > y), (y**2 * sp.cos(1 / y), True))
    r = limit(f, (x, y), (0, 0), return_result=True)
    assert r.status.name == "PROVED" and r.value == 0


def test_removable_sinc_like_multivariate_germs():
    from asymptotic.limits import limit

    x, y = sp.symbols("x y", real=True)
    cases = (
        (sp.sin(x * y) / (x * y), 1),
        (y * sp.sin(x) / x, 0),
        ((sp.exp(x * y) - 1) / y, 0),
        (sp.sin(x * y) / (sp.sin(x) * sp.sin(y)), 1),
    )
    for expression, expected in cases:
        result = limit(expression, (x, y), (0, 0), return_result=True)
        assert result.status.name == "PROVED" and result.value == expected


def test_oscillatory_composition_accumulation_range_is_not_a_limit():
    from asymptotic.limits import limit

    x, y = sp.symbols("x y", real=True)
    result = limit(
        sp.sin(1 / sp.Max(sp.Abs(x), sp.Abs(y))),
        (x, y),
        (0, 0),
        return_result=True,
    )
    assert result.status.name != "PROVED"
