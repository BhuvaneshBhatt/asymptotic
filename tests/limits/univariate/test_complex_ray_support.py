import pytest
import sympy as sp
from funcprops import ParametricContour

from asymptotic import complex_ray_limit, limit, one_sided_limit, path_limit
from asymptotic.limit_certificates import branch_boundary_certificate
from asymptotic.limit_models import LimitStatus


def test_log_and_sqrt_signed_fixed_rays():
    z = sp.Symbol("z")
    for d, sign in [(sp.I, 1), (-sp.I, -1), (1 + sp.I, 1), (1 - sp.I, -1)]:
        assert complex_ray_limit(sp.log(z), z, -1, ray=d) == sign * sp.I * sp.pi
        assert complex_ray_limit(sp.sqrt(z), z, -1, ray=d) == sign * sp.I
    assert complex_ray_limit(sp.log(z), z, -1, ray=1) == sp.I * sp.pi


def test_regular_rational_and_entire_complex_targets():
    z = sp.Symbol("z")
    p = sp.I
    r = complex_ray_limit(
        (z * z + 1) / (z - sp.I), z, p, ray=2 + sp.I, return_result=True
    )
    assert r.status is LimitStatus.PROVED and r.value == 2 * sp.I
    assert (
        r.variables == (z,)
        and r.target == (p,)
        and r.expression == (z * z + 1) / (z - sp.I)
    )
    assert complex_ray_limit(sp.exp(z) + sp.erf(z), z, p, ray=sp.I) == sp.exp(
        p
    ) + sp.erf(p)


def test_integral_cut_rays():
    z = sp.Symbol("z")
    for fn in [sp.Ei, sp.Ci, sp.Chi]:
        base = sp.Ei(-1) if fn is sp.Ei else fn(1)
        for d, sign in [(sp.I, 1), (-sp.I, -1)]:
            assert complex_ray_limit(fn(z), z, -1, ray=d) == base + sign * sp.I * sp.pi
    assert complex_ray_limit(sp.Ei(z), z, -1, ray=1) == sp.Ei(-1)
    # Numeric off-cut evaluations independently verify the branch convention.
    for fn in [sp.Ei, sp.Ci, sp.Chi]:
        value = complex_ray_limit(fn(z), z, -1, ray=-sp.I)
        assert (
            abs(complex((fn(-1 - sp.I / sp.Integer(10) ** 8) - value).evalf(25))) < 1e-6
        )


def test_approach_assumptions_are_pulled_back():
    z = sp.Symbol("z")
    x = sp.Symbol("x", real=True)
    assert one_sided_limit(sp.Abs(x) / x, x, 0, direction="+", assumptions=x > 0) == 1
    r = one_sided_limit(
        sp.Abs(x) / x, x, 0, direction="+", assumptions=x < 0, return_result=True
    )
    assert r.status is LimitStatus.UNKNOWN
    r = complex_ray_limit(
        z, z, 0, ray=sp.I, assumptions=sp.im(z) < 0, return_result=True
    )
    assert r.status is LimitStatus.UNKNOWN


def test_ray_validation_and_restricted_variable_domains():
    z = sp.Symbol("z")
    p = sp.Symbol("p", positive=True)
    for d in [0, sp.oo, sp.nan, z]:
        with pytest.raises(ValueError):
            complex_ray_limit(z, z, 0, ray=d)
    with pytest.raises(ValueError):
        complex_ray_limit(p, p, 0, ray=-1)
    with pytest.raises(ValueError):
        complex_ray_limit(p, p, 0, ray=sp.I)
    with pytest.raises(ValueError):
        one_sided_limit(p, p, 0, direction="-")
    with pytest.raises(ValueError):
        complex_ray_limit(z, z, sp.oo, ray=sp.I)


def test_cut_constants_do_not_erase_amplified_germs():
    t = sp.Symbol("t", positive=True)
    expr = (sp.log(-1 + sp.I * t) - sp.I * sp.pi) / t
    assert branch_boundary_certificate(expr, t, sp.S.Zero, sp.S.true, sp.S.true) is None
    expr = sp.sin((sp.log(-1 + sp.I * t) - sp.I * sp.pi) / t)
    assert branch_boundary_certificate(expr, t, sp.S.Zero, sp.S.true, sp.S.true) is None
    # Declining the shortcut must not invent a zero result.
    r = limit((sp.log(-1 + sp.I * t) - sp.I * sp.pi) / t, t, 0, return_result=True)
    assert r.status is LimitStatus.UNKNOWN or sp.simplify(r.value + sp.I) == 0


def test_attained_cut_witnesses_and_wrapper_metadata():
    x = sp.Symbol("x", real=True)
    r = limit(sp.log(-1 + sp.I * x), x, 0, return_result=True)
    assert r.status is LimitStatus.DOES_NOT_EXIST
    assert len(r.evidence) == 2 and {e.value for e in r.evidence} == {
        sp.I * sp.pi,
        -sp.I * sp.pi,
    }
    assert all(e.substitutions[0][0] == x for e in r.evidence)
    r = one_sided_limit(sp.ceiling(x), x, 0, direction="+", return_result=True)
    assert r.variables == (x,) and r.target == (0,) and r.domain == (x > 0)
    assert all(
        v == x
        for e in r.evidence
        if e.method == "attained_local_real_sign_chart"
        for v, s in e.substitutions
    )


def test_contour_assumptions_and_constant_path_guard():
    x, t = sp.symbols("x t", real=True)
    path = ParametricContour(t, (t, 0, 1))
    r = path_limit(x, x, 1, path=path, assumptions=x > 2, return_result=True)
    assert r.status is LimitStatus.UNKNOWN
    with pytest.raises(ValueError):
        path_limit(x, x, 0, path=ParametricContour(sp.S.Zero, (t, 0, 1)))


def test_unknown_integral_branch_and_unsupported_poles_stay_visible():
    z = sp.Symbol("z")
    d = sp.Symbol("d", finite=True, zero=False)
    for fn in [sp.Ei, sp.Ci, sp.Chi]:
        assert (
            complex_ray_limit(fn(z), z, -1, ray=d, return_result=True).status
            is LimitStatus.UNKNOWN
        )
    assert (
        complex_ray_limit(sp.acos(z), z, -2, ray=sp.I, return_result=True).status
        is LimitStatus.UNKNOWN
    )
    from asymptotic.fixed_ray_branch_germs import DirectionalInfinity

    pole = complex_ray_limit(1 / z, z, 0, ray=sp.I, return_result=True)
    assert pole.status is LimitStatus.PROVED and pole.value == DirectionalInfinity(
        -sp.I
    )


def test_finite_target_proved_by_parameter_assumptions():
    z, L = sp.symbols("z L")
    p = sp.pi / L
    assert one_sided_limit(z - p, z, p, direction="+", assumptions=L > 0) == 0
    assert complex_ray_limit(z - p, z, p, ray=sp.I, assumptions=L > 0) == 0
