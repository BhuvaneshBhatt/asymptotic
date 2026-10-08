import sympy as s

from asymptotic import complex_ray_limit, limit
from asymptotic.function_normalization import normalize_functions
from asymptotic.limit_models import LimitStatus
from asymptotic.path_limits import _finite_under_assumptions
from asymptotic.rational_pullback_germs import (
    local_real_sign_certificate,
    vanishing_remainder,
)


def test_rational_cut_pullbacks_with_quadratic_contact():
    z = s.Symbol("z")
    for f in (s.Ei, s.Ci, s.Chi):
        center = f(-1) if f is s.Ei else f(1)
        for sign in (1, -1):
            expr = f(-1 + sign * s.I * z**2 / (1 + z))
            assert (
                s.simplify(
                    complex_ray_limit(expr, z, 0, ray=1) - center - sign * s.I * s.pi
                )
                == 0
            )


def test_nonzero_does_not_imply_finite_and_explicit_finiteness():
    a = s.Symbol("a")
    L = s.Symbol("L", real=True)
    assert not _finite_under_assumptions(a, s.Ne(a, 0))
    assert _finite_under_assumptions(a, s.Q.finite(a))
    assert _finite_under_assumptions(s.pi / L, L > 0)
    assert _finite_under_assumptions(s.pi / L, s.Eq(L, 1))
    from asymptotic import one_sided_limit

    z = s.Symbol("z")
    result = one_sided_limit(z - s.I / 3, z, s.I / 3, direction="+", return_result=True)
    assert result.status is LimitStatus.PROVED and result.value == 0
    assert result.domain == s.And(s.Eq(s.im(z) - s.Rational(1, 3), 0), s.re(z) > 0)
    x = s.Symbol("x", real=True)
    assert (
        limit(s.sign(x), x, 0, assumptions=False, return_result=True).status
        is LimitStatus.UNKNOWN
    )
    assert (
        limit(x, x, 0, domain=False, return_result=True).status is LimitStatus.UNKNOWN
    )


def test_local_sign_and_step_attained_discontinuities():
    x = s.Symbol("x", real=True)
    for f in (s.sign, s.Heaviside):
        result = local_real_sign_certificate(f(x / (1 + x)), x, 0, s.true, s.true)
        assert result[0] is LimitStatus.DOES_NOT_EXIST
        assert len(result[2]) == 2
        assert all(e.substitutions for e in result[2])
    result = local_real_sign_certificate(s.Heaviside(x * x), x, 0, s.true, s.true)
    assert result[0] is LimitStatus.PROVED and result[1] == 1
    assert limit(s.sign(x * x), x, 0) == 1


def test_function_and_log_contracts():
    z = s.Symbol("z")
    expr = s.Function("unspecified")(z)
    assert normalize_functions(expr, (z,), (0,)) == (expr, ())
    assert (
        limit(s.log(-1 + s.I * z), z, 0, return_result=True).status
        is LimitStatus.DOES_NOT_EXIST
    )


def test_rational_remainder_planning():
    x = s.Symbol("x", positive=True)
    assert vanishing_remainder((x + 1) / (x * x + 2), x, s.oo)
    assert not vanishing_remainder(x / (x + 1), x, s.oo)


def test_bounded_recurrence_rational_punctured_denominator():
    from asymptotic.recurrence_simplification import simplify_recurrences

    x = s.Symbol("x", real=True)
    u = x / (1 + x)
    expression = s.polygamma(0, u + 1) - s.polygamma(0, u)
    result = simplify_recurrences(
        expression, variables=(x,), target=(0,), return_result=True
    )
    assert s.cancel(result.expression - (1 + x) / x) == 0
    assert result.evidence
    a = s.Symbol("a")
    unresolved = s.polygamma(0, a + 1) - s.polygamma(0, a)
    assert simplify_recurrences(unresolved) == unresolved
