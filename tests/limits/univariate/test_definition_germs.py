import pytest
import sympy as sp

from asymptotic import limit
from asymptotic.definition_limit_germs import (
    axis_argument_certificate,
    divergent_integer_certificate,
    local_integral_certificate,
    stirling_remainder_certificate,
)
from asymptotic.function_normalization import RealRoot, normalize_functions
from asymptotic.integral_definition_germs import gaussian_log_moment_certificate
from asymptotic.limit_models import LimitStatus
from asymptotic.reference_contracts import parse_reference
from asymptotic.reference_normalization import scalar_reference_namespace


@pytest.mark.parametrize("integer_function", [sp.floor, sp.ceiling])
def test_integer_divergence(integer_function):
    x = sp.Symbol("x", real=True)
    result = limit(integer_function(1 / x), x, 0, return_result=True)
    assert result.status is LimitStatus.DOES_NOT_EXIST
    assert {e.value for e in result.evidence} == {sp.oo, -sp.oo}
    right = divergent_integer_certificate(
        integer_function(1 / x), x, sp.S.Zero, x > 0, sp.S.true
    )
    assert right[0:2] == (LimitStatus.PROVED, sp.oo)


def test_axis_floor():
    x = sp.Symbol("x", real=True)
    expression = sp.floor(sp.arg(sp.I * (sp.sqrt(x * x + 3) - sp.sqrt(3))))
    assert limit(expression, x, 0) == 1
    assert limit(sp.floor(sp.arg(-sp.I * x * x)), x, 0) == -2
    assert (
        axis_argument_certificate(
            sp.floor(sp.arg(x)), x, sp.S.Zero, sp.S.true, sp.S.true
        )
        is None
    )


def test_axis_oscillation_guard():
    x = sp.Symbol("x", real=True)
    expression = sp.floor(sp.arg(sp.I * sp.sin(1 / x)))
    assert (
        axis_argument_certificate(expression, x, sp.S.Zero, sp.S.true, sp.S.true)
        is None
    )


def test_logarithmic_ei_sides():
    x = sp.Symbol("x", real=True)
    expression = sp.log(x) - sp.Ei(sp.log(1 - x))
    result = limit(expression, x, 0, return_result=True)
    assert result.status is LimitStatus.DOES_NOT_EXIST
    assert {e.value for e in result.evidence} == {
        -sp.EulerGamma,
        -sp.EulerGamma + sp.I * sp.pi,
    }
    assert limit(-(x**sp.I) / sp.log(x) + sp.I * sp.Ei(sp.I * sp.log(x)), x, 0) == sp.pi


@pytest.mark.parametrize("center", [sp.S.Zero, sp.pi / 2])
def test_integral_cancellation(center):
    x, t = sp.symbols("x t", real=True)
    f = sp.exp(sp.sin(t))
    expression = (sp.Integral(f, (t, x, center)) + f.subs(t, center) * (x - center)) / (
        x - center
    ) ** 2
    assert limit(expression, x, center) == -sp.diff(f, t).subs(t, center) / 2


def test_integral_definition():
    ns = scalar_reference_namespace()
    expression = parse_reference("Integrate(exp(sin(t)), (t, x, pi/2))", ns)
    x, t = sp.symbols("x t")
    assert expression == sp.Integral(sp.exp(sp.sin(t)), (t, x, sp.pi / 2))
    opaque = sp.Function("Integrate")(sp.exp(sp.sin(t)), sp.Tuple(t, x, sp.pi / 2))
    reduced, evidence = normalize_functions(opaque, (x,), (sp.pi / 2,))
    assert reduced == opaque
    assert evidence == ()


def test_integral_guards():
    x, t = sp.symbols("x t", real=True)
    for expression in (
        sp.Integral(1 / t, (t, x, 0)) / x,
        sp.Integral(sp.Abs(t), (t, x, 0)) / x**2,
        sp.Integral(sp.exp(t), (t, x, 0)) / sp.Abs(x),
        sp.Integral(sp.exp(t), (t, x, 0)) / x**8,
    ):
        assert (
            local_integral_certificate(expression, x, sp.S.Zero, sp.S.true, sp.S.true)
            is None
        )


def test_inverse_gaussian_moment():
    x = sp.Symbol("x", positive=True)
    y = sp.erfinv(1 - x * x)
    h = sp.hyper((sp.S.Half, sp.S.Half), (sp.Rational(3, 2), sp.Rational(3, 2)), -y * y)
    expression = (1 - x * x) * sp.log(y) - 2 * y * h / sp.sqrt(sp.pi)
    assert limit(expression, x, 0) == -sp.log(2) - sp.EulerGamma / 2
    outside = expression.subs(x * x, -x)
    result = gaussian_log_moment_certificate(
        outside, x, sp.S.Zero, sp.S.true, sp.S.true
    )
    assert result[0] is LimitStatus.UNKNOWN
    assert result[2].method == "inverse_error_complex_branch_required"


def test_bessel_real_root():
    x = sp.Symbol("x")
    expression = RealRoot(sp.besseli(1, 3 * x) / (x * sp.besseli(1, x) ** 3), 2)
    assert (
        limit(expression, x, sp.oo)
        == sp.sqrt(2) * 3 ** sp.Rational(3, 4) * sp.sqrt(sp.pi) / 3
    )
    unknown = RealRoot(sp.Function("f")(x), 2)
    assert normalize_functions(unknown, (x,), (sp.oo,))[0] == unknown


def test_stirling_remainders():
    x = sp.Symbol("x", positive=True)
    scale = sp.sqrt(2 * sp.pi) * x ** (x + sp.S.Half) * sp.exp(-x)
    linear = sp.gamma(x + 1) / sp.sqrt(2 * sp.pi) - (
        x ** (x + sp.S.Half) + x ** (x - sp.S.Half) / 12
    ) * sp.exp(-x)
    assert limit(linear, x, sp.oo) is sp.oo
    difference = sp.sqrt(2 * sp.pi) * x ** (x + sp.S.Half) * sp.exp(
        1 / (12 * x)
    ) - sp.factorial(x) * sp.exp(x)
    expression = sp.diff(difference**2, x) / sp.diff(sp.exp(2 * x), x)
    assert limit(expression, x, sp.oo) is sp.oo
    assert stirling_remainder_certificate(sp.gamma(x + 1) / scale - 1, x, sp.oo) is None
    # The subtracted approximation agrees through quadratic order; the cubic
    # coefficient follows independently from the logarithmic expansion.
    t = sp.Symbol("t")
    log_tail = t / 12 - t**3 / 360
    difference = (sp.exp(t / 12) - sp.exp(log_tail)).series(t, 0, 4).removeO()
    assert difference == t**3 / 360


def test_gamma_parameter_gap():
    x, a = sp.symbols("x a")
    result = limit(sp.sin(x) * sp.gamma(2 - a * x) / x, x, sp.oo, return_result=True)
    assert result.status is LimitStatus.UNKNOWN
    assert result.evidence[0].method == "gamma_tail_parameter_strata_required"
    assert limit(sp.sin(x) * sp.gamma(2) / x, x, sp.oo) == 0
