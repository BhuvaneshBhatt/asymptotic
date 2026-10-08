import sympy as sp

from asymptotic import limit, one_sided_limit
from asymptotic.limit_models import LimitStatus
from asymptotic.reference_normalization import (
    RegularizedHypergeometric0F1,
    scalar_reference_equal,
    scalar_reference_namespace,
)
from asymptotic.stratification import AsymptoticStratification
from asymptotic.univariate_perturbation_limits import (
    gamma_small_shift_certificate,
    iterated_digamma_exponential_certificate,
    zeta_pole_projection_certificate,
    zeta_tail_certificate,
)


def test_step_factorial_integer_products_and_negative_reciprocals():
    ns = scalar_reference_namespace()
    x = sp.Symbol("x", nonzero=True)
    h = sp.Symbol("h")
    assert ns["StepFactorialPower"](x, 3, h) == x * (x - h) * (x - 2 * h)
    assert ns["StepFactorialPower"](x, -3, h) == 1 / (
        (x + h) * (x + 2 * h) * (x + 3 * h)
    )
    unknown = sp.Symbol("unknown")
    cells = limit(ns["StepFactorialPower"](unknown, -3, h), h, 0, return_result=True)
    assert isinstance(cells, AsymptoticStratification)
    assert {cell.result.status for cell in cells.strata} == {
        LimitStatus.PROVED,
        LimitStatus.DOES_NOT_EXIST,
    }
    n = sp.Symbol("n")
    result = limit(ns["StepFactorialPower"](x, n, h), n, 0, return_result=True)
    assert isinstance(result, AsymptoticStratification)
    assert {cell.result.status for cell in result.strata} == {
        LimitStatus.PROVED,
        LimitStatus.UNKNOWN,
    }


def test_generalized_inverse_error_translation_keeps_first_argument():
    ns = scalar_reference_namespace()
    a, b = sp.symbols("a b")
    assert ns["inverse_error_increment"](a, b) == sp.erfinv(sp.erf(a) + b)
    assert ns["inverse_error_increment"](sp.oo, b) == sp.erfinv(1 + b)


def test_regularized_zero_f_one_does_not_vanish_at_parameter_poles():
    z = sp.Symbol("z")
    assert RegularizedHypergeometric0F1(0, z) == z * sp.hyper((), (2,), z)
    assert RegularizedHypergeometric0F1(-2, z) == z**3 * sp.hyper((), (4,), z) / 6
    assert RegularizedHypergeometric0F1(-2, 1) != 0


def test_spherical_comparison_is_explicit_and_preserves_real_signs():
    assert not scalar_reference_equal(-sp.oo, sp.zoo)
    assert scalar_reference_equal(-sp.oo, sp.zoo, "extended_complex")
    assert not scalar_reference_equal(-sp.oo, sp.oo, "extended_complex")
    assert not scalar_reference_equal(sp.S.One, sp.zoo, "extended_complex")


def test_zeta_tail_products_and_shift_signs():
    x = sp.Symbol("x", positive=True)
    assert limit(x * (sp.zeta(x) - 1), x, sp.oo) == 0
    assert limit((sp.zeta(x + 2) - 1) / (sp.zeta(x) - 1), x, sp.oo) == sp.Rational(1, 4)
    for c in (2, -3):
        expression = (sp.zeta(x + c * sp.exp(-x)) - sp.zeta(x)) * sp.exp(
            x * (1 + sp.log(2))
        )
        assert limit(expression, x, sp.oo) == -c * sp.log(2)


def test_zeta_cancellation_and_complex_shift_are_declined():
    x = sp.Symbol("x", positive=True)
    assert zeta_tail_certificate(sp.zeta(x) - 1 - 2 ** (-x), x, sp.oo) is None
    e = (sp.zeta(x + sp.I * sp.exp(-x)) - sp.zeta(x)) * sp.exp(x * (1 + sp.log(2)))
    assert zeta_tail_certificate(e, x, sp.oo) is None


def test_gamma_small_shifts_preserve_sign_and_decay():
    x = sp.Symbol("x", positive=True)
    for c in (1, -2):
        e = (sp.gamma(x + c * sp.exp(-x)) - sp.gamma(x)) * sp.exp(x)
        assert limit(e, x, sp.oo) == sp.sign(c) * sp.oo
    e = sp.gamma(x + sp.exp(-x * x)) - sp.gamma(x)
    assert gamma_small_shift_certificate(e, x, sp.oo)[1] == 0
    assert (
        gamma_small_shift_certificate(sp.gamma(x + 1) - sp.gamma(x), x, sp.oo) is None
    )


def test_zeta_pole_projection_preserves_stieltjes_constant_and_error():
    x = sp.Symbol("x", positive=True)
    e = x * (x + sp.im(sp.zeta(1 + sp.I / x)))
    assert limit(e, x, sp.oo) == -sp.stieltjes(1)
    assert (
        zeta_pole_projection_certificate(x**6 * sp.im(sp.zeta(1 + sp.I / x)), x, sp.oo)
        is None
    )


def test_gamma_exponential_gap_and_nested_log_growth():
    x = sp.Symbol("x", positive=True)
    e = sp.exp(sp.exp(1 / x) * sp.gamma(x - sp.exp(-x))) - sp.exp(sp.gamma(x))
    assert limit(e, x, sp.oo) == sp.oo
    assert limit(sp.exp(-x) * sp.log(sp.gamma(sp.gamma(x))), x, sp.oo) == sp.oo
    e = (
        (sp.gamma(sp.exp(1 / sp.gamma(x))) - sp.polygamma(0, x) - 1)
        * sp.exp(x)
        / sp.log(x) ** 2
    )
    assert limit(e, x, sp.oo) == -sp.oo


def test_iterated_digamma_bound_requires_enough_denominator_growth():
    x = sp.Symbol("x", positive=True)
    t = x
    for _ in range(3):
        t = sp.polygamma(0, t)
    tower = t
    for _ in range(3):
        tower = sp.exp(tower)
    assert limit(tower / x, x, sp.oo) == 0
    assert (
        iterated_digamma_exponential_certificate(tower / sp.sqrt(x), x, sp.oo) is None
    )


def test_minmax_eventual_selection_avoids_indeterminate_infinities():
    x = sp.Symbol("x", positive=True)
    e = sp.Max(x, sp.exp(x)) / sp.log(sp.Min(sp.exp(-x), sp.exp(-sp.exp(x))))
    assert limit(e, x, sp.oo) == -1


def test_regularized_hypergeometric_fixed_parameter_decay():
    x = sp.Symbol("x", positive=True)
    a = sp.Symbol("a")
    assert limit(x * RegularizedHypergeometric0F1(a, x) * sp.exp(-x * x), x, sp.oo) == 0


def test_catalan_translation_retains_first_cancellation_coefficient():
    x = sp.Symbol("x", positive=True)
    ns = scalar_reference_namespace()
    e = x * (-1 + sp.sqrt(sp.pi) * x ** sp.Rational(3, 2) * ns["catalan"](x) / 4**x)
    assert limit(e, x, sp.oo) == -sp.Rational(9, 8)


def test_dawson_translation_preserves_exact_represented_float():
    x = sp.Symbol("x")
    ns = scalar_reference_namespace()
    e = ns["dawson"](sp.sqrt(sp.log(x) + sp.Float(1)))
    expected = sp.sqrt(sp.pi) * sp.exp(-1) * sp.erfi(sp.sqrt(sp.log(2) + 1)) / 4
    assert sp.simplify(limit(e, x, sp.Float(2)) - expected) == 0
    assert (
        sp.simplify(one_sided_limit(e, x, sp.Float(2), direction="-") - expected) == 0
    )
