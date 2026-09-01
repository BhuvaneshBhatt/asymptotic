import pytest
import sympy as sp

from asymptotic import RemainderKind
from asymptotic.asymptotic_field import (
    IntegrationConstantLocation,
    asymptotic_differential_field,
)
from asymptotic.remainder_theorems import (
    certify_frechet_inverse_operator_remainder,
    certify_green_inverse_operator_remainder,
)


def test_shackell_integral_shadow_formula_replays_for_unevaluated_integral():
    x = sp.symbols("x", positive=True)
    field = asymptotic_differential_field(x, (1 / sp.log(x), 1 / x))
    extension = field.add_integral_extension(
        1 / x,
        primitive=sp.Integral(1 / x, x),
        leading_monomial=sp.log(x),
    )
    projection = field.project_integral(1, extension)
    assert projection.certified
    assert projection.replay() is True
    assert sp.simplify(projection.normalized_shadow - 1) == 0
    assert projection.normalized_ghost == 0
    assert projection.constant_location is IntegrationConstantLocation.FIXED


def test_integral_constant_can_belong_to_shadow():
    x = sp.symbols("x", positive=True)
    k = sp.symbols("k", nonzero=True, real=True)
    field = asymptotic_differential_field(x, (1 / x,))
    extension = field.add_integral_extension(
        1 / x,
        primitive=sp.Integral(1 / x, x),
        constant=k,
        leading_monomial=sp.log(x),
    )
    projection = field.project_integral(0, extension)
    assert projection.constant_location is IntegrationConstantLocation.SHADOW
    assert sp.simplify(projection.normalized_shadow - (1 + k / sp.log(x))) == 0
    assert projection.normalized_ghost == 0


def test_integral_constant_can_belong_to_ghost():
    x = sp.symbols("x", positive=True)
    k = sp.symbols("k", nonzero=True, real=True)
    field = asymptotic_differential_field(x, (1 / x,))
    extension = field.add_integral_extension(
        sp.exp(x),
        primitive=sp.Integral(sp.exp(x), x),
        constant=k,
        leading_monomial=sp.exp(x),
    )
    projection = field.project_integral(0, extension)
    assert projection.constant_location is IntegrationConstantLocation.GHOST
    assert sp.simplify(projection.normalized_shadow - 1) == 0
    assert sp.simplify(projection.normalized_ghost - k * sp.exp(-x)) == 0
    assert projection.certified


def test_integral_constant_that_changes_leading_scale_is_rejected():
    x = sp.symbols("x", positive=True)
    k = sp.symbols("k", nonzero=True, real=True)
    field = asymptotic_differential_field(x, (sp.exp(-x),))
    extension = field.add_integral_extension(
        sp.exp(-x),
        primitive=sp.Integral(sp.exp(-x), x),
        constant=k,
        leading_monomial=-sp.exp(-x),
    )
    with pytest.raises(ValueError, match="changes the leading monomial"):
        field.project_integral(0, extension)


def test_second_order_green_dichotomy_certifies_slow_forcing():
    x = sp.symbols("x", positive=True)
    delta = sp.Function("delta")
    operator = sp.diff(delta(x), x, 2) - delta(x)
    cert, green = certify_green_inverse_operator_remainder(
        sp.exp(-x / 2), operator, delta, x, sp.oo
    )
    assert cert.certified
    assert cert.conclusion.kind is RemainderKind.BIG_O
    assert green is not None and green.exact_right_inverse
    assert green.dichotomy is not None and green.dichotomy.certified
    assert len(green.dichotomy.stable_modes) == 1
    assert len(green.dichotomy.unstable_modes) == 1


def test_second_order_green_refuses_uncontrolled_slower_homogeneous_mode():
    x = sp.symbols("x", positive=True)
    delta = sp.Function("delta")
    operator = sp.diff(delta(x), x, 2) - delta(x)
    cert, _ = certify_green_inverse_operator_remainder(
        sp.exp(-3 * x / 2), operator, delta, x, sp.oo
    )
    assert not cert.certified
    assert cert.conclusion.kind is RemainderKind.UNKNOWN
    controls = [
        h for h in cert.hypotheses if str(h.predicate) == "green_homogeneous_modes_controlled"
    ]
    assert controls and controls[0].verdict is not True


def test_frechet_wrapper_dispatches_higher_order_green_theorem():
    x = sp.symbols("x", positive=True)
    delta = sp.Function("delta")
    operator = sp.diff(delta(x), x, 2) - delta(x)
    cert = certify_frechet_inverse_operator_remainder(sp.exp(-x / 2), operator, delta, x, sp.oo)
    assert cert.certified
    assert "Green" in cert.theorem


def test_asymptotically_constant_green_certifies_small_coefficient_perturbation():
    x = sp.symbols("x", positive=True)
    delta = sp.Function("delta")
    operator = sp.diff(delta(x), x, 2) + sp.diff(delta(x), x) / x - delta(x)
    cert, green = certify_green_inverse_operator_remainder(
        sp.exp(-x / 2), operator, delta, x, sp.oo
    )

    assert cert.certified
    assert cert.conclusion.kind is RemainderKind.BIG_O
    assert "asymptotically-constant" in cert.theorem
    assert green is not None and green.asymptotically_constant
    assert green.limiting_coefficients == (-1, 0, 1)
    assert green.perturbation_limits == (0, 0, 0)
    assert green.dichotomy is not None and green.dichotomy.certified
    # The limiting Green particular is asymptotic rather than an exact right
    # inverse for the variable-coefficient operator.
    assert not green.exact_right_inverse
    assert green.replay_asymptotic(x) is True
    assert sp.simplify(green.defect / sp.exp(-x / 2) + sp.Rational(2, 3) / x) == 0
    predicates = {str(h.predicate) for h in cert.hypotheses}
    assert "green_limiting_exact_right_inverse" in predicates
    assert "green_full_operator_defect_small" in predicates


def test_asymptotically_constant_green_normalizes_variable_leading_coefficient():
    x = sp.symbols("x", positive=True)
    delta = sp.Function("delta")
    operator = (1 + 1 / x) * sp.diff(delta(x), x, 2) - delta(x)
    cert, green = certify_green_inverse_operator_remainder(
        (1 + 1 / x) * sp.exp(-x / 2), operator, delta, x, sp.oo
    )

    assert cert.certified
    assert green is not None and green.asymptotically_constant
    assert green.limiting_coefficients == (-1, 0, 1)


def test_asymptotically_constant_green_refuses_nonconvergent_coefficients():
    x = sp.symbols("x", positive=True)
    delta = sp.Function("delta")
    operator = sp.diff(delta(x), x, 2) + sp.sin(x) * sp.diff(delta(x), x) - delta(x)
    cert, green = certify_green_inverse_operator_remainder(
        sp.exp(-x / 2), operator, delta, x, sp.oo
    )

    assert not cert.certified
    assert cert.conclusion.kind is RemainderKind.UNKNOWN
    assert green is not None and green.asymptotically_constant
    convergence = [
        h for h in cert.hypotheses if str(h.predicate) == "green_normalized_coefficients_converge"
    ]
    assert convergence and convergence[0].verdict is not True
