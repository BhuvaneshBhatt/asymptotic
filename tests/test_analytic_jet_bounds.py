"""Taylor upper bounds never supply denominator lower bounds."""

import pytest
import sympy as sp

from asymptotic.multivariate_limits_advanced import analytic_jet_radial_vanishing_limit


@pytest.mark.parametrize(
    "family",
    [
        "denominator_curve",
        "gamma_pole",
        "branch_jump",
        "sign_denominator",
        "root_denominator",
    ],
)
def test_jet_declines_singular_bounds(family):
    x, y = sp.symbols("x y", real=True)
    expressions = {
        "denominator_curve": (sp.sin(x * x + y * y) - x * y * sp.exp(x))
        / (sp.exp(x + y) - 1),
        "sign_denominator": x / (1 + sp.sign(x) + x),
        "root_denominator": x / (sp.sqrt(-1 + sp.I * x) + sp.I),
        "gamma_pole": x * sp.gamma(x) ** 2,
        "branch_jump": (sp.sqrt(-1 + sp.I * x) - sp.I) / sp.sqrt(x * x + y * y),
    }
    result = analytic_jet_radial_vanishing_limit(expressions[family], (x, y), (0, 0))
    assert not result.certified


def test_jet_coercive_denominator():
    x, y = sp.symbols("x y", real=True)
    result = analytic_jet_radial_vanishing_limit(
        sp.sin(x * x + y * y) ** 2 / (x * x + y * y), (x, y), (0, 0)
    )
    assert result.certified
    assert result.value == 0


def test_finite_center_does_not_imply_analyticity():
    class FiniteCenter(sp.Function):
        @classmethod
        def eval(cls, argument):
            if argument == 0:
                return sp.S.One

    x, y = sp.symbols("x y", real=True)
    result = analytic_jet_radial_vanishing_limit(x * FiniteCenter(x), (x, y), (0, 0))
    assert not result.certified
