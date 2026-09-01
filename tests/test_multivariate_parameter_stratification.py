import sympy as sp

from asymptotic import (
    dominant_balance_candidates,
    implicit_asymptotic,
    multivariate_dominant_balance_candidates,
)
from asymptotic.nonlinear_ode import (
    nonlinear_differential_dominant_balances,
    nonlinear_differential_transseries,
)
from asymptotic.stratification import AsymptoticStratification


def test_multivariate_scaling_path_anisotropic_balance():
    x, z, y = sp.symbols("x z y")
    balances = multivariate_dominant_balance_candidates(
        y**2 - x - z**2,
        y,
        (x, z),
        (2, 1),
    )
    assert len(balances) == 1
    balance = balances[0]
    assert balance.dependent_exponent == 1
    assert set(balance.coefficients) == {-sp.sqrt(2), sp.sqrt(2)}
    assert sp.expand(balance.transformed_equation) == y**2 - 2 * balance.path.parameter**2


def test_multivariate_scaling_path_changes_dominant_face():
    x, z, y = sp.symbols("x z y")
    balances = multivariate_dominant_balance_candidates(
        y**2 - x - z**2,
        y,
        (x, z),
        (4, 1),
    )
    assert len(balances) == 1
    assert balances[0].dependent_exponent == 1
    assert set(balances[0].coefficients) == {-1, 1}


def test_dominant_balance_automatically_stratifies_vanishing_coefficient():
    x, y, a = sp.symbols("x y a")
    result = dominant_balance_candidates((a + 1) * y - x, y, x)
    assert isinstance(result, AsymptoticStratification)
    assert result.exhaustive
    singular = result.select(sp.Eq(a, -1))
    generic = result.select(sp.Ne(a, -1))
    assert singular is not None and singular.result == ()
    assert generic is not None
    assert generic.result[0].coefficients == (1 / (a + 1),)


def test_implicit_solver_automatically_stratifies_parameter_case():
    x, y, a = sp.symbols("x y a")
    result = implicit_asymptotic((a + 1) * y - x, y, x, terms=2)
    assert isinstance(result, AsymptoticStratification)
    singular = result.select(sp.Eq(a, -1))
    generic = result.select(sp.Ne(a, -1))
    assert singular is not None and singular.result == ()
    assert generic is not None
    assert sp.simplify(generic.result[0].series.truncate() - x / (a + 1)) == 0


def test_nonlinear_ode_stratifies_characteristic_degeneracy_too():
    x, a = sp.symbols("x a")
    y = sp.Function("y")
    equation = (a + 1) * x * sp.diff(y(x), x) + y(x) - x

    balances = nonlinear_differential_dominant_balances(equation, y, x)
    assert isinstance(balances, AsymptoticStratification)
    exceptional = balances.select(sp.Eq(a, -2))
    ordinary = balances.select(sp.And(sp.Ne(a, -1), sp.Ne(a, -2)))
    assert exceptional is not None and exceptional.result == ()
    assert ordinary is not None
    assert ordinary.result[0].roots == (1 / (a + 2),)

    lifted = nonlinear_differential_transseries(equation, y, x, terms=2)
    assert isinstance(lifted, AsymptoticStratification)
    exceptional_lift = lifted.select(sp.Eq(a, -2))
    ordinary_lift = lifted.select(sp.And(sp.Ne(a, -1), sp.Ne(a, -2)))
    assert exceptional_lift is not None
    assert len(exceptional_lift.result) == 1
    assert sp.simplify(exceptional_lift.result[0].series + x * sp.log(x)) == 0
    assert exceptional_lift.result[0].steps[0].correction_kind == "logarithmic"
    assert ordinary_lift is not None
    assert sp.simplify(ordinary_lift.result[0].series - x / (a + 2)) == 0
