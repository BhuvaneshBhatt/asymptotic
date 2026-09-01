import sympy as sp

from asymptotic import implicit_asymptotic
from asymptotic.implicit import implicit_singularity_profile
from asymptotic.stratification import AsymptoticStratification


def test_turning_point_profile_detects_multiplicity_and_newton_scale():
    x, y = sp.symbols("x y", positive=True)
    profile = implicit_singularity_profile(y**2 - x, y, x)

    assert profile.multiplicity == 2
    assert profile.singular is True
    assert profile.turning_point is True
    assert profile.requires_blowup is True
    assert profile.scaling_exponents == (sp.Rational(1, 2),)

    branches = implicit_asymptotic(y**2 - x, y, x, terms=2)
    assert len(branches) == 2
    assert all(branch.method == "newton-puiseux-blowup" for branch in branches)
    assert all(branch.singularity is not None for branch in branches)


def test_cubic_multiple_root_uses_ramified_scaling_analysis():
    x, y = sp.symbols("x y", positive=True)
    profile = implicit_singularity_profile(y**3 - x**2, y, x)
    assert profile.multiplicity == 3
    assert profile.scaling_exponents == (sp.Rational(2, 3),)

    branches = implicit_asymptotic(y**3 - x**2, y, x, terms=2)
    assert len(branches) == 3
    assert all(branch.method == "newton-puiseux-blowup" for branch in branches)
    assert {branch.series.leading_term.exponent for branch in branches} == {sp.Rational(2, 3)}


def test_nonzero_center_parameter_multiplicity_is_stratified_before_lifting():
    x, y, a = sp.symbols("x y a")
    equation = (y - 1) ** 2 + a * (y - 1) - x
    result = implicit_asymptotic(equation, y, x, dependent_limit=1, terms=3)

    assert isinstance(result, AsymptoticStratification)
    by_condition = {str(stratum.condition): stratum.result for stratum in result.strata}
    singular = by_condition["Eq(a, 0)"]
    regular = by_condition["Ne(a, 0)"]

    assert len(singular) == 2
    assert all(branch.singularity.multiplicity == 2 for branch in singular)
    assert all(branch.method == "newton-puiseux-blowup" for branch in singular)
    assert {branch.series.leading_term.exponent for branch in singular} == {sp.Rational(0)}
    assert all(branch.singularity.multiplicity == 1 for branch in regular)
    assert all(branch.method == "regular-dominant-balance" for branch in regular)


def test_regular_implicit_root_is_not_mislabeled_singular():
    x, y = sp.symbols("x y", positive=True)
    profile = implicit_singularity_profile(sp.sin(y) - x, y, x)
    assert profile.multiplicity == 1
    assert profile.singular is False
    assert profile.requires_blowup is False
    branch = implicit_asymptotic(sp.sin(y) - x, y, x, terms=3)[0]
    assert branch.method == "regular-dominant-balance"
