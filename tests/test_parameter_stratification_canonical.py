import sympy as sp

from asymptotic.stratification import (
    normalize_parameter_condition,
    parameter_conditions_equivalent,
    stratify_parameter_cases,
)


def test_principal_radical_normalization_removes_repeated_factors_and_scalars():
    a = sp.symbols("a")
    assert parameter_conditions_equivalent(sp.Eq(4 * a**4, 0), sp.Eq(a, 0))
    assert parameter_conditions_equivalent(sp.Ne(-7 * a**3, 0), sp.Ne(a, 0))


def test_groebner_normalization_identifies_equivalent_linear_equality_ideals():
    a, b = sp.symbols("a b")
    left = sp.And(sp.Eq(a + b, 0), sp.Eq(a - b, 0))
    right = sp.And(sp.Eq(a, 0), sp.Eq(b, 0))
    assert parameter_conditions_equivalent(left, right)


def test_nonzero_atom_is_reduced_modulo_equality_ideal_and_detects_contradiction():
    a, b = sp.symbols("a b")
    condition = sp.And(sp.Eq(a - b, 0), sp.Ne(2 * a - 2 * b, 0))
    assert normalize_parameter_condition(condition) is sp.S.false


def test_boolean_equivalent_radical_branches_are_deduplicated():
    a = sp.symbols("a")
    condition = sp.Or(sp.Eq(a, 0), sp.Eq(a**2, 0))
    assert parameter_conditions_equivalent(normalize_parameter_condition(condition), sp.Eq(a, 0))


def test_stratum_order_is_deterministic_and_equivalent_results_coalesce():
    a = sp.symbols("a")
    first = stratify_parameter_cases(
        ((sp.Ne(a, 0), "generic"), (sp.Eq(a**2, 0), "special")),
        require_exhaustive=True,
    )
    second = stratify_parameter_cases(
        ((sp.Eq(-3 * a**3, 0), "special"), (sp.Ne(2 * a, 0), "generic")),
        require_exhaustive=True,
    )
    assert first.conditions == second.conditions
    assert tuple(s.result for s in first.strata) == tuple(s.result for s in second.strata)


def test_same_result_partially_redundant_branches_absorb_to_simpler_condition():
    a, b = sp.symbols("a b")
    strat = stratify_parameter_cases(
        (
            (sp.And(sp.Eq(a, 0), sp.Eq(b, 0)), "same"),
            (sp.And(sp.Eq(a, 0), sp.Ne(b, 0)), "same"),
            (sp.Ne(a, 0), "other"),
        ),
        require_exhaustive=True,
    )
    same = next(s for s in strat.strata if s.result == "same")
    assert parameter_conditions_equivalent(same.condition, sp.Eq(a, 0))
