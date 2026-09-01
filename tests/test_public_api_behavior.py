"""Behavioral contracts for root APIs not covered by focused test modules."""

import sympy as sp

from asymptotic import (
    AsymptoticTruncation,
    NestedExpansion,
    multiseries,
    nested_expansion,
)
from asymptotic.asymptotic_field import (
    AsymptoticDifferentialField,
    asymptotic_differential_field,
)
from asymptotic.decomposition import decompose_expression
from asymptotic.function_properties.query import branch_safe_substitution_decision
from asymptotic.function_properties.semantics import PropertyDecision
from asymptotic.nested import NestedForm
from asymptotic.parameter_auto import automatic_parameter_stratification
from asymptotic.stratification import AsymptoticStratification


def test_field_and_truncation_result_types_have_behavioral_contracts():
    x = sp.symbols("x", positive=True)
    field = asymptotic_differential_field(x, (1 / x,))
    assert isinstance(field, AsymptoticDifferentialField)
    assert sp.simplify(field.element(1 / x).differentiate().as_expr() + 1 / x**2) == 0

    truncation = multiseries(sp.exp(1 / x), x, terms=4).asymptotic_element().truncation(2)
    assert isinstance(truncation, AsymptoticTruncation)
    assert truncation.remainder.check() is True


def test_nested_result_types_have_behavioral_contracts():
    x = sp.symbols("x", positive=True)
    expansion = nested_expansion(sp.log(x) + 1 / x, x, depth=1)
    assert isinstance(expansion, NestedExpansion)
    form = expansion.forms[0]
    assert isinstance(form, NestedForm)
    assert sp.simplify(form.exact_expr - expansion.expr) == 0


def test_property_decision_and_structural_decomposition_are_actionable():
    x, z = sp.symbols("x z", positive=True)
    decision = branch_safe_substitution_decision(sp.log(z), z, 1)
    assert isinstance(decision, PropertyDecision)
    assert decision.verdict is True

    decomposition = decompose_expression(sp.exp(x) * sp.log(x), x)
    assert sp.simplify(decomposition.canonical - sp.exp(x) * sp.log(x)) == 0
    rebuilt = decomposition.rationalized.xreplace(dict(decomposition.substitutions))
    assert sp.simplify(rebuilt - decomposition.original) == 0


def test_automatic_parameter_stratification_returns_explicit_cases():
    a, x = sp.symbols("a x")
    result = automatic_parameter_stratification(
        (a * x,),
        lambda condition: condition,
        parameters=(a,),
    )
    assert isinstance(result, AsymptoticStratification)
    assert result.strata
