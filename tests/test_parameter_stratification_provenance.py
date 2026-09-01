import pytest
import sympy as sp

from asymptotic import (
    asymptotic_integrate,
    compose_transseries,
    transseries_from_expression,
)
from asymptotic.function_properties import (
    PropertyKnowledge,
    PropertyProvenance,
    analytic_at_decision,
    domain_contains_decision,
)
from asymptotic.function_properties.semantics import PropertyEnforcementError
from asymptotic.stratification import (
    AsymptoticStratification,
    ParameterStratum,
    evaluate_parameter_strata,
    stratify_parameter_cases,
    zero_nonzero_stratification,
)


def test_parameter_strata_are_disjoint_exhaustive_and_selectable():
    a = sp.symbols("a", real=True)
    strat = stratify_parameter_cases(
        ((a > 0, "positive"), (sp.Eq(a, 0), "zero"), (a < 0, "negative")),
        require_exhaustive=True,
    )
    assert isinstance(strat, AsymptoticStratification)
    assert strat.exhaustive is True
    assert strat.select(a > 0).result == "positive"
    assert strat.select(sp.Eq(a, 0)).result == "zero"


def test_overlapping_strata_are_rejected():
    a = sp.symbols("a", real=True)
    with pytest.raises(ValueError):
        stratify_parameter_cases(((a >= 0, 1), (a > 0, 2)))


def test_zero_nonzero_driver_evaluates_parameter_regimes():
    a = sp.symbols("a")
    strat = zero_nonzero_stratification((a,), lambda assumptions: assumptions)
    assert strat.exhaustive
    assert len(strat.strata) == 2
    assert any(s.condition.has(sp.Eq(a, 0)) for s in strat.strata)


def test_analyticity_decision_carries_provenance_and_enforces_branch_point():
    z = sp.symbols("z")
    at_zero = analytic_at_decision(sp.log(z), z, 0)
    assert at_zero.verdict is False
    assert at_zero.provenance
    assert at_zero.knowledge is PropertyKnowledge.EXACT

    at_one = analytic_at_decision(sp.log(z), z, 1)
    assert at_one.verdict is True


def test_domain_decision_is_tri_state_and_provenance_carrying():
    z = sp.symbols("z", real=True)
    assert domain_contains_decision(sp.log(z), z, 2, real=True).verdict is True
    assert domain_contains_decision(sp.log(z), z, -2, real=True).verdict is False


def test_composition_rejects_unknown_unregistered_outer_property_by_default():
    x, z = sp.symbols("x z", positive=True)
    F = sp.Function("F")
    inner = transseries_from_expression(1 + 1 / x, x, point=sp.oo)
    with pytest.raises(PropertyEnforcementError):
        compose_transseries(F(z), inner, argument=z)


def test_composition_records_property_decision_provenance():
    x, z = sp.symbols("x z", positive=True)
    inner = transseries_from_expression(1 + 1 / x, x, point=sp.oo)
    result = compose_transseries(sp.sin(z), inner, argument=z)
    decisions = result.metadata.get("property_decisions", [])
    assert decisions and decisions[-1].verdict is True
    assert result.metadata.get("operation_provenance")


def test_symbolic_integration_resonance_requires_stratification():
    x = sp.symbols("x", positive=True)
    a = sp.symbols("a", real=True)
    source = transseries_from_expression(x**a, x, point=sp.oo)
    with pytest.raises(PropertyEnforcementError):
        asymptotic_integrate(source, assumptions=sp.S.true)

    strat = evaluate_parameter_strata(
        (sp.Eq(a, -1), sp.Ne(a, -1)),
        lambda assumptions: asymptotic_integrate(
            source,
            assumptions=assumptions,
        ).truncate(),
        require_exhaustive=True,
    )
    assert len(strat.strata) == 2
    values = {sp.simplify(s.condition): s.result for s in strat.strata}
    assert any(sp.simplify(v - sp.log(x)) == 0 for v in values.values())


def test_manual_stratum_preserves_user_provenance():
    a = sp.symbols("a")
    prov = PropertyProvenance("test", reference="case split")
    s = ParameterStratum(sp.Eq(a, 0), 1, provenance=(prov,))
    assert s.provenance == (prov,)
