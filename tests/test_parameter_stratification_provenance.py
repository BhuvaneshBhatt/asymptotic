import pytest
import sympy as sp
from funcprops import (
    PropertyEnforcementError,
    PropertyKnowledge,
    PropertyProvenance,
    domain,
)

from asymptotic import integrate
from asymptotic._property_support import analytic_at_decision
from asymptotic.general_ops import (
    compose_transseries,
)
from asymptotic.stratification import (
    AsymptoticStratification,
    ParameterStratum,
    evaluate_parameter_strata,
    stratify_parameter_cases,
    zero_nonzero_stratification,
)
from asymptotic.transseries import (
    transseries_from_expression,
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


def test_analyticity_decision_carries_provenance_branch_point():
    z = sp.symbols("z")
    at_zero = analytic_at_decision(sp.log(z), z, 0)
    assert at_zero.verdict is False
    assert at_zero.provenance
    assert at_zero.knowledge is PropertyKnowledge.EXACT

    at_one = analytic_at_decision(sp.log(z), z, 1)
    assert at_one.verdict is True


def test_domain_decision_comes_from_shared_funcprops_backend():
    z = sp.symbols("z", real=True)
    natural_domain = domain(sp.log(z), z)
    assert sp.simplify(natural_domain.subs(z, 2)) is sp.S.true
    assert sp.simplify(natural_domain.subs(z, -2)) is sp.S.false


def test_composition_rejects_unknown_unregistered_by_default():
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
        integrate(source, assumptions=sp.S.true, return_result=True)

    strat = evaluate_parameter_strata(
        (sp.Eq(a, -1), sp.Ne(a, -1)),
        lambda assumptions: integrate(
            source, assumptions=assumptions, return_result=True
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


def test_empty_parameter_family():
    family = AsymptoticStratification(
        (),
        (ParameterStratum(sp.S.true, sp.Integer(42)),),
        assumptions=sp.S.false,
        exhaustive=True,
    )
    assert family.select() is None
    assert family.mathematical_value is None


def test_empty_selection_assumptions():
    a = sp.Symbol("a", real=True)
    family = AsymptoticStratification(
        (a,),
        (ParameterStratum(sp.S.true, sp.Integer(42)),),
        assumptions=a > 0,
    )
    assert family.select(sp.S.false) is None
    assert family.select(sp.Not(a > 0)) is None
    assert family.select().result == 42


def test_value_projection_protocol():
    from types import SimpleNamespace

    from asymptotic.conditional import mathematical_result

    family = SimpleNamespace(
        parameters=(),
        strata=(ParameterStratum(sp.S.true, sp.Integer(7)),),
        exhaustive=True,
    )
    assert mathematical_result(family) == 7
