from __future__ import annotations

import pytest
import sympy as sp

from asymptotic.limits import (
    LimitStatus,
    SimultaneousLimitDoesNotExist,
    limit,
    normalize_limit_target,
)


def test_target_normalization_scalar_sequence_and_mapping():
    x, y = sp.symbols("x y")
    assert normalize_limit_target(x, 2) == (sp.Integer(2),)
    assert normalize_limit_target((x, y), [1, 2]) == (sp.Integer(1), sp.Integer(2))
    assert normalize_limit_target((x, y), {y: 2, x: 1}) == (
        sp.Integer(1),
        sp.Integer(2),
    )
    with pytest.raises(ValueError):
        normalize_limit_target((x, y), [1])


def test_regular_composition_proves_limit():
    x, y = sp.symbols("x y")
    r = limit(sp.sin(x + y), (x, y), (0, 0), return_result=True)
    assert r.status is LimitStatus.PROVED
    assert r.value == 0
    assert r.exists is True
    assert r.value == 0


def test_variable_subset_reduces_to_two_sided_univariate_limit():
    x, y = sp.symbols("x y")
    r = limit(sp.sin(x) / x, (x, y), (0, 7), return_result=True)
    assert r.status is LimitStatus.PROVED
    assert r.value == 1
    assert r.reduced_variables == (x,)
    assert r.evidence[0].method == "variable_subset_reduction"


def test_path_conflict_proves_nonexistence():
    x, y = sp.symbols("x y", real=True)
    e = x * y / (x**2 + y**2)
    r = limit(e, (x, y), (0, 0), return_result=True)
    assert r.status is LimitStatus.DOES_NOT_EXIST
    assert r.exists is False
    assert r.evidence[-2].value != r.evidence[-1].value
    with pytest.raises(SimultaneousLimitDoesNotExist):
        limit(e, (x, y), (0, 0))


def test_monomial_curve_detects_failure_missed_by_straight_lines():
    x, y = sp.symbols("x y", real=True)
    e = x**2 * y / (x**4 + y**2)
    r = limit(e, (x, y), (0, 0), return_result=True)
    assert r.status is LimitStatus.DOES_NOT_EXIST
    witnesses = [q for q in r.evidence if q.method.startswith("attained_")]
    assert {q.value for q in witnesses} == {sp.S.Zero, sp.Rational(1, 2)}
    for witness in witnesses:
        sequence = dict(witness.substitutions)
        index = next(iter(set().union(*(v.free_symbols for v in sequence.values()))))
        assert tuple(sp.limit(sequence[v], index, sp.oo) for v in (x, y)) == (0, 0)
        assert (
            sp.limit(e.subs(sequence, simultaneous=True), index, sp.oo) == witness.value
        )
        assert sp.simplify((x**4 + y**2).subs(sequence)).is_positive is True


def test_ml2_rational_dominant_order_certifies_classic_limit():
    x, y = sp.symbols("x y", real=True)
    e = x**2 * y**2 / (x**2 + y**2)
    r = limit(e, (x, y), (0, 0), return_result=True)
    assert r.status is LimitStatus.PROVED
    assert r.value == 0
    assert r.exists is True


def test_simultaneous_semantics_are_not_iterated_semantics():
    x, y = sp.symbols("x y", real=True)
    e = x * y / (x**2 + y**2)
    assert sp.limit(sp.limit(e, x, 0), y, 0) == 0
    assert (
        limit(e, (x, y), (0, 0), return_result=True).status
        is LimitStatus.DOES_NOT_EXIST
    )


def test_piecewise_removable_germ_certifies_proof_route():
    import sys
    import types

    x, y = sp.symbols("x y", real=True)
    calls = {}

    class UnsupportedFunctionGraph(ValueError):
        pass

    class Closure:
        in_closure = True

    class Graph:
        formula = sp.Eq(sp.Symbol("_graph_value"), sp.Symbol("_graph_value"))
        auxiliary_variables = ()

    def point_in_closure(region, point, variables, return_result=False):
        calls["closure"] = (region, point, tuple(variables), return_result)
        return Closure()

    def graph(expr, target):
        calls["graph"] = (expr, target)
        return types.SimpleNamespace(
            formula=sp.Eq(target, x**2 * y**2 / (x**2 + y**2)), auxiliary_variables=()
        )

    def qe(formula, quantifiers=None, variables=None):
        calls.setdefault("qe", []).append(
            (formula, tuple(quantifiers), tuple(variables))
        )
        return sp.false if quantifiers and quantifiers[0][0] == "exists" else sp.true

    fake = types.ModuleType("semialg")
    fake.UnsupportedFunctionGraph = UnsupportedFunctionGraph
    fake.point_in_closure = point_in_closure
    fake.quantifier_eliminate = qe
    fake.semialgebraic_function_graph = graph
    from unittest.mock import patch

    e = sp.Piecewise((x + y, sp.Ne(x, 0)), (0, True))
    with patch.dict(sys.modules, {"semialg": fake}):
        r = limit(e, (x, y), (0, 0), return_result=True)
    assert r.status is LimitStatus.PROVED and r.value == 0
    assert r.exists is True


def test_ml2_restricted_domain_checks_approach_closure_before_proof():
    import sys
    import types

    x, y = sp.symbols("x y", real=True)

    class UnsupportedFunctionGraph(ValueError):
        pass

    class Closure:
        in_closure = False

    fake = types.ModuleType("semialg")
    fake.UnsupportedFunctionGraph = UnsupportedFunctionGraph
    fake.point_in_closure = lambda *a, **k: Closure()
    fake.quantifier_eliminate = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("QE should not run")
    )
    fake.semialgebraic_function_graph = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("graph should not run")
    )
    from unittest.mock import patch

    with patch.dict(sys.modules, {"semialg": fake}):
        r = limit(
            x + y, (x, y), (0, 0), domain=sp.And(x > 1, y > 1), return_result=True
        )
    assert r.status is LimitStatus.UNKNOWN
    assert r.evidence[-1].method == "semialgebraic_approach_closure"


def test_ml3_newton_support_finds_cubic_balance_curve():
    x, y = sp.symbols("x y", real=True)
    e = x**3 * y / (x**6 + y**2)
    r = limit(e, (x, y), (0, 0), return_result=True)
    assert r.status is LimitStatus.DOES_NOT_EXIST
    assert r.exists is False


def test_ml2_restricted_rational_domain_identity_is_certified():
    x, y = sp.symbols("x y", real=True)
    r = limit(x / y, (x, y), (0, 0), domain=sp.Eq(y, x), return_result=True)
    assert r.status is LimitStatus.PROVED
    assert r.value == 1
    assert any(ev.method == "rational_domain_identity" for ev in r.evidence)


def test_accumbounds_is_not_a_scalar_candidate():
    from asymptotic._limit_paths import _finite_candidate

    x = sp.symbols("x", real=True)
    assert _finite_candidate(sp.AccumBounds(-1, 1), (x,)) is None


def test_ml4_local_cad_closure_certifies_weighted_rational_limit():
    x, y = sp.symbols("x y", real=True)
    # Total-degree leading-form certification is insufficient because the
    # denominator's leading form x**2 vanishes on the unit sphere.  Local CAD
    # closure nevertheless proves that every positive-error bad set stays
    # away from the origin.
    e = x**2 * y / (x**2 + y**4)
    from asymptotic._limit_certificates import _rational_local_closure_certificate

    certificate = _rational_local_closure_certificate(
        e, (x, y), (sp.Integer(0), sp.Integer(0)), sp.Integer(0), sp.S.true
    )
    assert certificate is not None
    verdict, evidence = certificate
    assert verdict is True
    assert evidence.method == "rational_local_cad_closure"


def test_ml5_algebraic_graph_local_radical_limit():
    x, y = sp.symbols("x y", real=True)
    radius = sp.sqrt(x**2 + y**2)
    expr = sp.Piecewise((radius, x >= 0), (-radius, True))
    result = limit(expr, (x, y), (0, 0), return_result=True)
    assert result.status is LimitStatus.PROVED
    assert result.value == 0


def test_ml5_local_inequality_certificate_respects_restricted_domain():
    x, y = sp.symbols("x y", real=True)
    result = limit(
        sp.sqrt(x**2 + y**2),
        (x, y),
        (0, 0),
        domain=sp.Ge(x, 0),
        return_result=True,
    )
    assert result.status is LimitStatus.PROVED
    assert result.value == 0


def test_ml6_weighted_blowup_certifies_uniform_angular_limit():
    x, y = sp.symbols("x y", real=True)
    # Under x=r**2*u, y=r*v the residual and denominator are exactly
    # weighted homogeneous of orders 5 and 4.  The denominator angular form
    # u**2 + v**4 is bounded away from zero on u**2+v**2=1.
    expr = x**2 * y / (x**2 + y**4)
    result = limit(expr, (x, y), (0, 0), return_result=True)
    assert result.status is LimitStatus.PROVED
    assert result.value == 0


def test_ml6_radial_blowup_helper_certifies_uniform_angular_limit():
    from asymptotic._limit_certificates import _weighted_blowup_certificate

    x, y = sp.symbols("x y", real=True)
    expr = x**3 / (x**2 + y**2)
    evidence = _weighted_blowup_certificate(
        expr, (x, y), (sp.Integer(0), sp.Integer(0)), sp.Integer(0), sp.S.true
    )
    assert evidence is not None
    assert evidence.method == "radial_uniform_angular_optimization"


def test_ml6_general_optimizer_falls_back_to_certified_symbopt():
    import sys
    import types
    from unittest.mock import patch

    from asymptotic.angular_extrema import _certified_extremum

    x = sp.symbols("x", real=True)
    fake_semialg = types.ModuleType("semialg")
    fake_symbopt = types.ModuleType("symbopt")
    calls = {}

    class Result:
        certified = True
        optimum_value = sp.Integer(1)

    def maximize(objective, constraints, *, variables):
        calls["args"] = (objective, constraints, tuple(variables))
        return Result()

    fake_symbopt.maximize = maximize
    with patch.dict(sys.modules, {"semialg": fake_semialg, "symbopt": fake_symbopt}):
        result = _certified_extremum(x**2, sp.And(x >= -1, x <= 1), (x,), kind="max")
    assert result == (sp.Integer(1), "symbopt")
    assert set(calls["args"][1]) == {x >= -1, x <= 1}


def test_ml7_transcendental_composition_over_certified_rational_inner():
    x, y = sp.symbols("x y", real=True)
    inner = x**2 * y**2 / (x**2 + y**2)
    result = limit(sp.exp(inner), (x, y), (0, 0), return_result=True)
    assert result.status is LimitStatus.PROVED
    assert result.value == 1


def test_ml7_removable_transcendental_outer_singularity():
    x, y = sp.symbols("x y", real=True)
    radius2 = x**2 + y**2
    result = limit(sp.sin(radius2) / radius2, (x, y), (0, 0), return_result=True)
    assert result.status is LimitStatus.PROVED
    assert result.value == 1


def test_ml7_does_not_promote_unproved_inner_composition():
    x, y = sp.symbols("x y", real=True)
    inner = x * y / (x**2 + y**2)
    result = limit(sp.exp(inner), (x, y), (0, 0), return_result=True)
    assert result.status is LimitStatus.DOES_NOT_EXIST


def test_weighted_blowup_uses_compactness_angular_extrema():
    from asymptotic._limit_certificates import _weighted_blowup_certificate

    x, y = sp.symbols("x y", real=True)
    expr = x**4 * y**2 / (x**6 + y**2)
    evidence = _weighted_blowup_certificate(
        expr, (x, y), (sp.Integer(0), sp.Integer(0)), sp.Integer(0), sp.S.true
    )
    assert evidence is not None
    assert evidence.method == "weighted_blowup_uniform_angular_optimization"
    assert "positive_definite_even_form+compact_sphere" in evidence.statement
    assert "polynomial_compact_sphere" in evidence.statement


def test_polygamma_regular_composition_at_infinity():
    x = sp.symbols("x", positive=True)
    expr = sp.polygamma(2 + 1 / x, 3 + sp.exp(-x))
    result = limit(expr, x, sp.oo, return_result=True)
    assert result.status is LimitStatus.PROVED
    assert result.value == sp.polygamma(2, 3)
    assert any(e.method == "polygamma_regular_composition" for e in result.evidence)


def test_polygamma_regular_composition_rejects_pole():
    x = sp.symbols("x", positive=True)
    expr = sp.polygamma(2 + x, x)
    result = limit(expr, x, 0, domain=x > 0, return_result=True)
    assert not (
        result.status is LimitStatus.PROVED
        and any(e.method == "polygamma_regular_composition" for e in result.evidence)
    )


def test_polygamma_regular_composition_requires_certified_order():
    x = sp.symbols("x", positive=True)
    expr = sp.polygamma(2 + sp.sin(1 / x), 3 + x)
    result = limit(expr, x, 0, domain=x > 0, return_result=True)
    assert not (
        result.status is LimitStatus.PROVED
        and any(e.method == "polygamma_regular_composition" for e in result.evidence)
    )
