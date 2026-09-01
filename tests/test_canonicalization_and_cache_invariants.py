import sympy as sp

from asymptotic import (
    dominant_balance_candidates,
    series_reversion,
)
from asymptotic.canonical import (
    canonical_equal,
    canonical_expr,
)
from asymptotic.function_properties import nested_branch_safe_substitution_decision
from asymptotic.multivariate import (
    clear_weight_cone_cache,
    multivariate_scaling_regimes,
    weight_cone_cache_info,
)
from asymptotic.remainder_theorems import (
    certify_green_inverse_operator_remainder,
    characteristic_poly_cache_info,
    clear_characteristic_poly_cache,
)
from asymptotic.stratification import (
    ParameterStratum,
    stratify_parameter_cases,
)


def test_canonical_equality_ignores_unevaluated_commutative_arg_order():
    x, y, z = sp.symbols("x y z")
    left = sp.Add(x, y, z, evaluate=False)
    right = sp.Add(z, x, y, evaluate=False)
    assert left.args != right.args
    assert canonical_equal(left, right)
    assert canonical_expr(left) == canonical_expr(right)


def test_dominant_balance_certificate_replays_global_minimum_decision():
    x, y = sp.symbols("x y")
    candidates = dominant_balance_candidates(y**2 + x * y + x**3, y, x, stratify_parameters=False)
    assert candidates
    assert all(candidate.certificate is not None for candidate in candidates)
    assert all(candidate.replay() is True for candidate in candidates)


def test_equivalent_overlapping_parameter_strata_are_merged():
    a = sp.symbols("a", real=True)
    strat = stratify_parameter_cases(
        [ParameterStratum(a >= 0, sp.Integer(1)), ParameterStratum(a <= 0, sp.Integer(1))],
        parameters=(a,),
        require_disjoint=True,
        require_exhaustive=True,
    )
    assert len(strat.strata) == 1
    assert strat.exhaustive


def test_weight_cone_cache_is_bounded_and_reused():
    x, z, y = sp.symbols("x z y")
    clear_weight_cone_cache()
    eq = y**2 + x * y + z**3
    first = multivariate_scaling_regimes(eq, y, (x, z), stratify_parameters=False)
    info1 = weight_cone_cache_info()
    second = multivariate_scaling_regimes(eq, y, (x, z), stratify_parameters=False)
    info2 = weight_cone_cache_info()
    assert first == second
    assert info1.misses == 1
    assert info2.hits >= 1
    assert info2.maxsize == 256


def test_characteristic_poly_cache_reused_and_green_replays():
    x = sp.symbols("x", positive=True)
    delta = sp.Function("delta")
    operator = sp.diff(delta(x), x, 2) - delta(x)
    clear_characteristic_poly_cache()
    cert, green = certify_green_inverse_operator_remainder(
        sp.exp(-x / 2), operator, delta, x, sp.oo
    )
    info1 = characteristic_poly_cache_info()
    cert2, green2 = certify_green_inverse_operator_remainder(
        sp.exp(-x / 2), operator, delta, x, sp.oo
    )
    info2 = characteristic_poly_cache_info()
    assert cert.certified and cert2.certified
    assert green is not None and green.replay() is True
    assert green2 is not None and green2.replay() is True
    assert info1.misses == 1 and info2.hits >= 1


def test_nested_branch_trace_detects_inner_principal_cut_and_is_retained_on_reversion():
    x, y = sp.symbols("x y")
    expr = sp.log(sp.sqrt(x))
    bad = nested_branch_safe_substitution_decision(expr, x, -1)
    assert bad.verdict is False
    good = nested_branch_safe_substitution_decision(expr, x, 1)
    assert good.verdict is True

    branch = series_reversion(sp.log(1 + x), x, y, terms=3, branch=0)
    assert branch.branch_decisions
    assert all(decision.verdict is True for decision in branch.branch_decisions)


def test_ode_green_interchange_is_replayed_and_certified_without_importing_odeanalysis():
    from types import SimpleNamespace

    from asymptotic.ode_adapter import certify_green_operator_data

    x = sp.symbols("x", positive=True)
    lam = sp.Symbol("lambda_from_odeanalysis")
    data = SimpleNamespace(
        schema_version=1,
        variable=x,
        point=sp.oo,
        coefficients=(sp.Integer(2), sp.Integer(-3), sp.Integer(1)),
        order=2,
        characteristic_parameter=lam,
        characteristic_poly=lam**2 - 3 * lam + 2,
    )

    theorem, green = certify_green_operator_data(data, sp.exp(-x))

    assert theorem.certified
    assert green is not None
    assert green.replay(x) is True
    assert sp.simplify(green.particular + sp.exp(-x) / 6) == 0


def test_ode_green_interchange_rejects_stale_characteristic_poly():
    from types import SimpleNamespace

    import pytest

    from asymptotic.ode_adapter import (
        FormalODEAdapterError,
        certify_green_operator_data,
    )

    x = sp.symbols("x", positive=True)
    lam = sp.Symbol("lambda_from_odeanalysis")
    data = SimpleNamespace(
        schema_version=1,
        variable=x,
        point=sp.oo,
        coefficients=(sp.Integer(2), sp.Integer(-3), sp.Integer(1)),
        order=2,
        characteristic_parameter=lam,
        characteristic_poly=lam**2 + 1,
    )

    with pytest.raises(FormalODEAdapterError, match="failed replay"):
        certify_green_operator_data(data, sp.exp(-x))


def test_entailment_cache_is_context_sensitive_and_order_independent():
    from asymptotic.function_properties.semantics import (
        clear_entailment_cache,
        entailment_cache_info,
        entails,
    )

    a = sp.symbols("a", real=True)
    condition = a > 0
    cases = ((a > 1, True), (a < 0, False), (sp.S.true, None))

    def run(order):
        clear_entailment_cache()
        values = []
        for index in order:
            assumptions, expected = cases[index]
            value = entails(condition, assumptions)
            assert value is expected
            values.append(value)
        return tuple(values), entailment_cache_info()

    first, info1 = run((0, 1, 2, 0))
    second, info2 = run((2, 1, 0, 2))
    assert first[:3] == (True, False, None)
    assert second[:3] == (None, False, True)
    assert info1.hits >= 1 and info2.hits >= 1
    assert info1.misses == 3 and info2.misses == 3


def test_weight_cone_cache_keeps_assumption_contexts_separate():
    x, z, y, a = sp.symbols("x z y a", real=True)
    equation = y**2 + x * y + z**3
    clear_weight_cone_cache()
    positive = multivariate_scaling_regimes(
        equation, y, (x, z), assumptions=a > 0, stratify_parameters=False
    )
    info1 = weight_cone_cache_info()
    negative = multivariate_scaling_regimes(
        equation, y, (x, z), assumptions=a < 0, stratify_parameters=False
    )
    info2 = weight_cone_cache_info()
    positive_again = multivariate_scaling_regimes(
        equation, y, (x, z), assumptions=a > 0, stratify_parameters=False
    )
    info3 = weight_cone_cache_info()
    assert len(positive) == len(negative) == len(positive_again)
    assert tuple(item.cone.active_indices for item in positive) == tuple(
        item.cone.active_indices for item in negative
    )
    assert tuple(item.cone.active_indices for item in positive_again) == tuple(
        item.cone.active_indices for item in positive
    )
    assert info1.misses == 1
    assert info2.misses == 2
    assert info3.hits >= 1
