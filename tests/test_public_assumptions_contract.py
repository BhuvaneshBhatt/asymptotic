from __future__ import annotations

import inspect

import pytest
import sympy as sp
from funcprops import ConditionalExpression, PropertyEnforcementError

import asymptotic
from asymptotic.general_ops import compose_transseries
from asymptotic.relations import less
from asymptotic.transseries import transseries_from_expression

_PURE_OR_CARRIED_CONTEXT = {
    "as_element",
    "hyperasymptotic_series",
    "leading_term",
    "local_series",
    "mellin",
    "truncate",
    "certify_perturbation",
    "differentiate",
    "explain",
    "integrate",
    "multiple_scale_derivative",
    "resume_perturbation",
    "transseries_from_expression",
}


def test_public_symbolic_decision_apis_expose_neutral_assumptions():
    for name in sorted(asymptotic.__all__):
        function = getattr(asymptotic, name, None)
        if (
            not callable(function)
            or inspect.isclass(function)
            or name in _PURE_OR_CARRIED_CONTEXT
        ):
            continue
        signature = inspect.signature(function)
        parameter = signature.parameters.get("assumptions")
        assert parameter is not None, name
        assert parameter.default == sp.S.true, name


def test_same_relation_is_conditional_with_them():
    n = sp.symbols("n", positive=True)
    p = sp.symbols("p", real=True)
    unresolved = less(sp.log(n), n**p, n, sp.oo)
    resolved = less(sp.log(n), n**p, n, sp.oo, assumptions=p > 0)
    assert unresolved == ConditionalExpression(True, p > 0)
    assert resolved is True


def test_composition_assumptions_resolve_domain():
    x = sp.symbols("x", positive=True)
    a, z = sp.symbols("a z", real=True)
    inner = transseries_from_expression(a + 1 / x, x, point=sp.oo)
    with pytest.raises(PropertyEnforcementError):
        compose_transseries(sp.log(z), inner, argument=z)
    resolved = compose_transseries(sp.log(z), inner, argument=z, assumptions=a > 0)
    expression = resolved.truncate()
    assert expression.has(sp.log(a))
    assert sp.simplify(expression.coeff(x, -1) - 1 / a) == 0
