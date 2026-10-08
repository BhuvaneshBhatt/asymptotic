import pytest
import sympy as sp

from asymptotic import Multiseries
from asymptotic import series as expand
from asymptotic.nested import NestedExpansion
from asymptotic.puiseux import PuiseuxSeries


def test_expand_auto_uses_general_multiseries_at_infinity():
    x = sp.symbols("x", positive=True)
    result = expand(sp.exp(1 / x), x, terms=4, return_result=True)
    assert isinstance(result, Multiseries)
    assert result.default_terms == 4


def test_expand_auto_uses_local_puiseux_at_finite_point():
    x = sp.symbols("x", positive=True)
    result = expand(sp.sqrt(x) + x, x, point=0, terms=4, return_result=True)
    assert isinstance(result, PuiseuxSeries)
    assert result.ramification_index == 2


def test_expand_allows_explicit_nested_representation():
    x = sp.symbols("x", positive=True)
    result = expand(
        sp.log(x + sp.log(x)), x, method="nested", terms=2, return_result=True
    )
    assert isinstance(result, NestedExpansion)


def test_expand_rejects_invalid_user_options_early():
    x = sp.symbols("x")
    with pytest.raises(ValueError, match="choose one of"):
        expand(x, x, method="mystery", return_result=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive integer"):
        expand(x, x, terms=0, return_result=True)
