import pytest
import sympy as sp

from asymptotic import integrate, series
from asymptotic.transseries import TransseriesExpansion


def test_integrate_dispatches_symbolic_expression_asymptotic_analysis():
    x = sp.symbols("x", positive=True)
    result = integrate(1 / x**2, x, point=sp.oo, terms=3, return_result=True)
    assert isinstance(result, TransseriesExpansion)
    assert sp.simplify(sp.diff(result.truncate(), x) - 1 / x**2) == 0


def test_integrate_preserves_supported_representation_protocol():
    x = sp.symbols("x", positive=True)
    expansion = series(1 / x + 1 / x**2, x, point=sp.oo, terms=3, return_result=True)
    result = integrate(expansion, terms=3, return_result=True)
    assert type(result) is type(expansion)


def test_integrate_requires_variable_for_raw_symbolic_expression():
    x = sp.symbols("x")
    with pytest.raises(TypeError, match="variable is required"):
        integrate(1 / x, return_result=True)
