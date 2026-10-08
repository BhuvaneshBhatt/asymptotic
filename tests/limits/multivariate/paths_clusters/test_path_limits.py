import pytest
import sympy as sp
from funcprops import ParametricContour

import asymptotic
from asymptotic.limit_models import LimitStatus


def test_one_sided_limit_uses_native_limit_result():
    x = sp.symbols("x", real=True)
    result = asymptotic.one_sided_limit(
        sp.sin(x) / x, x, 0, direction="+", return_result=True
    )
    assert result.status is LimitStatus.PROVED
    assert result.value == 1
    assert result.evidence[0].method == "one_sided_pullback"


def test_path_limit_uses_terminal_contour_germ():
    x, t = sp.symbols("x t", real=True)
    path = ParametricContour(t**2, (t, 0, 1))
    result = asymptotic.path_limit(x, x, 1, path=path, return_result=True)
    assert result.status is LimitStatus.PROVED
    assert result.value == 1
    assert result.evidence[0].method == "path_pullback"


def test_path_limit_requires_requested_endpoint():
    x, t = sp.symbols("x t", real=True)
    path = ParametricContour(t, (t, 0, 1))
    with pytest.raises(ValueError, match="endpoint"):
        asymptotic.path_limit(x, x, 0, path=path, return_result=True)


def test_path_limit_requires_decidable_orientation():
    x, t, a = sp.symbols("x t a", real=True)
    path = ParametricContour(t, (t, 0, a))
    with pytest.raises(ValueError, match="orientation"):
        asymptotic.path_limit(x, x, a, path=path, return_result=True)
