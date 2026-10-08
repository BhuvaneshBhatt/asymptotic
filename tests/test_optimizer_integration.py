"""Exercise the optional optimizer's installed constraint contract."""

from unittest.mock import patch

import pytest
import sympy as sp

from asymptotic.angular_extrema import _certified_extremum


@pytest.mark.parametrize(
    "kind, degree, expected",
    [("max", 1, 1), ("min", 1, -1), ("max", 2, 1), ("min", 2, 0)],
)
def test_installed_optimizer(kind, degree, expected):
    pytest.importorskip("symbopt")
    x = sp.Symbol("x", real=True)
    backend = "semialgebraic_maximize" if kind == "max" else "semialgebraic_minimize"
    # Simulate an unavailable preferred route while retaining the actual
    # optional provider and its exact certification machinery.
    with patch("semialg." + backend, side_effect=NotImplementedError):
        result = _certified_extremum(
            x**degree, sp.And(x >= -1, x <= 1), (x,), kind=kind
        )
    assert result == (sp.Integer(expected), "symbopt")


def test_optimizer_rejects_disjunction():
    x = sp.Symbol("x", real=True)
    with patch("semialg.semialgebraic_maximize", side_effect=NotImplementedError):
        assert _certified_extremum(x, sp.Or(x < -1, x > 1), (x,), kind="max") is None
