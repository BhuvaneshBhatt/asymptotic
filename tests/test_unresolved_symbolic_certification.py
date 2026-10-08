import pytest
import sympy as sp

from asymptotic.limits import LimitStatus, limit


@pytest.mark.parametrize(
    "expr",
    [
        lambda x, y, a: (x + sp.I * y) ** a,
        lambda x, y, a: sp.log((x + sp.I * y) ** a),
        lambda x, y, a: sp.bessely(a, sp.sqrt(x * x + y * y)),
        lambda x, y, a: sp.besselj(a, sp.sqrt(x * x + y * y)),
    ],
)
def test_parameter_dependent_singular_substitution_is_not_proved(expr):
    x, y, a = sp.symbols("x y a", real=True)
    r = limit(expr(x, y, a), (x, y), (0, 0), return_result=True)
    assert r.status is LimitStatus.UNKNOWN
    assert r.value is None
