import sympy as sp

from asymptotic import limit
from asymptotic.limits import LimitStatus


def test_radial_order_does_not_treat_unbounded_exponential_as_bounded():
    x, y = sp.symbols("x y", real=True)
    result = limit((x + y) * sp.exp(-1 / (x + y)), (x, y), (0, 0), return_result=True)
    assert result.status is LimitStatus.DOES_NOT_EXIST


def test_denominator_term_majorant_requires_all_summands_nonnegative():
    x, y, z = sp.symbols("x y z", real=True)
    result = limit(
        x**3 * y**2 * z / (x**2 * y**2 - z**2), (x, y, z), (0, 0, 0), return_result=True
    )
    assert result.status is LimitStatus.DOES_NOT_EXIST
