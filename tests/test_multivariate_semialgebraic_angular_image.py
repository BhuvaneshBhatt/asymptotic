import sympy as sp

from asymptotic.angular_optimization import parameterized_angular_range
from asymptotic.semialgebraic_angular import semialgebraic_angular_image


def test_general_three_dimensional_linear_sphere_image():
    x, y, z = sp.symbols("x y z", real=True)
    sphere = sp.Eq(x * x + y * y + z * z, 1)
    result = semialgebraic_angular_image(x, (x, y, z), sphere)
    assert result.certified
    assert result.image == sp.Interval(-1, 1)


def test_general_three_dimensional_angular_optimizer_fallback():
    x, y, z = sp.symbols("x y z", real=True)
    result = parameterized_angular_range(x + y + z, (x, y, z))
    assert result.certified
    assert result.minimum is not None and result.maximum is not None


def test_semialgebraic_domain_can_disconnect_image():
    x, y = sp.symbols("x y", real=True)
    domain = sp.And(sp.Eq(x * x + y * y, 1), sp.Ne(x, 0))
    result = semialgebraic_angular_image(x, (x, y), domain)
    assert result.certified
    assert result.image.inf == -1 and result.image.sup == 1
