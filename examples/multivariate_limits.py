"""Representative simultaneous multivariate limits."""

import sympy as sp

from asymptotic.limits import LimitStatus, limit

x, y = sp.symbols("x y", real=True)

cases = {
    "uniform_bound": x * y / sp.sqrt(x**2 + y**2),
    "path_dependent": x * y / (x**2 + y**2),
    "radial": (x**2 + y**2) / (1 + x**2 + y**2),
    "cylindrical": sp.atan(1 / sp.Abs(x)),
}

results = {
    name: limit(expr, (x, y), (0, 0), return_result=True)
    for name, expr in cases.items()
}

assert results["uniform_bound"].status is LimitStatus.PROVED
assert results["uniform_bound"].value == 0
assert results["path_dependent"].status is LimitStatus.DOES_NOT_EXIST
assert results["radial"].status is LimitStatus.PROVED
assert results["radial"].value == 0
assert results["cylindrical"].status is LimitStatus.PROVED
assert results["cylindrical"].value == sp.pi / 2
