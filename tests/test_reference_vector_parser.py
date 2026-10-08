import sympy as sp

from tools.reference_expression_parser import parse_reference_expression


def test_scalar_arithmetic_broadcasts_over_reference_vector():
    x, y = sp.symbols("x y", real=True)
    got = parse_reference_expression("y + log(x)*(x, y) - 1 + y/x**2", {"x": x, "y": y})
    assert got == sp.Tuple(
        x * sp.log(x) + y - 1 + y / x**2, y * sp.log(x) + y - 1 + y / x**2
    )
