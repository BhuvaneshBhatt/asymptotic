import sympy as sp

from asymptotic.angular_optimization import parameterized_angular_strata
from asymptotic.parameter_cells import parameter_truth_cells


def test_truth_cells_support_two_parameter_predicates():
    a, b = sp.symbols("a b", real=True)
    cells = parameter_truth_cells((a > 0, b > 0), (a, b))
    assert len(cells) == 4
    assert {cell.signature for cell in cells} == {
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    }


def test_even_simplex_stationary_membership_two_parameters():
    x, y, a, b = sp.symbols("x y a b", real=True)
    # Descends to z**2 + a*z*(1-z) + b*z, whose stationary root
    # depends on both a and b. The exact cell engine, not sampling, decides
    # whether that root lies in [0,1].
    strata = parameterized_angular_strata(
        x**4 + a * x**2 * y**2 + b * x**2 * (x**2 + y**2), (x, y)
    )
    assert strata
    assert all(s.provider == "even_simplex_parameter_cell_qe" for s in strata)
