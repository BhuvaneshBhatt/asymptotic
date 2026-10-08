import sympy as sp

from asymptotic.multidimensional_saddles import (
    morse_bott_laplace_leading,
    orthant_laplace_leading,
    stationary_phase_leading,
)


def test_stationary_phase_quadratic_2d():
    x, y, n = sp.symbols("x y n", positive=True)
    r = stationary_phase_leading(1, x**2 + y**2, (x, y), (0, 0), n)
    assert r.certificate.certified
    assert sp.simplify(r.expression - sp.I * sp.pi / n) == 0


def test_stationary_phase_refuses_degenerate():
    x, n = sp.symbols("x n", positive=True)
    r = stationary_phase_leading(1, x**4, (x,), (0,), n)
    assert not r.certificate.certified and r.expression is None


def test_corner_laplace():
    x, y, n = sp.symbols("x y n", positive=True)
    r = orthant_laplace_leading(1, x + y, (x, y), (0, 0), n, active_boundaries=(0, 1))
    assert r.certificate.certified and sp.simplify(r.expression - n**-2) == 0


def test_mixed_boundary_gaussian():
    x, y, n = sp.symbols("x y n", positive=True)
    r = orthant_laplace_leading(
        1, x + y**2 / 2, (x, y), (0, 0), n, active_boundaries=(0,)
    )
    assert (
        r.certificate.certified
        and sp.simplify(r.expression - sp.sqrt(2 * sp.pi) * n ** sp.Rational(-3, 2))
        == 0
    )


def test_morse_bott_normal_factor():
    x, y, n = sp.symbols("x y n", positive=True)
    r = morse_bott_laplace_leading(
        1, x**2 / 2, (x, y), (0, 0), n, normal_directions=(0,), manifold_measure=3
    )
    assert (
        r.certificate.certified
        and sp.simplify(r.expression - 3 * sp.sqrt(2 * sp.pi / n)) == 0
    )


def test_morse_bott_rejects_indefinite_even_determinant():
    x, y, n = sp.symbols("x y n", positive=True)
    result = morse_bott_laplace_leading(
        1, -(x**2 + y**2) / 2, (x, y), (0, 0), n, normal_directions=(0, 1)
    )
    assert result.expression is None
    assert not result.certificate.certified
    assert "positive-definite normal Hessian" in result.certificate.obligations
