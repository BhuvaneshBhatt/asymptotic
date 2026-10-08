import sympy as sp

from asymptotic.limits import LimitStatus, limit
from asymptotic.multivariate_certificates import complex_hankel_projection_certificate


def _cases():
    x, y = sp.symbols("x y", real=True)
    r = sp.sqrt(x**2 + y**2)
    return (
        x,
        y,
        (
            sp.re(sp.hankel1(1, x) * (2 * x + y) / r - sp.hankel1(0, x)),
            sp.re(sp.hankel1(1, x) * (x + y) / r - sp.hankel1(0, x)),
        ),
    )


def test_principal_hankel_projection_has_two_boundary_cluster_values():
    x, y, cases = _cases()
    for expr in cases:
        cert = complex_hankel_projection_certificate(expr, (x, y), (0, 0))
        assert cert.certified
        assert cert.method == "complex_hankel_projection_dne"
        assert cert.data[:2] == (-1, 1)


def test_hankel_projection_cases_are_dne_through_public_api():
    x, y, cases = _cases()
    for expr in cases:
        result = limit(expr, (x, y), (0, 0), return_result=True)
        assert result.status is LimitStatus.DOES_NOT_EXIST
        assert result.evidence


def test_branch_blind_hankel_rewrite_side_value():
    x, y = sp.symbols("x y", real=True)
    # The two real-axis boundary values of Re(H_0^(1)(x)) are +1 and -1.
    expr = sp.re(sp.hankel1(0, x))
    cert = complex_hankel_projection_certificate(expr, (x, y), (0, 0))
    assert cert.certified and cert.data[:2] == (1, -1)
