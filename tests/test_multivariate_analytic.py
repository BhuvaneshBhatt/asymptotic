import sympy as sp

from asymptotic.limits import LimitStatus, limit
from asymptotic.multivariate_certificates import (
    analytic_equivalence_rewrite_certificate,
    analytic_jet_path_conflict_certificate,
    candidate_rational_limit_certificate,
    generalized_divided_difference_certificate,
    oscillatory_phase_sequence_certificate,
    radical_rationalization_certificate,
)


def _assert_limit(expr, variables, target, value, *, domain=sp.S.true):
    result = limit(expr, variables, target, domain=domain, return_result=True)
    if value is None:
        assert result.status is LimitStatus.DOES_NOT_EXIST
        assert result.value is None
    else:
        assert result.status is LimitStatus.PROVED
        assert result.value == value


def test_candidate_centered_rational_family_variety_cases():
    x, y, z = sp.symbols("x y z", real=True)
    cases = [
        (
            (x**3 + x**4 - 2 * x**2 * y - 4 * x**3 * y + 4 * x**2 * y**2)
            / (10 * x**2 + 3 * x**4 - 8 * x * y - 8 * x**3 * y + 8 * x**2 * y**2),
            None,
        ),
        ((x**2 + y**4) / (x**2 - x**4 / 3 + y**4), sp.Integer(1)),
        ((x**4 + y**4) / (x**2 - 2 * y**2), None),
        (
            -6 * (x**4 + y**4) / (-6 * x + x**3 + 6 * y - y**3),
            None,
        ),
        (
            (x**12 + y**12) / (x**10 - x**9 * y + x**6 * y**2 + y**6),
            sp.Integer(0),
        ),
        (
            (2 * x**2 * y - x**2 * z + 2 * y * z**2 - 3 * z**3)
            / (2 * x**2 - x * y + 3 * y**2 + x * z - 2 * y * z + 5 * z**2),
            sp.Integer(0),
        ),
        (
            (x**2 * y - x**2 * z + 2 * y * z**2 - z**3)
            / (x**2 - x * y + y**2 + x * z - 2 * y * z + z**2),
            None,
        ),
        (
            (x**2 * y - x**2 * z + 2 * y * z**2 - z**3)
            / (x**2 - x * y + y**2 + x * z - y * z + z**2),
            sp.Integer(0),
        ),
    ]
    for expr, expected in cases:
        variables = (x, y, z) if expr.has(z) else (x, y)
        target = (0,) * len(variables)
        certificate = candidate_rational_limit_certificate(expr, variables, target)
        assert certificate.certified
        if expected is None:
            assert certificate.value is sp.nan
        else:
            assert certificate.value == expected
        _assert_limit(expr, variables, target, expected)


def test_candidate_rational_signed_pole_handles_denominator_variety():
    x, y = sp.symbols("x y", real=True)
    expr = (-(x**2) + x**4 - y**2) / (x - y) ** 4
    certificate = candidate_rational_limit_certificate(expr, (x, y), (0, 0))
    assert certificate.certified and certificate.value == -sp.oo
    _assert_limit(expr, (x, y), (0, 0), -sp.oo)


def test_signed_pole_coercivity_does_radial_order():
    x, y = sp.symbols("x y", real=True)
    expr = (x**2 + y**10) / (x - y) ** 4
    certificate = candidate_rational_limit_certificate(expr, (x, y), (0, 0))
    assert not (certificate.certified and certificate.value in (sp.oo, -sp.oo))


def test_high_dimensional_weighted_coercivity_stays_on_cheap_path():
    x, y, z, w, t, ell = sp.symbols("x y z w t ell", real=True)
    expr = x**6 / (ell**2 + t**2 + w**6 + x**2 + y**2 + z**2)
    certificate = candidate_rational_limit_certificate(
        expr, (x, y, z, w, t, ell), (0, 0, 0, 0, 0, 0)
    )
    assert certificate.certified and certificate.value == 0
    assert certificate.data[0] == (3, 3, 3, 1, 3, 3)
    _assert_limit(
        expr,
        (x, y, z, w, t, ell),
        (0, 0, 0, 0, 0, 0),
        sp.Integer(0),
    )


def test_generalized_divided_differences_cover_inner_arguments():
    x, y, z = sp.symbols("x y z", real=True)
    cases = [
        ((x**2 - y**2) / (sp.cos(x) - sp.cos(y)), (x, y), -sp.Integer(2)),
        ((x - y) / (sp.sin(x) - sp.sin(y)), (x, y), sp.Integer(1)),
        (
            (-sp.sin(x**2 + y**2) + sp.sin(z)) / (-sp.tan(y**2) - sp.tan(x**2 - z)),
            (x, y, z),
            sp.Integer(1),
        ),
        (
            (sp.sin(z) - sp.sin(x**2 + y**2)) / (sp.tan(z - x**2) - sp.tan(y**2)),
            (x, y, z),
            sp.Integer(1),
        ),
    ]
    for expr, variables, expected in cases:
        target = (0,) * len(variables)
        certificate = generalized_divided_difference_certificate(
            expr, variables, target
        )
        assert certificate.certified and certificate.value == expected
        _assert_limit(
            expr,
            variables,
            target,
            expected,
        )


def test_analytic_equivalence_uses_certified_multivariate_remainder():
    x, y = sp.symbols("x y", real=True)
    cases = [
        (
            (2 - 2 * sp.cos(x**2 * y**2))
            / (x**10 + x**6 * y**2 + y**6 - x**9 * sp.sin(y)),
            sp.Integer(0),
        ),
        ((x**2 + sp.sin(y) ** 4) / (y**4 + sp.sin(x) ** 2), sp.Integer(1)),
        (
            (sp.sin(x) ** 2 + 2 * (1 - sp.cos(y))) / (x**2 + y**2),
            sp.Integer(1),
        ),
    ]
    for expr, expected in cases:
        certificate = analytic_equivalence_rewrite_certificate(expr, (x, y), (0, 0))
        assert certificate.certified and certificate.value == expected
        assert "remainder" in certificate.statement
        _assert_limit(expr, (x, y), (0, 0), expected)


def test_radical_rationalization_tracks_principal_deleted_variety():
    x, y = sp.symbols("x y", real=True)
    quotient = (x**2 - x * y) / (sp.sqrt(x) - sp.sqrt(y))
    for domain in (sp.S.true, sp.And(x > 0, y > 0)):
        certificate = radical_rationalization_certificate(
            quotient, (x, y), (0, 0), domain=domain
        )
        assert certificate.certified and certificate.value == 0
        direction = certificate.data[-1]
        assert direction[0] != direction[1]
        _assert_limit(
            quotient,
            (x, y),
            (0, 0),
            sp.Integer(0),
            domain=domain,
        )

    shifted = (sp.sqrt(x) - sp.sqrt(y + 1)) / (x - y - 1)
    certificate = radical_rationalization_certificate(shifted, (x, y), (4, 3))
    assert certificate.certified and certificate.value == sp.Rational(1, 4)
    _assert_limit(
        shifted,
        (x, y),
        (4, 3),
        sp.Rational(1, 4),
    )


def test_oscillatory_max_radius_dne_has_two_explicit_phase_sequences():
    x, y = sp.symbols("x y", real=True)
    expr = sp.sin(1 / sp.Max(sp.Abs(x), sp.Abs(y)))
    certificate = oscillatory_phase_sequence_certificate(expr, (x, y), (0, 0))
    assert certificate.certified
    seq_a, value_a, seq_b, value_b, n = certificate.data
    assert value_a == 1 and value_b == -1
    assert sp.limit(seq_a[0], n, sp.oo) == 0
    assert sp.limit(seq_b[0], n, sp.oo) == 0
    _assert_limit(
        expr,
        (x, y),
        (0, 0),
        None,
    )


def test_analytic_jet_conflicts_cover_noncoercive_and_weighted_dne():
    x, y = sp.symbols("x y", real=True)
    cases = [
        (
            (sp.sin(x**2 + y**2) - x * y * sp.exp(x)) / (sp.exp(x + y) - 1),
            (0, 0),
        ),
        (
            (-sp.exp(x**2) * x**2 * y**3 + sp.sin(x**4 + y**6))
            / (sp.exp(x**4 + y**6) - 1),
            (0, 0),
        ),
    ]
    for expr, target in cases:
        certificate = analytic_jet_path_conflict_certificate(expr, (x, y), target)
        assert certificate.certified
        assert certificate.method == "analytic_jet_path_dne"
        _assert_limit(expr, (x, y), target, None)
