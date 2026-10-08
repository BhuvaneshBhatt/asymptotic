import sympy as sp

from asymptotic.multivariate_expansion import (
    multivariate_equivalent,
    multivariate_little_o,
)


def test_relation_records_actual_certificate_backends():
    x, y = sp.symbols("x y", real=True)
    q = x**2 + y**2
    result = multivariate_little_o(x**2 * y**2, q, (x, y), (0, 0))
    assert result.certified
    assert "weighted_blowup_geometry" in result.certificate_backends
    assert "exact_zero_remainder" in result.certificate_backends


def test_equivalence_records_actual_certificate_backends():
    x, y = sp.symbols("x y", real=True)
    q = x**2 + y**2
    result = multivariate_equivalent(q + x**4, q, (x, y), (0, 0))
    assert result.certified
    assert result.certificate_backends


def test_expanded_relation_corpus_has_requested_stress_families():
    from tools.relation_certification_corpus import cases

    corpus = cases()
    families = {case.family for case in corpus}
    assert len(corpus) == 44
    assert {
        "3plus-Newton-fan",
        "nonradial-algebraic",
        "angular-transcendental",
        "mixed-restricted-domain",
        "sector-boundary-zero-pole",
    } <= families
    assert any(len(case.variables) >= 3 for case in corpus)
