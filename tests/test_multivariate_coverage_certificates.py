from asymptotic.coverage import CoverageCertificate, CoverageObligation, CoverageStatus


def test_complete_certificate_requires_discharged_obligations():
    c = CoverageCertificate.complete(
        "test", "covered", ("a",), (CoverageObligation("chart", "a", True, "proof"),)
    )
    assert c.certified


def test_partial_certificate_cannot_certify():
    c = CoverageCertificate.partial("test", "partial", ("a",), ("b",))
    assert c.status is CoverageStatus.PARTIAL
    assert not c.certified


def test_combination_preserves_missing_regimes():
    a = CoverageCertificate.complete("a", "a", ("a",))
    b = CoverageCertificate.partial("b", "b", (), ("b",))
    c = a.combine(b)
    assert not c.certified and c.missing_regimes == ("b",)
