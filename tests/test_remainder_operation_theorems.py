import sympy as sp

from asymptotic import AsymptoticRemainder, RemainderKind, transseries_from_expression
from asymptotic.remainder_theorems import (
    certify_algebraic_substitution_remainder,
    certify_finite_product_remainder,
    certify_finite_sum_remainder,
    certify_product_remainder,
    certify_quotient_remainder,
    certify_reciprocal_remainder,
    certify_unary_composition_remainder,
)


def test_binary_and_finite_product_propagate_prefix_weighted_errors():
    x = sp.symbols("x", positive=True)
    ra = AsymptoticRemainder.big_o(x**-2, x, sp.oo)
    rb = AsymptoticRemainder.little_o(x**-1, x, sp.oo)
    cert = certify_product_remainder(x, 1, ra, rb)
    assert cert.certified
    assert cert.conclusion.kind is RemainderKind.LITTLE_O
    finite = certify_finite_product_remainder(
        (x, 1, 2), (ra, rb, AsymptoticRemainder.exact_zero(x, sp.oo))
    )
    assert finite.certified


def test_finite_sum_uses_safe_envelope_when_scales_need_not_be_sharply_comparable():
    x = sp.symbols("x", positive=True)
    r1 = AsymptoticRemainder.big_o(x**-2, x, sp.oo)
    r2 = AsymptoticRemainder.big_o(sp.exp(-x), x, sp.oo)
    cert = certify_finite_sum_remainder((r1, r2))
    assert cert.certified
    assert cert.conclusion.kind is RemainderKind.BIG_O


def test_reciprocal_and_quotient_require_relative_smallness_and_nondegeneracy():
    x = sp.symbols("x", positive=True)
    r = AsymptoticRemainder.big_o(x**-1, x, sp.oo)
    reciprocal = certify_reciprocal_remainder(1, r)
    assert reciprocal.certified
    assert reciprocal.conclusion.kind is RemainderKind.BIG_O
    assert sp.simplify(reciprocal.conclusion.scale - x**-1) == 0

    quotient = certify_quotient_remainder(1 + x**-1, 1, r, r)
    assert quotient.certified

    bad = certify_reciprocal_remainder(x**-2, r)
    assert not bad.certified
    assert bad.conclusion.kind is RemainderKind.UNKNOWN


def test_polynomial_algebraic_substitution_handles_stationary_first_derivative():
    x, z = sp.symbols("x z", positive=True)
    r = AsymptoticRemainder.big_o(x**-1, x, sp.oo)
    cert = certify_algebraic_substitution_remainder(z**2, z, 0, r, output_variable=x, point=sp.oo)
    assert cert.certified
    assert cert.conclusion.kind is RemainderKind.BIG_O
    assert sp.simplify(cert.conclusion.scale - x**-2) == 0


def test_rational_algebraic_substitution_uses_certified_denominator_reciprocal():
    x, z = sp.symbols("x z", positive=True)
    r = AsymptoticRemainder.big_o(x**-1, x, sp.oo)
    cert = certify_algebraic_substitution_remainder(
        1 / (1 + z), z, 0, r, output_variable=x, point=sp.oo
    )
    assert cert.certified
    assert sp.simplify(cert.conclusion.scale - x**-1) == 0


def test_general_composition_finds_first_nonzero_taylor_derivative():
    x, z = sp.symbols("x z", positive=True)
    r = AsymptoticRemainder.big_o(x**-1, x, sp.oo)
    cert = certify_unary_composition_remainder(sp.cos(z), z, 0, r, output_variable=x, point=sp.oo)
    assert cert.certified
    assert cert.conclusion.kind is RemainderKind.BIG_O
    assert sp.simplify(cert.conclusion.scale - sp.Rational(1, 2) * x**-2) == 0


def test_transseries_reciprocal_records_geometric_truncation_remainder():
    x = sp.symbols("x", positive=True)
    source = transseries_from_expression(1 + x**-1, x, point=sp.oo, complete=True)
    reciprocal = source.reciprocal(terms=4)
    assert reciprocal.remainder.kind is RemainderKind.BIG_O
    assert sp.simplify(reciprocal.remainder.scale - x**-4) == 0
    assert reciprocal.remainder.check() is True


def test_exact_reciprocal_still_requires_eventual_nonvanishing():
    x = sp.symbols("x", real=True)
    exact = AsymptoticRemainder.exact_zero(x, sp.oo)
    cert = certify_reciprocal_remainder(sp.sin(x), exact)
    assert not cert.certified
    assert cert.conclusion.kind is RemainderKind.UNKNOWN


def test_exact_scaling_certificate_replays_and_is_attached_to_scalar_product():
    import sympy as sp

    from asymptotic.remainder import AsymptoticRemainder
    from asymptotic.remainder_theorems import certify_scaling_remainder
    from asymptotic.transseries import TransseriesExpansion, TransseriesTerm

    x = sp.symbols("x", positive=True)
    remainder = AsymptoticRemainder.little_o(1 / x**2, x, sp.oo, exact_expression=1 / x**3)
    cert = certify_scaling_remainder(3, remainder)
    assert cert.certified
    assert cert.replay() is True

    series = TransseriesExpansion.from_terms(
        x, sp.oo, (TransseriesTerm(1, 1 / x),), remainder=remainder
    )
    scaled = 3 * series
    attached = scaled.metadata["remainder_certificates"][-1]
    assert attached.theorem == "exact scaling remainder theorem"
    assert attached.replay() is True


def test_addition_attaches_replayable_finite_sum_certificate():
    import sympy as sp

    from asymptotic.remainder import AsymptoticRemainder
    from asymptotic.transseries import TransseriesExpansion, TransseriesTerm

    x = sp.symbols("x", positive=True)
    left = TransseriesExpansion.from_terms(
        x,
        sp.oo,
        (TransseriesTerm(1, 1 / x),),
        remainder=AsymptoticRemainder.big_o(1 / x**2, x, sp.oo, exact_expression=1 / x**2),
    )
    right = TransseriesExpansion.from_terms(
        x,
        sp.oo,
        (TransseriesTerm(1, 1 / x**2),),
        remainder=AsymptoticRemainder.little_o(1 / x**2, x, sp.oo, exact_expression=1 / x**3),
    )
    result = left + right
    cert = result.metadata["remainder_certificates"][-1]
    assert cert.theorem == "finite sum remainder theorem"
    assert cert.replay() is True


def test_antiderivative_certificate_uses_direct_exact_error_replay():
    import sympy as sp

    from asymptotic.remainder import AsymptoticRemainder, RemainderKind
    from asymptotic.remainder_theorems import certify_antiderivative_remainder

    x = sp.symbols("x", positive=True)
    remainder = AsymptoticRemainder.little_o(1 / x**2, x, sp.oo, exact_expression=1 / x**3)
    cert = certify_antiderivative_remainder(remainder)
    assert cert.certified
    assert cert.conclusion.kind is RemainderKind.LITTLE_O
    assert cert.replay() is True
