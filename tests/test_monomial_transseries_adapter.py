from types import SimpleNamespace

import sympy as sp

from asymptotic import (
    GrowthComparison,
    TransseriesExpansion,
)
from asymptotic.monomial import (
    AsymptoticMonomial,
    RamificationModel,
    canonical_asymptotic_monomial,
    compare_asymptotic_monomials,
    ramification_index,
)
from asymptotic.ode_adapter import from_formal_ode_data
from asymptotic.transseries import (
    TransseriesTerm,
    compare_monomials,
)


def test_ramification_model_and_common_cover_are_canonical():
    x = sp.symbols("x", positive=True)
    assert ramification_index(2, 3, 4) == 12

    _, square = canonical_asymptotic_monomial(sp.sqrt(x), x, point=0, ramification_index=2)
    _, cube = canonical_asymptotic_monomial(
        x ** sp.Rational(1, 3), x, point=0, ramification_index=3
    )
    assert square.power == 1
    assert cube.power == 1
    assert compare_asymptotic_monomials(square, cube) is GrowthComparison.SMALLER


def test_hierarchy_cancels_common_exponential_levels_before_ordering():
    x = sp.symbols("x", positive=True)
    _, plus = canonical_asymptotic_monomial(sp.exp(x**2 + x), x, point=sp.oo)
    _, minus = canonical_asymptotic_monomial(sp.exp(x**2 - x), x, point=sp.oo)
    assert compare_asymptotic_monomials(plus, minus) is GrowthComparison.LARGER

    assert compare_monomials(sp.exp(-x), x**-100, x, point=sp.oo) is GrowthComparison.SMALLER
    assert (
        compare_monomials(
            x ** sp.Rational(3, 2) * sp.log(x), x ** sp.Rational(3, 2), x, point=sp.oo
        )
        is GrowthComparison.LARGER
    )
    assert (
        compare_monomials(
            x ** sp.Rational(3, 2),
            x ** sp.Rational(1, 2) * sp.log(x) ** 100,
            x,
            point=sp.oo,
        )
        is GrowthComparison.LARGER
    )


def test_transseries_arithmetic_combines_structural_monomials():
    x = sp.symbols("x", positive=True)
    _, m = canonical_asymptotic_monomial(sp.exp(-x) / x, x, point=sp.oo)
    _, n = canonical_asymptotic_monomial(1 / x, x, point=sp.oo)

    left = TransseriesExpansion.from_terms(
        x,
        sp.oo,
        [TransseriesTerm(2, m), TransseriesTerm(3, n)],
    )
    right = TransseriesExpansion.from_terms(
        x,
        sp.oo,
        [TransseriesTerm(-2, m), TransseriesTerm(4, n)],
    )
    total = left + right
    assert len(total.terms) == 1
    assert sp.simplify(total.truncate() - 7 / x) == 0

    product = left * TransseriesExpansion.from_terms(
        x,
        sp.oo,
        [TransseriesTerm(5, n)],
    )
    got = sp.expand(product.truncate())
    assert sp.simplify(got - (10 * sp.exp(-x) / x**2 + 15 / x**2)) == 0
    assert product.valuation() is not None
    assert sp.simplify(product.valuation().expression - 15 / x**2) == 0


def test_term_division_uses_monomial_group():
    x = sp.symbols("x", positive=True)
    _, numerator = canonical_asymptotic_monomial(sp.exp(-2 * x) / x**3, x, point=sp.oo)
    _, denominator = canonical_asymptotic_monomial(sp.exp(-x) / x, x, point=sp.oo)
    quotient = TransseriesTerm(6, numerator) / TransseriesTerm(2, denominator)
    assert sp.simplify(quotient.expression - 3 * sp.exp(-x) / x**2) == 0


def test_formal_ode_adapter_consumes_schema_without_importing_odeanalysis():
    x = sp.symbols("x", positive=True)
    h = sp.Dummy("h", positive=True)
    t = sp.Dummy("t", positive=True)
    vector = SimpleNamespace(
        source_exponent=sp.Rational(1, 2),
        exponent=sp.Rational(1, 2),
        ramified_exponent=sp.Rational(1, 2),
        logarithmic_degree=0,
        local_parameter=t,
        amplitude_parameter=sp.sqrt(t) * (1 + sp.Rational(5, 48) * t**3),
        expression=sp.exp(sp.Rational(2, 3) * x ** sp.Rational(3, 2)) / x ** sp.Rational(1, 4),
    )
    block = SimpleNamespace(
        index=0,
        local_coordinate=h,
        local_parameter=t,
        ramification_index=2,
        local_exponential_polynomial=sp.Rational(2, 3) / h ** sp.Rational(3, 2),
        semisimple_exponent=sp.ImmutableMatrix([[sp.Rational(1, 2)]]),
        nilpotent_exponent=sp.ImmutableMatrix([[0]]),
        formal_exponent_matrix=sp.ImmutableMatrix([[sp.Rational(1, 2)]]),
        basis_vectors=(vector,),
        cover_monodromy=sp.ImmutableMatrix([[-1]]),
        has_logarithms=False,
    )
    data = SimpleNamespace(
        schema_version=1,
        point=sp.oo,
        ramification_index=2,
        blocks=(block,),
        cover_monodromy=sp.ImmutableMatrix([[-1]]),
        local_monodromy=None,
        stokes=None,
        complete=True,
        limitation=None,
    )

    converted = from_formal_ode_data(data, x)
    assert converted.dimension == 1
    solution = converted.solutions[0]
    assert len(solution.terms) == 2
    first = solution.terms[0]
    assert isinstance(first.monomial, AsymptoticMonomial)
    assert sp.simplify(first.monomial.exponential - sp.Rational(2, 3) / t**3) == 0
    expected = sp.exp(sp.Rational(2, 3) * x ** sp.Rational(3, 2)) * (
        x ** sp.Rational(-1, 4) + sp.Rational(5, 48) * x ** sp.Rational(-7, 4)
    )
    assert sp.simplify(solution.truncate() - expected) == 0


def test_ramified_logarithm_is_structural_not_opaque():
    x = sp.symbols("x", positive=True)
    t = sp.Dummy("t", positive=True)
    ram = RamificationModel(x, sp.oo, 2, t)
    monomial = AsymptoticMonomial(ram, sp.S.Zero, sp.Rational(1, 2), 2)
    assert monomial.power == sp.Rational(1, 2)
    assert monomial.log_power == 2
    assert sp.simplify(monomial.parameter_expression - 4 * sp.sqrt(t) * sp.log(t) ** 2) == 0
