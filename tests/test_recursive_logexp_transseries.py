import sympy as sp

from asymptotic import (
    GrowthComparison,
    transseries_from_expression,
)
from asymptotic.logexp_transseries import (
    RecursiveLogExpMonomial,
    canonical_recursive_logexp_monomial,
)
from asymptotic.transseries import compare_monomials


def test_recursive_monomial_canonicalizes_nested_exp_and_logs():
    x = sp.symbols("x", positive=True)
    c, m = canonical_recursive_logexp_monomial(
        3 * sp.exp(sp.exp(x) + x) * x ** sp.Rational(3, 2) * sp.log(sp.log(x)) ** 2,
        x,
        point=sp.oo,
    )
    assert c == 3
    assert isinstance(m, RecursiveLogExpMonomial)
    assert m.height >= 2
    assert (
        sp.simplify(
            m.expression - sp.exp(sp.exp(x) + x) * x ** sp.Rational(3, 2) * sp.log(sp.log(x)) ** 2
        )
        == 0
    )


def test_recursive_monomial_group_operations_are_exact():
    x = sp.symbols("x", positive=True)
    _, a = canonical_recursive_logexp_monomial(sp.exp(sp.exp(x)) * sp.log(x) ** 2, x)
    _, b = canonical_recursive_logexp_monomial(sp.exp(-sp.exp(x) + x) * sp.log(x) ** -1, x)
    product = a * b
    assert sp.simplify(product.expression - sp.exp(x) * sp.log(x)) == 0
    quotient = a / b
    assert sp.simplify(quotient.expression - sp.exp(2 * sp.exp(x) - x) * sp.log(x) ** 3) == 0


def test_recursive_nested_hierarchy_comparison():
    x = sp.symbols("x", positive=True)
    assert (
        compare_monomials(sp.exp(sp.exp(x)), sp.exp(x**100), x, point=sp.oo)
        is GrowthComparison.LARGER
    )
    assert (
        compare_monomials(sp.exp(sp.sqrt(sp.log(x))), sp.log(x) ** 100, x, point=sp.oo)
        is GrowthComparison.LARGER
    )
    assert (
        compare_monomials(sp.log(x), sp.log(sp.log(x)), x, point=sp.oo) is GrowthComparison.LARGER
    )


def test_transseries_parser_accepts_nested_logexp_terms_and_multiplies():
    x = sp.symbols("x", positive=True)
    s = transseries_from_expression(sp.exp(sp.exp(x)) + x * sp.log(sp.log(x)), x, point=sp.oo)
    assert len(s.terms) == 2
    assert any(isinstance(t.monomial, RecursiveLogExpMonomial) for t in s.terms)
    squared = (s * s).normalized()
    assert sp.simplify(squared.truncate() - (sp.exp(sp.exp(x)) + x * sp.log(sp.log(x))) ** 2) == 0


def test_recursive_exp_log_closure_on_transseries():
    x = sp.symbols("x", positive=True)
    base = transseries_from_expression(sp.exp(x) + x, x, point=sp.oo, complete=True)
    nested = base.exp()
    assert sp.simplify(nested.truncate() - sp.exp(sp.exp(x) + x)) == 0
    recovered = nested.log()
    assert sp.simplify(recovered.truncate() - (sp.exp(x) + x)) == 0
