import sympy as sp

from asymptotic.frontier import SparseTerm, compose_analytic_terms


def test_analytic_composition_frontier_exponential():
    # exp(z + 2 z^2) = 1 + z + 5/2 z^2 + ...
    tail = [SparseTerm(1, 1), SparseTerm(2, 2)]
    out = compose_analytic_terms(0, tail, lambda p, c: sp.exp(c), 3)
    assert [term.exponent for term in out] == [0, 1, 2]
    assert [sp.simplify(term.coefficient) for term in out] == [1, 1, sp.Rational(5, 2)]


def test_analytic_composition_frontier_logarithm():
    # log(1 + z + z^2) = z + z^2/2 + ...
    tail = [SparseTerm(1, 1), SparseTerm(2, 1)]

    def derivative(p, c):
        if p == 0:
            return sp.log(c)
        return (-1) ** (p - 1) * sp.factorial(p - 1) / c**p

    out = compose_analytic_terms(1, tail, derivative, 2)
    assert [term.exponent for term in out] == [1, 2]
    assert sp.simplify(out[0].coefficient - 1) == 0
    assert sp.simplify(out[1].coefficient - sp.Rational(1, 2)) == 0
