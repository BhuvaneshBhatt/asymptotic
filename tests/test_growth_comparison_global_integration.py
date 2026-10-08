import pytest
import sympy as sp

from asymptotic.limits import LimitStatus, limit

CASES = [
    ("x**N/x**M", "N > M", "0"),
    ("x**N/x**M", "N < M", "oo"),
    ("log(1/x)**(-N)/log(1/x)**(-M)", "N > M & M > 0", "0"),
    ("log(1/x)**(-N)/log(1/x)**(-M)", "M > N & N > 0", "oo"),
    ("exp(-1/x**q)/x**N", "q > 0 & N > 0", "0"),
    ("x**N/exp(-1/x**q)", "q > 0 & N > 0", "oo"),
    ("exp(-exp(exp(1/x)))/exp(-exp(1/x))", None, "0"),
    ("exp(-1/x)*log(1/x)**7/x**3", None, "0"),
    ("x**3/(exp(-1/x)*log(1/x)**7)", None, "oo"),
    ("exp(-1/sqrt(x))/x**3", None, "0"),
    ("x**3/exp(-1/sqrt(x))", None, "oo"),
]


def parse(s, loc):
    return (
        sp.And(*(sp.sympify(p.strip(), locals=loc) for p in s.split("&")))
        if s
        else sp.S.true
    )


@pytest.mark.parametrize("expr,condition,expected", CASES)
def test_growth_comparison_proof_lifts_to_global_limit(expr, condition, expected):
    x, N, M, q = sp.symbols("x N M q", real=True)
    loc = locals()
    r = limit(
        sp.sympify(expr, locals=loc),
        (x,),
        (0,),
        domain=x > 0,
        assumptions=parse(condition, loc),
        return_result=True,
    )
    assert r.status is LimitStatus.PROVED
    assert r.value == sp.sympify(expected, locals=loc)
    assert any(
        e.method
        in {"growth_comparison", "univariate_reduction", "one_sided_domain_limit"}
        for e in r.evidence
    )
