import sympy as sp
from hypothesis import given, settings
from hypothesis import strategies as st

from asymptotic.limits import LimitStatus, limit
from asymptotic.multivariate_certificates import candidate_rational_limit_certificate

x, y = sp.symbols("x y", real=True)


def _proved(expr, expected):
    r = limit(expr, (x, y), (0, 0), return_result=True)
    assert r.status is LimitStatus.PROVED
    assert sp.simplify(r.value - expected) == 0


def test_candidate_rational_adversarial_table():
    cases = [
        ((x**2 + y**4) / (x**2 + y**4 + x**4), 1),
        ((x**4 + y**8) / (x**2 + y**4), 0),
        ((3 * x**2 + 3 * y**2 + x**4) / (x**2 + y**2), 3),
        ((x**2 + y**10) / (x - y) ** 4, None),
        ((x**4 - y**4) / (x**4 + y**4), None),
    ]
    for expr, expected in cases:
        cert = candidate_rational_limit_certificate(expr, (x, y), (0, 0))
        if expected is None:
            assert not (cert.certified and cert.value in (sp.oo, -sp.oo))
        else:
            assert cert.certified and sp.simplify(cert.value - expected) == 0
            _proved(expr, sp.Integer(expected))


@settings(max_examples=16, deadline=None)
@given(
    a=st.integers(min_value=1, max_value=7),
    b=st.integers(min_value=1, max_value=7),
    c=st.integers(min_value=1, max_value=7),
)
def test_higher_order_perturbations_do_not_change_zero_limit(a, b, c):
    expr = (a * x**4 + b * y**6 + c * x**2 * y**4) / (x**2 + y**2)
    _proved(expr, sp.Integer(0))


@settings(max_examples=12, deadline=None)
@given(k=st.integers(min_value=1, max_value=9))
def test_nonzero_candidate_is_invariant_and_scaling(k):
    base = (x**2 + y**4) / (x**2 + y**4 + x**4)
    swapped = base.xreplace({x: y, y: x})
    _proved(base, sp.Integer(1))
    _proved(swapped, sp.Integer(1))
    _proved(k * base, sp.Integer(k))


@settings(max_examples=12, deadline=None)
@given(a=st.integers(min_value=1, max_value=5), b=st.integers(min_value=1, max_value=5))
def test_invertible_diagonal_coordinate_changes_certified_limit(a, b):
    base = (x**4 + y**4) / (x**2 + y**2)
    changed = base.xreplace({x: a * x, y: b * y})
    _proved(base, sp.Integer(0))
    _proved(changed, sp.Integer(0))
