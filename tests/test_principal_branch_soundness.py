import sympy as sp

from asymptotic.limits import LimitStatus, limit

x, y = sp.symbols("x y", real=True)


def test_arg_negative_real_cut_has_two_side_germs():
    r = limit(sp.arg(-1 + x + sp.I * y), (x, y), (0, 0), return_result=True)
    assert r.status is LimitStatus.DOES_NOT_EXIST
    assert {e.value for e in r.evidence} >= {sp.pi, -sp.pi}


def test_fractional_power_negative_real_cut_has_two_side_germs():
    r = limit(
        (-1 + x + sp.I * y) ** sp.Rational(1, 2), (x, y), (0, 0), return_result=True
    )
    assert r.status is LimitStatus.DOES_NOT_EXIST
    assert {e.value for e in r.evidence} >= {sp.I, -sp.I}


def test_branch_cut_guard_does_not_reject_regular_positive_target():
    for f, expected in (
        (sp.arg(1 + x + sp.I * y), 0),
        ((1 + x + sp.I * y) ** sp.Rational(1, 2), 1),
    ):
        r = limit(f, (x, y), (0, 0), return_result=True)
        assert r.status is LimitStatus.PROVED and r.value == expected


def test_tangential_contact_is_not_falsely_declared_a_cut_crossing():
    f = sp.arg(-1 + x**2 + sp.I * y**2)
    r = limit(f, (x, y), (0, 0), return_result=True)
    assert not (
        r.status is LimitStatus.DOES_NOT_EXIST
        and any(e.method.startswith("principal_cut_") for e in r.evidence)
    )
