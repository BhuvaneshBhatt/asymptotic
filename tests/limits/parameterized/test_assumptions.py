import sympy as sp

from asymptotic.limits import LimitStatus, limit


def test_symbolic_positive_power_parameter_refines_before_dispatch():
    x, N = sp.symbols("x N", real=True)
    r = limit(
        x ** (N + 2) / x**N, (x,), (0,), assumptions=sp.Gt(N, 0), return_result=True
    )
    assert r.status in {LimitStatus.PROVED, LimitStatus.UNKNOWN}
    if r.status is LimitStatus.PROVED:
        assert r.value == 0


def test_assumptions_never_force_unordered_symbolic_power_result():
    x, N, M = sp.symbols("x N M", real=True)
    r = limit(x**N / x**M, (x,), (0,), return_result=True)
    assert r.status is not LimitStatus.DOES_NOT_EXIST
