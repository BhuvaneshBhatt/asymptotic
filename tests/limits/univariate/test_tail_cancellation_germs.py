"""Independent germs, original-domain witnesses, and exceptional strata."""

import json
from pathlib import Path

import mpmath as mp
import sympy as sp

from asymptotic import limit, one_sided_limit
from asymptotic.limit_models import LimitStatus
from asymptotic.local_tail_germs import (
    ci_trig_cancellation_certificate as ci_cert,
)
from asymptotic.local_tail_germs import (
    radical_trig_tail_certificate as radical_cert,
)
from asymptotic.local_tail_germs import (
    reciprocal_gamma_exponential_domination_certificate as gamma_cert,
)
from asymptotic.local_tail_germs import (
    secant_odd_pi_pole_certificate as sec_cert,
)
from asymptotic.local_tail_germs import (
    upper_gamma_fixed_argument_tail_certificate as upper_cert,
)
from asymptotic.reference_normalization import scalar_reference_namespace


def row_expression(index):
    p = (
        Path(__file__).resolve().parents[2]
        / "data/univariate_limit_reference_cases.json"
    )
    return sp.sympify(
        json.loads(p.read_text())[index]["expression"],
        locals=scalar_reference_namespace(),
    )


def test_ci_exact_cancellation_and_both_principal_boundary_values():
    x = sp.Symbol("x")
    e = row_expression(15)
    expected = (sp.EulerGamma + sp.log(2)) / 2
    assert limit(e, x, 0) == expected
    with mp.workdps(70):
        f = sp.lambdify(x, e, "mpmath")
        for h in (mp.mpf("1e-5"), -mp.mpf("1e-5")):
            assert abs(f(h) - (mp.euler + mp.log(2)) / 2) < mp.mpf("1e-9")


def test_ci_declines_amplified_remainders_and_wrong_scaling_branch():
    x = sp.Symbol("x")
    assert ci_cert((sp.Ci(2 * x) - sp.log(x)) / x**2, x, 0) is None
    assert ci_cert(sp.Ci(-2 * x) - sp.log(x), x, 0) is None
    assert ci_cert(sp.Ci(2 * x) - sp.log(x) + sp.tan(1 / x), x, 0) is None


def test_secant_pole_has_attained_tail_poles():
    x = sp.Symbol("x")
    e = 2 * x / (1 + sp.sec((x + 1) * sp.Abs(sp.pi - x)))
    result = limit(e, x, 0, return_result=True)
    assert result.status is LimitStatus.DOES_NOT_EXIST
    witnesses = [ev for ev in result.evidence if ev.substitutions]
    assert len(witnesses) == 2 and {ev.value for ev in witnesses} == {sp.oo, -sp.oo}
    for ev in witnesses:
        variable, sequence = ev.substitutions[0]
        n = next(iter(sequence.free_symbols))
        assert variable == x
        # For n>=4, |x|<=1/4 and 0<|delta|<pi/2 using 3<pi<4:
        # delta=x*(pi-1-x). Hence cos(phase) and 1+sec(phase)
        # are nonzero; these are attained values, not interval enclosures.
        for k in (4, 20, 100):
            h = sequence.subs(n, k)
            delta = h * (sp.pi - 1 - h)
            assert bool(sp.Abs(delta) > 0) and bool(sp.Abs(delta) < sp.pi / 2)
        with mp.workdps(40):
            f = sp.lambdify(x, e, "mpmath")
            h = sequence.subs(n, 1000)
            actual = f(mp.mpf(str(h.evalf(40))))
            assert mp.sign(actual) == (1 if ev.value is sp.oo else -1)
    assert one_sided_limit(e, x, 0, direction="+") is -sp.oo
    assert one_sided_limit(e, x, 0, direction="-") is sp.oo


def test_secant_family_multiple_orders_and_exception_rejection():
    x = sp.Symbol("x", positive=True)
    a = sp.Symbol("a")
    assert sec_cert(x**4 / (1 + sp.sec(sp.pi + x * x)), x, 0)[1] == -2
    assert sec_cert(x**5 / (1 + sp.sec(sp.pi + x * x)), x, 0)[1] == 0
    assert sec_cert(x / (1 + sp.sec(sp.pi + a * x)), x, 0) is None
    assert sec_cert(x / (1 + sp.sec(sp.pi + sp.Abs(x + a))), x, 0) is None
    assert sec_cert(x / (1 + sp.sec(sp.pi + sp.sin(1 / x))), x, 0) is None


def test_upper_gamma_exact_shift_and_complex_fixed_argument():
    n = sp.Symbol("n", positive=True)
    w = sp.Symbol("w")
    e = (n + 1) * sp.uppergamma(n + 1, w) * sp.exp(w) / sp.gamma(n + 2)
    assert upper_cert(e, n, sp.oo)[1] == sp.exp(w)
    assert limit(e, n, sp.oo) == sp.exp(w)
    with mp.workdps(60):
        for w in (mp.mpf("2"), mp.mpc("-2", "1")):
            observed = mp.gammainc(31, w, mp.inf) / mp.gamma(31)
            assert abs(observed - 1) < mp.mpf("1e-22")


def test_upper_gamma_declines_moving_unproved_denominators():
    n = sp.Symbol("n", positive=True)
    w, p = sp.symbols("w p")
    assert upper_cert(sp.uppergamma(n, n) / sp.gamma(n), n, sp.oo) is None
    assert (
        upper_cert(sp.uppergamma(n, w) / ((p * n + 1) * sp.gamma(n)), n, sp.oo) is None
    )
    # Even a finite leading ratio cannot erase an undefined lower coefficient.
    e = sp.uppergamma(n, w) * (n + 1 / (p - 1)) / (sp.gamma(n) * (n + 2 / (p - 1)))
    assert upper_cert(e, n, sp.oo) is None
    infinite = sp.Symbol("infinite", finite=False)
    assert upper_cert(sp.uppergamma(n, infinite) / sp.gamma(n), n, sp.oo) is None


def test_two_gamma_factorial_domination_zero_base():
    k = sp.Symbol("k", positive=True)
    x, v = sp.symbols("x v")
    e = x ** (2 * k) / (sp.gamma(k - v + 1) * sp.gamma(k + v + 1))
    assert gamma_cert(e, k, sp.oo)[1] == 0
    assert limit(e, k, sp.oo) == 0
    with mp.workdps(50):
        value = (mp.mpc("2", "1") ** 60) / (mp.gamma(31 - mp.j) * mp.gamma(31 + mp.j))
        assert abs(value) < mp.mpf("1e-40")
    assert gamma_cert(x ** (-k) / (sp.gamma(k + 1) * sp.gamma(k + 2)), k, sp.oo) is None
    assert (
        gamma_cert(2 ** (k * k) / (sp.gamma(k + 1) * sp.gamma(k + 2)), k, sp.oo) is None
    )


def test_radical_cosine_ratio_retains_first_nonzero_phase_terms():
    n = sp.Symbol("n", positive=True)
    e = row_expression(928).subs(sp.Symbol("n"), n)
    assert radical_cert(e, n, sp.oo)[1] == 1
    assert limit(e, n, sp.oo) == 1
    with mp.workdps(60):
        f = sp.lambdify(n, e, "mpmath")
        assert abs(f(mp.mpf("1e8")) - 1) < mp.mpf("5e-8")
    assert radical_cert(n**5 * e, n, sp.oo) is None
    assert radical_cert(sp.cos(sp.sqrt(-n)), n, sp.oo) is None
