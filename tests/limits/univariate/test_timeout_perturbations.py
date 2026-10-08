import mpmath as mp
import sympy as sp

from asymptotic import limit, one_sided_limit
from asymptotic.perturbation_scale_germs import (
    gamma_root_correction_certificate as gamma_cert,
)
from asymptotic.perturbation_scale_germs import (
    principal_lambert_analytic_scale_certificate as w_cert,
)
from asymptotic.perturbation_scale_germs import (
    real_log_exp_tower_certificate as log_cert,
)


def test_gamma_root_fixed_shift_and_positive_polynomial_tail():
    x = sp.Symbol("x", positive=True)
    a = sp.Symbol("a", positive=True)
    for z in (x + 3, 2 * x * x + 1):
        e = sp.rf(z, a) ** (1 / a) - z
        assert gamma_cert(e, x, sp.oo)[1] == (a - 1) / 2
    assert limit(sp.rf(x * x + 1, a) ** (1 / a) - x * x, x, sp.oo) == (a + 1) / 2
    with mp.workdps(50):
        a = mp.mpf("0.7")
        z = mp.mpf("1e7")
        observed = mp.exp((mp.loggamma(z + a) - mp.loggamma(z)) / a) - z
        assert abs(observed - (a - 1) / 2) < mp.mpf("1e-7")


def test_gamma_root_declines_missing_hypotheses_and_amplification():
    x = sp.Symbol("x", positive=True)
    a = sp.Symbol("a")
    assert gamma_cert(sp.rf(x, a) ** (1 / a) - x, x, sp.oo) is None
    a = sp.Symbol("a", positive=True)
    assert gamma_cert(x * (sp.rf(x, a) ** (1 / a) - x), x, sp.oo) is None
    assert gamma_cert(sp.rf(-x, a) ** (1 / a) + x, x, sp.oo) is None


def test_principal_w_complex_tail_quadratic_cancellations():
    n = sp.Symbol("n")
    q = sp.LambertW(n) / n
    t = sp.sqrt(q)
    assert limit((sp.exp(3 * t / 2) - 1 - 3 * t / 2) / q, n, -sp.oo) == sp.Rational(
        9, 8
    )
    assert limit((1 - sp.cos(t / 2)) / q, n, -sp.oo) == sp.Rational(1, 8)
    with mp.workdps(50):
        n = -mp.mpf("1e12")
        q = mp.lambertw(n) / n
        t = mp.sqrt(q)
        assert abs((mp.exp(3 * t / 2) - 1 - 3 * t / 2) / q - mp.mpf(9) / 8) < mp.mpf(
            "1e-5"
        )
        assert abs((1 - mp.cos(t / 2)) / q - mp.mpf(1) / 8) < mp.mpf("1e-10")


def test_lambert_certificate_declines_nonanalytic_nonprincipal_germs():
    x = sp.Symbol("x", positive=True)
    q = sp.LambertW(-x) / (-x)
    assert w_cert(sp.log(q), x, sp.oo) is None
    assert w_cert(1 / q, x, sp.oo) is None
    assert w_cert(sp.exp(sp.sqrt(sp.LambertW(-x, 1) / (-x))), x, sp.oo) is None


def test_real_log_exp_tower_preserves_principal_branch():
    x = sp.Symbol("x", positive=True)
    e = sp.log(
        sp.exp(
            sp.exp(
                sp.log(sp.exp(1 / (sp.log(x) + 1), evaluate=False), evaluate=False)
                ** -2
            ),
            evaluate=False,
        ),
        evaluate=False,
    )
    assert log_cert(e, x, 0)[1] is sp.oo
    assert one_sided_limit(e, x, 0, direction="+") is sp.oo
    z = sp.Symbol("z")
    assert log_cert(e.subs(x, z), z, 0) is None
    bad = sp.log(sp.exp(sp.I * sp.log(x), evaluate=False), evaluate=False)
    assert log_cert(bad, x, 0) is None
