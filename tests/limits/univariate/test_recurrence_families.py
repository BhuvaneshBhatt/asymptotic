"""Exact residuals, independent numeric checks and exceptional-stratum guards."""

import mpmath as mp
import sympy as sp

from asymptotic.recurrence_simplification import simplify_recurrences as reduce
from asymptotic.reference_normalization import RegularizedHypergeometric0F1
from asymptotic.special_functions import StruveH, StruveL


def test_orthogonal_polynomial_degree_families():
    n = sp.Symbol("n", integer=True, positive=True)
    z = sp.Symbol("z")
    specs = [
        (sp.hermite, (z,), 2 * z, 2 * (n + 1)),
        (sp.hermite_prob, (z,), z, n + 1),
        (sp.chebyshevt, (z,), 2 * z, 1),
        (sp.chebyshevu, (z,), 2 * z, 1),
        (sp.legendre, (z,), (2 * n + 3) * z / (n + 2), (n + 1) / (n + 2)),
        (sp.laguerre, (z,), (2 * n + 3 - z) / (n + 2), (n + 1) / (n + 2)),
        (
            sp.assoc_laguerre,
            (sp.Rational(2, 3), z),
            (2 * n + sp.Rational(11, 3) - z) / (n + 2),
            (n + sp.Rational(5, 3)) / (n + 2),
        ),
        (
            sp.gegenbauer,
            (sp.Rational(2, 3), z),
            2 * (n + sp.Rational(5, 3)) * z / (n + 2),
            (n + sp.Rational(4, 3)) / (n + 2),
        ),
        (sp.assoc_legendre, (1, z), (2 * n + 3) * z / (n + 1), (n + 2) / (n + 1)),
    ]
    for head, args, A, C in specs:
        e = head(n + 2, *args) - A * head(n + 1, *args) + C * head(n, *args)
        r = reduce(e, budget_seconds=1, return_result=True)
        assert r.expression == 0, (head, r)
        assert sp.simplify(e.subs(n, 3)) == 0
    a, b = sp.Rational(2, 3), sp.Rational(1, 4)
    p = n + 1
    t = 2 * p + a + b
    e = (
        2 * (p + 1) * (p + a + b + 1) * t * sp.jacobi(n + 2, a, b, z)
        - (t + 1) * (t * (t + 2) * z + a * a - b * b) * sp.jacobi(n + 1, a, b, z)
        + 2 * (p + a) * (p + b) * (t + 2) * sp.jacobi(n, a, b, z)
    )
    assert reduce(e, budget_seconds=1, max_ops=400) == 0
    assert sp.simplify(e.subs(n, 2)) == 0


def test_discrete_contract_and_parameter_denominator_guards():
    n = sp.Symbol("n")
    z = sp.Symbol("z")
    e = (
        sp.hermite(n + 2, z)
        - 2 * z * sp.hermite(n + 1, z)
        + 2 * (n + 1) * sp.hermite(n, z)
    )
    assert reduce(e) == e
    n = sp.Symbol("n", integer=True, nonnegative=True)
    a, b = sp.symbols("a b")
    e = sp.jacobi(n, a, b, z) + sp.jacobi(n + 2, a, b, z)
    assert reduce(e) == e


def test_struve_inhomogeneous_terms_and_principal_powers():
    v = sp.Symbol("v")
    z = sp.Symbol("z", nonzero=True)
    term = (z / 2) ** (v + 1) / (sp.sqrt(sp.pi) * sp.gamma(v + sp.Rational(5, 2)))
    h = StruveH(v + 2, z) + StruveH(v, z) - 2 * (v + 1) * StruveH(v + 1, z) / z - term
    log_value = (
        StruveL(v + 2, z) - StruveL(v, z) + 2 * (v + 1) * StruveL(v + 1, z) / z + term
    )
    assert reduce(h, budget_seconds=1) == 0
    assert reduce(log_value, budget_seconds=1) == 0
    with mp.workdps(40):
        for f in (mp.struveh, mp.struvel):
            v = mp.mpf("0.7")
            z = mp.mpc("1.2", "0.4")
            source = (z / 2) ** (v + 1) / (mp.sqrt(mp.pi) * mp.gamma(v + mp.mpf("2.5")))
            residual = (
                f(v, z)
                + (1 if f == mp.struveh else -1) * f(v + 2, z)
                - 2 * (v + 1) * f(v + 1, z) / z
                - source
            )
            assert abs(residual) < mp.mpf("1e-35")


def test_lower_gamma_harmonic_and_lerch_shifts():
    a = sp.Symbol("a", positive=True)
    z = sp.Symbol("z", nonzero=True)
    s = sp.Symbol("s")
    assert (
        reduce(
            sp.lowergamma(a + 1, z) - a * sp.lowergamma(a, z) + z**a * sp.exp(-z),
            budget_seconds=1,
        )
        == 0
    )
    assert (
        reduce(
            sp.harmonic(a + 1, s) - sp.harmonic(a, s) - (a + 1) ** (-s),
            budget_seconds=1,
        )
        == 0
    )
    e = z * sp.lerchphi(z, s, a + 1) - sp.lerchphi(z, s, a) + a ** (-s)
    assert reduce(e, budget_seconds=1) == 0
    bad = sp.Symbol("b")
    e = sp.lerchphi(z, s, bad + 1) - sp.lerchphi(z, s, bad)
    assert reduce(e) == e
    with mp.workdps(40):
        z, a, s = mp.mpf(".3"), mp.mpf(".7"), mp.mpf("1.2")
        assert abs(
            z * mp.lerchphi(z, s, a + 1) - mp.lerchphi(z, s, a) + a ** (-s)
        ) < mp.mpf("1e-35")


def test_factorial_pochhammer_subfactorial_and_argument_shifts():
    n = sp.Symbol("n", integer=True, positive=True)
    a = sp.Symbol("a")
    assert (
        reduce(sp.factorial(n + 2) / sp.factorial(n), budget_seconds=1)
        == n * n + 3 * n + 2
    )
    for head, sign in ((sp.rf, 1), (sp.ff, -1)):
        assert (
            reduce(head(a, n + 1) - (a + sign * n) * head(a, n), budget_seconds=1) == 0
        )
    assert (
        reduce(
            sp.subfactorial(n + 2)
            - (n + 1) * (sp.subfactorial(n + 1) + sp.subfactorial(n)),
            budget_seconds=1,
        )
        == 0
    )
    z = sp.Symbol("z")

    def B(x):
        return sp.bernoulli(5, x, evaluate=False)

    def E(x):
        return sp.euler(4, x, evaluate=False)

    assert reduce(B(z + 1) - B(z) - 5 * z**4, budget_seconds=1) == 0
    assert reduce(E(z + 1) + E(z) - 2 * z**4, budget_seconds=1) == 0


def test_hypergeometric_contiguous_families_and_permuted_upper_slots():
    a = sp.Symbol("a", positive=True)
    b = sp.Symbol("b", positive=True)
    z = sp.Symbol("z", negative=True)

    def F(c):
        return sp.hyper((), (c,), z)

    assert reduce(b * (b + 1) * (F(b) - F(b + 1)) - z * F(b + 2), budget_seconds=1) == 0

    def M(c):
        return sp.hyper((c,), (b,), z)

    e = (b - a - 1) * M(a) + (2 * a + 2 - b + z) * M(a + 1) - (a + 1) * M(a + 2)
    assert reduce(e, budget_seconds=1) == 0

    def M(c):
        return sp.hyper((a,), (c,), z)

    e = b * (b + 1) * M(b) - (b + 1) * (b + z) * M(b + 1) + z * (b + 1 - a) * M(b + 2)
    assert reduce(e, budget_seconds=1) == e
    assert reduce(e, assumptions=sp.Ne(a - b - 1, 0), budget_seconds=1) == 0
    c = sp.Symbol("c", positive=True)

    def F(p):
        return sp.hyper((p, b), (c,), z)

    e = (
        (c - a - 1) * F(a)
        + (2 * a + 2 - c + (b - a - 1) * z) * F(a + 1)
        + (a + 1) * (z - 1) * F(a + 2)
    )
    assert reduce(e, budget_seconds=1) == 0

    def R(p):
        return RegularizedHypergeometric0F1(p, z, evaluate=False)

    assert reduce(R(a) - (a + 1) * R(a + 1) - z * R(a + 2), budget_seconds=1) == 0


def test_hypergeometric_numeric_independent_validation():
    with mp.workdps(40):
        a, b, c, z = map(mp.mpf, ("0.7", "1.3", "2.1", "0.2"))

        def F(p):
            return mp.hyp2f1(p, b, c, z)

        assert abs(
            (c - a - 1) * F(a)
            + (2 * a + 2 - c + (b - a - 1) * z) * F(a + 1)
            + (a + 1) * (z - 1) * F(a + 2)
        ) < mp.mpf("1e-35")

        def M(p):
            return mp.hyp1f1(p, b, z)

        assert abs(
            (b - a - 1) * M(a) + (2 * a + 2 - b + z) * M(a + 1) - (a + 1) * M(a + 2)
        ) < mp.mpf("1e-35")


def test_recurrence_work_budgets():
    n = sp.Symbol("n", positive=True, integer=True)
    z = sp.Symbol("z")
    e = sp.hermite(n + 8, z) + sp.hermite(n, z)
    assert reduce(e, max_shift=4) == e
    e = sp.hermite(n + 2, z) + sp.hermite(n, z)
    assert reduce(e, max_steps=1) == e
    assert reduce(e, budget_seconds=0) == e
    assert reduce(e, max_ops=1) == e


def test_parabolic_cylinder_kummer_u_and_whittaker_order_recurrences():
    from asymptotic.special_functions import (
        HypergeometricU as U,
    )
    from asymptotic.special_functions import (
        ParabolicCylinderD as D,
    )
    from asymptotic.special_functions import (
        WhittakerM as M,
    )
    from asymptotic.special_functions import (
        WhittakerW as W,
    )

    a = sp.Symbol("a", positive=True)
    z = sp.Symbol("z", positive=True)
    b = sp.Rational(1, 3)
    mu = sp.Rational(2, 3)
    p = a + 1
    assert (
        reduce(D(a + 2, z) - z * D(a + 1, z) + (a + 1) * D(a, z), budget_seconds=1) == 0
    )
    assert (
        reduce(
            U(a, b, z)
            + (b - 2 * p - z) * U(p, b, z)
            + p * (p - b + 1) * U(p + 1, b, z),
            budget_seconds=1,
        )
        == 0
    )
    assert (
        reduce(
            (a - b) * U(b, a, z) + (1 - p - z) * U(b, p, z) + z * U(b, p + 1, z),
            budget_seconds=1,
        )
        == 0
    )
    assert (
        reduce(
            (p - mu - sp.Rational(1, 2)) * M(a, mu, z)
            + (z - 2 * p) * M(p, mu, z)
            + (p + mu + sp.Rational(1, 2)) * M(p + 1, mu, z),
            budget_seconds=1,
        )
        == 0
    )
    assert (
        reduce(
            W(p + 1, mu, z)
            + (2 * p - z) * W(p, mu, z)
            + (p - mu - sp.Rational(1, 2)) * (p + mu - sp.Rational(1, 2)) * W(a, mu, z),
            budget_seconds=1,
        )
        == 0
    )
    with mp.workdps(40):
        a, b, mu, z = map(mp.mpf, (".7", ".3", ".6", "1.2"))
        p = a + 1
        assert abs(
            mp.pcfd(a + 2, z) - z * mp.pcfd(a + 1, z) + (a + 1) * mp.pcfd(a, z)
        ) < mp.mpf("1e-35")
        assert abs(
            mp.hyperu(a, b, z)
            + (b - 2 * p - z) * mp.hyperu(p, b, z)
            + p * (p - b + 1) * mp.hyperu(p + 1, b, z)
        ) < mp.mpf("1e-35")
        assert abs(
            (a - b) * mp.hyperu(b, a, z)
            + (1 - p - z) * mp.hyperu(b, p, z)
            + z * mp.hyperu(b, p + 1, z)
        ) < mp.mpf("1e-35")
        assert abs(
            (p - mu - mp.mpf(".5")) * mp.whitm(a, mu, z)
            + (z - 2 * p) * mp.whitm(p, mu, z)
            + (p + mu + mp.mpf(".5")) * mp.whitm(p + 1, mu, z)
        ) < mp.mpf("1e-35")
        assert abs(
            mp.whitw(p + 1, mu, z)
            + (2 * p - z) * mp.whitw(p, mu, z)
            + (p - mu - mp.mpf(".5")) * (p + mu - mp.mpf(".5")) * mp.whitw(a, mu, z)
        ) < mp.mpf("1e-35")


def test_regularized_contiguous_relations_at_parameter_poles():
    from asymptotic.special_functions import (
        RegularizedHypergeometric1F1 as R,
    )
    from asymptotic.special_functions import (
        RegularizedHypergeometric2F1 as F,
    )

    a = sp.Symbol("a", positive=True)
    b = sp.Symbol("b")
    z = sp.Symbol("z", negative=True)
    e = (
        (b - a - 1) * R(a, b, z)
        + (2 * a + 2 - b + z) * R(a + 1, b, z)
        - (a + 1) * R(a + 2, b, z)
    )
    assert reduce(e, budget_seconds=1, require_defined_germ=True) == 0
    assert reduce(e.subs(b, -2), budget_seconds=1, require_defined_germ=True) == 0
    e = R(a, b, z) - (b + z) * R(a, b + 1, z) + z * (b + 1 - a) * R(a, b + 2, z)
    assert reduce(e, assumptions=sp.Ne(a - b - 1, 0), budget_seconds=1) == 0
    e = (
        (-2 - a - 1) * F(a, b, -2, z)
        + (2 * a + 2 + 2 + (b - a - 1) * z) * F(a + 1, b, -2, z)
        + (a + 1) * (z - 1) * F(a + 2, b, -2, z)
    )
    assert reduce(e, budget_seconds=1, require_defined_germ=True) == 0


def test_function_branch_contracts():
    from asymptotic.function_normalization import RealRoot, normalize_functions

    x = sp.Symbol("x", positive=True)
    expr = RealRoot(x, 2)
    assert expr == sp.sqrt(x)
    unknown = sp.Function("undefined_function")(x)
    assert normalize_functions(unknown, (x,), (sp.oo,)) == (unknown, ())


def test_recurrence_evidence():
    from asymptotic import limit
    from asymptotic.special_functions import ParabolicCylinderD as D

    x = sp.Symbol("x", positive=True)
    e = (
        D(sp.Rational(7, 3), x)
        - x * D(sp.Rational(4, 3), x)
        + sp.Rational(4, 3) * D(sp.Rational(1, 3), x)
    )
    result = limit(e, x, sp.oo, return_result=True)
    assert result.value == 0 and result.expression == e
    assert any(
        v.method == "integer_shift_recurrence_ParabolicCylinderD"
        for v in result.evidence
    )


def test_kelvin_coupled_recurrences_and_real_domain_guards():
    from asymptotic.special_functions import KelvinBei, KelvinBer, KelvinKei, KelvinKer

    v = sp.Symbol("v", real=True)
    z = sp.Symbol("z", positive=True)
    for R, image in ((KelvinBer, KelvinBei), (KelvinKer, KelvinKei)):
        c = sp.sqrt(2) * (v + 1) / z
        assert (
            reduce(
                R(v + 2, z) + R(v, z) + c * (R(v + 1, z) - image(v + 1, z)),
                budget_seconds=1,
            )
            == 0
        )
        assert (
            reduce(
                image(v + 2, z) + image(v, z) + c * (R(v + 1, z) + image(v + 1, z)),
                budget_seconds=1,
            )
            == 0
        )
        bad = sp.Symbol("t")
        e = R(v + 2, bad) + R(v, bad)
        assert reduce(e) == e
    with mp.workdps(40):
        v, z = mp.mpf(".7"), mp.mpf("1.2")
        c = mp.sqrt(2) * (v + 1) / z
        for R, image in ((mp.ber, mp.bei), (mp.ker, mp.kei)):
            assert abs(
                R(v + 2, z) + R(v, z) + c * (R(v + 1, z) - image(v + 1, z))
            ) < mp.mpf("1e-35")
            assert abs(
                image(v + 2, z) + image(v, z) + c * (R(v + 1, z) + image(v + 1, z))
            ) < mp.mpf("1e-35")


def test_spherical_harmonic_degree_recurrence():
    n = sp.Symbol("n", integer=True, positive=True)
    theta, phi = sp.symbols("theta phi", real=True)
    p = n + 1
    m = sp.S.One
    high = sp.sqrt(((p + 1) ** 2 - m**2) / ((2 * p + 1) * (2 * p + 3)))
    low = sp.sqrt((p**2 - m**2) / ((2 * p - 1) * (2 * p + 1)))
    e = (
        high * sp.Ynm(n + 2, m, theta, phi)
        - sp.cos(theta) * sp.Ynm(n + 1, m, theta, phi)
        + low * sp.Ynm(n, m, theta, phi)
    )
    assert reduce(e, budget_seconds=1, max_ops=400) == 0
    assert (
        abs(
            complex(
                e.subs({n: 3, theta: sp.Rational(3, 5), phi: sp.Rational(1, 4)}).evalf(
                    35
                )
            )
        )
        < 1e-30
    )


def test_ferrers_q_branch_interval_and_degree_recurrence():
    from asymptotic.special_functions import LegendreQ as Q

    n = sp.Symbol("n", positive=True)
    z = sp.Symbol("z", real=True)
    p = n + 1
    e = (p + 1) * Q(p + 1, 0, z) - (2 * p + 1) * z * Q(p, 0, z) + p * Q(n, 0, z)
    assert reduce(e) == e
    assert reduce(e, assumptions=sp.And(z > -1, z < 1), budget_seconds=1) == 0
    with mp.workdps(40):
        n, z = mp.mpf(".7"), mp.mpf(".2")
        p = n + 1
        assert abs(
            (p + 1) * mp.legenq(p + 1, 0, z, type=2)
            - (2 * p + 1) * z * mp.legenq(p, 0, z, type=2)
            + p * mp.legenq(n, 0, z, type=2)
        ) < mp.mpf("1e-35")


def test_gauss_lower_parameter_and_parameter_recurrences():
    from asymptotic.special_functions import RegularizedHypergeometric2F1 as R

    c = sp.Symbol("c", positive=True)
    a, b = sp.Rational(1, 3), sp.Rational(2, 3)
    z = sp.Symbol("z", negative=True)
    p = c + 1

    def F(q):
        return sp.hyper((a, b), (q,), z)

    e = (
        p * (p - 1) * (z - 1) * F(c)
        + p * (p - 1 - (2 * p - a - b - 1) * z) * F(p)
        + (p - a) * (p - b) * z * F(p + 1)
    )
    assert reduce(e, budget_seconds=1) == 0
    e = (
        (z - 1) * R(a, b, c, z)
        + (p - 1 - (2 * p - a - b - 1) * z) * R(a, b, p, z)
        + (p - a) * (p - b) * z * R(a, b, p + 1, z)
    )
    assert reduce(e, budget_seconds=1) == 0
    with mp.workdps(40):
        a, b, c, z = map(mp.mpf, (".3", ".6", "1.2", ".2"))
        p = c + 1

        def F(q):
            return mp.hyp2f1(a, b, q, z)

        assert abs(
            p * (p - 1) * (z - 1) * F(c)
            + p * (p - 1 - (2 * p - a - b - 1) * z) * F(p)
            + (p - a) * (p - b) * z * F(p + 1)
        ) < mp.mpf("1e-35")


def test_translated_heads_cannot_inherit_uncertified_generic_germs():
    from asymptotic import limit, one_sided_limit
    from asymptotic.limit_models import LimitStatus
    from asymptotic.special_functions import StruveL

    x = sp.Symbol("x", positive=True)
    t = sp.Symbol("t", real=True)
    examples = [
        (sp.assoc_legendre(sp.Rational(1, 3), 1, t), t, -1),
        (
            sp.sqrt(x)
            * (
                sp.sqrt(2)
                * x ** (-x - sp.Rational(1, 4))
                * sp.factorial(x)
                * sp.laguerre(x, -1)
                * sp.exp(-2 * sp.sqrt(x) + x + sp.Rational(1, 2))
                - 1
            ),
            x,
            sp.oo,
        ),
    ]
    for expr, var, point in examples:
        r = limit(expr, var, point, return_result=True)
        assert r.status is LimitStatus.UNKNOWN and r.value is None
        assert any(
            e.method == "recurrence_family_germ_prerequisite" for e in r.evidence
        )
    r = one_sided_limit(examples[0][0], t, -1, direction="+", return_result=True)
    assert r.status is LimitStatus.UNKNOWN
    # The exact subtracted tail now has a separate certified theorem.
    r = limit(
        sp.pi * x * (sp.besseli(0, x) - StruveL(0, x)) / 2, x, sp.oo, return_result=True
    )
    assert r.status is LimitStatus.PROVED and r.value == 1
    assert r.evidence[0].method == "struve_bessel_exact_subtraction_remainder"
    # The exact Struve recurrence must still close before the missing-germ guard.
    from asymptotic.special_functions import StruveH

    v = sp.Rational(1, 3)
    expr = (
        StruveH(v + 2, x)
        + StruveH(v, x)
        - 2 * (v + 1) * StruveH(v + 1, x) / x
        - (x / 2) ** (v + 1) / (sp.sqrt(sp.pi) * sp.gamma(v + sp.Rational(5, 2)))
    )
    r = limit(expr, x, sp.oo, return_result=True)
    assert r.value == 0 and any(
        e.method == "integer_shift_recurrence_StruveH" for e in r.evidence
    )
