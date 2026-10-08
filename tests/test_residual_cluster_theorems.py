import sympy as sp

from asymptotic.limits import LimitStatus, limit

x, y, t = sp.symbols("x y t", real=True)


def result(e, vars=(x, y), target=(0, 0)):
    return limit(e, vars, target, return_result=True)


def test_uniform_radial_vanishing_handles_inverse_radial_denominator():
    r = result(sp.asin(x * y) / sp.sqrt(x * x + y * y))
    assert r.status is LimitStatus.PROVED and r.value == 0
    assert any(e.method == "uniform_radial_order" for e in r.evidence)


def test_uniform_bounded_oscillation_sum_vanishes():
    r = result(x**2 * y**2 * (1 - sp.cos(1 / (x * y))))
    assert r.status is LimitStatus.PROVED and r.value == 0


def test_phase_tail_image_proves_dne_for_nonradial_pole():
    r = result(sp.sin(1 / (x - y)))
    assert r.status is LimitStatus.DOES_NOT_EXIST


def test_phase_tail_image_survives_vanishing_additive_perturbation():
    r = result(-sp.sin(y - 1 / x) + sp.Abs(x) + sp.Abs(y))
    assert r.status is LimitStatus.DOES_NOT_EXIST


def test_phase_tail_image_handles_ratio_at_nonzero_target():
    r = result(sp.cos(x / y), target=(1, 0))
    assert r.status is LimitStatus.DOES_NOT_EXIST


def test_log_of_vanishing_linear_form_diverges_negative():
    r = result(sp.log(sp.Abs(x + y)))
    assert r.status is LimitStatus.PROVED and r.value == -sp.oo
    assert any(e.method == "log_vanishing_inner" for e in r.evidence)


def test_log_vanishing_shortcut_rejects_rational_zero_over_zero_germ():
    from asymptotic.multivariate_limits_advanced import coercive_outer_divergence_limit

    inner = sp.Mul(x, sp.Pow(x**2 + y**2, -1, evaluate=False), evaluate=False)
    expr = sp.log(sp.Abs(inner))
    shortcut = coercive_outer_divergence_limit(expr, (x, y), (0, 0))
    assert shortcut is None or shortcut.provider != "log_vanishing_inner"


def test_log_vanishing_shortcut_accepts_continuous_rational_germ():
    r = result(sp.log(sp.Abs(x / (1 + x**2 + y**2))))
    assert r.status is LimitStatus.PROVED and r.value == -sp.oo
    assert any(e.method == "log_vanishing_inner" for e in r.evidence)


def test_log_vanishing_shortcut_uses_reduced_rational_form():
    raw = sp.Mul(
        x * (1 + x**2 + y**2),
        sp.Pow(1 + x**2 + y**2, -1, evaluate=False),
        evaluate=False,
    )
    from asymptotic.multivariate_limits_advanced import coercive_outer_divergence_limit

    shortcut = coercive_outer_divergence_limit(sp.log(sp.Abs(raw)), (x, y), (0, 0))
    assert shortcut is not None
    assert shortcut.certified and shortcut.value == -sp.oo
    assert shortcut.provider == "log_vanishing_inner"


def test_positive_vanishing_denominator_diverges():
    r = result(sp.cos(x * y) / (sp.Abs(x) * sp.Abs(y)))
    assert r.status is LimitStatus.PROVED and r.value == sp.oo


def test_exp_positive_reciprocal_diverges():
    r = result(sp.exp(1 / (t**2 * x**2)), vars=(t, x), target=(0, 0))
    assert r.status is LimitStatus.PROVED and r.value == sp.oo


def test_bounded_phase_without_unbounded_tail_is_not_misclassified():
    # phase x+y tends to zero, so periodicity cannot be used as a DNE proof.
    r = result(sp.sin(x + y))
    assert r.status is LimitStatus.PROVED and r.value == 0
