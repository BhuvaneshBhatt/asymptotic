import sympy as sp

from asymptotic.parameter_uniformity import ParameterInterval, uniform_parameter_expand


def test_uniform_exp_constant_over_parameter_interval():
    x, a = sp.symbols("x a", real=True)
    r = uniform_parameter_expand(
        sp.exp(a * x),
        x,
        parameters=(ParameterInterval(a, -2, 2),),
        order=4,
        radius_bound=sp.Rational(1, 10),
    )
    assert r.certified and r.remainder_certificate.uniform_variables == (a,)
    assert (
        sp.simplify(
            r.remainder_certificate.constant
            - 2**4 * sp.exp(sp.Rational(1, 5)) / sp.factorial(4)
        )
        == 0
    )


def test_geometric_requires_uniform_distance_from_pole():
    x, a = sp.symbols("x a", real=True)
    good = uniform_parameter_expand(
        1 / (1 + a * x),
        x,
        parameters=(ParameterInterval(a, 0, 2),),
        order=3,
        radius_bound=sp.Rational(1, 4),
    )
    bad = uniform_parameter_expand(
        1 / (1 + a * x),
        x,
        parameters=(ParameterInterval(a, 0, 2),),
        order=3,
        radius_bound=1,
    )
    assert good.certified and not bad.certified and bad.obligations


def test_log_uniform_remainder():
    x, a = sp.symbols("x a", real=True)
    r = uniform_parameter_expand(
        sp.log(1 + a * x),
        x,
        parameters=(ParameterInterval(a, -1, 1),),
        order=4,
        radius_bound=sp.Rational(1, 2),
    )
    assert r.certified and r.remainder_scale == x**4


def test_polynomial_exact_for_parameter_box():
    x, a, b = sp.symbols("x a b", real=True)
    r = uniform_parameter_expand(
        a * x + b * x**2,
        x,
        parameters=(ParameterInterval(a, -1, 1), ParameterInterval(b, 0, 3)),
        order=3,
    )
    assert r.certified and r.remainder_certificate.constant == 0


def test_unbounded_symbolic_parameter_declines():
    x, a, b = sp.symbols("x a b", real=True)
    r = uniform_parameter_expand(
        sp.exp((a + b) * x),
        x,
        parameters=(ParameterInterval(a, -1, 1),),
        order=3,
        radius_bound=sp.Rational(1, 10),
    )
    assert not r.certified and r.obligations
