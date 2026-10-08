import sympy as sp

from asymptotic.multivariate_expansion import UniformRemainderCertificate
from asymptotic.uniform_remainder import (
    UniformRemainder,
    UniformRemainderKind,
    from_weighted_certificate,
)


def test_uniform_sum_keeps_noncomparable_guessing_dominance():
    r, u = sp.symbols("r u", positive=True)
    a = UniformRemainder.big_o(
        r**2, r, constant=2, radius_bound=1, uniform_variables=(u,)
    )
    b = UniformRemainder.big_o(
        r**3, r, constant=5, radius_bound=sp.Rational(1, 2), uniform_variables=(u,)
    )
    c = a.add(b)
    assert c.certified and c.kind is UniformRemainderKind.BIG_O
    assert c.radius_bound == sp.Rational(1, 2)
    assert sp.simplify(c.scale - (2 * r**2 + 5 * r**3)) == 0


def test_uniform_product_propagates_little_o_strength():
    r, u = sp.symbols("r u", positive=True)
    a = UniformRemainder(r, UniformRemainderKind.LITTLE_O, r**2, 2, 1, (u,))
    b = UniformRemainder.big_o(
        r**3, r, constant=5, radius_bound=1, uniform_variables=(u,)
    )
    c = a.product(b)
    assert (
        c.kind is UniformRemainderKind.LITTLE_O and c.scale == r**5 and c.constant == 10
    )


def test_parameter_dependent_scaling_declines_without_uniform_bound():
    r, u = sp.symbols("r u", positive=True)
    a = UniformRemainder.big_o(
        r**2, r, constant=2, radius_bound=1, uniform_variables=(u,)
    )
    assert not a.scale_by(u).certified


def test_radial_substitution_preserves_uniform_variables():
    r, s, u = sp.symbols("r s u", positive=True)
    a = UniformRemainder.big_o(
        r**2, r, constant=2, radius_bound=1, uniform_variables=(u,)
    )
    b = a.substitute(s**2, s)
    assert b.certified and b.scale == s**4 and b.uniform_variables == (u,)


def test_weighted_certificate_adapter():
    r, u = sp.symbols("r u", positive=True)
    old = UniformRemainderCertificate(
        sp.Integer(4), sp.Integer(7), sp.Rational(1, 3), "test", "bound"
    )
    new = from_weighted_certificate(old, r, uniform_variables=(u,))
    assert (
        new.certified
        and new.scale == r**4
        and new.constant == 7
        and new.radius_bound == sp.Rational(1, 3)
    )


def test_multivariate_expansion_exposes_shared_uniform_contract():
    from asymptotic.multivariate_expansion import multivariate_expansion

    x, y = sp.symbols("x y", real=True)
    expansion = multivariate_expansion(
        1 / (1 + x**2 + y**2), (x, y), (0, 0), order=2, weights=(1, 1)
    )
    assert expansion.remainder_certificate is not None
    shared = expansion.uniform_remainder()
    assert shared.certified
    assert shared.radius == expansion.radius_symbol
    assert shared.uniform_variables == expansion.angular_variables
