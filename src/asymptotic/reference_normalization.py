"""Neutral normalization helpers for exact multivariate reference cases."""

from __future__ import annotations

import sympy as sp
from sympy.logic.boolalg import Boolean

from .step_factorials import StepFactorialPower as StepFactorialPower


def scalar_reference_assumptions(text, namespace=None):
    """Read a source assumption list as a conjunction, preserving its clauses."""
    from .reference_contracts import parse_reference

    value = parse_reference(text, namespace or scalar_reference_namespace())
    if isinstance(value, bool):
        return sp.S.true if value else sp.S.false
    if isinstance(value, (tuple, list, sp.Tuple)):
        if not all(isinstance(clause, Boolean) for clause in value):
            raise ValueError("reference assumption list contains a non-logical clause")
        return sp.And(*value)
    if not isinstance(value, Boolean):
        raise ValueError("reference assumption is not a logical expression")
    return value


class Normal(sp.Function):
    """Finite part of a formal series, with no unspecified-function semantics."""

    nargs = 1


class Series(sp.Function):
    """Formal expansion request whose specification is a variable/center/order tuple."""

    nargs = 2


class RegularizedHypergeometric0F1(sp.Function):
    """Entire-parameter 0F1/gamma; never substitute a spurious 0*infinity."""

    nargs = 2

    @classmethod
    def eval(cls, b, z):
        if b.is_Integer and b <= 0:
            m = int(-b)
            if m <= 128:
                return z ** (m + 1) * sp.hyper((), (m + 2,), z) / sp.factorial(m + 1)
        if b.is_positive is True or (b.is_number and b.is_integer is False):
            return sp.hyper((), (b,), z) / sp.gamma(b)


def scalar_reference_equal(
    actual, expected, comparison_codomain="exact", *, assumptions=sp.S.true
):
    """Compare proved values in an explicit codomain, retaining raw direction."""
    if comparison_codomain not in ("exact", "extended_complex"):
        raise ValueError("unsupported reference comparison codomain")
    from .branch_constant_normalization import normalize_unit_roots

    def branch_constants(value):
        replacements = {}
        for atom in value.atoms(sp.Function):
            if atom.func.__name__ == "Chi" and atom.args[0].is_negative is True:
                replacements[atom] = atom.func(-atom.args[0]) + sp.I * sp.pi
            elif atom.func is sp.elliptic_k and atom.args == (sp.Integer(2),):
                replacements[atom] = (
                    sp.sqrt(sp.pi)
                    * (1 - sp.I)
                    * sp.gamma(sp.Rational(1, 4))
                    / (4 * sp.gamma(sp.Rational(3, 4)))
                )
        value = value.xreplace(replacements)
        return value.replace(
            lambda a: (
                a.func is sp.conjugate
                and a.args[0].func.__name__ == "Chi"
                and a.args[0].args[0].is_positive is True
            ),
            lambda a: a.args[0],
        )

    actual, expected = (
        branch_constants(normalize_unit_roots(value)) for value in (actual, expected)
    )
    if actual == expected:
        return True
    if (
        actual.func == expected.func
        and actual.func in (sp.acos, sp.asin, sp.atan, sp.log, sp.atanh)
        and not (actual.free_symbols | expected.free_symbols)
        and max(sp.count_ops(actual), sp.count_ops(expected)) <= 50
    ):
        if sp.cancel(actual.args[0] - expected.args[0]) == 0:
            return True
    if comparison_codomain == "extended_complex" and expected is sp.zoo:
        if actual.func is sp.Piecewise:
            return actual.args[-1][1] is sp.S.true and all(
                scalar_reference_equal(value, expected, comparison_codomain)
                for value, _ in actual.args
            )
        from .fixed_ray_branch_germs import DirectionalInfinity

        if actual.func is DirectionalInfinity:
            return True
        if actual.is_Mul and any(a in (sp.oo, -sp.oo) for a in actual.args):
            phase = sp.Mul(*(a for a in actual.args if a not in (sp.oo, -sp.oo)))
            return phase.is_finite is True and phase.is_zero is False
        return actual in (sp.oo, -sp.oo, sp.zoo)
    # Directional infinities carry the normalized phase d/Abs(d).
    # Compare only exact finite numeric directions, without merging them with zoo.
    if expected.func.__name__ == "DirectionalInfinity" and len(expected.args) == 1:
        d = expected.args[0]
        phase = None
        if actual.func.__name__ == "DirectionalInfinity":
            phase = actual.args[0]
        elif actual is sp.oo:
            phase = sp.S.One
        elif actual is -sp.oo:
            phase = -sp.S.One
        elif actual.is_Mul and any(a in (sp.oo, -sp.oo) for a in actual.args):
            phase = sp.Mul(*(a for a in actual.args if a not in (sp.oo, -sp.oo)))
            if -sp.oo in actual.args:
                phase = -phase
        if phase is not None and max(sp.count_ops(phase), sp.count_ops(d)) <= 60:
            if (
                phase.is_finite is True
                and d.is_finite is True
                and phase.is_zero is False
                and d.is_zero is False
            ):
                difference = sp.trigsimp(
                    sp.expand_complex(phase / sp.Abs(phase) - d / sp.Abs(d))
                )
                return sp.trigsimp(sp.expand_complex(sp.simplify(difference))) == 0
        return False
    if actual in (sp.oo, -sp.oo, sp.zoo) or expected in (sp.oo, -sp.oo, sp.zoo):
        return False
    difference = actual - expected
    if (
        not difference.free_symbols
        and sp.count_ops(difference) <= 120
        and difference.has(sp.sinh, sp.cosh)
        and difference.has(sp.asin, sp.acos)
    ):
        angle_form = difference.replace(sp.acos, lambda z: sp.pi / 2 - sp.asin(z))
        if sp.cancel(angle_form.rewrite(sp.exp)) == 0:
            return True
    if (
        not difference.free_symbols
        and sp.count_ops(difference) <= 80
        and difference.has(sp.tan, sp.sec, sp.cot, sp.csc)
    ):
        if sp.simplify(difference.rewrite(sp.cos)) == 0:
            return True
    if sp.simplify(difference) == 0:
        return True
    if max(sp.count_ops(actual), sp.count_ops(expected)) > 120:
        return False
    from ._symbolic_policy import bounded_ask

    clauses = sp.And.make_args(assumptions)
    positive = {
        parameter: sp.Dummy("positive_reference_parameter", positive=True)
        for parameter in difference.free_symbols
        if (parameter > 0) in clauses
        or bounded_ask(sp.Q.positive(parameter), assumptions) is True
    }
    difference = difference.xreplace(positive)
    # Factor inside radicals before refining their sign, so sqrt(b**2)
    # is replaced by b only in the certified positive parameter chart.
    difference = difference.replace(
        lambda node: (
            node.is_Pow
            and node.exp in (sp.S.Half, -sp.S.Half)
            and sp.count_ops(node.base) <= 80
        ),
        lambda node: sp.Pow(
            sp.factor(sp.trigsimp(node.base.rewrite(sp.cos))), node.exp
        ),
    )
    expanded = sp.expand(difference)
    logs = sorted(expanded.atoms(sp.log), key=sp.default_sort_key)
    for index, first in enumerate(logs):
        c = expanded.coeff(first)
        if c == 0 or c.has(sp.log):
            continue
        for second in logs[index + 1 :]:
            if expanded.coeff(second) != -c:
                continue
            z = first.args[0]
            if sp.cancel(z + second.args[0]) != 0:
                continue
            imag = sp.im(z)
            if imag.is_positive is True:
                jump = sp.I * sp.pi
            elif imag.is_negative is True:
                jump = -sp.I * sp.pi
            elif z.is_positive is True:
                jump = -sp.I * sp.pi
            elif z.is_negative is True:
                jump = sp.I * sp.pi
            else:
                numerator, denominator = sp.fraction(z)
                factor = numerator / sp.I
                if (
                    denominator.is_Pow
                    and denominator.exp == sp.S.Half
                    and sp.im(denominator.base).is_zero is False
                    and factor.is_real is True
                    and (factor.is_positive is True or factor.is_negative is True)
                ):
                    # A principal square root off the real cut has positive
                    # real part; i/sqrt(w) consequently has positive imaginary part.
                    jump = sp.I * sp.pi if factor.is_positive else -sp.I * sp.pi
                else:
                    continue
            expanded = expanded - c * first + c * second + c * jump
            break
    if sp.simplify(expanded) == 0:
        return True
    expanded = expanded.replace(
        lambda node: node.func is sp.acos,
        lambda node: sp.pi / 2 - sp.asin(node.args[0]),
    )
    return sp.simplify(sp.trigsimp(expanded.rewrite(sp.exp))) == 0


def scalar_reference_namespace():
    """Return the SymPy vocabulary and native mathematical function classes."""
    from .analytic_limits import SquareWave
    from .fixed_ray_branch_germs import DirectionalInfinity
    from .function_normalization import (
        InverseRegularizedGamma,
        NearestInteger,
        RealRoot,
        TriangleWave,
        dawson,
        hypergeometric_1f1,
        inverse_error_increment,
    )
    from .rational_approximation import rational_approximation

    namespace = {name: getattr(sp, name) for name in dir(sp)}
    from . import special_functions, weierstrass_origin

    for module in (special_functions, weierstrass_origin):
        namespace.update(
            {
                name: value
                for name, value in vars(module).items()
                if isinstance(value, type) and issubclass(value, sp.Function)
            }
        )
    namespace.update(
        {
            name: value
            for name, value in globals().items()
            if isinstance(value, type) and issubclass(value, sp.Function)
        }
    )
    namespace.update(
        SquareWave=SquareWave,
        DirectionalInfinity=DirectionalInfinity,
        NearestInteger=NearestInteger,
        TriangleWave=TriangleWave,
        RealRoot=RealRoot,
        rational_approximation=rational_approximation,
        InverseRegularizedGamma=InverseRegularizedGamma,
        dawson=dawson,
        inverse_error_increment=inverse_error_increment,
        hypergeometric_1f1=hypergeometric_1f1,
    )

    def exact_binomial(top, bottom):
        top, bottom = map(sp.sympify, (top, bottom))
        return (
            sp.binomial(top, bottom, evaluate=False)
            if top.is_Integer and top < 0 and bottom.free_symbols
            else sp.binomial(top, bottom)
        )

    from .conditional import conditional_expression
    from .reference_contracts import reference_membership

    namespace.update(
        EllipticNomeQ=special_functions.EllipticNome,
        InverseEllipticNomeQ=special_functions.InverseEllipticNome,
        Contains=reference_membership,
        Alternatives=lambda *args: sp.Tuple(*args),
        conditional_value=conditional_expression,
        Unequal=sp.Ne,
        Beta=sp.beta,
        Integrate=sp.Integral,
        CubeRoot=lambda z: RealRoot(z, 3),
        Reals=sp.S.Reals,
        Integers=sp.S.Integers,
    )
    namespace["binomial"] = exact_binomial
    return namespace


def multivariate_reference_namespace():
    """Bind exact scalar source names for multivariate reference evaluation.

    Registered real roots retain their real-domain checks. Unknown user
    functions and functions with unsettled endpoint conventions stay unbound.
    """
    from .discontinuous_functions import SignedFractionalPart, unit_step
    from .fixed_ray_branch_germs import DirectionalInfinity

    namespace = scalar_reference_namespace()
    namespace.update(
        FractionalPart=SignedFractionalPart,
        UnitStep=unit_step,
        ArcCot=sp.acot,
        ArcSec=sp.asec,
        Directedoo=DirectionalInfinity,
        Log10=lambda argument: sp.log(argument) / sp.log(10),
        Re=sp.re,
        Im=sp.im,
        Conjugate=sp.conjugate,
        Arg=sp.arg,
        Erfi=sp.erfi,
        FresnelC=sp.fresnelc,
        FresnelS=sp.fresnels,
        HankelH1=sp.hankel1,
        HankelH2=sp.hankel2,
        Sech=sp.sech,
    )
    return namespace


def real_algebraic_root(poly, variable, index):
    """Exact indexed real root with certified ordering."""
    poly = sp.Poly(poly, variable)
    rr = [r for r in poly.all_roots() if r.is_real is True]
    return rr[int(index)]


def normalize_domain_membership(variable, set_expr):
    """Translate exact real-domain membership into a Boolean domain condition."""
    if set_expr == sp.S.Reals:
        return (
            sp.S.true if variable.is_real is True else sp.Contains(variable, sp.S.Reals)
        )
    return sp.Contains(variable, set_expr)


def normalize_complex_argument(expr):
    return sp.arg(sp.sympify(expr))


def normalize_modular(expr, modulus):
    return sp.Mod(sp.sympify(expr), sp.sympify(modulus))


def normalize_derivative(expr, variable, order=1):
    """Evaluate exact symbolic derivatives before local-limit dispatch."""
    return sp.diff(sp.sympify(expr), variable, int(order))


__all__ = [
    "normalize_complex_argument",
    "normalize_derivative",
    "normalize_domain_membership",
    "normalize_modular",
    "real_algebraic_root",
]
