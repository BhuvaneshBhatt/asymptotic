"""Algebra of radial/logarithmic and complex local asymptotic germs."""

from __future__ import annotations

from dataclasses import dataclass, replace

import sympy as sp


@dataclass(frozen=True)
class GermOrder:
    radial: sp.Expr = sp.S.Zero
    logarithmic: sp.Expr = sp.S.Zero
    loglog: sp.Expr = sp.S.Zero

    def __add__(self, other):
        return GermOrder(
            *(
                sp.simplify(a + b)
                for a, b in zip(self.as_tuple(), other.as_tuple(), strict=True)
            )
        )

    def __sub__(self, other):
        return GermOrder(
            *(
                sp.simplify(a - b)
                for a, b in zip(self.as_tuple(), other.as_tuple(), strict=True)
            )
        )

    def scale(self, c):
        return GermOrder(*(sp.simplify(c * a) for a in self.as_tuple()))

    def as_tuple(self):
        return self.radial, self.logarithmic, self.loglog


@dataclass(frozen=True)
class LocalAsymptoticGerm:
    """``r^a (log r)^b (log|log r|)^c A(u)`` with exact leading data."""

    order: GermOrder
    angular_factor: sp.Expr = sp.S.One
    radial_variable: sp.Symbol | None = None
    angular_variables: tuple[sp.Symbol, ...] = ()
    coefficient: sp.Expr = sp.S.One
    domain: sp.Expr = sp.S.true
    provider: str = "local_germ"

    def __mul__(self, other):
        if not isinstance(other, LocalAsymptoticGerm):
            return replace(self, coefficient=sp.simplify(self.coefficient * other))
        self._compatible(other)
        return replace(
            self,
            order=self.order + other.order,
            angular_factor=sp.simplify(self.angular_factor * other.angular_factor),
            coefficient=sp.simplify(self.coefficient * other.coefficient),
            provider="germ_product",
        )

    def __truediv__(self, other):
        if not isinstance(other, LocalAsymptoticGerm):
            return replace(self, coefficient=sp.simplify(self.coefficient / other))
        self._compatible(other)
        return replace(
            self,
            order=self.order - other.order,
            angular_factor=sp.simplify(self.angular_factor / other.angular_factor),
            coefficient=sp.simplify(self.coefficient / other.coefficient),
            provider="germ_quotient",
        )

    def power(self, power):
        power = sp.sympify(power)
        return replace(
            self,
            order=self.order.scale(power),
            angular_factor=sp.simplify(self.angular_factor**power),
            coefficient=sp.simplify(self.coefficient**power),
            provider="germ_power",
        )

    def conjugate(self):
        return replace(
            self,
            angular_factor=sp.conjugate(self.angular_factor),
            coefficient=sp.conjugate(self.coefficient),
            provider="germ_conjugate",
        )

    def real_part(self):
        return replace(
            self,
            angular_factor=sp.re(self.coefficient * self.angular_factor),
            coefficient=sp.S.One,
            provider="germ_real_part",
        )

    def imag_part(self):
        return replace(
            self,
            angular_factor=sp.im(self.coefficient * self.angular_factor),
            coefficient=sp.S.One,
            provider="germ_imag_part",
        )

    def differentiate_radial(self):
        """Differentiate the leading germ, including logarithmic resonance."""
        a, b, c = self.order.as_tuple()
        r = self.radial_variable
        if r is None:
            raise ValueError("radial variable is required")
        if a != 0:
            return replace(
                self,
                order=GermOrder(a - 1, b, c),
                coefficient=sp.simplify(self.coefficient * a),
                provider="germ_derivative",
            )
        if b != 0:
            return replace(
                self,
                order=GermOrder(-1, b - 1, c),
                coefficient=sp.simplify(self.coefficient * b),
                provider="germ_derivative",
            )
        if c != 0:
            return replace(
                self,
                order=GermOrder(-1, -1, c - 1),
                coefficient=sp.simplify(self.coefficient * c),
                provider="germ_derivative",
            )
        return replace(self, coefficient=sp.S.Zero, provider="germ_derivative")

    def expression(self):
        r = self.radial_variable
        if r is None:
            raise ValueError("radial variable is required")
        a, b, c = self.order.as_tuple()
        return sp.simplify(
            self.coefficient
            * r**a
            * sp.log(r) ** b
            * sp.log(sp.Abs(sp.log(r))) ** c
            * self.angular_factor
        )

    def _compatible(self, other):
        if (
            self.radial_variable != other.radial_variable
            or self.angular_variables != other.angular_variables
        ):
            raise ValueError("germs use different local coordinates")


@dataclass(frozen=True)
class ComplexLocalGerm:
    """``z^alpha (log z)^k A(u)`` with branch metadata."""

    power: sp.Expr
    log_power: sp.Expr = sp.S.Zero
    angular_factor: sp.Expr = sp.S.One
    coefficient: sp.Expr = sp.S.One
    branch_index: int = 0
    monodromy_state: object | None = None
    provider: str = "complex_local_germ"

    def __mul__(self, other):
        if not isinstance(other, ComplexLocalGerm):
            return replace(self, coefficient=sp.simplify(self.coefficient * other))
        if self.branch_index != other.branch_index:
            raise ValueError("incompatible branches")
        return ComplexLocalGerm(
            sp.simplify(self.power + other.power),
            sp.simplify(self.log_power + other.log_power),
            sp.simplify(self.angular_factor * other.angular_factor),
            sp.simplify(self.coefficient * other.coefficient),
            self.branch_index,
            self.monodromy_state or other.monodromy_state,
            "complex_germ_product",
        )

    def __truediv__(self, other):
        if not isinstance(other, ComplexLocalGerm):
            return replace(self, coefficient=sp.simplify(self.coefficient / other))
        if self.branch_index != other.branch_index:
            raise ValueError("incompatible branches")
        return ComplexLocalGerm(
            sp.simplify(self.power - other.power),
            sp.simplify(self.log_power - other.log_power),
            sp.simplify(self.angular_factor / other.angular_factor),
            sp.simplify(self.coefficient / other.coefficient),
            self.branch_index,
            self.monodromy_state or other.monodromy_state,
            "complex_germ_quotient",
        )

    def with_monodromy(self, state, divisor_index=0):
        w = state.winding_for(divisor_index)
        coefficient = sp.simplify(self.coefficient * w.multiplier)
        angular = sp.simplify(
            self.angular_factor * sp.exp(w.log_increment * self.log_power)
        )
        return replace(
            self,
            coefficient=coefficient,
            angular_factor=angular,
            branch_index=sp.simplify(self.branch_index + w.winding),
            monodromy_state=state,
            provider="complex_germ_monodromy",
        )

    def conjugate(self):
        return replace(
            self,
            power=sp.conjugate(self.power),
            angular_factor=sp.conjugate(self.angular_factor),
            coefficient=sp.conjugate(self.coefficient),
            branch_index=-self.branch_index,
            provider="complex_germ_conjugate",
        )


@dataclass(frozen=True)
class GermRule:
    function: object
    builder: object
    statement: str


_GERM_RULES = {}


def register_germ_rule(function, builder, statement):
    _GERM_RULES[function] = GermRule(function, builder, statement)


def function_germ(function, *args, **kwargs):
    rule = _GERM_RULES.get(function)
    if rule is None:
        return None
    return rule.builder(*args, **kwargs)


def _besselj_germ(order, *, radial_variable, angular_factor=sp.S.One):
    return LocalAsymptoticGerm(
        GermOrder(order),
        angular_factor,
        radial_variable,
        coefficient=1 / (2**order * sp.gamma(order + 1)),
        provider="besselj_germ_data",
    )


register_germ_rule(
    sp.besselj,
    _besselj_germ,
    "J_nu(z) ~ (z/2)^nu/Gamma(nu+1) at zero on a compatible branch",
)


def _erf_germ(*, radial_variable, angular_factor=sp.S.One):
    return LocalAsymptoticGerm(
        GermOrder(1),
        angular_factor,
        radial_variable,
        coefficient=2 / sp.sqrt(sp.pi),
        provider="erf_germ_data",
    )


def _erfi_germ(*, radial_variable, angular_factor=sp.S.One):
    return LocalAsymptoticGerm(
        GermOrder(1),
        angular_factor,
        radial_variable,
        coefficient=2 / sp.sqrt(sp.pi),
        provider="erfi_germ_data",
    )


def _gamma_germ(*, radial_variable, angular_factor=sp.S.One):
    return LocalAsymptoticGerm(
        GermOrder(-1),
        1 / angular_factor,
        radial_variable,
        coefficient=sp.S.One,
        provider="gamma_pole_germ_data",
    )


register_germ_rule(sp.erf, _erf_germ, "erf(z) ~ 2 z/sqrt(pi) at zero")
register_germ_rule(sp.erfi, _erfi_germ, "erfi(z) ~ 2 z/sqrt(pi) at zero")
register_germ_rule(sp.gamma, _gamma_germ, "Gamma(z) ~ 1/z at zero")


__all__ = [
    "ComplexLocalGerm",
    "GermOrder",
    "GermRule",
    "LocalAsymptoticGerm",
    "function_germ",
    "register_germ_rule",
    "shifted_gamma_germ",
]


def _bessely_germ(order, *, radial_variable, angular_factor=sp.S.One):
    order = sp.sympify(order)
    if order.is_integer is not True or order.is_nonnegative is not True:
        return None
    if order == 0:
        return LocalAsymptoticGerm(
            GermOrder(0),
            sp.log(radial_variable * angular_factor / 2) + sp.EulerGamma,
            radial_variable,
            coefficient=2 / sp.pi,
            provider="bessely0_log_germ_data",
        )
    return LocalAsymptoticGerm(
        GermOrder(-order),
        angular_factor ** (-order),
        radial_variable,
        coefficient=-(sp.gamma(order) / sp.pi) * 2**order,
        provider="bessely_integer_germ_data",
    )


def shifted_gamma_germ(pole, *, radial_variable, angular_factor=sp.S.One):
    """Gamma(-n+z) ~ (-1)^n/(n! z), n a nonnegative integer."""
    pole = sp.sympify(pole)
    n = -pole
    if n.is_integer is not True or n.is_nonnegative is not True:
        return None
    return LocalAsymptoticGerm(
        GermOrder(-1),
        1 / angular_factor,
        radial_variable,
        coefficient=(-1) ** n / sp.factorial(n),
        provider="gamma_shifted_pole_germ_data",
    )


register_germ_rule(
    sp.bessely, _bessely_germ, "integer-order Bessel Y local singular germ"
)
