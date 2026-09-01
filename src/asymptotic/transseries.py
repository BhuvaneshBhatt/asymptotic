from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import cmp_to_key

import sympy as sp

from ._power_simplify import analytic_powsimp, formal_powsimp, mixed_powsimp
from ._symbolic_errors import SYMBOLIC_ERRORS
from .complex_domain import (
    ComplexBranchMetadata,
    ComplexSector,
    merge_complex_germ_metadata,
)
from .context import AsymptoticContext, GrowthComparison, context_for
from .logexp_transseries import (
    RecursiveLogExpMonomial,
    canonical_recursive_logexp_monomial,
)
from .monomial import (
    AsymptoticMonomial,
    canonical_asymptotic_monomial,
    compare_asymptotic_monomials,
)
from .remainder import (
    AsymptoticRemainder,
    AsymptoticTruncation,
)


@dataclass(frozen=True)
class TransseriesValuation:
    """Leading asymptotic monomial data for an exact coefficient.

    ``leading_term = leading_coefficient * monomial``. ``monomial`` stores the
    SymPy expression while ``canonical_monomial`` stores the structural
    power/log/exponential representation when canonicalization succeeds.
    """

    expression: sp.Expr
    leading_term: sp.Expr
    leading_coefficient: sp.Expr
    monomial: sp.Expr
    point: sp.Expr = 0
    canonical_monomial: AsymptoticMonomial | None = None


@dataclass(frozen=True)
class TransseriesTerm:
    """One coefficient times one generalized asymptotic monomial.

    Existing callers may still pass a SymPy expression as ``monomial``.
    Adapter-created and arithmetic-normalized terms use
    :class:`AsymptoticMonomial` directly.
    """

    coefficient: sp.Expr
    monomial: sp.Expr | AsymptoticMonomial | RecursiveLogExpMonomial

    def __post_init__(self) -> None:
        object.__setattr__(self, "coefficient", sp.sympify(self.coefficient))
        if not isinstance(self.monomial, (AsymptoticMonomial, RecursiveLogExpMonomial)):
            object.__setattr__(self, "monomial", formal_powsimp(sp.sympify(self.monomial)))

    @property
    def monomial_expression(self) -> sp.Expr:
        if isinstance(self.monomial, (AsymptoticMonomial, RecursiveLogExpMonomial)):
            return self.monomial.expression
        return self.monomial

    @property
    def expression(self) -> sp.Expr:
        return mixed_powsimp(self.coefficient, self.monomial_expression)

    def canonical(
        self,
        variable: sp.Symbol,
        *,
        point: sp.Expr = 0,
        parameter: sp.Symbol | None = None,
    ) -> TransseriesTerm:
        if isinstance(self.monomial, RecursiveLogExpMonomial):
            if self.monomial.variable != variable or self.monomial.point != sp.sympify(point):
                raise ValueError("recursive monomial uses a different variable or asymptotic point")
            return self
        if isinstance(self.monomial, AsymptoticMonomial):
            if parameter is None or self.monomial.parameter == parameter:
                return self
            target = self.monomial.ramification.__class__(
                variable, point, self.monomial.ramification.index, parameter
            )
            return TransseriesTerm(self.coefficient, self.monomial.on_ramification(target))
        try:
            coefficient, monomial = canonical_asymptotic_monomial(
                self.monomial,
                variable,
                point=point,
                parameter=parameter,
            )
        except ValueError:
            coefficient, monomial = canonical_recursive_logexp_monomial(
                self.monomial, variable, point=point
            )
        return TransseriesTerm(sp.simplify(self.coefficient * coefficient), monomial)

    def __mul__(self, other: TransseriesTerm) -> TransseriesTerm:
        """Multiply finite transseries prefixes while propagating remainder semantics."""
        if not isinstance(other, TransseriesTerm):
            return NotImplemented
        if isinstance(self.monomial, AsymptoticMonomial) and isinstance(
            other.monomial, AsymptoticMonomial
        ):
            monomial: sp.Expr | AsymptoticMonomial | RecursiveLogExpMonomial = (
                self.monomial * other.monomial
            )
        elif isinstance(self.monomial, RecursiveLogExpMonomial) and isinstance(
            other.monomial, RecursiveLogExpMonomial
        ):
            monomial = self.monomial * other.monomial
        else:
            monomial = formal_powsimp(self.monomial_expression * other.monomial_expression)
        return TransseriesTerm(sp.simplify(self.coefficient * other.coefficient), monomial)

    def __truediv__(self, other: TransseriesTerm) -> TransseriesTerm:
        if not isinstance(other, TransseriesTerm):
            return NotImplemented
        if sp.sympify(other.coefficient).is_zero is True:
            raise ZeroDivisionError("division by a zero transseries term")
        if isinstance(self.monomial, AsymptoticMonomial) and isinstance(
            other.monomial, AsymptoticMonomial
        ):
            monomial: sp.Expr | AsymptoticMonomial | RecursiveLogExpMonomial = (
                self.monomial / other.monomial
            )
        elif isinstance(self.monomial, RecursiveLogExpMonomial) and isinstance(
            other.monomial, RecursiveLogExpMonomial
        ):
            monomial = self.monomial / other.monomial
        else:
            monomial = formal_powsimp(self.monomial_expression / other.monomial_expression)
        return TransseriesTerm(sp.simplify(self.coefficient / other.coefficient), monomial)


def _merge_metadata(*items: dict[str, object]) -> dict[str, object]:
    """Merge operation/provenance metadata without discarding prior evidence."""

    out = {}
    for metadata in items:
        for key, value in metadata.items():
            if key in {"property_decisions", "operation_provenance", "remainder_certificates"}:
                existing = list(out.get(key, []))
                incoming = list(value) if isinstance(value, (list, tuple)) else [value]
                existing.extend(v for v in incoming if v not in existing)
                out[key] = existing
            elif key not in out:
                out[key] = value
            elif out[key] != value:
                out[key] = (out[key], value)
    return out


@dataclass(frozen=True)
class TransseriesExpansion:
    """Finite exact prefix of a generalized transseries branch.

    Terms are stored in descending asymptotic magnitude. Structural monomials
    support exact finite-prefix arithmetic alongside expression-backed
    construction.
    """

    expression: sp.Expr
    variable: sp.Symbol
    point: sp.Expr
    terms: tuple[TransseriesTerm, ...]
    center: sp.Expr = sp.S.Zero
    complete: bool = False
    metadata: dict[str, object] = field(default_factory=dict, compare=False, hash=False, repr=False)
    remainder: AsymptoticRemainder | None = field(
        default=None, compare=False, hash=False, repr=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "expression", sp.sympify(self.expression))
        object.__setattr__(self, "point", sp.sympify(self.point))
        object.__setattr__(self, "center", sp.sympify(self.center))
        object.__setattr__(self, "terms", tuple(self.terms))
        remainder = self.remainder
        if remainder is None:
            remainder = (
                AsymptoticRemainder.exact_zero(self.variable, self.point)
                if self.complete
                else AsymptoticRemainder.unknown(self.variable, self.point)
            )
        elif remainder.variable != self.variable or remainder.point != self.point:
            raise ValueError("transseries remainder uses a different variable or asymptotic point")
        if self.complete and not remainder.is_exact:
            raise ValueError("complete=True requires an exact-zero remainder")
        object.__setattr__(self, "remainder", remainder)

    @property
    def leading_term(self) -> TransseriesTerm | None:
        return self.terms[0] if self.terms else None

    def asymptotic_element(self):
        """View this transseries through the common asymptotic-field protocol."""
        from .algebra import asymptotic_element

        return asymptotic_element(self)

    def truncate(self, n: int | None = None) -> sp.Expr:
        selected = self.terms if n is None else self.terms[: int(n)]
        return analytic_powsimp(
            sp.expand(self.center + sum((term.expression for term in selected), sp.S.Zero))
        )

    def truncation(self, n: int | None = None) -> AsymptoticTruncation:
        """Return a prefix together with rigorous remainder semantics.

        Unlike :meth:`truncate`, this does not silently discard information.
        If known terms are omitted, their exact sum is retained and a certified
        ``O(first omitted monomial)`` bound is attached whenever the pre-existing
        remainder is known to be smaller (or zero).
        """

        normalized = self.normalized()
        count = len(normalized.terms) if n is None else max(0, min(int(n), len(normalized.terms)))
        prefix = normalized.truncate(count)
        omitted = normalized.terms[count:]
        remainder = normalized.remainder
        if remainder is None:
            raise RuntimeError("normalized transseries is missing its remainder")
        if not omitted:
            return AsymptoticTruncation(prefix, remainder, count, len(normalized.terms))

        omitted_expr = sp.powsimp(sp.expand(sum((term.expression for term in omitted), sp.S.Zero)))
        first_scale = omitted[0].monomial_expression
        finite_tail = AsymptoticRemainder.big_o(
            first_scale,
            self.variable,
            self.point,
            exact_expression=omitted_expr,
            source="finite omitted transseries tail",
        )
        combined = finite_tail.add(remainder)
        return AsymptoticTruncation(prefix, combined, count, len(normalized.terms))

    def prefix(self, n: int) -> TransseriesExpansion:
        """Return a shorter expansion while retaining its certified remainder."""

        trunc = self.truncation(n)
        normalized = self.normalized()
        kept = normalized.terms[: max(0, int(n))]
        return TransseriesExpansion.from_terms(
            self.variable,
            self.point,
            kept,
            center=self.center,
            complete=trunc.remainder.is_exact,
            metadata=dict(self.metadata),
            remainder=trunc.remainder,
        )

    def canonical_terms(self) -> tuple[TransseriesTerm, ...]:
        return tuple(term.canonical(self.variable, point=self.point) for term in self.terms)

    def valuation(self) -> TransseriesTerm | None:
        """Return the dominant nonzero term of this finite prefix."""

        normalized = self.normalized()
        return normalized.leading_term

    def normalized(self) -> TransseriesExpansion:
        """Canonicalize, combine equal monomials, remove zeros, and order terms."""

        grouped = {}
        fallback = []
        canonical_parameter = next(
            (
                term.monomial.parameter
                for term in self.terms
                if isinstance(term.monomial, AsymptoticMonomial)
            ),
            sp.Dummy("t", positive=True),
        )
        for term in self.terms:
            if term.coefficient == 0:
                continue
            try:
                item = term.canonical(
                    self.variable, point=self.point, parameter=canonical_parameter
                )
            except ValueError:
                fallback.append(term)
                continue
            if not isinstance(item.monomial, (AsymptoticMonomial, RecursiveLogExpMonomial)):
                raise TypeError("canonical term did not produce a structural monomial")
            grouped[item.monomial] = sp.simplify(grouped.get(item.monomial, 0) + item.coefficient)

        combined = [
            TransseriesTerm(coeff, mon) for mon, coeff in grouped.items() if sp.simplify(coeff) != 0
        ]
        combined.extend(fallback)
        ordered = ordered_transseries_terms(combined, self.variable, point=self.point)
        expr = analytic_powsimp(
            sp.expand(self.center + sum((term.expression for term in ordered), sp.S.Zero))
        )
        return TransseriesExpansion(
            expr,
            self.variable,
            self.point,
            ordered,
            self.center,
            self.complete,
            dict(self.metadata),
            self.remainder,
        )

    @classmethod
    def from_terms(
        cls,
        variable: sp.Symbol,
        point: sp.Expr,
        terms: Iterable[TransseriesTerm],
        *,
        center: sp.Expr = 0,
        complete: bool = False,
        metadata: dict[str, object] | None = None,
        remainder: AsymptoticRemainder | None = None,
    ) -> TransseriesExpansion:
        raw = tuple(terms)
        expr = analytic_powsimp(
            sp.expand(sp.sympify(center) + sum((term.expression for term in raw), sp.S.Zero))
        )
        return cls(
            expr, variable, point, raw, center, complete, metadata or {}, remainder
        ).normalized()

    def _check_compatible(self, other: TransseriesExpansion) -> None:
        if self.variable != other.variable or self.point != other.point:
            raise ValueError("transseries expansions use different variables or asymptotic points")

    def __add__(self, other: TransseriesExpansion | sp.Expr) -> TransseriesExpansion:
        if isinstance(other, TransseriesExpansion):
            self._check_compatible(other)
            from .remainder_theorems import certify_finite_sum_remainder

            certificate = certify_finite_sum_remainder((self.remainder, other.remainder))
            metadata = _merge_metadata(self.metadata, other.metadata)
            metadata.setdefault("remainder_certificates", []).append(certificate)
            return TransseriesExpansion.from_terms(
                self.variable,
                self.point,
                self.terms + other.terms,
                center=sp.simplify(self.center + other.center),
                complete=certificate.conclusion.is_exact,
                metadata=metadata,
                remainder=certificate.conclusion,
            )
        scalar = sp.sympify(other)
        if self.variable in scalar.free_symbols:
            return NotImplemented
        return TransseriesExpansion.from_terms(
            self.variable,
            self.point,
            self.terms,
            center=sp.simplify(self.center + scalar),
            complete=self.complete,
            metadata=dict(self.metadata),
            remainder=self.remainder,
        )

    __radd__ = __add__

    def __neg__(self) -> TransseriesExpansion:
        from .remainder_theorems import certify_scaling_remainder

        certificate = certify_scaling_remainder(-1, self.remainder)
        metadata = dict(self.metadata)
        metadata.setdefault("remainder_certificates", []).append(certificate)
        return TransseriesExpansion.from_terms(
            self.variable,
            self.point,
            (TransseriesTerm(-term.coefficient, term.monomial) for term in self.terms),
            center=-self.center,
            complete=certificate.conclusion.is_exact,
            metadata=metadata,
            remainder=certificate.conclusion,
        )

    def __sub__(self, other: TransseriesExpansion | sp.Expr) -> TransseriesExpansion:
        return self + (-other if isinstance(other, TransseriesExpansion) else -sp.sympify(other))

    def __mul__(self, other: TransseriesExpansion | sp.Expr) -> TransseriesExpansion:
        """Multiply finite transseries prefixes while propagating remainder semantics."""
        if isinstance(other, TransseriesExpansion):
            self._check_compatible(other)
            products = []
            for left in self.terms:
                if other.center != 0:
                    products.append(
                        TransseriesTerm(sp.simplify(left.coefficient * other.center), left.monomial)
                    )
            for right in other.terms:
                if self.center != 0:
                    products.append(
                        TransseriesTerm(
                            sp.simplify(right.coefficient * self.center), right.monomial
                        )
                    )
            from .instrumentation import record_symbolic_event

            record_symbolic_event("term_products", len(self.terms) * len(other.terms))
            for left in self.terms:
                for right in other.terms:
                    products.append(left * right)
            from .remainder_theorems import certify_product_remainder

            certificate = certify_product_remainder(
                self.truncate(), other.truncate(), self.remainder, other.remainder
            )
            metadata = _merge_metadata(self.metadata, other.metadata)
            metadata.setdefault("remainder_certificates", []).append(certificate)
            return TransseriesExpansion.from_terms(
                self.variable,
                self.point,
                products,
                center=sp.simplify(self.center * other.center),
                complete=certificate.conclusion.is_exact,
                metadata=metadata,
                remainder=certificate.conclusion,
            )
        scalar = sp.sympify(other)
        if self.variable in scalar.free_symbols:
            return NotImplemented
        from .remainder_theorems import certify_scaling_remainder

        certificate = certify_scaling_remainder(scalar, self.remainder)
        metadata = dict(self.metadata)
        metadata.setdefault("remainder_certificates", []).append(certificate)
        return TransseriesExpansion.from_terms(
            self.variable,
            self.point,
            (
                TransseriesTerm(sp.simplify(scalar * term.coefficient), term.monomial)
                for term in self.terms
            ),
            center=sp.simplify(scalar * self.center),
            complete=certificate.conclusion.is_exact,
            metadata=metadata,
            remainder=certificate.conclusion,
        )

    __rmul__ = __mul__

    def _dominant_prefix_scale(self) -> sp.Expr | None:
        normalized = self.normalized()
        if normalized.center != 0:
            return sp.S.One
        if normalized.leading_term is not None:
            return normalized.leading_term.monomial_expression
        return None

    def _product_remainder(self, other: TransseriesExpansion) -> AsymptoticRemainder:
        """Return the certified binary product remainder."""

        from .remainder_theorems import certify_product_remainder

        return certify_product_remainder(
            self.truncate(), other.truncate(), self.remainder, other.remainder
        ).conclusion

    def differentiate(self, order: int = 1) -> TransseriesExpansion:
        """Differentiate the represented finite prefix and recanonicalize it."""

        if order < 0:
            raise ValueError("order must be nonnegative")
        expr = sp.diff(self.truncate(), self.variable, order)
        result = transseries_from_expression(
            expr, self.variable, point=self.point, complete=self.complete
        )
        from .remainder_theorems import certify_differentiation_remainder

        certificate = certify_differentiation_remainder(self.remainder, order)
        remainder = certificate.conclusion
        metadata = dict(self.metadata)
        metadata.setdefault("remainder_certificates", []).append(certificate)
        return TransseriesExpansion.from_terms(
            result.variable,
            result.point,
            result.terms,
            center=result.center,
            complete=remainder.is_exact,
            metadata=metadata,
            remainder=remainder,
        )

    def integrate(self, *, constant: sp.Expr = 0, terms: int | None = None) -> TransseriesExpansion:
        """Scale-aware asymptotic integration of the finite prefix."""

        from .general_ops import asymptotic_integrate

        return asymptotic_integrate(
            self, constant=constant, terms=6 if terms is None else int(terms)
        )

    def exp(self, *, terms: int = 6) -> TransseriesExpansion:
        """Exponentiate a finite transseries, raising the LE height when needed.

        For an infinitesimal tail about a finite center this uses the ordinary
        Taylor algebra.  Otherwise the represented finite prefix becomes the
        exponent of one recursive LE monomial, which is the exact finite-height
        transmonomial corresponding to the available prefix.
        """

        if terms < 1:
            raise ValueError("terms must be positive")
        normalized = self.normalized()
        tail = TransseriesExpansion.from_terms(self.variable, self.point, normalized.terms)
        if normalized.center != 0 and normalized.terms:
            unit = TransseriesTerm(1, sp.S.One).canonical(self.variable, point=self.point)
            leading = tail.leading_term
            if (
                leading is not None
                and compare_monomials(
                    leading.monomial, unit.monomial, self.variable, point=self.point
                )
                is GrowthComparison.SMALLER
            ):
                return normalized._compose_analytic_taylor(sp.exp, terms=terms)
        if not normalized.terms:
            return TransseriesExpansion.from_terms(
                self.variable,
                self.point,
                (),
                center=sp.exp(normalized.center),
                complete=normalized.complete,
            )
        prefix = normalized.truncate()
        monomial = RecursiveLogExpMonomial(self.variable, self.point, prefix, ())
        from .remainder_theorems import certify_unary_composition_remainder

        z = sp.Dummy("z")
        certificate = certify_unary_composition_remainder(
            sp.exp(z),
            z,
            prefix,
            normalized.remainder,
            output_variable=self.variable,
            point=self.point,
        )
        metadata = {**normalized.metadata, "logexp_height": monomial.height}
        metadata.setdefault("remainder_certificates", []).append(certificate)
        return TransseriesExpansion.from_terms(
            self.variable,
            self.point,
            (TransseriesTerm(1, monomial),),
            complete=certificate.conclusion.is_exact,
            metadata=metadata,
            remainder=certificate.conclusion,
        )

    def log(self, *, terms: int = 6) -> TransseriesExpansion:
        """Take the logarithm using recursive LE factorization.

        If a nonzero finite center dominates, use Taylor expansion.  Otherwise
        factor the dominant term ``L`` and apply
        ``log(S)=log(L)+log(1+(S-L)/L)``.
        """

        if terms < 1:
            raise ValueError("terms must be positive")
        normalized = self.normalized()
        if normalized.center != 0:
            if not normalized.terms:
                return TransseriesExpansion.from_terms(
                    self.variable,
                    self.point,
                    (),
                    center=sp.log(normalized.center),
                    complete=normalized.complete,
                )
            unit = TransseriesTerm(1, sp.S.One).canonical(self.variable, point=self.point)
            leading = normalized.leading_term
            if (
                leading is not None
                and compare_monomials(
                    leading.monomial, unit.monomial, self.variable, point=self.point
                )
                is GrowthComparison.SMALLER
            ):
                return normalized._compose_analytic_taylor(sp.log, terms=terms)

        if normalized.center == 0 and normalized.leading_term is not None:
            leading = normalized.leading_term.canonical(self.variable, point=self.point)
            if isinstance(leading.monomial, RecursiveLogExpMonomial):
                log_monomial = leading.monomial.logarithm_expression
            else:
                log_monomial = sp.log(leading.monomial_expression)
            leading_log = sp.log(leading.coefficient) + log_monomial
            rest = TransseriesExpansion.from_terms(
                self.variable, self.point, normalized.terms[1:], complete=True
            )
            result = transseries_from_expression(
                leading_log, self.variable, point=self.point, complete=True
            )
            if rest.terms:
                u = rest.divide_by_term(leading)
                one_plus_u = TransseriesExpansion.from_terms(
                    self.variable, self.point, u.terms, center=1, complete=True
                )
                result = result + one_plus_u._compose_analytic_taylor(sp.log, terms=terms)
            result = result.normalized()
            from .remainder_theorems import certify_unary_composition_remainder

            z = sp.Dummy("z")
            certificate = certify_unary_composition_remainder(
                sp.log(z),
                z,
                normalized.truncate(),
                normalized.remainder,
                output_variable=self.variable,
                point=self.point,
            )
            metadata = dict(result.metadata)
            metadata.setdefault("remainder_certificates", []).append(certificate)
            # The factorization/Taylor truncation may itself have a bound. Add
            # the propagated input error rather than replacing that bound.
            remainder = result.remainder.add(certificate.conclusion)
            return TransseriesExpansion.from_terms(
                result.variable,
                result.point,
                result.terms,
                center=result.center,
                complete=remainder.is_exact,
                metadata=metadata,
                remainder=remainder,
            )

        # Mixed/undecidable finite prefix: preserve exact recursive logarithm
        # as a finite-height expression and parse its additive decomposition.
        result = transseries_from_expression(
            sp.log(normalized.truncate()), self.variable, point=self.point, complete=False
        )
        from .remainder_theorems import certify_unary_composition_remainder

        z = sp.Dummy("z")
        certificate = certify_unary_composition_remainder(
            sp.log(z),
            z,
            normalized.truncate(),
            normalized.remainder,
            output_variable=self.variable,
            point=self.point,
        )
        metadata = dict(result.metadata)
        metadata.setdefault("remainder_certificates", []).append(certificate)
        return TransseriesExpansion.from_terms(
            result.variable,
            result.point,
            result.terms,
            center=result.center,
            complete=certificate.conclusion.is_exact,
            metadata=metadata,
            remainder=certificate.conclusion,
        )

    def constant_power(self, exponent: sp.Expr, *, terms: int = 6) -> TransseriesExpansion:
        """Raise a finite transseries to a variable-independent power."""

        exponent = sp.sympify(exponent)
        if self.variable in exponent.free_symbols:
            raise ValueError("exponent must be independent of the asymptotic variable")
        return (exponent * self.log()).exp(terms=terms)

    def inverse_asymptotic(
        self,
        inverse_variable: sp.Symbol | None = None,
        *,
        terms: int = 6,
        branch: int | None = 0,
        assumptions: sp.Expr | bool = sp.S.true,
        allow_unknown_properties: bool = False,
    ):
        """Asymptotically revert the represented finite prefix as a function."""

        from .reversion import inverse_asymptotic

        return inverse_asymptotic(
            self.truncate(),
            self.variable,
            inverse_variable,
            point=self.point,
            terms=terms,
            branch=branch,
            assumptions=assumptions,
            allow_unknown_properties=allow_unknown_properties,
        )

    def reciprocal(self, terms: int = 6) -> TransseriesExpansion:
        """Return a finite geometric reciprocal expansion about the leading term."""

        if terms < 1:
            raise ValueError("terms must be positive")
        normalized = self.normalized()
        if normalized.center != 0:
            leading = TransseriesTerm(normalized.center, sp.S.One).canonical(
                self.variable, point=self.point
            )
            rest = TransseriesExpansion.from_terms(self.variable, self.point, normalized.terms)
        elif normalized.leading_term is not None:
            leading = normalized.leading_term.canonical(self.variable, point=self.point)
            rest = TransseriesExpansion.from_terms(self.variable, self.point, normalized.terms[1:])
        else:
            raise ZeroDivisionError("cannot invert a zero transseries")

        unit_tail = (
            rest.divide_by_term(leading)
            if rest.terms or rest.center != 0
            else TransseriesExpansion.from_terms(self.variable, self.point, ())
        )
        # 1/(L(1+u)) = L^-1 sum (-u)^k.  Truncate by powers of u;
        # normalization then orders and combines the resulting monomials.
        one = TransseriesExpansion.from_terms(
            self.variable, self.point, (), center=1, complete=True
        )
        total = one
        power = one
        for k in range(1, terms):
            power = power * unit_tail
            total = total + ((-1) ** k) * power
        inverse_leading = TransseriesTerm(
            1, AsymptoticMonomial(leading.monomial.ramification) / leading.monomial
        )
        inverse_leading = TransseriesTerm(
            sp.simplify(1 / leading.coefficient), inverse_leading.monomial
        )
        result = (
            total * TransseriesExpansion.from_terms(self.variable, self.point, (inverse_leading,))
        ).normalized()

        # Two independent errors contribute: truncating the finite geometric
        # reciprocal of the represented prefix, and the pre-existing input
        # remainder propagated through reciprocal stability.
        from .remainder_theorems import (
            _classify_exact_error,
            certify_finite_sum_remainder,
            certify_reciprocal_remainder,
        )

        finite_prefix = normalized.truncate()
        geometric_error = sp.simplify(1 / finite_prefix - result.truncate())
        u = unit_tail.truncate()
        leading_expr = leading.expression
        geometric_scale = sp.simplify(u**terms / leading_expr)
        geometric_remainder = _classify_exact_error(
            geometric_error,
            geometric_scale,
            self.variable,
            self.point,
            source="finite geometric reciprocal truncation",
        )
        input_certificate = certify_reciprocal_remainder(finite_prefix, normalized.remainder)
        combined_certificate = certify_finite_sum_remainder(
            (geometric_remainder, input_certificate.conclusion)
        )
        metadata = dict(result.metadata)
        metadata.setdefault("remainder_certificates", []).extend(
            (input_certificate, combined_certificate)
        )
        return TransseriesExpansion.from_terms(
            result.variable,
            result.point,
            result.terms,
            center=result.center,
            complete=combined_certificate.conclusion.is_exact,
            metadata=metadata,
            remainder=combined_certificate.conclusion,
        )

    def __truediv__(self, other: TransseriesExpansion | sp.Expr) -> TransseriesExpansion:
        """Divide by another expansion using its certified reciprocal."""

        if isinstance(other, TransseriesExpansion):
            self._check_compatible(other)
            return self * other.reciprocal()
        scalar = sp.sympify(other)
        if self.variable in scalar.free_symbols:
            return NotImplemented
        if scalar == 0:
            raise ZeroDivisionError("division by zero")
        return self * sp.simplify(1 / scalar)

    def _compose_analytic_taylor(
        self,
        outer: sp.Expr | sp.FunctionClass,
        *,
        argument: sp.Symbol | None = None,
        terms: int = 6,
        assumptions: sp.Expr | bool = sp.S.true,
        allow_unknown_properties: bool = False,
    ) -> TransseriesExpansion:
        """Taylor-compose an outer analytic function at this finite center."""

        if terms < 1:
            raise ValueError("terms must be positive")
        z = argument or sp.Dummy("z")
        outer_expr = (
            outer(z) if callable(outer) and not isinstance(outer, sp.Expr) else sp.sympify(outer)
        )
        if z not in outer_expr.free_symbols:
            symbols = tuple(outer_expr.free_symbols - {self.variable})
            if argument is None and len(symbols) == 1:
                z = symbols[0]
            elif argument is None:
                raise ValueError("outer expression requires an explicit argument symbol")
        from .function_properties import analytic_at_decision, require_decision

        decision = analytic_at_decision(outer_expr, z, self.center, assumptions=assumptions)
        require_decision(
            decision,
            operation="Taylor composition",
            allow_unknown=allow_unknown_properties,
        )
        epsilon = sp.Dummy("eps")
        try:
            taylor = sp.series(
                outer_expr.xreplace({z: self.center + epsilon}), epsilon, 0, terms
            ).removeO()
        except SYMBOLIC_ERRORS as exc:
            raise NotImplementedError(
                "outer function is not Taylor-expandable at the transseries center"
            ) from exc
        tail = TransseriesExpansion.from_terms(self.variable, self.point, self.terms)
        result = TransseriesExpansion.from_terms(self.variable, self.point, (), center=0)
        poly = sp.Poly(sp.expand(taylor), epsilon)
        for (degree,), coefficient in poly.terms():
            power = TransseriesExpansion.from_terms(
                self.variable, self.point, (), center=1, complete=True
            )
            for _ in range(degree):
                power = power * tail
            result = result + coefficient * power
        result = result.normalized()
        metadata = dict(result.metadata)
        metadata.setdefault("property_decisions", []).append(decision)

        # Taylor's theorem supplies a real remainder statement; do not confuse
        # the finite Taylor polynomial with a complete representation. For an
        # analytic outer function the perturbation caused by an already-known
        # input O/o remainder is bounded by a locally bounded derivative.
        input_remainder = self.remainder
        if input_remainder is None:
            raise RuntimeError("transseries is missing its input remainder")
        from .remainder_theorems import certify_unary_composition_remainder

        certificate = certify_unary_composition_remainder(
            outer_expr,
            z,
            self.truncate(),
            input_remainder,
            output_variable=self.variable,
            point=self.point,
        )
        output_remainder = certificate.conclusion
        metadata.setdefault("remainder_certificates", []).append(certificate)
        derivative_n = sp.simplify(sp.diff(outer_expr, z, terms))
        if derivative_n != 0 and self.terms:
            leading_tail = (
                TransseriesExpansion.from_terms(self.variable, self.point, self.terms)
                .normalized()
                .leading_term
            )
            if leading_tail is not None:
                taylor_remainder = AsymptoticRemainder.big_o(
                    leading_tail.monomial_expression**terms,
                    self.variable,
                    self.point,
                    source=f"analytic Taylor remainder of order {terms}",
                )
                output_remainder = output_remainder.add(taylor_remainder)
        return TransseriesExpansion.from_terms(
            result.variable,
            result.point,
            result.terms,
            center=result.center,
            complete=output_remainder.is_exact,
            metadata=metadata,
            remainder=output_remainder,
        )

    def compose(
        self,
        outer: sp.Expr | sp.FunctionClass | TransseriesExpansion,
        *,
        argument: sp.Symbol | None = None,
        terms: int = 6,
        assumptions: sp.Expr | bool = sp.S.true,
        allow_unknown_properties: bool = False,
    ) -> TransseriesExpansion:
        """Compose an outer finite LE expression/transseries with this series."""

        from .general_ops import compose_transseries

        return compose_transseries(
            outer,
            self,
            argument=argument,
            terms=terms,
            assumptions=assumptions,
            allow_unknown_properties=allow_unknown_properties,
        )

    def divide_by_term(self, divisor: TransseriesTerm) -> TransseriesExpansion:
        """Divide a finite prefix by a nonzero monomial term exactly."""

        if self.center != 0:
            # Treat the center as the unit monomial.  Canonicalizing it here
            # avoids silently dropping a valid term in the quotient.
            unit = TransseriesTerm(self.center, sp.S.One).canonical(self.variable, point=self.point)
            source = (unit,) + self.terms
        else:
            source = self.terms
        divisor = divisor.canonical(self.variable, point=self.point)
        return TransseriesExpansion.from_terms(
            self.variable,
            self.point,
            (term.canonical(self.variable, point=self.point) / divisor for term in source),
            complete=self.complete,
            remainder=self.remainder.scale_by(1 / divisor.expression),
        )


def _split_leading_term(leading: sp.Expr, variable: sp.Symbol) -> tuple[sp.Expr, sp.Expr]:
    """Separate a variable-independent scalar from an asymptotic monomial."""

    leading = analytic_powsimp(sp.sympify(leading))
    scalar, monomial = leading.as_independent(variable, as_Add=False)
    scalar = sp.simplify(scalar)
    monomial = formal_powsimp(sp.simplify(monomial))
    if scalar == 0:
        return sp.S.Zero, sp.S.One
    if monomial == 1 and variable in scalar.free_symbols:
        return sp.S.One, leading
    return scalar, monomial


def _canonical_or_none(
    monomial: sp.Expr,
    variable: sp.Symbol,
    point: sp.Expr,
) -> AsymptoticMonomial | None:
    try:
        coefficient, canonical = canonical_asymptotic_monomial(monomial, variable, point=point)
    except ValueError:
        return None
    if sp.simplify(coefficient - 1) != 0:
        # Canonical monomials exclude scalar factors. If a scalar remains, keep
        # the separately stored leading coefficient rather than misclassifying
        # the monomial.
        return None
    return canonical


def transseries_valuation(
    expr: sp.Expr,
    variable: sp.Symbol,
    *,
    point: sp.Expr = 0,
    context: AsymptoticContext | None = None,
) -> TransseriesValuation | None:
    """Return a generalized leading-monomial valuation."""

    expr = sp.sympify(expr)
    ctx = context_for(variable, point, context)
    if ctx.is_zero(expr) is True:
        return None

    # ``Expr.as_leading_term(variable)`` is intrinsically a zero-germ
    # operation.  At either infinite endpoint, localize first so exponential
    # and logarithmic scales are not accidentally valued as constants.
    if point in (sp.oo, -sp.oo):
        local = sp.Dummy("_valuation_h", positive=True)
        sign = sp.S.One if point is sp.oo else -sp.S.One
        localized = analytic_powsimp(expr.subs(variable, sign / local))
        local_value = transseries_valuation(localized, local, point=0)
        if local_value is None:
            return None
        inverse_substitution = {local: sign / variable}
        leading_term = analytic_powsimp(local_value.leading_term.subs(inverse_substitution))
        monomial = formal_powsimp(local_value.monomial.subs(inverse_substitution))
        leading_coefficient = analytic_powsimp(
            local_value.leading_coefficient.subs(inverse_substitution)
        )
        return TransseriesValuation(
            expression=expr,
            leading_term=leading_term,
            leading_coefficient=leading_coefficient,
            monomial=monomial,
            point=point,
            canonical_monomial=_canonical_or_none(monomial, variable, point),
        )

    # Pull out exact common factors before asking an additive series engine to
    # discover them.  This is crucial for coefficients such as
    # ``-exp(2/h) - h*exp(2/h)``: the common exponential is the Hardy
    # monomial, while ``-(1+h)`` contributes only its limiting coefficient.
    try:
        factored = sp.factor_terms(expr)
    except (TypeError, ValueError, NotImplementedError):
        factored = expr
    if factored != expr and isinstance(factored, sp.Mul):
        factored_value = transseries_valuation(factored, variable, point=point, context=ctx)
        if factored_value is not None:
            return TransseriesValuation(
                expression=expr,
                leading_term=factored_value.leading_term,
                leading_coefficient=factored_value.leading_coefficient,
                monomial=factored_value.monomial,
                point=point,
                canonical_monomial=factored_value.canonical_monomial,
            )

    # Multiplicative valuation is exact in the monomial group and avoids a
    # common failure mode of generic series expansion: factors such as
    # ``(1+h)*exp(1/h)`` should have leading monomial ``exp(1/h)``, not the
    # entire product treated as an opaque monomial.
    if isinstance(expr, sp.Mul):
        leading_coefficient = sp.S.One
        monomial = sp.S.One
        for factor in expr.args:
            if variable not in factor.free_symbols:
                leading_coefficient *= factor
                continue
            factor_value = transseries_valuation(factor, variable, point=point, context=ctx)
            if factor_value is None:
                return None
            leading_coefficient *= factor_value.leading_coefficient
            monomial *= factor_value.monomial
        leading_coefficient = analytic_powsimp(leading_coefficient)
        monomial = formal_powsimp(monomial)
        leading_term = mixed_powsimp(leading_coefficient, monomial)
        return TransseriesValuation(
            expression=expr,
            leading_term=leading_term,
            leading_coefficient=leading_coefficient,
            monomial=monomial,
            point=point,
            canonical_monomial=_canonical_or_none(monomial, variable, point),
        )

    structural: sp.Expr | None
    try:
        structural = analytic_powsimp(expr.as_leading_term(variable))
    except SYMBOLIC_ERRORS:
        structural = None

    if structural is not None and (structural != expr or not expr.is_Add):
        coefficient, monomial = _split_leading_term(structural, variable)
        if ctx.is_zero(coefficient) is not True:
            return TransseriesValuation(
                expression=expr,
                leading_term=structural,
                leading_coefficient=coefficient,
                monomial=monomial,
                point=point,
                canonical_monomial=_canonical_or_none(monomial, variable, point),
            )

    leading: sp.Expr | None = None
    try:
        from .multiseries import multiseries

        expansion = multiseries(expr, variable, point=point, terms=1)
        leading = analytic_powsimp(expansion.leading_term(recursive=True))
    except SYMBOLIC_ERRORS:
        leading = None

    if leading is None or ctx.is_zero(leading) is True:
        if structural is None:
            return None
        leading = structural
    else:
        try:
            refined = analytic_powsimp(leading.as_leading_term(variable))
        except SYMBOLIC_ERRORS:
            refined = None
        if refined is not None and refined != leading:
            relation, _ = ctx.compare_growth(refined, leading)
            if relation is GrowthComparison.SAME_ORDER:
                leading = refined

    if structural is not None and structural != leading:
        relation, _ = ctx.compare_growth(structural, leading)
        if relation is GrowthComparison.SAME_ORDER and sp.count_ops(structural) < sp.count_ops(
            leading
        ):
            leading = structural

    coefficient, monomial = _split_leading_term(leading, variable)
    if ctx.is_zero(coefficient) is True:
        return None
    return TransseriesValuation(
        expression=expr,
        leading_term=leading,
        leading_coefficient=coefficient,
        monomial=monomial,
        point=point,
        canonical_monomial=_canonical_or_none(monomial, variable, point),
    )


def transseries_from_expression(
    expr: sp.Expr,
    variable: sp.Symbol,
    *,
    point: sp.Expr = 0,
    complete: bool = False,
    metadata: dict[str, object] | None = None,
    remainder: AsymptoticRemainder | None = None,
    sector: ComplexSector | None = None,
    branch: ComplexBranchMetadata | None = None,
) -> TransseriesExpansion:
    """Convert a finite additive exp/power/log expression to native terms.

    Constants are stored in ``center``.  Every variable-dependent summand must
    belong to the canonical multiplicative monomial group; unsupported terms
    are rejected rather than silently wrapped as opaque expressions.
    """

    metadata = merge_complex_germ_metadata(metadata, sector=sector, branch=branch)
    expr = analytic_powsimp(sp.expand(sp.sympify(expr)))
    center = sp.S.Zero
    terms = []
    parameter = sp.Dummy("t", positive=True)
    for summand in sp.Add.make_args(expr):
        if variable not in summand.free_symbols:
            center += summand
            continue
        try:
            coefficient, monomial = canonical_asymptotic_monomial(
                summand, variable, point=point, parameter=parameter
            )
        except ValueError:
            coefficient, monomial = canonical_recursive_logexp_monomial(
                summand, variable, point=point
            )
        terms.append(TransseriesTerm(coefficient, monomial))
    return TransseriesExpansion.from_terms(
        variable,
        point,
        terms,
        center=sp.simplify(center),
        complete=complete,
        metadata=metadata,
        remainder=remainder,
    )


def compare_monomials(
    left: sp.Expr | AsymptoticMonomial,
    right: sp.Expr | AsymptoticMonomial,
    variable: sp.Symbol,
    *,
    point: sp.Expr = 0,
    context: AsymptoticContext | None = None,
) -> GrowthComparison:
    """Compare generalized monomials, preferring the structural hierarchy."""

    if isinstance(left, RecursiveLogExpMonomial) and isinstance(right, RecursiveLogExpMonomial):
        return left.compare(right)

    try:
        if isinstance(left, AsymptoticMonomial):
            left_m = left
        else:
            _, left_m = canonical_asymptotic_monomial(left, variable, point=point)
        if isinstance(right, AsymptoticMonomial):
            right_m = right
        else:
            _, right_m = canonical_asymptotic_monomial(right, variable, point=point)
        result = compare_asymptotic_monomials(left_m, right_m)
        if result is not GrowthComparison.UNKNOWN:
            return result
    except ValueError:
        pass

    ctx = context_for(variable, point, context)
    left_expr = (
        left.expression if isinstance(left, (AsymptoticMonomial, RecursiveLogExpMonomial)) else left
    )
    right_expr = (
        right.expression
        if isinstance(right, (AsymptoticMonomial, RecursiveLogExpMonomial))
        else right
    )
    relation, _ = ctx.compare_growth(left_expr, right_expr)
    if relation is not GrowthComparison.UNKNOWN:
        return relation
    from .exp_log_scale import compare_log_exp_scales

    return compare_log_exp_scales(left_expr, right_expr, variable, point=point)


def ordered_transseries_terms(
    terms: tuple[TransseriesTerm, ...] | list[TransseriesTerm],
    variable: sp.Symbol,
    *,
    point: sp.Expr = 0,
    context: AsymptoticContext | None = None,
) -> tuple[TransseriesTerm, ...]:
    """Order terms from asymptotically largest to smallest when decidable."""

    ctx = context_for(variable, point, context)

    def cmp(a: TransseriesTerm, b: TransseriesTerm) -> int:
        relation = compare_monomials(a.monomial, b.monomial, variable, point=point, context=ctx)
        if relation is GrowthComparison.LARGER:
            return -1
        if relation is GrowthComparison.SMALLER:
            return 1
        akey = sp.default_sort_key(a.expression)
        bkey = sp.default_sort_key(b.expression)
        return (akey > bkey) - (akey < bkey)

    return tuple(sorted(terms, key=cmp_to_key(cmp)))
