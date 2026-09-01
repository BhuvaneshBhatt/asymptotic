from __future__ import annotations

import heapq
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import count

import sympy as sp

from ._ordering import exponent_sort_key as _exp_key
from ._power_simplify import analytic_powsimp
from ._symbolic_errors import SYMBOLIC_ERRORS
from .context import AsymptoticContext, GrowthComparison, context_for
from .obligations import (
    AsymptoticKnowledge,
    AsymptoticObligation,
    CoefficientExpansionObligation,
    ComparabilityFactorObligation,
    ExponentialScaleObligation,
    GrowthComparisonObligation,
    LogarithmicScaleObligation,
    ZeroTestObligation,
)
from .scale import AsymptoticScale, ScaleDiscovery
from .sparse import LazySparseSeries
from .tower import ExpLogTower


@dataclass(frozen=True)
class MultiseriesTerm:
    exponent: sp.Expr
    coefficient: sp.Expr

    def as_expr(self, scale_element: sp.Expr) -> sp.Expr:
        return self.coefficient * scale_element**self.exponent


def _combine_terms(terms: Iterable[MultiseriesTerm], n: int | None = None) -> list[MultiseriesTerm]:
    grouped = {}
    for term in terms:
        exponent = sp.simplify(term.exponent)
        grouped[exponent] = grouped.get(exponent, sp.S.Zero) + term.coefficient
    result = []
    for exponent, coefficient in grouped.items():
        coefficient = sp.simplify(coefficient)
        if coefficient != 0 and coefficient.is_zero is not True:
            result.append(MultiseriesTerm(exponent, coefficient))
    result.sort(key=lambda term: _exp_key(term.exponent))
    return result if n is None else result[:n]


class Multiseries:
    """Demand-driven recursive multiseries.

    At level ``k`` the expression is expanded in ``scale[k]``; coefficients
    remain exact expressions and can themselves be expanded on demand in lower
    scale elements.  Elementary analytic composition is handled by the package's
    own heap frontier; SymPy's univariate series engine is retained as a fallback
    for unsupported formal expressions.
    """

    def __init__(
        self,
        expr: sp.Expr,
        scale: AsymptoticScale,
        *,
        level: int | None = None,
        context: AsymptoticContext | None = None,
        default_terms: int = 6,
        knowledge: AsymptoticKnowledge | None = None,
        allow_series_fallback: bool = True,
    ) -> None:
        self.expr = sp.sympify(expr)
        self.scale = scale
        self.level = len(scale) - 1 if level is None else level
        self.context = context_for(scale.variable, scale.point, context)
        self.default_terms = default_terms
        self.knowledge = knowledge or AsymptoticKnowledge()
        self.allow_series_fallback = allow_series_fallback
        self._terms: list[MultiseriesTerm] = []
        self._formal_cache: tuple[sp.Expr, tuple[sp.Symbol, ...]] | None = None
        self._dynamic_extensions = 0
        self.max_dynamic_extensions = 12
        self.obligation_history: list[AsymptoticObligation] = []
        self.tower = ExpLogTower.from_expr(sp.Tuple(self.expr, *self.scale.exprs), scale.variable)

    @property
    def scale_element(self) -> sp.Expr:
        if self.level < 0:
            return sp.S.One
        return self.scale.elements[self.level].expr

    def _formal(self) -> tuple[sp.Expr, tuple[sp.Symbol, ...]]:
        if self._formal_cache is None:
            self._formal_cache = self.scale.formalize(self.expr)
        return self._formal_cache

    @staticmethod
    def _extract_terms(series_expr: sp.Expr, z: sp.Symbol) -> list[MultiseriesTerm]:
        expanded = sp.expand(
            series_expr.removeO() if hasattr(series_expr, "removeO") else series_expr
        )
        raw = {}
        for term in sp.Add.make_args(expanded):
            powers = term.as_powers_dict()
            exponent = sp.sympify(powers.get(z, 0))
            coefficient = sp.simplify(term / (z**exponent))
            raw[exponent] = sp.simplify(raw.get(exponent, 0) + coefficient)
        return _combine_terms((MultiseriesTerm(e, c) for e, c in raw.items()))

    def _extend_exponential_scale(
        self,
        request: ExponentialScaleObligation,
        syms: tuple[sp.Symbol, ...],
    ) -> bool:
        """Resolve an exponential-scale obligation by adding a scale element.

        If the active formal variable is ``z`` and an exponent contains, say,
        ``z**-1 + z``, the negative-power part is deformalized back to ``x``.
        Its eventual sign determines the vanishing representative: ``exp(-H)``
        for ``H -> +oo`` and ``exp(H)`` for ``H -> -oo``.  The scale is then
        re-sorted by comparability class and the expression is re-formalized.
        """

        if self._dynamic_extensions >= self.max_dynamic_extensions:
            return False
        reverse = {sym: elem.expr for sym, elem in zip(syms, self.scale.elements)}
        divergent = analytic_powsimp(sp.simplify(request.divergent_part.xreplace(reverse)))
        sign = self.context.eventual_sign(divergent)
        if sign not in (-1, 1):
            limit = self.context.limit(divergent)
            if limit is sp.oo:
                sign = 1
            elif limit is -sp.oo:
                sign = -1
            else:
                return False

        candidate = sp.exp(-sign * divergent)
        try:
            new_scale, changed = self.scale.with_element(candidate, self.context)
        except SYMBOLIC_ERRORS:
            return False
        if not changed:
            return False

        old_scale = self.scale
        old_level = self.level
        old_active = old_scale.elements[old_level].expr if old_level >= 0 else None
        self.scale = new_scale
        candidate_index = next(
            (
                i
                for i, item in enumerate(new_scale.elements)
                if sp.simplify(item.expr - candidate) == 0
            ),
            len(new_scale) - 1,
        )
        if old_level == len(old_scale) - 1:
            self.level = len(new_scale) - 1
        elif old_active is not None:
            active_index = next(
                (
                    i
                    for i, item in enumerate(new_scale.elements)
                    if sp.simplify(item.expr - old_active) == 0
                ),
                old_level,
            )
            self.level = max(active_index, candidate_index)

        self._formal_cache = None
        self._terms = []
        self._dynamic_extensions += 1
        self.tower = ExpLogTower.from_expr(
            sp.Tuple(self.expr, *self.scale.exprs), self.scale.variable
        )
        return True

    def _extend_logarithmic_scale(
        self,
        request: LogarithmicScaleObligation,
    ) -> bool:
        """Insert the lower scale needed to represent ``log(active_scale)``.

        For a positive scale element ``t -> 0``, ``log(t) -> -oo``.  The
        canonical missing lower representative is therefore the vanishing
        reciprocal of its magnitude, e.g.

            t = x**-1       ->  1/log(x)
            t = exp(-x)     ->  1/x
            t = exp(-exp(x))->  exp(-x).

        The exact logarithm is retained; the sign is used only to choose a
        positive vanishing representative when it is decidable.
        """

        if self._dynamic_extensions >= self.max_dynamic_extensions or self.level < 0:
            return False
        active = self.scale_element
        log_active = sp.log(active)
        sign = self.context.eventual_sign(log_active)
        if sign not in (-1, 1):
            limit = self.context.limit(log_active)
            if limit is sp.oo:
                sign = 1
            elif limit is -sp.oo:
                sign = -1
            else:
                return False
        candidate = analytic_powsimp(sp.simplify(sign / log_active))
        try:
            new_scale, changed = self.scale.with_element(candidate, self.context)
        except SYMBOLIC_ERRORS:
            return False
        if not changed:
            return False

        old_active = active
        self.scale = new_scale
        self.level = next(
            (
                i
                for i, item in enumerate(new_scale.elements)
                if sp.simplify(item.expr - old_active) == 0
            ),
            self.level,
        )
        self._formal_cache = None
        self._terms = []
        self._dynamic_extensions += 1
        self.tower = ExpLogTower.from_expr(
            sp.Tuple(self.expr, *self.scale.exprs), self.scale.variable
        )
        return True

    def _deformalize_expr(self, expr: sp.Expr, syms: tuple[sp.Symbol, ...]) -> sp.Expr:
        reverse = {sym: elem.expr for sym, elem in zip(syms, self.scale.elements)}
        return analytic_powsimp(sp.simplify(sp.sympify(expr).xreplace(reverse)))

    def _resolve_zero_test(
        self, obligation: ZeroTestObligation, syms: tuple[sp.Symbol, ...]
    ) -> bool:
        expression = self._deformalize_expr(obligation.expression, syms)
        result = self.context.is_zero(expression)
        if result is None:
            return False
        self.knowledge.set(obligation, result)
        return True

    def _resolve_growth_comparison(
        self, obligation: GrowthComparisonObligation, syms: tuple[sp.Symbol, ...]
    ) -> bool:
        left = self._deformalize_expr(obligation.left, syms)
        right = self._deformalize_expr(obligation.right, syms)
        if obligation.logarithmic:
            result = self.context.compare_log_growth(left, right)
        else:
            result = self.context.compare_growth(left, right)
        if result[0] is GrowthComparison.UNKNOWN:
            return False
        self.knowledge.set(obligation, result)
        return True

    def _resolve_coefficient_expansion(
        self, obligation: CoefficientExpansionObligation, syms: tuple[sp.Symbol, ...]
    ) -> bool:
        expression = self._deformalize_expr(obligation.expression, syms)
        level = min(obligation.lower_level, self.level - 1)
        if level < 0:
            self.knowledge.set(obligation, expression)
            return True
        child = Multiseries(
            expression,
            self.scale,
            level=level,
            context=self.context,
            default_terms=obligation.terms,
            knowledge=self.knowledge,
            allow_series_fallback=self.allow_series_fallback,
        )
        try:
            value = child.truncate(obligation.terms)
        except SYMBOLIC_ERRORS:
            return False
        self.obligation_history.extend(child.obligation_history)
        self.knowledge.set(obligation, value)
        return True

    def _resolve_comparability_factor(
        self, obligation: ComparabilityFactorObligation, syms: tuple[sp.Symbol, ...]
    ) -> bool:
        expression = self._deformalize_expr(obligation.expression, syms)
        for formal_candidate in obligation.candidates:
            candidate = self._deformalize_expr(formal_candidate, syms)
            relation, ratio = self.context.compare_growth(expression, candidate)
            if relation is GrowthComparison.SAME_ORDER and ratio is not None:
                residual = sp.simplify(expression - ratio * candidate)
                self.knowledge.set(obligation, (candidate, ratio, residual))
                return True
        self.knowledge.set(obligation, None)
        return True

    def _resolve_asymptotic_obligation(
        self,
        obligation: AsymptoticObligation,
        syms: tuple[sp.Symbol, ...],
    ) -> bool:
        """Resolve a sparse request without assuming it mutates the scale.

        Resolvers either (a) add a fact to shared asymptotic knowledge and retry
        the same formal expression, or (b) update the scale/tower and then
        re-formalize. This is the central suspension/resumption boundary.
        """

        self.obligation_history.append(obligation)
        if isinstance(obligation, ZeroTestObligation):
            return self._resolve_zero_test(obligation, syms)
        if isinstance(obligation, GrowthComparisonObligation):
            return self._resolve_growth_comparison(obligation, syms)
        if isinstance(obligation, CoefficientExpansionObligation):
            return self._resolve_coefficient_expansion(obligation, syms)
        if isinstance(obligation, ComparabilityFactorObligation):
            return self._resolve_comparability_factor(obligation, syms)
        if isinstance(obligation, ExponentialScaleObligation):
            return self._extend_exponential_scale(obligation, syms)
        if isinstance(obligation, LogarithmicScaleObligation):
            return self._extend_logarithmic_scale(obligation)
        return False

    def resolve_obligation(self, obligation: AsymptoticObligation) -> bool:
        """Resolve an obligation against this multiseries context.

        This public hook is useful to custom sparse nodes and extension modules:
        they can construct an obligation, ask the owning multiseries to resolve
        it, and then read the answer from :attr:`knowledge`. Scale-mutating and
        fact-only obligations use the same interface.
        """
        _, syms = self._formal()
        return self._resolve_asymptotic_obligation(obligation, syms)

    def _deformalize_and_filter(
        self,
        terms: Iterable[MultiseriesTerm],
        syms: tuple[sp.Symbol, ...],
    ) -> list[MultiseriesTerm]:
        reverse = {sym: elem.expr for sym, elem in zip(syms, self.scale.elements)}
        output = []
        for term in terms:
            coefficient = analytic_powsimp(sp.simplify(term.coefficient.xreplace(reverse)))
            zero = self.context.is_zero(coefficient)
            if zero is True:
                continue
            output.append(MultiseriesTerm(sp.simplify(term.exponent), coefficient))
        return _combine_terms(output)

    def terms(self, n: int | None = None) -> tuple[MultiseriesTerm, ...]:
        """Return up to ``n`` nonzero terms from the active scale level.

        Sparse evaluation is resumed one formal term at a time so exact
        coefficient cancellation does not force geometric oversampling.
        Recoverable obligations may extend the scale and restart formalization;
        unsupported nodes use the optional SymPy-series SymPy fallback.
        """

        n = self.default_terms if n is None else int(n)
        if n <= 0:
            return ()
        if self.level < 0:
            return (MultiseriesTerm(sp.S.Zero, self.expr),)[:n]
        if len(self._terms) >= n:
            return tuple(self._terms[:n])

        # A proven exact zero must terminate immediately; otherwise an infinite
        # lazy stream whose every coefficient vanishes after deformalization
        # would have no first surviving term.
        if self.context.is_zero(self.expr) is True:
            self._terms = []
            return ()

        formal, syms = self._formal()
        z = syms[self.level]

        log_z = None
        if self.level > 0:
            lower_scale = AsymptoticScale(
                self.scale.variable, self.scale.elements[: self.level], self.scale.point
            )
            try:
                lower_formal, lower_syms = lower_scale.formalize(sp.log(self.scale_element))
                lower_to_full = {a: b for a, b in zip(lower_syms, syms[: self.level])}
                candidate = analytic_powsimp(lower_formal.xreplace(lower_to_full))
                if not candidate.has(z):
                    log_z = candidate
            except SYMBOLIC_ERRORS:
                log_z = None

        native = None
        filtered = []
        sparse = LazySparseSeries(formal, z, self.context, log_z=log_z, knowledge=self.knowledge)
        sparse_request = 1

        # Pull exactly one additional formal term at a time.  If deformalized
        # coefficient cancellation removes it, advance the same persistent
        # frontier and ask for one more.  There is no n->2n oversampling and no
        # refinement-count heuristic.
        while True:
            sparse_terms = sparse.terms(sparse_request)
            if sparse_terms is None:
                obligation = sparse.obligation
                if obligation is None:
                    break
                old_scale = self.scale.exprs
                if not self._resolve_asymptotic_obligation(obligation, syms):
                    break
                if self.scale.exprs != old_scale:
                    return self.terms(n)
                sparse.resume(obligation)
                continue

            native = [MultiseriesTerm(t.exponent, t.coefficient) for t in sparse_terms]
            filtered = self._deformalize_and_filter(native, syms)
            if len(filtered) >= n:
                self._terms = filtered
                return tuple(filtered[:n])

            root_state = next((st for st in sparse.node_states if st.expr == formal), None)
            if root_state is not None and root_state.exhausted:
                self._terms = filtered
                return tuple(filtered[:n])
            sparse_request += 1
            # This is a safety bound on pathological unresolved infinite zero
            # streams, not an oversampling strategy. Exact-zero inputs are
            # handled above; normal nodes terminate or emit before this point.
            if sparse_request > 10000:
                break

        # Unsupported formal nodes may use SymPy's series engine only when the
        # caller explicitly permits that SymPy fallback.
        if not self.allow_series_fallback:
            if native is not None:
                self._terms = self._deformalize_and_filter(native, syms)
                return tuple(self._terms[:n])
            raise NotImplementedError(
                f"native sparse expansion cannot handle {formal} in scale variable {z}"
            )

        order = max(4, n + 3)
        filtered = []
        for _ in range(8):
            try:
                series_expr = sp.series(formal, z, 0, order)
            except SYMBOLIC_ERRORS as exc:
                if native is not None:
                    filtered = self._deformalize_and_filter(native, syms)
                    break
                raise NotImplementedError(
                    f"Could not expand {formal} in scale variable {z}; provide an explicit scale or simplify the expression"
                ) from exc
            extracted = self._extract_terms(series_expr, z)
            filtered = self._deformalize_and_filter(extracted, syms)
            if len(filtered) >= n or not series_expr.has(sp.Order):
                break
            order *= 2
        self._terms = filtered
        return tuple(filtered[:n])

    def coefficient_series(self, index: int, n: int | None = None) -> Multiseries:
        term = self.terms(index + 1)[index]
        return Multiseries(
            term.coefficient,
            self.scale,
            level=self.level - 1,
            context=self.context,
            default_terms=self.default_terms if n is None else n,
            knowledge=self.knowledge,
            allow_series_fallback=self.allow_series_fallback,
        )

    def leading_term(self, *, recursive: bool = False) -> sp.Expr:
        if self.level < 0:
            return self.expr
        first_terms = self.terms(1)
        if not first_terms:
            return sp.S.Zero
        first = first_terms[0]
        coeff = first.coefficient
        if recursive and self.level > 0 and coeff.has(self.scale.variable):
            coeff = self.coefficient_series(0).leading_term(recursive=True)
        return analytic_powsimp(coeff * self.scale_element**first.exponent)

    def truncate(
        self, n: int | None = None, *, recursive_coefficients: int | None = None
    ) -> sp.Expr:
        if self.level < 0:
            return self.expr
        result = sp.S.Zero
        for i, term in enumerate(self.terms(n)):
            coeff = term.coefficient
            if recursive_coefficients and self.level > 0 and coeff.has(self.scale.variable):
                coeff = self.coefficient_series(i).truncate(recursive_coefficients)
            result += coeff * self.scale_element**term.exponent
        return analytic_powsimp(result)

    def truncation(self, n: int | None = None, *, recursive_coefficients: int | None = None):
        """Return a finite prefix with conservative remainder semantics.

        The first omitted active-scale term certifies a big-O tail.  Coefficients
        are retained exactly unless recursive coefficient truncation is requested.
        """
        from .remainder import AsymptoticRemainder, AsymptoticTruncation

        count = self.default_terms if n is None else max(0, int(n))
        prefix = self.truncate(count, recursive_coefficients=recursive_coefficients)
        exact_error = analytic_powsimp(sp.expand(self.expr - prefix))
        if self.context.is_zero(exact_error) is True:
            remainder = AsymptoticRemainder.exact_zero(
                self.scale.variable, self.scale.point, source="exact multiseries truncation"
            )
            return AsymptoticTruncation(prefix, remainder, count, count)

        known = self.terms(count + 1)
        if len(known) > count:
            omitted = known[count]
            scale = analytic_powsimp(omitted.coefficient * self.scale_element**omitted.exponent)
            candidate = AsymptoticRemainder.big_o(
                scale,
                self.scale.variable,
                self.scale.point,
                exact_expression=exact_error,
                source="first omitted multiseries term",
            )
            if candidate.check(context=self.context) is True:
                return AsymptoticTruncation(prefix, candidate, count, len(known))
            remainder = AsymptoticRemainder.unknown(
                self.scale.variable,
                self.scale.point,
                exact_expression=exact_error,
                source="first omitted multiseries term did not certify the complete tail",
            )
            return AsymptoticTruncation(prefix, remainder, count, len(known))

        remainder = AsymptoticRemainder.unknown(
            self.scale.variable,
            self.scale.point,
            exact_expression=exact_error,
            source="multiseries tail scale not resolved",
        )
        return AsymptoticTruncation(prefix, remainder, count, len(known))

    def asymptotic_element(self):
        """View this multiseries through the common asymptotic-field protocol."""
        from .algebra import asymptotic_element

        return asymptotic_element(self)

    def differentiate(self, order: int = 1) -> Multiseries:
        """Differentiate the represented exact expression.

        The derivative is re-entered through scale discovery because
        differentiation can move terms between comparability classes. Shared
        asymptotic knowledge is retained.
        """
        if order < 0:
            raise ValueError("order must be nonnegative")
        derived = sp.diff(self.expr, self.scale.variable, order)
        if order == 0:
            return Multiseries(
                derived,
                self.scale,
                level=self.level,
                context=self.context,
                default_terms=self.default_terms,
                knowledge=self.knowledge,
            )
        result = multiseries(
            derived,
            self.scale.variable,
            point=self.scale.point,
            terms=self.default_terms,
        )
        result.knowledge = self.knowledge
        return result

    def integrate(
        self,
        *,
        constant: sp.Expr = 0,
        terms: int | None = None,
    ) -> Multiseries:
        """Integrate exactly when possible, otherwise integrate a lazy prefix.

        Scale discovery is rerun on the primitive, so logarithms and other new
        scale classes introduced by integration are represented explicitly.
        The fallback is intentionally marked by its finite lazy prefix: it is
        used only when SymPy cannot express an elementary antiderivative.
        """
        x = self.scale.variable
        primitive = sp.integrate(self.expr, x)
        if primitive.has(sp.Integral):
            n = self.default_terms if terms is None else int(terms)
            prefix = self.truncate(n, recursive_coefficients=n)
            primitive = sp.integrate(prefix, x)
            if primitive.has(sp.Integral):
                raise NotImplementedError(
                    f"Could not integrate {self.expr} or its {n}-term asymptotic prefix"
                )
        primitive = analytic_powsimp(sp.simplify(primitive + sp.sympify(constant)))
        result = multiseries(primitive, x, point=self.scale.point, terms=self.default_terms)
        result.knowledge = self.knowledge
        return result

    def inverse_asymptotic(
        self,
        inverse_variable: sp.Symbol | None = None,
        *,
        terms: int | None = None,
        branch: int | None = 0,
    ):
        """Return an asymptotic branch of the functional inverse.

        Inversion is performed from the exact represented expression rather
        than a truncation, and therefore keeps cancellation/zero decisions in
        the same certified context as the multiseries itself.
        """
        from .reversion import inverse_asymptotic

        return inverse_asymptotic(
            self.expr,
            self.scale.variable,
            inverse_variable,
            point=self.scale.point,
            terms=self.default_terms if terms is None else int(terms),
            branch=branch,
            context=self.context,
        )

    def __repr__(self) -> str:
        if self.level < 0:
            return f"Multiseries({self.expr!r}; constants)"
        return f"Multiseries({self.expr!r}; in {self.scale_element!r})"


def add_term_streams(
    a: Iterable[MultiseriesTerm], b: Iterable[MultiseriesTerm]
) -> Iterator[MultiseriesTerm]:
    ia, ib = iter(a), iter(b)
    ta = next(ia, None)
    tb = next(ib, None)
    while ta is not None or tb is not None:
        if ta is None:
            yield tb
            tb = next(ib, None)
            continue
        if tb is None:
            yield ta
            ta = next(ia, None)
            continue
        if _exp_key(ta.exponent) < _exp_key(tb.exponent):
            yield ta
            ta = next(ia, None)
        elif _exp_key(tb.exponent) < _exp_key(ta.exponent):
            yield tb
            tb = next(ib, None)
        else:
            coeff = sp.simplify(ta.coefficient + tb.coefficient)
            if coeff != 0:
                yield MultiseriesTerm(ta.exponent, coeff)
            ta = next(ia, None)
            tb = next(ib, None)


def multiply_term_lists(
    a: list[MultiseriesTerm], b: list[MultiseriesTerm], n: int
) -> list[MultiseriesTerm]:
    """First ``n`` product terms by a heap frontier over index pairs."""

    if not a or not b or n <= 0:
        return []
    ticket = count()
    heap = []
    seen = set()

    def push(i: int, j: int) -> None:
        if i >= len(a) or j >= len(b) or (i, j) in seen:
            return
        seen.add((i, j))
        exponent = sp.simplify(a[i].exponent + b[j].exponent)
        heapq.heappush(heap, (_exp_key(exponent), next(ticket), i, j))

    push(0, 0)
    result = []
    while heap and len(result) < n:
        key, _, i, j = heapq.heappop(heap)
        exponent = sp.simplify(a[i].exponent + b[j].exponent)
        coeff = sp.simplify(a[i].coefficient * b[j].coefficient)
        batch = [(i, j)]
        deferred = []
        while heap and heap[0][0] == key:
            _, _, ii, jj = heapq.heappop(heap)
            ee = sp.simplify(a[ii].exponent + b[jj].exponent)
            if ee == exponent or (ee - exponent).is_zero is True:
                coeff += a[ii].coefficient * b[jj].coefficient
                batch.append((ii, jj))
            else:
                deferred.append((ii, jj))
        for ii, jj in deferred:
            heapq.heappush(
                heap, (_exp_key(sp.simplify(a[ii].exponent + b[jj].exponent)), next(ticket), ii, jj)
            )
        coeff = sp.simplify(coeff)
        if coeff != 0:
            result.append(MultiseriesTerm(exponent, coeff))
        for ii, jj in batch:
            push(ii + 1, jj)
            push(ii, jj + 1)
    return result


def multiseries(
    expr: sp.Expr,
    variable: sp.Symbol,
    *,
    scale: AsymptoticScale | Iterable[sp.Expr] | None = None,
    point: sp.Expr = sp.oo,
    terms: int = 6,
    allow_series_fallback: bool = True,
) -> Multiseries:
    """Create a lazy multiseries, discovering an asymptotic scale when omitted."""
    discovery: ScaleDiscovery | None = None
    knowledge: AsymptoticKnowledge | None = None
    if scale is None:
        knowledge = AsymptoticKnowledge()
        discovery = ScaleDiscovery(expr, variable, point, knowledge=knowledge)
        scale_obj = discovery.discover()
    elif isinstance(scale, AsymptoticScale):
        scale_obj = scale
    else:
        scale_obj = AsymptoticScale.from_exprs(variable, tuple(scale), point)
    result = Multiseries(
        expr,
        scale_obj,
        default_terms=terms,
        knowledge=knowledge,
        allow_series_fallback=allow_series_fallback,
    )
    if discovery is not None:
        result.obligation_history.extend(discovery.obligation_history)
        result.scale_discovery = discovery
    else:
        result.scale_discovery = None
    return result
