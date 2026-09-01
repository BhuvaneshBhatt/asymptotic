from __future__ import annotations

from dataclasses import dataclass
from functools import cmp_to_key

import sympy as sp

from ._power_simplify import analytic_powsimp
from ._symbolic_errors import SYMBOLIC_ERRORS
from .context import AsymptoticContext, GrowthComparison, context_for
from .decomposition import decompose_expression
from .mrv import mrv_decomposition
from .obligations import AsymptoticKnowledge, GrowthComparisonObligation
from .periodic import periodic_decomposition
from .tower import ExpLogTower


@dataclass(frozen=True)
class ScaleElement:
    expr: sp.Expr
    name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "expr", sp.sympify(self.expr))

    def __str__(self) -> str:
        return self.name or str(self.expr)

    def asymptotic_element(self, variable: sp.Symbol, *, point: sp.Expr = sp.oo):
        """View this scale representative through the common field protocol."""
        from .algebra import asymptotic_element

        return asymptotic_element(self, variable, point=point)


@dataclass(frozen=True)
class AsymptoticScale:
    """A Shackell-style scale ordered from slowest to fastest vanishing."""

    variable: sp.Symbol
    elements: tuple[ScaleElement, ...]
    point: sp.Expr = sp.oo

    @classmethod
    def from_exprs(
        cls,
        variable: sp.Symbol,
        exprs: list[sp.Expr] | tuple[sp.Expr, ...],
        point: sp.Expr = sp.oo,
    ) -> AsymptoticScale:
        return cls(variable, tuple(ScaleElement(sp.sympify(e)) for e in exprs), point)

    def __len__(self) -> int:
        return len(self.elements)

    @property
    def exprs(self) -> tuple[sp.Expr, ...]:
        return tuple(e.expr for e in self.elements)

    def element(self, index: int):
        """Return one scale representative as a common asymptotic element."""
        return self.elements[index].asymptotic_element(self.variable, point=self.point)

    def validate(self, ctx: AsymptoticContext | None = None) -> None:
        ctx = context_for(self.variable, self.point, ctx)
        for item in self.elements:
            lim = ctx.limit(item.expr)
            if lim != 0:
                raise ValueError(f"Scale element {item.expr} does not tend to zero; limit={lim}")
        for left, right in zip(self.elements, self.elements[1:]):
            cmp, _ = ctx.compare_log_growth(left.expr, right.expr)
            if cmp is not GrowthComparison.SMALLER:
                raise ValueError(
                    f"Scale is not strictly ordered by comparability class: {left.expr}, {right.expr} ({cmp})"
                )

    def symbols(self) -> tuple[sp.Symbol, ...]:
        return tuple(sp.Dummy(f"t{i + 1}", positive=True) for i in range(len(self.elements)))

    def with_element(
        self,
        expr: sp.Expr,
        ctx: AsymptoticContext | None = None,
    ) -> tuple[AsymptoticScale, bool]:
        """Insert a vanishing scale representative in comparability order.

        The method is intentionally conservative.  A candidate that is in an
        existing comparability class is treated as an alias and does not grow
        the scale.  Otherwise it is inserted according to ``compare_log_growth``.
        Unknown pairwise comparisons fall back to deterministic SymPy ordering;
        validation can subsequently reject a genuinely unresolved ordering.
        """

        ctx = context_for(self.variable, self.point, ctx)
        candidate = analytic_powsimp(sp.simplify(sp.sympify(expr)))
        if ctx.limit(candidate) != 0:
            raise ValueError(f"Dynamic scale candidate {candidate} does not tend to zero")

        for old in self.elements:
            relation, _ = ctx.compare_log_growth(candidate, old.expr)
            if relation is GrowthComparison.SAME_ORDER:
                return self, False
            if sp.simplify(candidate - old.expr) == 0:
                return self, False

        exprs = [item.expr for item in self.elements] + [candidate]

        def cmp(a: sp.Expr, b: sp.Expr) -> int:
            relation, _ = ctx.compare_log_growth(a, b)
            if relation is GrowthComparison.SMALLER:
                return -1
            if relation is GrowthComparison.LARGER:
                return 1
            return (sp.default_sort_key(a) > sp.default_sort_key(b)) - (
                sp.default_sort_key(a) < sp.default_sort_key(b)
            )

        exprs.sort(key=cmp_to_key(cmp))
        return AsymptoticScale.from_exprs(self.variable, exprs, self.point), True

    def _factor_exponential_aliases(
        self,
        expr: sp.Expr,
        syms: tuple[sp.Symbol, ...],
        ctx: AsymptoticContext,
    ) -> sp.Expr:
        """Factor exp nodes against existing exponential scale classes.

        If t_j = exp(b) and exp(a) has a/b -> c != 0, rewrite

            exp(a) = t_j**c * exp(a - c*b).

        This captures exact aliases such as exp(-2*x) = exp(-x)**2 and also
        useful same-class factorizations such as exp(-2*x + 1/x).
        """

        exponential_scales = []
        for elem, sym in zip(self.elements, syms):
            e = analytic_powsimp(elem.expr)
            if e.func is sp.exp:
                exponential_scales.append((e.args[0], sym))

        if not exponential_scales:
            return expr

        replacements = {}
        for node in sp.preorder_traversal(expr):
            if node.func is not sp.exp:
                continue
            a = node.args[0]
            best: tuple[int, sp.Expr, sp.Symbol, sp.Expr] | None = None
            for index, (b, sym) in enumerate(exponential_scales):
                try:
                    c = ctx.limit(sp.cancel(a / b))
                except SYMBOLIC_ERRORS:
                    continue
                if c.is_finite is not True or c.is_zero is not False:
                    continue
                residual = sp.simplify(a - c * b)
                # Accept only if the residual is asymptotically smaller than b;
                # otherwise the limit comparison was not useful enough.
                if residual != 0:
                    rel, _ = ctx.compare_growth(residual, b)
                    if rel is not GrowthComparison.SMALLER:
                        continue
                score = sp.count_ops(residual)
                if best is None or score < best[0]:
                    best = (score, c, sym, residual)
            if best is not None:
                _, c, sym, residual = best
                replacement = sym**c
                if residual != 0:
                    replacement *= sp.exp(residual)
                replacements[node] = analytic_powsimp(replacement)
        return expr.xreplace(replacements) if replacements else expr

    def formalize(self, expr: sp.Expr) -> tuple[sp.Expr, tuple[sp.Symbol, ...]]:
        """Rewrite scale elements, reciprocals, and same-class aliases formally.

        Exact scale elements are replaced first.  Exponential expressions in an
        existing comparability class are factored into powers of that scale plus
        a lower-order residual, so e.g. a scale containing ``exp(-x)`` also
        recognizes ``exp(-2*x)`` without adding a redundant scale element.
        """

        expr = analytic_powsimp(sp.expand_power_exp(sp.sympify(expr)))
        syms = self.symbols()
        ctx = AsymptoticContext(self.variable, self.point)
        expr = self._factor_exponential_aliases(expr, syms, ctx)

        mapping = {}
        pairs = sorted(
            zip(self.elements, syms), key=lambda p: sp.count_ops(p[0].expr), reverse=True
        )
        for elem, sym in pairs:
            e = analytic_powsimp(elem.expr)
            mapping[e] = sym
            mapping[analytic_powsimp(1 / e)] = 1 / sym

        out = expr
        for _ in range(4):
            new = out.subs(mapping, simultaneous=True)
            new = analytic_powsimp(new)
            if new == out:
                break
            out = new
        return out, syms


@dataclass
class ScaleDiscovery:
    """Demand-driven exp-log scale discovery using asymptotic obligations.

    Candidate generation follows the actual dependency-ordered exp/log tower;
    there is no fixed iterated-log depth. Pairwise ordering and comparability
    tests are represented as ``GrowthComparisonObligation`` objects and their
    answers are stored in shared ``AsymptoticKnowledge``.
    """

    expr: sp.Expr
    variable: sp.Symbol
    point: sp.Expr = sp.oo
    context: AsymptoticContext | None = None
    knowledge: AsymptoticKnowledge | None = None

    def __post_init__(self) -> None:
        self.expr = sp.sympify(self.expr)
        self.context = context_for(self.variable, self.point, self.context)
        self.knowledge = self.knowledge or AsymptoticKnowledge()
        self.decomposition = decompose_expression(self.expr, self.variable)
        self.periodic = periodic_decomposition(
            self.decomposition.canonical, self.variable, point=self.point, context=self.context
        )
        self._mrv = None
        # Periodic factors are coefficient-like objects, not ordered growth
        # scales. Build the exp-log tower from the nonoscillatory envelope.
        self.tower = ExpLogTower.from_expr(self.periodic.envelope, self.variable)
        self.obligation_history: list[GrowthComparisonObligation] = []

    @property
    def mrv(self):
        # MRV analysis can be substantially more expensive than tower
        # construction for deeply iterated logarithms, so compute it only when
        # callers need it. Scale candidate generation itself remains tower-led.
        if self._mrv is None:
            self._mrv = mrv_decomposition(
                self.periodic.envelope,
                self.variable,
                self.point,
                context=self.context,
                structural=decompose_expression(self.periodic.envelope, self.variable),
            )
        return self._mrv

    def _compare(self, left: sp.Expr, right: sp.Expr) -> tuple[GrowthComparison, sp.Expr | None]:
        obligation = GrowthComparisonObligation(
            node=sp.Tuple(left, right),
            left=left,
            right=right,
            logarithmic=True,
            reason="scale discovery requires a comparability-class ordering",
        )
        cached = self.knowledge.get(obligation, None)
        if cached is not None:
            return cached
        self.obligation_history.append(obligation)
        result = self.context.compare_log_growth(left, right)
        if result[0] is not GrowthComparison.UNKNOWN:
            self.knowledge.set(obligation, result)
        return result

    def _log_candidates(self) -> list[sp.Expr]:
        candidates = []
        # Every logarithmic generator already present in the tower is a genuine
        # dependency. If it diverges, its reciprocal is a lower vanishing
        # scale representative. This naturally handles arbitrary iterated-log
        # depth without a hard-coded loop bound.
        for ext in self.tower.extensions:
            if ext.kind != "log":
                continue
            generator = analytic_powsimp(ext.generator)
            lim = self.context.limit(generator)
            if lim not in (sp.oo, -sp.oo):
                continue
            sign = 1 if lim is sp.oo else -1
            candidates.append(analytic_powsimp(sign / generator))
        return candidates

    def _exp_candidates(self) -> list[sp.Expr]:
        candidates = []
        for ext in self.tower.extensions:
            if ext.kind != "exp":
                continue
            node = analytic_powsimp(ext.generator)
            lim = self.context.limit(node)
            if lim == 0:
                candidates.append(node)
            elif lim is sp.oo:
                candidates.append(sp.exp(-ext.argument))
        return candidates

    def candidates(self) -> list[sp.Expr]:
        candidates: list[sp.Expr] = [1 / self.variable]
        candidates.extend(self._log_candidates())
        candidates.extend(self._exp_candidates())
        unique = []
        for candidate in candidates:
            candidate = analytic_powsimp(sp.simplify(candidate))
            if self.context.limit(candidate) != 0:
                continue
            if not any(sp.simplify(candidate - old) == 0 for old in unique):
                unique.append(candidate)
        return unique

    def discover(self) -> AsymptoticScale:
        reps = []
        for candidate in self.candidates():
            equivalent = False
            for old in reps:
                relation, _ = self._compare(candidate, old)
                if relation is GrowthComparison.SAME_ORDER:
                    equivalent = True
                    break
            if not equivalent:
                reps.append(candidate)

        def cmp(a: sp.Expr, b: sp.Expr) -> int:
            relation, _ = self._compare(a, b)
            if relation is GrowthComparison.SMALLER:
                return -1
            if relation is GrowthComparison.LARGER:
                return 1
            # UNKNOWN is not promoted to a mathematical claim. Deterministic
            # ordering is used only to make the returned container stable; the
            # unresolved obligation remains visible in obligation_history.
            return (sp.default_sort_key(a) > sp.default_sort_key(b)) - (
                sp.default_sort_key(a) < sp.default_sort_key(b)
            )

        reps.sort(key=cmp_to_key(cmp))
        return AsymptoticScale.from_exprs(self.variable, reps, self.point)


def discover_scale(expr: sp.Expr, x: sp.Symbol, point: sp.Expr = sp.oo) -> AsymptoticScale:
    """Discover a dependency-driven exp-log scale.

    For callers that need the comparison obligations and cached knowledge,
    instantiate :class:`ScaleDiscovery` directly and call ``discover()``.
    """

    return ScaleDiscovery(sp.sympify(expr), x, point).discover()
