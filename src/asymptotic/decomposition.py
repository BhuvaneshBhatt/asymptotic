from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


@dataclass(frozen=True)
class CompositionLayer:
    """One exact univariate layer ``outer(inner)`` in a decomposition."""

    symbol: sp.Symbol
    outer: sp.Expr
    inner: sp.Expr

    def apply(self, value: sp.Expr) -> sp.Expr:
        return self.outer.xreplace({self.symbol: value})


@dataclass(frozen=True)
class StructuralDecomposition:
    """Exact structural preprocessing result used before scale discovery."""

    original: sp.Expr
    canonical: sp.Expr
    variable: sp.Symbol
    composition: tuple[CompositionLayer, ...]
    rationalized: sp.Expr
    substitutions: tuple[tuple[sp.Symbol, sp.Expr], ...]

    def reconstruct_rationalized(self) -> sp.Expr:
        return self.rationalized.xreplace(dict(self.substitutions))


def _dependent_args(expr: sp.Expr, x: sp.Symbol) -> list[sp.Expr]:
    return [arg for arg in expr.args if arg.has(x)]


def _shared_inner(expr: sp.Expr, x: sp.Symbol) -> sp.Expr | None:
    """Return a nontrivial exact inner expression carrying all x-dependence.

    A candidate is accepted only when replacing every exact occurrence by one
    dummy symbol removes ``x`` from the outer expression.  The search is purely
    structural and bounded by the existing expression tree; it performs no
    simplification or equation solving.
    """

    candidates = []
    seen = set()
    for node in sp.preorder_traversal(expr):
        if node in seen or node in (expr, x) or not node.has(x):
            continue
        seen.add(node)
        candidates.append(node)
    candidates.sort(key=sp.count_ops, reverse=True)
    t = sp.Dummy("u")
    for candidate in candidates:
        outer = expr.xreplace({candidate: t})
        if not outer.has(x) and outer.has(t):
            return candidate
    return None


def maximal_univariate_decomposition(expr: sp.Expr, x: sp.Symbol) -> tuple[CompositionLayer, ...]:
    """Peel the maximal exact chain of one-variable outer compositions.

    The implementation deliberately uses only exact substitution identities.
    At a node with exactly one x-dependent child, that child is replaced by a
    fresh symbol to create an outer map and the walk continues inward.  This
    covers function composition as well as affine/rational wrappers such as
    ``log(1 + sin(exp(-x)))``.
    """

    expr = sp.sympify(expr)
    layers = []
    current = expr
    while current != x and current.has(x):
        deps = _dependent_args(current, x)
        if len(deps) == 1:
            inner = deps[0]
        else:
            inner = _shared_inner(current, x)
            if inner is None:
                break
        t = sp.Dummy("u")
        outer = current.xreplace({inner: t})
        if outer == t:
            current = inner
            continue
        layers.append(CompositionLayer(t, outer, inner))
        current = inner
    return tuple(layers)


def canonicalize_transcendentals(expr: sp.Expr) -> sp.Expr:
    """Apply branch-safe exact canonicalizations useful to asymptotics."""

    expr = sp.sympify(expr)
    # powsimp/expand_power_exp are exact for the integer/exponential structures
    # used here and avoid branch-changing forceful power-base rewrites.
    out = sp.expand_power_exp(expr)
    out = sp.powsimp(out, force=False)
    # Canonical reciprocal trig/hyperbolic forms reduce duplicate generators.
    repl = {}
    for node in sp.preorder_traversal(out):
        if node.func is sp.cot:
            repl[node] = 1 / sp.tan(node.args[0])
        elif node.func is sp.sec:
            repl[node] = 1 / sp.cos(node.args[0])
        elif node.func is sp.csc:
            repl[node] = 1 / sp.sin(node.args[0])
        elif node.func is sp.coth:
            repl[node] = 1 / sp.tanh(node.args[0])
        elif node.func is sp.sech:
            repl[node] = 1 / sp.cosh(node.args[0])
        elif node.func is sp.csch:
            repl[node] = 1 / sp.sinh(node.args[0])
    return sp.powsimp(out.xreplace(repl), force=False) if repl else out


def rational_decomposition(
    expr: sp.Expr, x: sp.Symbol
) -> tuple[sp.Expr, tuple[tuple[sp.Symbol, sp.Expr], ...]]:
    """Rationalize compatible transcendental families by exact identities.

    The strongest transformation is the tangent-half-angle map for
    trigonometric functions sharing an argument.  Exponentials are represented
    by exact auxiliary variables, allowing rational/algebraic preprocessing
    without losing the reconstruction map.  No branch-sensitive inverse
    identity is applied here.
    """

    expr = canonicalize_transcendentals(expr)
    replacements = {}
    back = []
    trig_args = []
    for node in sp.preorder_traversal(expr):
        if (
            node.func in {sp.sin, sp.cos, sp.tan}
            and node.args[0].has(x)
            and node.args[0] not in trig_args
        ):
            trig_args.append(node.args[0])
    for i, arg in enumerate(trig_args):
        t = sp.Dummy(f"trig{i}")
        back.append((t, sp.tan(arg / 2)))
        replacements[sp.sin(arg)] = 2 * t / (1 + t**2)
        replacements[sp.cos(arg)] = (1 - t**2) / (1 + t**2)
        replacements[sp.tan(arg)] = 2 * t / (1 - t**2)

    exp_nodes = []
    for node in sp.preorder_traversal(expr):
        if node.func is sp.exp and node.has(x) and node not in exp_nodes:
            exp_nodes.append(node)

    # Integer powers of one exponential generator are globally exact:
    # exp(k*g) == exp(g)**k for integer k.  Group rational coefficients
    # by a rational gcd so exp(x/2) and exp(3*x/2), for example, share
    # exp(x/2) without introducing any branch-sensitive power identity.
    selected = [
        node
        for node in exp_nodes
        if not any(node != other and other.args[0].has(node) for other in exp_nodes)
    ]
    groups: dict[sp.Expr, list[tuple[sp.Expr, sp.Rational]]] = {}
    leftovers = []
    for node in selected:
        coeff, core = node.args[0].as_coeff_Mul(rational=True)
        if coeff.is_Rational and coeff != 0 and core.has(x):
            groups.setdefault(core, []).append((node, sp.Rational(coeff)))
        else:
            leftovers.append(node)

    def rational_gcd(values: list[sp.Rational]) -> sp.Rational:
        nums = [abs(int(value.p)) for value in values]
        dens = [int(value.q) for value in values]
        num = nums[0]
        den = dens[0]
        for item in nums[1:]:
            num = int(sp.igcd(num, item))
        for item in dens[1:]:
            den = int(sp.ilcm(den, item))
        return sp.Rational(num, den)

    exp_index = 0
    for core, members in groups.items():
        base_coeff = rational_gcd([coeff for _, coeff in members])
        if base_coeff == 0:
            leftovers.extend(node for node, _ in members)
            continue
        base_expr = sp.exp(base_coeff * core)
        e = sp.Dummy(f"exp{exp_index}", positive=True)
        exp_index += 1
        back.append((e, base_expr))
        for node, coeff in members:
            power = sp.cancel(coeff / base_coeff)
            if power.is_integer is not True:
                leftovers.append(node)
                continue
            replacements[node] = e ** int(power)

    for node in leftovers:
        if node in replacements:
            continue
        e = sp.Dummy(f"exp{exp_index}", positive=True)
        exp_index += 1
        back.append((e, node))
        replacements[node] = e

    rat = expr.xreplace(replacements) if replacements else expr
    return sp.cancel(rat), tuple(back)


def decompose_expression(expr: sp.Expr, x: sp.Symbol) -> StructuralDecomposition:
    """Return canonical, compositional, and rationalized views of an expression."""
    expr = sp.sympify(expr)
    canonical = canonicalize_transcendentals(expr)
    composition = maximal_univariate_decomposition(canonical, x)
    rationalized, substitutions = rational_decomposition(canonical, x)
    return StructuralDecomposition(
        original=expr,
        canonical=canonical,
        variable=x,
        composition=composition,
        rationalized=rationalized,
        substitutions=substitutions,
    )
