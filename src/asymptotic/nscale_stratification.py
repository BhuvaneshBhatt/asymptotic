"""Budgeted n-scale dominance stratification.

The scale-stratification engine builds finite total preorders only for scales that actually compete in
additive subexpressions.  Comparable blocks retain explicit ratio coordinates;
strict block order records little-o facts.  The construction is combinatorial
and never uses numerical sampling as proof.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, pairwise

import sympy as sp

from .proof_obligations import ObligationKind, ProofObligation
from .stratified_expansion import (
    CertifiedPowerExpansion,
    ExpansionRegime,
    RegimeRelation,
    SmallQuantity,
    StratificationCoverageCertificate,
    StratifiedExpansion,
)


@dataclass(frozen=True)
class DominanceComparison:
    left: sp.Symbol
    right: sp.Symbol


def relevant_scale_comparisons(
    expr: sp.Expr, variables: tuple[sp.Symbol, ...]
) -> tuple[DominanceComparison, ...]:
    """Return pairwise walls that can change an additive dominant balance."""
    wanted = set(variables)
    pairs: set[tuple[sp.Symbol, sp.Symbol]] = set()
    for node in sp.preorder_traversal(expr):
        if not isinstance(node, sp.Add):
            continue
        present = sorted((s for s in wanted if node.has(s)), key=sp.default_sort_key)
        for a, b in combinations(present, 2):
            pairs.add((a, b))
    return tuple(
        DominanceComparison(a, b)
        for a, b in sorted(
            pairs, key=lambda p: (sp.default_sort_key(p[0]), sp.default_sort_key(p[1]))
        )
    )


def _ordered_partitions(items: tuple[sp.Symbol, ...]):
    """Generate ordered set partitions once each (Fubini regimes)."""
    if not items:
        yield ()
        return
    first, rest = items[0], items[1:]
    for part in _ordered_partitions(rest):
        # Join each comparable block.
        for i in range(len(part)):
            yield part[:i] + (part[i] + (first,),) + part[i + 1 :]
        # Or create a new strict-order block in every position.
        for i in range(len(part) + 1):
            yield part[:i] + ((first,),) + part[i:]


def _canonical(partition):
    return tuple(tuple(sorted(block, key=sp.default_sort_key)) for block in partition)


def _regime(partition) -> ExpansionRegime:
    facts: list[SmallQuantity] = []
    coords: list[tuple[sp.Symbol, sp.Expr]] = []
    # Blocks are ordered from asymptotically smallest to largest.  Adjacent
    # little-o facts suffice transitively and avoid quadratic metadata.
    for small_block, large_block in pairwise(partition):
        facts.append(
            SmallQuantity(
                sp.simplify(small_block[0] / large_block[0]), RegimeRelation.LITTLE_O
            )
        )
    # Preserve a spanning forest of ratio coordinates inside comparable blocks.
    for block_index, block in enumerate(partition):
        if len(block) < 2:
            continue
        root = block[0]
        for edge_index, member in enumerate(block[1:], 1):
            ratio = sp.Dummy(f"rho_{block_index}_{edge_index}", positive=True)
            coords.append((ratio, sp.simplify(member / root)))
    statement = " <o< ".join("~".join(map(sp.sstr, block)) for block in partition)
    return ExpansionRegime(
        facts=tuple(facts), scaled_coordinates=tuple(coords), statement=statement
    )


def nscale_stratified_expand(
    expr, variables, *, target=None, order=3, branch_budget=64
):
    """Construct a finite exact dominance atlas for relevant positive scales.

    The atlas enumerates feasible total preorders of the relevant scale graph.
    If the number of cells exceeds ``branch_budget``, no partial atlas is called
    complete: a SYMBOLIC_BUDGET obligation is returned instead.
    """
    expr = sp.sympify(expr)
    variables = tuple(variables)
    target = (
        tuple(sp.S.Zero for _ in variables)
        if target is None
        else tuple(map(sp.sympify, target))
    )
    if any(t != 0 for t in target) or len(variables) < 3:
        return None
    if branch_budget < 1:
        raise ValueError("branch_budget must be positive")
    comparisons = relevant_scale_comparisons(expr, variables)
    active = tuple(
        sorted(
            {s for c in comparisons for s in (c.left, c.right)}, key=sp.default_sort_key
        )
    )
    if len(active) < 3:
        return None
    # Current certified kernel requires the relevant comparison graph to be
    # connected. Disconnected components have no proved relative scale order.
    edges = {frozenset((c.left, c.right)) for c in comparisons}
    seen = {active[0]}
    changed = True
    while changed:
        changed = False
        for edge in edges:
            if seen & edge and not edge <= seen:
                seen |= edge
                changed = True
    if seen != set(active):
        return None

    unique = []
    seen_parts = set()
    for raw in _ordered_partitions(active):
        part = _canonical(raw)
        if part in seen_parts:
            continue
        seen_parts.add(part)
        unique.append(part)
        if len(unique) > branch_budget:
            obligation = ProofObligation(
                ObligationKind.SYMBOLIC_BUDGET,
                f"n-scale stratification exceeds branch budget {branch_budget}",
                provider="nscale_stratified_expand",
                expression=sp.sstr(expr),
            )
            return StratifiedExpansion(
                expr,
                variables,
                target,
                (),
                StratificationCoverageCertificate(
                    False, "dominance atlas exceeds symbolic branch budget", ()
                ),
                (obligation,),
            )

    branches = tuple(
        CertifiedPowerExpansion(
            expr,
            expr,
            sp.S.Zero,
            _regime(part),
            order,
            True,
            "nscale_exact_dominance_cell",
        )
        for part in unique
    )
    return StratifiedExpansion(
        expr,
        variables,
        target,
        branches,
        StratificationCoverageCertificate(
            True,
            "all total preorders of the relevant positive scale graph",
            tuple(sp.S.true for _ in branches),
        ),
    )
