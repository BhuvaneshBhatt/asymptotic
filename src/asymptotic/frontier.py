from __future__ import annotations

import heapq
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import count

import sympy as sp

from ._ordering import exponent_sort_key as _exp_key


@dataclass(frozen=True)
class SparseTerm:
    exponent: sp.Expr
    coefficient: sp.Expr


class AnalyticCompositionFrontier:
    """Lazy best-first enumerator for ``g(c + eta)``.

    ``eta_terms`` must have strictly positive, increasing exponents.  The
    Taylor coefficient callback returns ``g^(p)(c) / p!``.  Internally, the
    frontier enumerates nondecreasing tuples of indices, exactly the candidate
    family in Shackell §4.3.1, but uses a heap instead of repeatedly scanning
    the waiting lists.

    The class is intentionally independent of SymPy's ``series`` machinery and
    can therefore be reused by exp/log/binomial composition.
    """

    def __init__(
        self,
        eta_terms: Iterable[SparseTerm],
        taylor_coefficient: Callable[[int], sp.Expr],
    ) -> None:
        self.terms = tuple(eta_terms)
        if any(sp.sympify(term.exponent).is_positive is False for term in self.terms):
            raise ValueError("analytic-composition tail exponents must be positive")
        self.taylor_coefficient = taylor_coefficient
        self._heap: list[tuple[tuple[object, ...], int, tuple[int, ...]]] = []
        self._seen: set[tuple[int, ...]] = set()
        self._ticket = count()
        if self.terms:
            self._push((0,))

    def _weight(self, indices: tuple[int, ...]) -> sp.Expr:
        return sp.simplify(sum((self.terms[i].exponent for i in indices), sp.S.Zero))

    def _push(self, indices: tuple[int, ...]) -> None:
        if indices in self._seen:
            return
        if not indices or indices[-1] >= len(self.terms):
            return
        self._seen.add(indices)
        heapq.heappush(
            self._heap,
            (_exp_key(self._weight(indices)), next(self._ticket), indices),
        )

    def _coefficient(self, indices: tuple[int, ...]) -> sp.Expr:
        p = len(indices)
        counts = Counter(indices)
        # eta**p contributes p!/prod(m_i!) copies of a given nondecreasing
        # tuple.  Multiplying by g^(p)(c)/p! leaves g^(p)(c)/prod(m_i!).
        multiplicity = sp.factorial(p)
        for value in counts.values():
            multiplicity /= sp.factorial(value)
        product = sp.prod(self.terms[i].coefficient for i in indices)
        # Keep coefficient algebra lazy.  Full ``simplify`` here can trigger
        # expensive assumption/root analysis on symbolic log/exp coefficients;
        # zero testing is performed later after deformalization.
        return self.taylor_coefficient(p) * multiplicity * product

    def _advance(self, indices: tuple[int, ...]) -> None:
        # Minimal larger tuples under componentwise order: increment the final
        # occurrence of each distinct index while preserving nondecreasing order.
        for position in range(len(indices) - 1, -1, -1):
            if position < len(indices) - 1 and indices[position] == indices[position + 1]:
                continue
            candidate = list(indices)
            candidate[position] += 1
            if candidate[position] >= len(self.terms):
                continue
            if position + 1 < len(candidate) and candidate[position] > candidate[position + 1]:
                continue
            self._push(tuple(candidate))
        # When the all-zero p-tuple is consumed, introduce the (p+1)-power.
        if all(i == 0 for i in indices):
            self._push((0,) * (len(indices) + 1))

    def terms_up_to(self, n: int) -> list[SparseTerm]:
        if n <= 0:
            return []
        result = []
        while self._heap and len(result) < n:
            key, _, indices = heapq.heappop(self._heap)
            exponent = self._weight(indices)
            coefficient = self._coefficient(indices)
            batch = [indices]
            while self._heap and self._heap[0][0] == key:
                _, _, other = heapq.heappop(self._heap)
                other_exp = self._weight(other)
                if sp.simplify(other_exp - exponent) == 0:
                    coefficient += self._coefficient(other)
                    batch.append(other)
                else:
                    heapq.heappush(
                        self._heap,
                        (_exp_key(other_exp), next(self._ticket), other),
                    )
                    break
            if coefficient != 0:
                result.append(SparseTerm(exponent, coefficient))
            for item in batch:
                self._advance(item)
        return result


def compose_analytic_terms(
    constant: sp.Expr,
    tail_terms: Iterable[SparseTerm],
    derivative: Callable[[int, sp.Expr], sp.Expr],
    n: int,
) -> list[SparseTerm]:
    """Return the first terms of an analytic composition about ``constant``.

    ``derivative(p, constant)`` must return the p-th derivative of the outer
    function evaluated at the expansion point.  The constant term is emitted
    separately; positive-exponent terms are enumerated lazily by the frontier.
    """

    if n <= 0:
        return []
    c0 = derivative(0, constant)
    result = [] if c0 == 0 else [SparseTerm(sp.S.Zero, c0)]
    if len(result) >= n:
        return result[:n]

    def taylor(p: int) -> sp.Expr:
        return derivative(p, constant) / sp.factorial(p)

    frontier = AnalyticCompositionFrontier(tuple(tail_terms), taylor)
    result.extend(frontier.terms_up_to(n - len(result)))
    return result[:n]
