"""Small exact integer helpers shared by ramification code."""

from __future__ import annotations

from math import lcm


def integer_lcm(a: int, b: int) -> int:
    """Return the standard nonnegative least common multiple.

    All ramification callers pass positive denominators; defining zero by the
    standard ``math.lcm`` convention removes the previous private inconsistency.
    """

    return lcm(int(a), int(b))
