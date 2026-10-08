"""Small exact integer helpers shared by ramification code."""

from __future__ import annotations

from math import lcm


def integer_lcm(a: int, b: int) -> int:
    """Return the standard nonnegative least common multiple.

    Ramification callers pass positive denominators; zero follows the standard
    ``math.lcm`` convention.
    """

    return lcm(int(a), int(b))
