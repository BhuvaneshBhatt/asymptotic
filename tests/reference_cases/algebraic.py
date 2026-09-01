"""Puiseux, reversion, and implicit reference cases."""

import sympy as sp

from asymptotic import (
    implicit_asymptotic,
    series_reversion,
)
from asymptotic.implicit import implicit_singularity_profile

from . import CapabilityStatus, ReferenceCase


def _square_root_turning_point() -> bool:
    x, y = sp.symbols("x y", positive=True)
    profile = implicit_singularity_profile(y**2 - x, y, x)
    branches = implicit_asymptotic(y**2 - x, y, x, terms=3)
    prefixes = {sp.expand(branch.truncate()) for branch in branches}
    return profile.multiplicity == 2 and profile.turning_point is True and len(prefixes) == 2


def _reversion_round_trip() -> bool:
    x, y = sp.symbols("x y")
    branch = series_reversion(x + x**2, x, y, terms=5, branch=0)
    inverse = branch.truncate()
    defect = sp.series((x + x**2).subs(x, inverse), y, 0, 5).removeO() - y
    return sp.expand(defect) == 0


CASES = (
    ReferenceCase(
        "square-root-turning-point", "implicit", CapabilityStatus.FORMAL, _square_root_turning_point
    ),
    ReferenceCase(
        "quadratic-reversion", "reversion", CapabilityStatus.FORMAL, _reversion_round_trip
    ),
)
