"""Multivariate and parameter-stratification reference cases."""

import sympy as sp

from asymptotic import implicit_asymptotic
from asymptotic.multivariate import multivariate_scaling_regimes
from asymptotic.stratification import AsymptoticStratification

from . import CapabilityStatus, ReferenceCase


def _weight_cones() -> bool:
    x, z, y = sp.symbols("x z y", positive=True)
    result = multivariate_scaling_regimes(y**2 - x - z**2, y, (x, z))
    return len(result) >= 2


def _multiplicity_strata() -> bool:
    x, y, a = sp.symbols("x y a")
    result = implicit_asymptotic((y - 1) ** 2 + a * (y - 1) - x, y, x, dependent_limit=1, terms=3)
    if not isinstance(result, AsymptoticStratification):
        return False
    conditions = {sp.simplify(stratum.condition) for stratum in result.strata}
    return any(condition.has(sp.Eq) or condition.has(sp.Ne) for condition in conditions)


CASES = (
    ReferenceCase(
        "two-variable-weight-cones", "multivariate", CapabilityStatus.FORMAL, _weight_cones
    ),
    ReferenceCase(
        "parameter-dependent-multiplicity",
        "parameter-strata",
        CapabilityStatus.FORMAL,
        _multiplicity_strata,
    ),
)
