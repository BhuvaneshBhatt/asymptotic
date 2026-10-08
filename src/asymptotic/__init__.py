"""Primary public API for symbolic asymptotic analysis.

The root namespace contains common mathematical workflows and principal result
types. Specialist theorem, geometry, coefficient, and certificate APIs remain
public from their defining submodules.
"""

from __future__ import annotations

from .algebra import as_element
from .analytic_limits import SquareWave, analytic_limit
from .calculus import truncate
from .cluster_limits import Circle, ExactClusterResult, cluster_set
from .complex_limits import complex_limit
from .context import AsymptoticContext
from .discrete_limits import discrete_limit
from .dsolve import DSolveResult
from .evidence import (
    Evidence,
    EvidenceStatus,
)
from .fixed_ray_branch_germs import DirectionalInfinity
from .limits import limit
from .mathematical_api import (
    compose,
    differentiate,
    dsolve,
    expectation,
    hyperasymptotic_series,
    implicit,
    integrate,
    inverse,
    leading_term,
    lindstedt_poincare,
    local_series,
    maximize,
    mellin,
    minimize,
    multiseries,
    nested_series,
    probability,
    product,
    puiseux_series,
    regular_perturbation,
    relation,
    root,
    rsolve,
    series,
    solve,
    sum,
)
from .multiseries import Multiseries
from .optimization import OptimizationResult, argmax, argmin
from .path_limits import (
    complex_ray_limit,
    one_sided_limit,
    path_limit,
)
from .probability import StatisticalResult
from .products import ProductResult
from .public_result import (
    PublicResult,
    explain,
)
from .relations import big_o, equivalent, little_o
from .remainder import (
    Remainder,
    RemainderKind,
    Truncation,
)
from .rsolve import RSolveResult
from .scale import (
    Scale,
    discover_scale,
)
from .solve import SolveResult
from .stratified_expansion import stratified_expand as stratified_series
from .sums import SumResult
from .transseries import TransseriesExpansion

__version__ = "0.2.0"


__all__ = [
    "AsymptoticContext",
    "DSolveResult",
    "DirectionalInfinity",
    "Evidence",
    "EvidenceStatus",
    "Multiseries",
    "OptimizationResult",
    "ProductResult",
    "PublicResult",
    "RSolveResult",
    "Remainder",
    "RemainderKind",
    "Scale",
    "SolveResult",
    "SquareWave",
    "StatisticalResult",
    "SumResult",
    "TransseriesExpansion",
    "Truncation",
    "__version__",
    "analytic_limit",
    "argmax",
    "argmin",
    "as_element",
    "big_o",
    "complex_limit",
    "complex_ray_limit",
    "cluster_set",
    "Circle",
    "ExactClusterResult",
    "compose",
    "differentiate",
    "discover_scale",
    "discrete_limit",
    "dsolve",
    "equivalent",
    "expectation",
    "explain",
    "hyperasymptotic_series",
    "implicit",
    "integrate",
    "inverse",
    "leading_term",
    "limit",
    "lindstedt_poincare",
    "little_o",
    "local_series",
    "maximize",
    "mellin",
    "minimize",
    "multiseries",
    "nested_series",
    "one_sided_limit",
    "path_limit",
    "probability",
    "product",
    "puiseux_series",
    "regular_perturbation",
    "relation",
    "root",
    "rsolve",
    "series",
    "solve",
    "stratified_series",
    "sum",
    "truncate",
]


def __dir__() -> list[str]:
    """Expose only the primary namespace to interactive discovery."""
    return sorted(set(__all__) | {"__all__", "__doc__", "__name__", "__package__"})
