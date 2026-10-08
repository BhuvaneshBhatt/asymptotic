"""Reference comparison keeps directional and spherical infinity distinct."""

import pytest
import sympy as sp

from asymptotic.fixed_ray_branch_germs import DirectionalInfinity
from tools.run_reference_corpus_shard import same_infinite_direction


@pytest.mark.parametrize(
    "first,second,expected",
    [
        (DirectionalInfinity(-sp.I), -sp.oo * sp.I, True),
        (-sp.oo * sp.I, DirectionalInfinity(-sp.I), True),
        (DirectionalInfinity(sp.I), sp.oo * sp.I, True),
        (DirectionalInfinity(sp.I), DirectionalInfinity(2 * sp.I), True),
        (DirectionalInfinity(-sp.I), sp.oo * sp.I, False),
        (DirectionalInfinity(-sp.I), sp.zoo, False),
        (DirectionalInfinity(-sp.I), sp.oo, False),
        (DirectionalInfinity(-sp.I), -sp.I, False),
        (DirectionalInfinity(sp.I), sp.oo + sp.I * sp.oo, False),
    ],
)
def test_reference_directions(first, second, expected):
    assert same_infinite_direction(first, second) is expected
