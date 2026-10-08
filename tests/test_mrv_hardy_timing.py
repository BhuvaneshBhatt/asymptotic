import sympy as sp

from asymptotic import root
from asymptotic.mrv_profile import mrv_hardy_timing


def test_mrv_hardy_timing_attributes_newton_lifting_stages():
    y, n = sp.symbols("y n", positive=True)
    with mrv_hardy_timing() as timing:
        root_result = root(
            y**2 - (1 + 1 / n) * sp.exp(2 * n),
            y,
            parameter=n,
            terms=4,
            branch=1,
            return_result=True,
        )
    expected = sp.exp(n) * (1 + 1 / (2 * n) - 1 / (8 * n**2) + 1 / (16 * n**3))
    assert sp.simplify(root_result - expected) == 0
    snapshot = timing.snapshot()
    assert snapshot["recursive_calls"] >= 4
    assert snapshot["max_depth"] == 4
    for stage in (
        "zerotest",
        "simplification",
        "series_and_balance",
        "recursive_root_construction",
    ):
        assert snapshot["calls"][stage] > 0
        assert snapshot["seconds"][stage] >= 0
