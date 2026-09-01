"""Automatic Newton--Puiseux handoff at a turning point."""

import sympy as sp

from asymptotic import implicit_asymptotic
from asymptotic.implicit import implicit_singularity_profile


def main() -> None:
    x, y = sp.symbols("x y", positive=True)
    profile = implicit_singularity_profile(y**2 - x, y, x)
    branches = implicit_asymptotic(y**2 - x, y, x, terms=3)
    print("profile:", profile)
    print("branch count:", len(branches))
    for branch in branches:
        print(branch.truncate())


if __name__ == "__main__":
    main()
