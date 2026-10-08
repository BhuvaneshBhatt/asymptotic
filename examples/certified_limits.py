"""Compare simultaneous existence, path dependence and a signed real pole."""

import sympy as sp

from asymptotic import limit, one_sided_limit
from asymptotic.limit_models import LimitStatus


def main():
    x, y = sp.symbols("x y", real=True)
    assert limit(x * y / sp.sqrt(x * x + y * y), (x, y), (0, 0)) == 0
    result = limit(x * y / (x * x + y * y), (x, y), (0, 0), return_result=True)
    assert result.status is LimitStatus.DOES_NOT_EXIST
    assert len({item.value for item in result.evidence}) >= 2
    assert one_sided_limit(1 / x, x, 0, direction="+") == sp.oo
    assert one_sided_limit(1 / x, x, 0, direction="-") == -sp.oo
    print(
        "Simultaneous limit: 0; path-dependent expression: no limit; signed poles verified."
    )


if __name__ == "__main__":
    main()
