"""Optional python-flint helpers.

Arb is excellent for certified numerical enclosures, but it does not replace the
symbolic zero-equivalence and comparability decisions required by multiseries.
The core package therefore does not depend on python-flint.  This module exposes
feature detection for certified constant/sign fallback support
without changing the public asymptotic API.
"""

from __future__ import annotations

try:
    import flint  # type: ignore
except ImportError:  # pragma: no cover - environment dependent
    flint = None


def flint_available() -> bool:
    return flint is not None
