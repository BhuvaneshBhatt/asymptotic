import json
from pathlib import Path

import sympy as s

from asymptotic.reference_normalization import scalar_reference_equal

ROOT = Path(__file__).resolve().parents[3]
ROWS = json.loads(
    (ROOT / "tests/data/univariate_limit_reference_cases.json").read_text()
)


def test_mapped_harmonic_orders():
    applications = ROWS[90]["applications"]
    assert len(applications) == 20
    orders = [s.sympify(c["expression"]).args[1] for c in applications]
    assert orders == list(range(3, 42, 2))
    assert [s.sympify(c["expected"]) for c in applications] == [
        s.zeta(order) for order in orders
    ]
    assert len(ROWS) == 3536 and sum(r["multiplicity"] for r in ROWS) == 5128


def test_application_directions():
    assert [c["direction"] for c in ROWS[28]["applications"]] == ["+", "-"]
    assert ROWS[1075]["applications"][0]["direction"] == "-"
    for row in ROWS:
        if row.get("executable", True):
            if not row.get("applications"):
                assert "unbound_argument(" not in row["expression"]
            assert all(
                "unbound_argument(" not in child["expression"]
                for child in row.get("applications", [])
            )


def test_directed_infinity_comparison_preserves_avoids_zoo():
    D = s.Function("DirectionalInfinity")
    assert scalar_reference_equal(
        s.oo * s.sign(-2 - 2 * s.I), D((-1 - s.I) / s.sqrt(2))
    )
    assert scalar_reference_equal(s.oo, D(7))
    assert scalar_reference_equal(-s.oo, D(-7))
    theta = s.sqrt(2) * s.pi / 2
    assert scalar_reference_equal(
        -s.oo * s.sign((-s.Rational(1, 2)) ** (1 - s.sqrt(2) / 2)),
        D((s.cot(theta) - s.I) / s.sqrt(s.cot(theta) ** 2 + 1)),
    )
    assert not scalar_reference_equal(s.oo, D(s.I))
    assert not scalar_reference_equal(s.zoo, D(s.I))
    assert not scalar_reference_equal(s.oo, D(0))
    assert not scalar_reference_equal(s.oo, D(s.Symbol("d")))
