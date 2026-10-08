"""Independent checks of parameter-stratified reference contracts."""

import json
from pathlib import Path

import sympy as s

from asymptotic.reference_normalization import scalar_reference_namespace

ROOT = Path(__file__).resolve().parents[3]
ROWS = {
    r["id"][-4:]: r
    for r in json.loads(
        (ROOT / "tests/data/univariate_limit_reference_cases.json").read_text()
    )
}


def test_atan_zero_strata_retain_the_original_source_and_direction():
    for id in ("0763", "0764"):
        row = ROWS[id]
        assert row["expression"] == "atan(x/eps)"
        assert row["expected_kind"] == "stratified"
        assert [c["condition"] for c in row["parameter_cases"]] == [
            "Eq(x,0)",
            "Ne(x,0)",
        ]
        zero = row["parameter_cases"][0]
        assert zero["expected"] == "0" and zero["parameter_substitutions"] == {"x": "0"}
    assert ROWS["0763"]["direction"] == "+"


def test_exp_power_strata_cover_finite_complex_parameters():
    row = ROWS["3416"]
    ns = scalar_reference_namespace()
    conditions = [s.sympify(c["condition"], locals=ns) for c in row["parameter_cases"]]
    n = s.Symbol("n")
    for z in (-3, -1, 0, -3 + s.I, -1 + s.I, s.I, 2 + s.I):
        assert sum(bool(c.subs(n, z)) for c in conditions) == 1
    assert len(conditions) == 5 and row["parameter_domain"] == "finite complex n"
    x = s.Symbol("x", positive=True)
    assert s.exp(-x * x ** (-1)) == s.exp(-1)


def test_nonreal_power_phase_sequences_are_exact_and_pole_free():
    k = s.Symbol("k", positive=True, integer=True)
    # n=-1+i has unit amplitude; n=i has growing amplitude.
    for realpart in (0, 1):
        for theta, phase in ((0, 1), (s.pi, -1)):
            logx = 2 * s.pi * k + theta
            assert s.simplify(s.exp(s.I * logx) - phase) == 0
            if realpart == 0:
                assert s.exp(-phase) == (s.exp(-1) if phase == 1 else s.E)
            assert s.exp(logx).is_positive is True


def test_decimal_reference_correction_has_an_exact_nonzero_offset():
    row = ROWS["1032"]
    assert row["expected_kind"] == "dne"
    constant = s.Rational(s.Float("0.909297426825682"))
    lower = sum(
        (-1) ** k * s.Rational(2) ** (2 * k + 1) / s.factorial(2 * k + 1)
        for k in range(40)
    )
    upper = lower + s.Rational(2) ** 81 / s.factorial(81)
    assert lower < upper < constant
    assert s.sympify(row["expression"]) == s.sympify(
        "(sin(h + 2.0) - 0.909297426825682)/h"
    )


def test_unresolved_convention_and_binding_silently_rewritten():
    assert ROWS["0559"]["expected"] == "0"
    assert ROWS["3479"]["expected"] == "0"
    assert ROWS["0777"]["assumptions"] == "Eq(a, 0)"
    assert ROWS["1088"]["point"] == "x/3"
    assert all(
        ROWS[id].get("executable", True) for id in ("0559", "3479", "0777", "1088")
    )


def test_rationalize_translation_preserves_source_and_expectation():
    row = ROWS["1365"]
    ns = scalar_reference_namespace()
    target = s.sympify(row["point"], locals=ns)
    expected = s.sympify(row["expected"], locals=ns)
    assert target - s.sqrt(s.pi) == expected
    assert "N(sqrt(pi), 100)" in row["point"]
