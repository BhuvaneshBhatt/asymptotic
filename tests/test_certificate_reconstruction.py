"""Replayable certificates survive ordinary object reconstruction."""

import pickle

import sympy as sp

from asymptotic.remainder_theorems import certify_green_inverse_operator_remainder


def test_green_certificate_survives_pickle_round_trip_and_replays():
    x = sp.symbols("x", positive=True)
    delta = sp.Function("delta")
    operator = sp.diff(delta(x), x, 2) + sp.diff(delta(x), x) / x - delta(x)
    cert, green = certify_green_inverse_operator_remainder(
        sp.exp(-x / 2), operator, delta, x, sp.oo
    )
    assert cert.certified and green is not None
    restored = pickle.loads(pickle.dumps(green))
    assert restored.replay_asymptotic(x) is True
    assert restored.limiting_coefficients == green.limiting_coefficients
