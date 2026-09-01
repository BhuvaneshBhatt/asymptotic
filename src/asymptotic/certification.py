"""Uniform replay checks for theorem-backed public results."""

from __future__ import annotations


def _replay_object(obj: object) -> bool | None:
    replay = getattr(obj, "replay", None)
    if callable(replay):
        try:
            value = replay()
        except (TypeError, ValueError, NotImplementedError):
            return False
        return value if value in (True, False, None) else bool(value)
    certificate = getattr(obj, "certificate", None)
    if certificate is not None:
        return _replay_object(certificate)
    return None


def replay_certification(result: object) -> bool | None:
    """Replay the evidence behind a public result's certification status.

    ``EXACT`` results need no theorem certificate and return ``True``.
    ``CERTIFIED`` results must carry a replayable certificate directly or be
    derived solely from replayable certified sources with a certified
    remainder.  Formal and unknown results return ``None``.
    """

    status = getattr(result, "status", None)
    if status == "EXACT":
        return True
    if status != "CERTIFIED":
        return None

    direct = _replay_object(result)
    if direct is not None:
        return direct

    probability = getattr(result, "probability", None)
    if probability is not None:
        return replay_certification(probability)

    sources = tuple(getattr(result, "sources", ()) or ())
    if sources:
        checks = []
        for source in sources:
            source_status = getattr(source, "status", None)
            if source_status is not None:
                checks.append(replay_certification(source))
            else:
                checks.append(_replay_object(source))
        if checks and all(check is True for check in checks):
            remainder = getattr(result, "remainder", None)
            return remainder is None or bool(getattr(remainder, "is_certified", False))
        return False
    return False
