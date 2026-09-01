"""Every root API must appear in a behavioral test outside import contracts."""

from pathlib import Path

import asymptotic


def test_every_root_api_is_named_in_a_behavioral_test():
    test_dir = Path(__file__).parent
    behavioral_text = "\n".join(
        path.read_text()
        for path in test_dir.glob("test_*.py")
        if path.name
        not in {
            "test_public_api_contract.py",
            "test_public_api_behavior_coverage.py",
        }
    )
    missing = [
        name
        for name in sorted(asymptotic.__all__)
        if name != "__version__" and name not in behavioral_text
    ]
    assert not missing, f"root APIs without a behavioral test: {missing}"
