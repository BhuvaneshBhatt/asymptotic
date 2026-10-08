import runpy
from pathlib import Path


def test_uniform_source_coherence_audit():
    module = runpy.run_path(
        str(Path(__file__).parents[1] / "tools/audit_uniform_asymptotics.py")
    )
    findings = module["source_convention_audit"](Path(__file__).parents[1])
    assert findings
    assert all(item.passed for item in findings)
