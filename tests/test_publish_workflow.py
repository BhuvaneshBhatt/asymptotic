import re
from pathlib import Path

from .suite_layout import SHARDS


def test_publish_workflow_uses_trusted_publishing_after_release_gates():
    text = Path(".github/workflows/publish.yml").read_text()

    assert "tags:" in text and '"v*"' in text
    assert "Require tag to match package version" in text
    assert "python tools/run_test_shard.py" in text
    assert 'python-version: ["3.10", "3.11", "3.12", "3.13"]' in text
    assert "python -m twine check dist/*" in text
    assert "tests/test_installed_wheel.py" in text
    assert "pypa/gh-action-pypi-publish@release/v1" in text
    assert "id-token: write" in text
    assert "environment:" in text and "name: pypi" in text


def test_public_sum_signature_has_no_internal_normalization_hook():
    import inspect

    from asymptotic import asymptotic_sum

    assert "_stirling_normalization" not in inspect.signature(asymptotic_sum).parameters


def test_publish_workflow_shards_match_release_layout():
    text = Path(".github/workflows/publish.yml").read_text()
    block = text.split("shard:", 1)[1].split("runs-on:", 1)[0]
    declared = set(re.findall(r"^\s+- ([a-z][a-z-]+)\s*$", block, re.MULTILINE))
    expected = set(SHARDS) - {"artifact"}
    assert declared == expected
