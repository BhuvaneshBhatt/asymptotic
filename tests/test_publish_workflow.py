import re
from pathlib import Path

from .suite_layout import SHARDS


def test_publish_workflow_uses_trusted_publishing_after_release_gates():
    text = Path(".github/workflows/publish.yml").read_text()

    assert "tags:" in text and '"v*"' in text
    assert "Require tag to match package version" in text
    assert "python tools/run_test_shard.py" in text
    assert 'python-version: ["3.11", "3.12", "3.13", "3.14"]' in text
    assert "python -m twine check dist/*" in text
    assert "tests/test_installed_wheel.py" in text
    assert "pypa/gh-action-pypi-publish@release/v1" in text
    assert "id-token: write" in text
    assert "environment:" in text and "name: pypi" in text


def test_public_sum_signature_has_no_internal_normalization_hook():
    import inspect

    from asymptotic import sum

    assert "_stirling_normalization" not in inspect.signature(sum).parameters


def test_publish_workflow_shards_match_release_layout():
    text = Path(".github/workflows/publish.yml").read_text()
    block = text.split("shard:", 1)[1].split("runs-on:", 1)[0]
    declared = set(re.findall(r"^\s+- ([a-z][a-z-]+)\s*$", block, re.MULTILINE))
    expected = set(SHARDS) - {"artifact"}
    assert declared == expected


def test_regular_workflow_shards_match_layout():
    text = Path(".github/workflows/tests.yml").read_text()
    block = text.split("shard:", 1)[1].split("runs-on:", 1)[0]
    declared = set(re.findall(r"^\s+- ([a-z][a-z-]+)\s*$", block, re.MULTILINE))
    assert declared == set(SHARDS) - {"artifact"}


def test_workflows_check_formatting():
    for name in ("tests.yml", "publish.yml"):
        text = (Path(".github/workflows") / name).read_text()
        assert "ruff format --check src tests benchmarks tools examples" in text


def test_publish_waits_for_backend_and_wheel_checks():
    text = Path(".github/workflows/publish.yml").read_text()
    assert (
        "needs: [static, shards, optional-backends, reference-corpus-aggregate]" in text
    )
    assert "needs: [build, wheel-validation]" in text
    block = text.split("  wheel-validation:", 1)[1].split("  publish:", 1)[0]
    assert 'python-version: ["3.11", "3.12", "3.13", "3.14"]' in block
    assert "python -m pip install dist/*.whl pytest" in block
