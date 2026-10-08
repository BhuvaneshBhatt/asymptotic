"""Contracts for the sequential isolated-suite coordinator."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "run_isolated_suite.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("isolated_suite_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_isolated_runner_uses_one_blocking_child_at_a_time():
    text = RUNNER.read_text()
    assert "subprocess.run(" in text
    assert "Popen(" not in text
    assert "concurrency=1" in text


def test_isolated_runner_disables_plugin_autoload_and_caller_addopts():
    runner = _load_runner()
    env = runner._environment()
    assert env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert "PYTEST_ADDOPTS" not in env


def test_isolated_runner_module_mapping_parameterized_nodeid():
    runner = _load_runner()
    nodeid = "tests/test_example.py::test_case[value::with-colons]"
    assert runner._module(nodeid) == "tests/test_example.py"


def test_isolated_runner_slug_is_filesystem_safe_and_bounded():
    runner = _load_runner()
    slug = runner._slug("tests/test_x.py::test_a[a/b c]")
    assert "/" not in slug
    assert " " not in slug
    assert len(slug) <= 180
