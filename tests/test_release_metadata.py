"""Keep package, runtime, CI, and user-facing release metadata synchronized."""

import tomllib
from pathlib import Path

import asymptotic

ROOT = Path(__file__).resolve().parents[1]


def test_release_version_is_consistent():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["version"] == "0.2.0"
    assert asymptotic.__version__ == project["version"]


def test_supported_python_floor_is_consistent():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["requires-python"] == ">=3.11"
    assert "Python 3.11" in (ROOT / "README.md").read_text()
    for workflow in ("tests.yml", "publish.yml"):
        text = (ROOT / ".github" / "workflows" / workflow).read_text()
        assert '["3.11", "3.12", "3.13", "3.14"]' in text
        assert '"3.10"' not in text


def test_semialg_is_a_required_dependency():
    dependencies = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
        "dependencies"
    ]
    assert any(item.startswith("semialg>=") for item in dependencies)


def test_ode_dependency_uses_distribution_version():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["optional-dependencies"]["ode"] == ["odeanalysis>=0.1.0"]
