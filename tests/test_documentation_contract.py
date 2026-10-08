"""Structural checks for user-facing documentation."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


def test_readme_doc_links_are_absolute():
    text = README.read_text()
    links = re.findall(r"\[[^]]+\]\(([^)]+)\)", text)
    doc_links = [link for link in links if "docs/" in link or link.endswith(".md")]
    assert doc_links
    assert all(
        link.startswith("https://github.com/BhuvaneshBhatt/asymptotic/")
        for link in doc_links
    )


def test_primary_guides_exist():
    expected = {
        "index.md",
        "getting-started.md",
        "limits.md",
        "multivariate-limits.md",
        "certification.md",
        "capabilities.md",
        "testing.md",
    }
    existing = {path.name for path in (ROOT / "docs").glob("*.md")}
    assert expected <= existing


def test_readme_explains_unknown():
    text = README.read_text()
    assert "UNKNOWN" in text
    assert "PROVED" in text
    assert "DOES_NOT_EXIST" in text
