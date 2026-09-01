"""README documentation links must survive rendering on PyPI."""

import re
from pathlib import Path

GITHUB_DOC_PREFIX = "https://github.com/BhuvaneshBhatt/asymptotic/blob/main/"


def test_readme_documentation_links_are_absolute_and_resolve_in_repository():
    readme = Path("README.md").read_text()
    links = re.findall(r"\[[^]]+\]\(([^)]+)\)", readme)
    relative_docs = [link for link in links if link.startswith(("docs/", "examples/"))]
    assert relative_docs == []

    repository_links = [link for link in links if link.startswith(GITHUB_DOC_PREFIX)]
    assert repository_links
    missing = []
    for link in repository_links:
        relative = link.removeprefix(GITHUB_DOC_PREFIX).split("#", 1)[0]
        if not Path(relative).exists():
            missing.append(relative)
    assert not missing, f"README links point to missing repository files: {missing}"
