from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _active_text_files():
    for base in ("src", "tests", "tools", "examples"):
        for path in (ROOT / base).rglob("*"):
            if (
                path.is_file()
                and path.suffix in {".py", ".md"}
                and path.name != Path(__file__).name
            ):
                yield path


def test_retired_semialg_syntactic_closure_name_is_not_used():
    retired = "_syntactically_closed_polynomial_formula"
    offenders = [
        str(path.relative_to(ROOT))
        for path in _active_text_files()
        if retired in path.read_text(errors="ignore")
    ]
    assert offenders == []


def test_retired_direct_limit_rule_name_is_not_used():
    retired = "_apply_direct_limit_rules"
    offenders = [
        str(path.relative_to(ROOT))
        for path in _active_text_files()
        if retired in path.read_text(errors="ignore")
    ]
    assert offenders == []
