from __future__ import annotations

from pathlib import Path

from agents.tools.code_analysis_tool import CodeAnalysisTool


def test_detects_long_function(tmp_path: Path):
    body_lines = "\n".join(f"    x{i} = {i}" for i in range(70))
    source = f"def big_function():\n{body_lines}\n    return x0\n"
    path = tmp_path / "mod.py"
    path.write_text(source)

    smells, functions = CodeAnalysisTool.analyze_file(path)
    assert any(s.kind == "long_function" for s in smells)
    assert functions[0].n_lines > CodeAnalysisTool.MAX_FUNCTION_LINES


def test_detects_too_many_args(tmp_path: Path):
    path = tmp_path / "mod.py"
    path.write_text("def f(a, b, c, d, e, f, g):\n    return a\n")
    smells, _ = CodeAnalysisTool.analyze_file(path)
    assert any(s.kind == "too_many_args" for s in smells)


def test_detects_bare_except(tmp_path: Path):
    path = tmp_path / "mod.py"
    path.write_text("try:\n    pass\nexcept:\n    pass\n")
    smells, _ = CodeAnalysisTool.analyze_file(path)
    assert any(s.kind == "bare_except" for s in smells)


def test_finds_structurally_duplicated_functions(tmp_path: Path):
    path = tmp_path / "mod.py"
    path.write_text(
        "def add_a(x, y):\n"
        "    total = x + y\n"
        "    total = total + 1\n"
        "    total = total + 1\n"
        "    total = total + 1\n"
        "    return total\n"
        "\n"
        "def add_b(m, n):\n"
        "    total = m + n\n"
        "    total = total + 1\n"
        "    total = total + 1\n"
        "    total = total + 1\n"
        "    return total\n"
    )
    _, functions = CodeAnalysisTool.analyze_file(path)
    duplicates = CodeAnalysisTool.find_duplicates(functions, min_lines=3)
    assert len(duplicates) == 1
    assert {f.name for f in duplicates[0]} == {"add_a", "add_b"}


def test_parse_handles_syntax_error_gracefully(tmp_path: Path):
    path = tmp_path / "broken.py"
    path.write_text("def f(:\n")
    smells, functions = CodeAnalysisTool.analyze_file(path)
    assert smells == []
    assert functions == []
