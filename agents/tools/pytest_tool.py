"""
agents.tools.pytest_tool — Ejecuta pytest y parsea sus reportes.

Formatos verificados generándolos de verdad antes de escribir este módulo
(no son una suposición sobre la documentación):
- JUnit XML (`--junitxml=`): `<testsuites><testsuite errors= failures=
  skipped= tests=><testcase classname= name=><failure message="...">
  traceback</failure></testcase></testsuite></testsuites>`.
- Cobertura JSON (`pytest-cov` con `--cov-report=json`): nivel superior
  `{"meta", "files", "totals"}`; cada entrada de `files` tiene `summary`
  con `percent_covered`, `covered_lines`, `num_statements`, `missing_lines`.

Ambos usan solo dependencias que YA están en pyproject.toml (`pytest`,
`pytest-cov`) — no se añade nada nuevo.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from agents.tools.process_tool import ProcessResult, run_command
from agents.tools.registry import register_tool


@dataclass
class TestFailure:
    classname: str
    name: str
    message: str


@dataclass
class TestRunSummary:
    total: int
    failures: int
    errors: int
    skipped: int
    passed: int
    duration_seconds: float
    failed_tests: list[TestFailure] = field(default_factory=list)


@register_tool("pytest")
class PytestTool:
    @staticmethod
    def run(
        root: Path, *, path: str = "tests/", markers: str | None = None, junit_xml_path: Path | None = None,
        timeout: int = 600,
    ) -> ProcessResult:
        args = ["uv", "run", "pytest", path, "-v"]
        if markers:
            args += ["-m", markers]
        if junit_xml_path:
            args.append(f"--junitxml={junit_xml_path}")
        return run_command(args, cwd=root, timeout=timeout)

    @staticmethod
    def parse_junit_xml(path: Path) -> TestRunSummary:
        tree = ET.parse(path)
        root = tree.getroot()
        # pytest anida <testsuite> dentro de <testsuites>, pero por robustez
        # se busca el primer <testsuite> esté donde esté (raíz o anidado).
        suite = root if root.tag == "testsuite" else root.find("testsuite")
        if suite is None:
            raise ValueError(f"'{path}' no contiene ningún <testsuite> reconocible.")

        failed_tests = []
        for testcase in suite.findall("testcase"):
            failure = testcase.find("failure")
            error = testcase.find("error")
            node = failure if failure is not None else error
            if node is not None:
                failed_tests.append(TestFailure(
                    classname=testcase.get("classname", ""),
                    name=testcase.get("name", ""),
                    message=(node.get("message") or "").strip()[:500],
                ))

        total = int(suite.get("tests", 0))
        failures = int(suite.get("failures", 0))
        errors = int(suite.get("errors", 0))
        skipped = int(suite.get("skipped", 0))
        return TestRunSummary(
            total=total, failures=failures, errors=errors, skipped=skipped,
            passed=total - failures - errors - skipped,
            duration_seconds=float(suite.get("time", 0.0)),
            failed_tests=failed_tests,
        )

    @staticmethod
    def run_with_coverage(
        root: Path, *, module: str, path: str = "tests/", coverage_json_path: Path | None = None, timeout: int = 600,
    ) -> ProcessResult:
        cov_path = coverage_json_path or (root / "coverage.json")
        args = ["uv", "run", "pytest", path, f"--cov={module}", f"--cov-report=json:{cov_path}", "-q"]
        return run_command(args, cwd=root, timeout=timeout)

    @staticmethod
    def parse_coverage_json(path: Path) -> dict:
        data = json.loads(path.read_text(encoding="utf-8"))
        per_file = {
            filename: info["summary"]["percent_covered"]
            for filename, info in data.get("files", {}).items()
        }
        return {"total_percent_covered": data.get("totals", {}).get("percent_covered"), "per_file": per_file}
