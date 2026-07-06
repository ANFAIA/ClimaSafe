"""
agents.tools.git_tool — Envoltorio sobre el binario `git`.

Se usa `git` por subprocess a propósito, no GitPython: es una dependencia
menos y cubre el 100% de lo que necesita `GitAgent` (diff, log, status).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from agents.tools.process_tool import ProcessResult, run_command
from agents.tools.registry import register_tool

_CONVENTIONAL_COMMIT_RE = re.compile(
    r"^(?P<type>feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)"
    r"(?P<scope>\([^)]+\))?(?P<breaking>!)?: (?P<subject>.+)$"
)

# Heurística de clasificación por ruta de archivo tocado -> tipo de Conventional Commit.
# No es infalible (un cambio en tests/ podría ser un "fix" real), pero da una
# sugerencia razonable de partida — el commit final lo revisa una persona.
_PATH_TYPE_HINTS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^tests?/"), "test"),
    (re.compile(r"^docs?/"), "docs"),
    (re.compile(r"README|CHANGELOG"), "docs"),
    (re.compile(r"^\.github/"), "ci"),
    (re.compile(r"Dockerfile|docker-compose"), "build"),
    (re.compile(r"pyproject\.toml|uv\.lock|requirements"), "build"),
]


@register_tool("git")
@dataclass
class GitTool:
    repo_root: Path

    def _git(self, *args: str, check: bool = False) -> ProcessResult:
        return run_command(["git", *args], cwd=self.repo_root, check=check)

    def is_repo(self) -> bool:
        return self._git("rev-parse", "--is-inside-work-tree").ok

    def status_porcelain(self) -> list[tuple[str, str]]:
        """Devuelve [(código_estado, ruta), ...] tal y como `git status --porcelain`."""
        result = self._git("status", "--porcelain")
        entries = []
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            code, path = line[:2].strip(), line[3:].strip()
            entries.append((code, path))
        return entries

    def diff(self, *, staged: bool = False, name_only: bool = False) -> str:
        args = ["diff"]
        if staged:
            args.append("--staged")
        if name_only:
            args.append("--name-only")
        return self._git(*args).stdout

    def diff_stat(self, *, staged: bool = False) -> str:
        args = ["diff", "--stat"]
        if staged:
            args.insert(1, "--staged")
        return self._git(*args).stdout

    def changed_files(self, *, staged: bool = False) -> list[str]:
        raw = self.diff(staged=staged, name_only=True)
        files = [f for f in raw.splitlines() if f.strip()]
        if not files and staged:
            # nada en staging: mirar también working tree, es lo que suele querer el usuario
            files = [f for f in self.diff(staged=False, name_only=True).splitlines() if f.strip()]
        return files

    def log(self, *, max_count: int = 20, since_tag: str | None = None) -> list[dict[str, str]]:
        """Devuelve una lista de commits como dicts {hash, subject, author, date}."""
        rev_range = f"{since_tag}..HEAD" if since_tag else "HEAD"
        fmt = "%H%x1f%s%x1f%an%x1f%ad"
        result = self._git(
            "log", rev_range, f"--max-count={max_count}", f"--pretty=format:{fmt}", "--date=short"
        )
        commits = []
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            parts = line.split("\x1f")
            if len(parts) == 4:
                commits.append(
                    {"hash": parts[0], "subject": parts[1], "author": parts[2], "date": parts[3]}
                )
        return commits

    def last_tag(self) -> str | None:
        result = self._git("describe", "--tags", "--abbrev=0")
        return result.stdout.strip() or None if result.ok else None

    def current_branch(self) -> str:
        return self._git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()

    def commit(self, message: str) -> ProcessResult:
        return self._git("commit", "-m", message)

    def add(self, *paths: str) -> ProcessResult:
        return self._git("add", *paths)

    def create_tag(self, tag: str, *, message: str | None = None) -> ProcessResult:
        if message:
            return self._git("tag", "-a", tag, "-m", message)
        return self._git("tag", tag)

    def tag_exists(self, tag: str) -> bool:
        return self._git("rev-parse", tag).ok

    @staticmethod
    def parse_conventional_commit(message: str) -> dict[str, str] | None:
        """Extrae {type, scope, breaking, subject} de un mensaje Conventional Commits, o None."""
        match = _CONVENTIONAL_COMMIT_RE.match(message.strip().splitlines()[0])
        if not match:
            return None
        data = match.groupdict()
        data["scope"] = (data["scope"] or "").strip("()")
        data["breaking"] = bool(data["breaking"])
        return data

    @staticmethod
    def guess_commit_type(changed_files: list[str]) -> str:
        """Heurística: infiere el tipo Conventional Commit dominante a partir de las rutas tocadas."""
        if not changed_files:
            return "chore"
        counts: dict[str, int] = {}
        for path in changed_files:
            matched = False
            for pattern, commit_type in _PATH_TYPE_HINTS:
                if pattern.search(path):
                    counts[commit_type] = counts.get(commit_type, 0) + 1
                    matched = True
                    break
            if not matched:
                counts["feat"] = counts.get("feat", 0) + 1
        return max(counts, key=counts.get)
