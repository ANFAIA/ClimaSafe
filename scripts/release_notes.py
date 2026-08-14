#!/usr/bin/env python3
"""
release_notes.py — genera el changelog / release notes de un rango de commits
(Conventional Commits), replicando la lógica de agrupación del GitAgent del
arnés (agents/agents/git_agent.py) en stdlib puro, para que CI no dependa del
arnés.

Uso:
    release_notes.py <range> <version> [--release-notes]

  <range>    rango git: 'v0.0.66..HEAD' o 'HEAD' (primer release, sin tags)
  <version>  versión a etiquetar, p. ej. '0.0.67' (para la cabecera v0.0.67)
  --release-notes  antepone la cabecera de release notes; sin él emite solo
             la sección de changelog (para CHANGELOG.md).

El markdown sale por stdout. Requiere `git` en el PATH.
"""

from __future__ import annotations

import re
import subprocess
import sys
from datetime import date

_CONVENTIONAL_COMMIT_RE = re.compile(
    r"^(?P<type>feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)"
    r"(?P<scope>\([^)]+\))?(?P<breaking>!)?: (?P<subject>.+)$"
)

# Mismas etiquetas de sección que ya usa el GitAgent del arnés.
SECTION_TITLES = {
    "feat": "### Añadido",
    "fix": "### Corrección de bugs",
    "docs": "### Documentación",
    "refactor": "### Refactorización",
    "perf": "### Rendimiento",
    "test": "### Tests",
    "build": "### Build / dependencias",
    "ci": "### CI",
    "chore": "### Mantenimiento",
    "revert": "### Reversiones",
}


def _parse_conventional(message: str) -> tuple[str, str] | None:
    match = _CONVENTIONAL_COMMIT_RE.match(message.strip().splitlines()[0])
    if not match:
        return None
    return match.group("type"), match.group("subject")


def git_log(range_: str) -> list[str]:
    out = subprocess.run(
        ["git", "log", range_, "--pretty=format:%s", "--date=short"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in out.stdout.splitlines() if line.strip()]


def build_markdown(subjects: list[str]) -> str:
    grouped: dict[str, list[str]] = {}
    unclassified: list[str] = []
    for subject in subjects:
        parsed = _parse_conventional(subject)
        if parsed:
            grouped.setdefault(parsed[0], []).append(parsed[1])
        else:
            unclassified.append(subject)

    lines: list[str] = []
    for commit_type, title in SECTION_TITLES.items():
        if commit_type in grouped:
            lines.append(title)
            lines.append("")
            lines.extend(f"- {msg}" for msg in grouped[commit_type])
            lines.append("")
    if unclassified:
        lines.append("### Otros")
        lines.append("")
        lines.extend(f"- {msg}" for msg in unclassified)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    range_, version = sys.argv[1], sys.argv[2]
    release_notes_mode = "--release-notes" in sys.argv

    subjects = git_log(range_)
    if not subjects:
        print(f"Sin commits en el rango {range_} — nada que publicar.")
        return 0

    body = build_markdown(subjects)
    stamp = date.today().isoformat()
    if release_notes_mode:
        header = f"# Release v{version} — {stamp}\n\n"
    else:
        header = f"## [v{version}] — {stamp}\n\n"
    sys.stdout.write(header + body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
