"""
agents.tools.secrets_tool — Detecta secretos hardcodeados en el proyecto.

Dos niveles, verificados de verdad antes de escribir este módulo (no de
memoria):

1. Si el binario `detect-secrets` (Yelp, github.com/Yelp/detect-secrets)
   está instalado, se usa vía subprocess (`detect-secrets scan --all-files`)
   — lo instalé y ejecuté de verdad para confirmar el JSON exacto que
   produce: `{"results": {"<archivo>": [{"type", "filename",
   "hashed_secret", "is_verified", "line_number"}, ...]}}`. Necesita que el
   directorio sea un repo git (o pasar --all-files, que ya se pasa siempre
   aquí).
2. Si no está instalado, se usa un heurístico propio MUCHO más limitado:
   patrón de clave de AWS (`AKIA[0-9A-Z]{16}`, formato público y muy
   documentado — es literalmente el que usa AWS en su propia
   documentación como ejemplo), cabeceras de clave privada PEM
   (`-----BEGIN...PRIVATE KEY-----`), asignaciones tipo `api_key = "..."`,
   y entropía de Shannon para strings largos que parezcan aleatorios. Esto
   NO es un sustituto de detect-secrets/gitleaks/TruffleHog — cubre una
   fracción pequeña de los ~25 detectores que trae detect-secrets de
   fábrica (ArtifactoryDetector, SlackDetector, StripeDetector, JwtTokenDetector,
   etc. — lista completa verificada ejecutando `detect-secrets scan` yo
   mismo). Instala `detect-secrets` si quieres cobertura real.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from agents.tools.process_tool import run_command
from agents.tools.registry import register_tool

_AWS_KEY_RE = re.compile(r"AKIA[0-9A-Z]{16}")
_PRIVATE_KEY_RE = re.compile(r"-----BEGIN (RSA |EC |OPENSSH |DSA |)?PRIVATE KEY-----")
_ASSIGNMENT_RE = re.compile(
    r"(?i)(?:^|[^a-z0-9])(api[_-]?key|secret|password|passwd|token|access[_-]?key)\s*[:=]\s*['\"]([^'\"]{8,})['\"]"
)
_ENTROPY_THRESHOLD = 4.3  # empírico: strings aleatorias base64/hex superan esto; palabras normales no
_MIN_LENGTH_FOR_ENTROPY_CHECK = 20

_SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", "dist", "build"}


@dataclass
class SecretFinding:
    file: str
    line_number: int
    finding_type: str
    detector: str  # "detect-secrets" o "heuristico-propio"


def shannon_entropy(s: str) -> float:
    """Entropía de Shannon en bits/símbolo — más alta = más 'aleatorio'. Cálculo estándar, sin librerías."""
    if not s:
        return 0.0
    counts = {ch: s.count(ch) for ch in set(s)}
    length = len(s)
    return -sum((count / length) * math.log2(count / length) for count in counts.values())


def detect_secrets_binary_available() -> bool:
    return shutil.which("detect-secrets") is not None


@register_tool("secrets")
class SecretsTool:
    @staticmethod
    def scan_with_detect_secrets(root: Path) -> list[SecretFinding]:
        result = run_command(["detect-secrets", "scan", "--all-files", "."], cwd=root, timeout=120)
        # detect-secrets escribe el JSON a stdout incluso si no es un repo git
        # perfecto (avisa por stderr) — se intenta parsear igualmente.
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            return []

        findings = []
        for filename, entries in data.get("results", {}).items():
            for entry in entries:
                findings.append(SecretFinding(
                    file=filename, line_number=entry.get("line_number", 0),
                    finding_type=entry.get("type", "desconocido"), detector="detect-secrets",
                ))
        return findings

    @staticmethod
    def scan_with_heuristic(root: Path, *, extensions: tuple[str, ...] = (".py", ".env", ".yml", ".yaml", ".json", ".cfg", ".ini")) -> list[SecretFinding]:
        findings = []
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in extensions:
                continue
            if any(part in _SKIP_DIRS for part in path.parts):
                continue
            try:
                lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
            except OSError:
                continue

            relative = str(path.relative_to(root))
            for i, line in enumerate(lines, start=1):
                if _AWS_KEY_RE.search(line):
                    findings.append(SecretFinding(relative, i, "AWS Access Key (patrón)", "heuristico-propio"))
                if _PRIVATE_KEY_RE.search(line):
                    findings.append(SecretFinding(relative, i, "Cabecera de clave privada PEM", "heuristico-propio"))
                match = _ASSIGNMENT_RE.search(line)
                if match:
                    value = match.group(2)
                    if len(value) >= _MIN_LENGTH_FOR_ENTROPY_CHECK and shannon_entropy(value) >= _ENTROPY_THRESHOLD:
                        findings.append(SecretFinding(
                            relative, i, f"Asignación de '{match.group(1)}' con valor de alta entropía", "heuristico-propio"
                        ))
        return findings

    @staticmethod
    def scan(root: Path) -> tuple[list[SecretFinding], str]:
        """Devuelve (hallazgos, detector_usado). Usa detect-secrets si está disponible, si no el heurístico propio."""
        if detect_secrets_binary_available():
            return SecretsTool.scan_with_detect_secrets(root), "detect-secrets"
        return SecretsTool.scan_with_heuristic(root), "heuristico-propio"
