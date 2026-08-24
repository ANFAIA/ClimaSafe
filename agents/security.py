"""
agents.security — Security/Policy Layer entre el LLM y las tools.

Toda tool call pasa por `approve_tool_call()`. Ninguna tool se ejecuta
sin pasar por ella. La función verifica:
  1. Catálogo cerrado de tools por agente
  2. Validación de argumentos por tool (tipos y rangos)
  3. Restricción de filesystem por agente
  4. Control de red (lista blanca de dominios)
  5. Protección de secretos
  6. Human-in-the-loop para acciones destructivas
  7. Rate/resource limits
  8. Auditoría de cada intento
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Secret detection patterns
# ---------------------------------------------------------------------------
_SECRET_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)api[_-]?key\s*[:=]\s*['\"]?\S{8,}"),
    re.compile(r"(?i)secret\s*[:=]\s*['\"]?\S{8,}"),
    re.compile(r"(?i)password\s*[:=]\s*['\"]?\S{8,}"),
    re.compile(r"(?i)token\s*[:=]\s*['\"]?\S{8,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9\-._~+/]+=*", re.IGNORECASE),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (RSA |EC |OPENSSH |DSA |)?PRIVATE KEY-----"),
]

# Nombres de claves que siempre se consideran secretos
_SECRET_KEY_NAMES: frozenset[str] = frozenset({
    "api_key", "apikey", "api-key", "secret", "password", "passwd",
    "token", "access_token", "auth_token", "bearer", "private_key",
    "secret_key", "secretkey", "ssh_key",
})

# Archivos sensibles que NUNCA se pueden leer
_SENSITIVE_FILES: frozenset[str] = frozenset({
    ".env", ".env.local", ".env.production", ".env.staging",
    ".env.development", ".env.test",
    ".git/config", ".git/credentials",
})

# ---------------------------------------------------------------------------
# Binary whitelist for process_tool sandbox
# ---------------------------------------------------------------------------
ALLOWED_BINARIES: frozenset[str] = frozenset({
    "git", "python", "python3", "uv", "pip", "ruff", "pytest",
    "make", "bash", "sh", "ls", "cat", "grep", "find", "wc",
    "head", "tail", "diff", "sort", "uniq", "jq",
    "docker", "curl", "wget",
})

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ArgumentSchema:
    """Schema de validación de un argumento de tool."""
    param_name: str
    param_type: type
    required: bool = True
    min_value: float | None = None
    max_value: float | None = None
    allowed_values: list[Any] | None = None
    max_length: int | None = None
    pattern: str | None = None


@dataclass
class ToolCall:
    """Representa una petición de ejecución de tool."""
    agent: str
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ApprovalResult:
    """Resultado de la función de aprobación."""
    approved: bool
    reason: str = ""
    tool: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    checks_passed: list[str] = field(default_factory=list)
    audit_entry: dict[str, Any] = field(default_factory=dict)


@dataclass
class RateLimits:
    """Tope configurable de recursos por petición."""
    max_executions: int = 100
    timeout_seconds: int = 300
    max_memory_mb: int = 1024
    max_cpu_percent: float = 80.0


@dataclass
class AgentPolicy:
    """Política de seguridad de un agente."""
    allowed_tools: set[str]
    argument_schemas: dict[str, list[ArgumentSchema]] = field(default_factory=dict)
    fs_read_dirs: set[str] = field(default_factory=set)
    fs_write_dirs: set[str] = field(default_factory=set)
    network_domains: set[str] = field(default_factory=set)
    destructive_tools: set[str] = field(default_factory=set)
    requires_confirmation: set[str] = field(default_factory=set)
    rate_limits: RateLimits = field(default_factory=RateLimits)


# ---------------------------------------------------------------------------
# Audit log (JSONL append-only)
# ---------------------------------------------------------------------------
_AUDIT_ENTRIES: list[dict[str, Any]] = []
_AUDIT_LOG_PATH: Path | None = None


def _audit_log_path() -> Path:
    global _AUDIT_LOG_PATH
    if _AUDIT_LOG_PATH is None:
        _AUDIT_LOG_PATH = Path("agents/workspace/audit/security_audit.jsonl")
        _AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    return _AUDIT_LOG_PATH


def _record_audit(entry: dict[str, Any]) -> None:
    """Registra una entrada de auditoría. Nunca lanza excepciones."""
    _AUDIT_ENTRIES.append(entry)
    try:
        log_path = _audit_log_path()
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


def get_audit_entries() -> list[dict[str, Any]]:
    """Devuelve las entradas de auditoría en memoria (para tests)."""
    return list(_AUDIT_ENTRIES)


def clear_audit_entries() -> None:
    """Limpia las entradas en memoria (para tests)."""
    _AUDIT_ENTRIES.clear()


# ---------------------------------------------------------------------------
# Approval counters (rate limiting in-memory)
# ---------------------------------------------------------------------------
_call_counters: dict[str, int] = {}


def reset_counters() -> None:
    """Reinicia contadores (para tests)."""
    _call_counters.clear()


def _get_call_count(key: str) -> int:
    return _call_counters.get(key, 0)


def _increment_call_count(key: str) -> None:
    _call_counters[key] = _get_call_count(key) + 1


# ---------------------------------------------------------------------------
# Confirmation store (human-in-the-loop)
# ---------------------------------------------------------------------------
_pending_confirmations: dict[str, ToolCall] = {}


def request_confirmation(call: ToolCall) -> str:
    """
    Registra una petición que requiere confirmación humana.
    Devuelve un confirmation_id que el humano usa para aprobar.
    """
    cid = f"confirm-{call.agent}-{call.tool}-{int(call.timestamp * 1000)}"
    _pending_confirmations[cid] = call
    return cid


def approve_confirmation(confirmation_id: str) -> bool:
    """Aprueba una petición pendiente. Devuelve True si existía."""
    return _pending_confirmations.pop(confirmation_id, None) is not None


def get_pending_confirmations() -> dict[str, ToolCall]:
    """Devuelve las confirmaciones pendientes (para tests)."""
    return dict(_pending_confirmations)


def clear_confirmations() -> None:
    """Limpia confirmaciones pendientes (para tests)."""
    _pending_confirmations.clear()


# ---------------------------------------------------------------------------
# Secret detection
# ---------------------------------------------------------------------------
def _contains_secrets(value: Any) -> bool:
    """Detecta si un valor contiene secretos (tokens, claves, passwords)."""
    if isinstance(value, str):
        for pattern in _SECRET_PATTERNS:
            if pattern.search(value):
                return True
    elif isinstance(value, dict):
        for key, val in value.items():
            # Check if the key name itself suggests a secret
            if key.lower().replace("-", "_") in _SECRET_KEY_NAMES:
                if isinstance(val, str) and len(val) > 0:
                    return True
            if _contains_secrets(val):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_secrets(v) for v in value)
    return False


def _redact_secrets(value: Any) -> Any:
    """Redacta secretos en un valor (para auditoría)."""
    if isinstance(value, str):
        for pattern in _SECRET_PATTERNS:
            if pattern.search(value):
                return "[REDACTED]"
        return value
    elif isinstance(value, dict):
        return {k: _redact_secrets(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return [_redact_secrets(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# Filesystem validation
# ---------------------------------------------------------------------------
def _validate_filesystem(agent: str, args: dict[str, Any], policy: AgentPolicy) -> str | None:
    """Valida que las rutas en args estén dentro de los directorios permitidos."""
    fs_args = ("path", "file", "filepath", "directory", "dir", "target", "source")
    # Determine if this is a write operation
    is_write = args.get("action", "").lower() in ("write", "create", "update", "append") or "content" in args

    for key in fs_args:
        if key in args and isinstance(args[key], str):
            path_str = args[key]
            # Expand ~ to home directory
            if path_str.startswith("~"):
                path_str = str(Path.home() / path_str[2:]) if path_str.startswith("~/") else str(Path.home() / path_str[1:])
            # Resolve relative paths against cwd
            elif not os.path.isabs(path_str):
                path_str = str(Path.cwd() / path_str)
            path_str = str(Path(path_str).resolve())

            # Check sensitive files first - always blocked
            path_name = Path(path_str).name
            path_parts = set(Path(path_str).parts)
            if path_name in _SENSITIVE_FILES or any(s in path_parts for s in _SENSITIVE_FILES):
                return f"Acceso a archivo sensible '{path_name}' bloqueado por seguridad"

            if is_write:
                # Write operations only check write dirs
                in_write = any(
                    str(Path(d).resolve()) in path_str or path_str.startswith(str(Path(d).resolve()) + os.sep)
                    for d in policy.fs_write_dirs if d
                )
                if not in_write:
                    return f"Escritura en '{path_str}' no permitida (write dirs: {policy.fs_write_dirs})"
            else:
                # Read operations check read dirs
                in_read = any(
                    str(Path(d).resolve()) in path_str or path_str.startswith(str(Path(d).resolve()) + os.sep)
                    for d in policy.fs_read_dirs if d
                )
                if not in_read:
                    return f"Lectura de '{path_str}' no permitida (read dirs: {policy.fs_read_dirs})"
    return None


# ---------------------------------------------------------------------------
# Network validation
# ---------------------------------------------------------------------------
def _validate_network(agent: str, args: dict[str, Any], policy: AgentPolicy) -> str | None:
    """Valida que las URLs en args estén en los dominios permitidos."""
    url_args = ("url", "endpoint", "api_url", "base_url", "host")
    for key in url_args:
        if key in args and isinstance(args[key], str):
            url = args[key]
            parsed = urlparse(url)
            if parsed.hostname:
                domain = parsed.hostname.lower()
                if not any(domain == d or domain.endswith("." + d) for d in policy.network_domains if d):
                    return f"Dominio '{domain}' no está en la lista blanca: {policy.network_domains}"
    return None


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------
def _validate_arguments(tool: str, args: dict[str, Any], policy: AgentPolicy) -> str | None:
    """Valida argumentos contra el esquema declarado."""
    schemas = policy.argument_schemas.get(tool, [])
    for schema in schemas:
        value = args.get(schema.param_name)

        # Check required
        if schema.required and value is None:
            return f"Argumento requerido faltante: '{schema.param_name}'"

        if value is None:
            continue

        # Check type
        if not isinstance(value, schema.param_type):
            return f"Argumento '{schema.param_name}': se esperaba {schema.param_type.__name__}, se recibió {type(value).__name__}"

        # Check range
        if schema.min_value is not None and isinstance(value, (int, float)):
            if value < schema.min_value:
                return f"Argumento '{schema.param_name}': {value} < mínimo {schema.min_value}"
        if schema.max_value is not None and isinstance(value, (int, float)):
            if value > schema.max_value:
                return f"Argumento '{schema.param_name}': {value} > máximo {schema.max_value}"

        # Check allowed values
        if schema.allowed_values is not None and value not in schema.allowed_values:
            return f"Argumento '{schema.param_name}': '{value}' no está en valores permitidos {schema.allowed_values}"

        # Check max length
        if schema.max_length is not None:
            if isinstance(value, str) and len(value) > schema.max_length:
                return f"Argumento '{schema.param_name}': longitud {len(value)} > máximo {schema.max_length}"
            if isinstance(value, (list, tuple)) and len(value) > schema.max_length:
                return f"Argumento '{schema.param_name}': longitud {len(value)} > máximo {schema.max_length}"

        # Check pattern
        if schema.pattern is not None and isinstance(value, str):
            if not re.match(schema.pattern, value):
                return f"Argumento '{schema.param_name}': '{value[:50]}' no cumple patrón {schema.pattern}"

    return None


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
def _check_rate_limit(agent: str, tool: str, policy: AgentPolicy) -> str | None:
    """Verifica rate limits."""
    key = f"{agent}:{tool}"
    count = _get_call_count(key)
    limits = policy.rate_limits

    if count >= limits.max_executions:
        return f"Rate limit: {count} llamadas a '{tool}' (tope: {limits.max_executions})"

    _increment_call_count(key)
    return None


# ---------------------------------------------------------------------------
# Sandbox: process_tool binary whitelist
# ---------------------------------------------------------------------------
def _validate_process_command(args: dict[str, Any]) -> str | None:
    """Valida que el binario esté en la lista blanca y el cwd sea el repo."""
    cmd = args.get("args") or args.get("command")
    if not cmd:
        return None

    if isinstance(cmd, str):
        parts = cmd.split()
    elif isinstance(cmd, list):
        parts = cmd
    else:
        return None

    if not parts:
        return None

    binary = parts[0]
    # Resolve binary name (strip path)
    binary_name = Path(binary).name

    if binary_name not in ALLOWED_BINARIES:
        return f"Binario '{binary_name}' no está en la lista blanca: {sorted(ALLOWED_BINARIES)}"

    # Validate cwd is within repo
    cwd = args.get("cwd")
    if cwd:
        cwd_resolved = str(Path(str(cwd)).resolve())
        repo_root = str(Path.cwd().resolve())
        if not cwd_resolved.startswith(repo_root):
            return f"cwd '{cwd_resolved}' está fuera del repositorio ({repo_root})"

    return None


# ---------------------------------------------------------------------------
# Agent policies
# ---------------------------------------------------------------------------
AGENT_POLICIES: dict[str, AgentPolicy] = {}


def register_policy(agent: str, policy: AgentPolicy) -> None:
    """Registra la política de un agente."""
    AGENT_POLICIES[agent] = policy


def get_policy(agent: str) -> AgentPolicy | None:
    """Obtiene la política de un agente."""
    return AGENT_POLICIES.get(agent)


def register_default_policies() -> None:
    """Registra las políticas por defecto de todos los agentes conocidos."""
    # --- harness ---
    register_policy("harness", AgentPolicy(
        allowed_tools={"git", "process", "filesystem", "code_analysis", "dependency", "validate"},
        argument_schemas={
            "git": [
                ArgumentSchema("args", list, required=True, max_length=10),
            ],
            "process": [
                ArgumentSchema("args", list, required=True, max_length=10),
            ],
        },
        fs_read_dirs={".", "agents/", "progress/"},
        fs_write_dirs={"featureslist.json", "progress/", "agents/workspace/"},
        network_domains=set(),
        destructive_tools={"git"},
        requires_confirmation={"git"},
        rate_limits=RateLimits(max_executions=50, timeout_seconds=600),
    ))

    # --- explorer ---
    register_policy("explorer", AgentPolicy(
        allowed_tools={"code_analysis", "filesystem", "research", "rest", "stats"},
        argument_schemas={},
        fs_read_dirs={"."},
        fs_write_dirs=set(),
        network_domains={"localhost"},
        rate_limits=RateLimits(max_executions=20, timeout_seconds=120),
    ))

    # --- implementer ---
    register_policy("implementer", AgentPolicy(
        allowed_tools={"git", "process", "filesystem", "code_analysis", "validate", "pytest", "dependency", "research"},
        argument_schemas={
            "git": [
                ArgumentSchema("args", list, required=True, max_length=10),
            ],
            "process": [
                ArgumentSchema("args", list, required=True, max_length=10),
            ],
        },
        fs_read_dirs={"."},
        fs_write_dirs={"climasafeai/", "tests/", "agents/", "reports/", "data/", "documentacion/"},
        network_domains={"localhost"},
        destructive_tools={"git"},
        requires_confirmation=set(),
        rate_limits=RateLimits(max_executions=100, timeout_seconds=600),
    ))

    # --- reviewer ---
    register_policy("reviewer", AgentPolicy(
        allowed_tools={"code_analysis", "filesystem", "validate", "pytest", "stats"},
        argument_schemas={},
        fs_read_dirs={"."},
        fs_write_dirs=set(),
        network_domains=set(),
        rate_limits=RateLimits(max_executions=20, timeout_seconds=120),
    ))

    # --- git ---
    register_policy("git", AgentPolicy(
        allowed_tools={"git", "process"},
        argument_schemas={
            "git": [
                ArgumentSchema("args", list, required=True, max_length=10),
            ],
            "process": [
                ArgumentSchema("args", list, required=True, max_length=10),
            ],
        },
        fs_read_dirs={".", "agents/workspace/"},
        fs_write_dirs={"agents/workspace/"},
        network_domains=set(),
        destructive_tools={"git"},
        requires_confirmation=set(),
        rate_limits=RateLimits(max_executions=30, timeout_seconds=300),
    ))

    # --- audit ---
    register_policy("audit", AgentPolicy(
        allowed_tools={"code_analysis", "filesystem", "stats", "rest"},
        argument_schemas={},
        fs_read_dirs={"agents/workspace/", "progress/", "logs/", "."},
        fs_write_dirs=set(),
        network_domains=set(),
        rate_limits=RateLimits(max_executions=20, timeout_seconds=120),
    ))

    # --- data ---
    register_policy("data", AgentPolicy(
        allowed_tools={"filesystem", "dataframe_analysis", "data_io", "duckdb", "stats", "rest", "validate"},
        argument_schemas={
            "rest": [
                ArgumentSchema("timeout", int, required=False, min_value=1, max_value=60),
            ],
        },
        fs_read_dirs={"data/", "climasafeai/", "."},
        fs_write_dirs={"data/processed/", "data/raw/", "data/interim/"},
        network_domains={"archive-api.open-meteo.com", "api.open-meteo.org", "nominatim.openstreetmap.org"},
        rate_limits=RateLimits(max_executions=50, timeout_seconds=600),
    ))

    # --- documentation ---
    register_policy("documentation", AgentPolicy(
        allowed_tools={"filesystem", "code_analysis", "stats"},
        argument_schemas={},
        fs_read_dirs={"documentacion/", ".", "README.md"},
        fs_write_dirs={"documentacion/", "reports/", "README.md"},
        network_domains=set(),
        rate_limits=RateLimits(max_executions=20, timeout_seconds=120),
    ))

    # --- deploy ---
    register_policy("deploy", AgentPolicy(
        allowed_tools={"git", "docker", "process", "filesystem", "cicd"},
        argument_schemas={},
        fs_read_dirs={"."},
        fs_write_dirs={"."},
        network_domains={"github.com", "ghcr.io", "docker.io"},
        destructive_tools={"docker", "git"},
        requires_confirmation={"docker"},
        rate_limits=RateLimits(max_executions=20, timeout_seconds=600),
    ))


# ---------------------------------------------------------------------------
# Main approval function
# ---------------------------------------------------------------------------
def approve_tool_call(call: ToolCall) -> ApprovalResult:
    """
    Función de aprobación ÚNICA por la que pasa toda tool call.
    Ninguna tool se ejecuta sin pasar por ella.

    Verifica:
    1. Tool existe en el catálogo
    2. Agente tiene permiso para usar la tool
    3. Argumentos son válidos
    4. Rutas filesystem están permitidas
    5. URLs están en dominios permitidos
    6. No hay secretos en los argumentos
    7. Rate limits no se han excedido
    8. Acciones destructivas tienen confirmación
    9. Sandbox para process_tool (binarios y cwd)
    """
    checks_passed: list[str] = []
    entry: dict[str, Any] = {
        "timestamp": time.time(),
        "agent": call.agent,
        "tool": call.tool,
        "args_keys": sorted(call.args.keys()),
    }

    # Get policy
    policy = get_policy(call.agent)
    if policy is None:
        reason = f"Agente '{call.agent}' no tiene política de seguridad definida"
        entry.update({"decision": "denied", "reason": reason})
        _record_audit(entry)
        return ApprovalResult(approved=False, reason=reason, audit_entry=entry)

    # 1. Tool in catalog
    if call.tool not in policy.allowed_tools:
        reason = f"Tool '{call.tool}' no está en el catálogo de '{call.agent}': {sorted(policy.allowed_tools)}"
        entry.update({"decision": "denied", "reason": reason})
        _record_audit(entry)
        return ApprovalResult(approved=False, reason=reason, audit_entry=entry)
    checks_passed.append("catalog")

    # 2. Argument validation
    arg_error = _validate_arguments(call.tool, call.args, policy)
    if arg_error:
        reason = f"Argumentos inválidos: {arg_error}"
        entry.update({"decision": "denied", "reason": reason})
        _record_audit(entry)
        return ApprovalResult(approved=False, reason=reason, audit_entry=entry)
    checks_passed.append("arguments")

    # 3. Secret detection
    if _contains_secrets(call.args):
        reason = "Argumentos contienen secretos (tokens, claves o passwords detectados)"
        entry.update({"decision": "denied", "reason": reason})
        _record_audit(entry)
        return ApprovalResult(approved=False, reason=reason, audit_entry=entry)
    checks_passed.append("no_secrets")

    # 4. Filesystem restriction
    fs_error = _validate_filesystem(call.agent, call.args, policy)
    if fs_error:
        reason = f"Filesystem: {fs_error}"
        entry.update({"decision": "denied", "reason": reason})
        _record_audit(entry)
        return ApprovalResult(approved=False, reason=reason, audit_entry=entry)
    checks_passed.append("filesystem")

    # 5. Network control
    net_error = _validate_network(call.agent, call.args, policy)
    if net_error:
        reason = f"Red: {net_error}"
        entry.update({"decision": "denied", "reason": reason})
        _record_audit(entry)
        return ApprovalResult(approved=False, reason=reason, audit_entry=entry)
    checks_passed.append("network")

    # 6. Sandbox for process_tool
    if call.tool == "process":
        sandbox_error = _validate_process_command(call.args)
        if sandbox_error:
            reason = f"Sandbox: {sandbox_error}"
            entry.update({"decision": "denied", "reason": reason})
            _record_audit(entry)
            return ApprovalResult(approved=False, reason=reason, audit_entry=entry)
        checks_passed.append("sandbox")

    # 7. Rate limits
    rate_error = _check_rate_limit(call.agent, call.tool, policy)
    if rate_error:
        reason = rate_error
        entry.update({"decision": "denied", "reason": reason})
        _record_audit(entry)
        return ApprovalResult(approved=False, reason=reason, audit_entry=entry)
    checks_passed.append("rate_limit")

    # 8. Human-in-the-loop for destructive actions
    if call.tool in policy.requires_confirmation and call.tool in policy.destructive_tools:
        reason = f"Tool '{call.tool}' requiere confirmación humana (acción destructiva)"
        entry.update({"decision": "needs_confirmation", "reason": reason})
        _record_audit(entry)
        return ApprovalResult(approved=False, reason=reason, audit_entry=entry)

    # All checks passed
    entry.update({
        "decision": "approved",
        "reason": "",
        "checks_passed": checks_passed,
    })
    _record_audit(entry)

    return ApprovalResult(
        approved=True,
        tool=call.tool,
        args=call.args,
        checks_passed=checks_passed,
        audit_entry=entry,
    )


# Initialize default policies on import
register_default_policies()
