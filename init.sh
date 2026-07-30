#!/usr/bin/env bash
#
# init.sh — Puerta de entrada del arnés de ClimaSafeAI
#
# Decide si el proyecto está en un estado suficientemente bueno para que un
# agente de IA empiece a trabajar. Si algo falla, el agente PARA — no intenta
# arreglarlo por su cuenta ni sigue implementando encima de un proyecto roto.
#
# Uso:
#   ./init.sh              verificación completa (estructura + tests)
#   ./init.sh --quick      omite la suite de tests (solo checks estructurales)
#   ./init.sh --json       salida JSON para consumo por agentes
#
# Códigos de salida:
#   0  ENTORNO LISTO — puedes trabajar
#   1  ENTORNO BLOQUEADO — para y reporta el fallo al usuario
#
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT" || exit 1

QUICK=0
JSON=0
for arg in "$@"; do
  case "$arg" in
    --quick) QUICK=1 ;;
    --json)  JSON=1 ;;
    -h|--help)
      sed -n '3,18p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "init.sh: opción desconocida '$arg'" >&2; exit 1 ;;
  esac
done

ERRORS=0
WARNINGS=0
REPORT=""

record() {  # record <status> <check> <detail>
  REPORT="${REPORT}${1}"$'\t'"${2}"$'\t'"${3}"$'\n'
}

ok()   { record "ok"   "$1" "$2"; [ "$JSON" -eq 1 ] || printf '  \033[32m✔\033[0m %-22s %s\n' "$1" "$2"; }
warn() { record "warn" "$1" "$2"; WARNINGS=$((WARNINGS + 1)); [ "$JSON" -eq 1 ] || printf '  \033[33m⚠\033[0m %-22s %s\n' "$1" "$2"; }
fail() { record "fail" "$1" "$2"; ERRORS=$((ERRORS + 1));   [ "$JSON" -eq 1 ] || printf '  \033[31m✘\033[0m %-22s %s\n' "$1" "$2"; }

section() { [ "$JSON" -eq 1 ] || printf '\n\033[1m%s\033[0m\n' "$1"; }

[ "$JSON" -eq 1 ] || printf '\n\033[1m━━ init.sh · arnés de ClimaSafeAI ━━\033[0m\n'

# ─────────────────────────────────────────────────────────────────────────────
#  1. Entorno de ejecución
# ─────────────────────────────────────────────────────────────────────────────
section "Entorno"

PY=""
for candidate in "python3.13" python3 python; do
  if command -v "$candidate" >/dev/null 2>&1; then PY="$candidate"; break; fi
done

if [ -z "$PY" ]; then
  fail "python" "no se encontró ningún intérprete de Python en PATH"
else
  PY_VERSION="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)"
  if [ -z "$PY_VERSION" ]; then
    fail "python" "'$PY' existe pero no ejecuta código"
  elif [ "$PY_VERSION" = "3.13" ]; then
    ok "python" "$PY $PY_VERSION"
  else
    warn "python" "$PY es $PY_VERSION, el proyecto declara 3.13 — usa el intérprete de .venv"
  fi
fi

if command -v uv >/dev/null 2>&1; then
  ok "uv" "$(uv --version 2>/dev/null | head -1)"
  # --no-sync: la puerta MIRA, no toca. Un `uv run` normal sincroniza el venv
  # contra las dependencias base antes de ejecutar y DESINSTALA todo lo que
  # venga de un extra (torch, mlflow, xgboost, fastapi...). Una puerta que
  # rompe el entorno que está verificando no es una puerta.
  # Si falta algo, se restaura con `make setup`, no en mitad de un check.
  RUNNER="uv run --no-sync"
else
  warn "uv" "no instalado — se usará '$PY' directamente (https://docs.astral.sh/uv/)"
  RUNNER="$PY -m"
fi

if [ -d ".venv" ]; then
  ok "venv" ".venv presente"

  # Integridad de los paquetes instalados.
  #
  # Un apagado sucio dejó 203 ficheros .py truncados a 0 bytes repartidos por 39
  # paquetes (numpy, pandas, mcp...). Los síntomas no apuntaban al venv: el bot de
  # Telegram decía "mcp no está instalado" y `uv sync` no lo arreglaba, porque uv
  # enlaza los ficheros con hardlinks a su caché y la caché tenía el mismo daño.
  # Se compara contra el RECORD de cada dist-info: solo hace stat, así que
  # cuesta menos de un segundo.
  #
  # Se busca el daño real (fichero vacío o desaparecido), no cualquier
  # diferencia de tamaño: un tamaño distinto pero NO vacío es normal cuando dos
  # wheels comparten rutas y una pisa a la otra. Sin este matiz, un venv sano
  # con torch daba 47 falsos positivos en las cabeceras de nvidia-nccl y
  # bloqueaba la puerta para siempre.
  VENV_DANIO="$("$PY" - <<'PYEOF' 2>/dev/null
import csv, pathlib, sys

sp = next(pathlib.Path(".venv/lib").glob("python3.*/site-packages"), None)
if sp is None:
    sys.exit(0)
pkgs, ficheros = set(), 0
for rec in sp.glob("*.dist-info/RECORD"):
    for row in csv.reader(rec.open(encoding="utf-8", errors="replace")):
        if len(row) < 3 or not row[2].isdigit():
            continue
        declarado = int(row[2])
        if declarado == 0:
            continue  # el propio wheel lo declara vacío (__init__.py de espacio de nombres)
        f = sp / row[0]
        try:
            danio = f.stat().st_size == 0
        except OSError:
            danio = True  # declarado en el RECORD y no está
        if danio:
            pkgs.add(rec.parent.name.split("-")[0])
            ficheros += 1
if ficheros:
    print(f"{ficheros} {' '.join(sorted(pkgs))}")
PYEOF
)"
  if [ -n "$VENV_DANIO" ]; then
    N_FICHEROS="${VENV_DANIO%% *}"
    PKGS_DANIO="${VENV_DANIO#* }"
    fail "venv íntegro" "$N_FICHEROS fichero(s) corruptos. Arréglalo con: uv cache clean $PKGS_DANIO && uv sync --reinstall"
  else
    ok "venv íntegro" "los paquetes instalados coinciden con su RECORD"
  fi
else
  warn "venv" "sin .venv — ejecuta 'make setup' antes de tocar código"
fi

# ─────────────────────────────────────────────────────────────────────────────
#  2. Ficheros del arnés
# ─────────────────────────────────────────────────────────────────────────────
section "Arnés"

for required in AGENTS.md featureslist.json progress/current.md progress/history.md; do
  if [ -f "$required" ]; then
    ok "$(basename "$required")" "presente"
  else
    fail "$(basename "$required")" "FALTA — el arnés está incompleto ($required)"
  fi
done

MISSING_AGENTS=""
for agent_def in lider implementer reviewer explorer; do
  [ -f ".opencode/agents/${agent_def}.md" ] || MISSING_AGENTS="$MISSING_AGENTS $agent_def"
done
if [ -z "$MISSING_AGENTS" ]; then
  ok "subagentes" "lider, implementer, reviewer, explorer"
else
  fail "subagentes" "faltan definiciones en .opencode/agents/:$MISSING_AGENTS"
fi

# ─────────────────────────────────────────────────────────────────────────────
#  3. Backlog — featureslist.json bien formado
# ─────────────────────────────────────────────────────────────────────────────
section "Backlog"

if [ -n "$PY" ] && [ -f "featureslist.json" ]; then
  BACKLOG_OUT="$("$PY" - <<'PYEOF'
import json
import sys

VALID_STATUS = ("pending", "in_progress", "done", "blocked")
REQUIRED = ("id", "title", "description", "acceptance_criteria", "status")

try:
    with open("featureslist.json", encoding="utf-8") as fh:
        doc = json.load(fh)
except json.JSONDecodeError as exc:
    print("fail\tJSON inválido: %s" % exc)
    sys.exit(0)
except OSError as exc:
    print("fail\tno se pudo leer: %s" % exc)
    sys.exit(0)

features = doc.get("features") if isinstance(doc, dict) else None
if not isinstance(features, list):
    print("fail\tse esperaba un objeto con la clave 'features' (lista)")
    sys.exit(0)

problems = []
seen = set()
for i, feat in enumerate(features):
    if not isinstance(feat, dict):
        problems.append("feature #%d no es un objeto" % i)
        continue
    missing = [k for k in REQUIRED if k not in feat]
    if missing:
        problems.append("feature #%d sin campos: %s" % (i, ", ".join(missing)))
        continue
    fid = feat["id"]
    if fid in seen:
        problems.append("id duplicado: %s" % fid)
    seen.add(fid)
    if feat["status"] not in VALID_STATUS:
        problems.append("%s: status '%s' no válido (%s)" % (fid, feat["status"], "|".join(VALID_STATUS)))
    criteria = feat["acceptance_criteria"]
    if not isinstance(criteria, list) or not criteria:
        problems.append("%s: acceptance_criteria debe ser una lista no vacía" % fid)

for feat in features:
    if not isinstance(feat, dict):
        continue
    for dep in feat.get("depends_on", []):
        if dep not in seen:
            problems.append("%s: depends_on '%s' no existe en el backlog" % (feat.get("id"), dep))

for problem in problems:
    print("fail\t%s" % problem)

if problems:
    sys.exit(0)

counts = {status: 0 for status in VALID_STATUS}
for feat in features:
    counts[feat["status"]] += 1

running = [f["id"] for f in features if f["status"] == "in_progress"]
if len(running) > 1:
    print("warn\t%d features en in_progress a la vez: %s" % (len(running), ", ".join(running)))

print("ok\t%d features · %d pending · %d in_progress · %d done · %d blocked" % (
    len(features), counts["pending"], counts["in_progress"], counts["done"], counts["blocked"]))

nxt = next((f for f in features if f["status"] == "in_progress"), None)
nxt = nxt or next((f for f in features if f["status"] == "pending"), None)
if nxt is None:
    print("ok\tsin trabajo pendiente — backlog vacío")
else:
    print("next\t%s — %s [%s]" % (nxt["id"], nxt["title"], nxt["status"]))
PYEOF
)"
  while IFS=$'\t' read -r status detail; do
    [ -z "$status" ] && continue
    case "$status" in
      ok)   ok   "featureslist" "$detail" ;;
      warn) warn "featureslist" "$detail" ;;
      fail) fail "featureslist" "$detail" ;;
      next) ok   "siguiente tarea" "$detail" ;;
    esac
  done <<< "$BACKLOG_OUT"
else
  fail "featureslist" "no verificable (falta Python o featureslist.json)"
fi

# ─────────────────────────────────────────────────────────────────────────────
#  4. Código del proyecto
# ─────────────────────────────────────────────────────────────────────────────
section "Proyecto"

if [ -d "climasafeai" ]; then
  ok "paquete" "climasafeai/ presente"
else
  fail "paquete" "falta el paquete climasafeai/"
fi

if [ -f "pyproject.toml" ]; then
  ok "pyproject" "presente"
else
  fail "pyproject" "falta pyproject.toml"
fi

if [ -d "tests" ]; then
  TEST_COUNT="$(find tests -name 'test_*.py' -type f 2>/dev/null | wc -l | tr -d ' ')"
  if [ "$TEST_COUNT" -gt 0 ]; then
    ok "tests" "$TEST_COUNT ficheros de test"
  else
    fail "tests" "tests/ existe pero no contiene ningún test_*.py"
  fi
else
  fail "tests" "falta el directorio tests/"
fi

# ─────────────────────────────────────────────────────────────────────────────
#  5. Suite de tests — la prueba de que el proyecto no está roto
# ─────────────────────────────────────────────────────────────────────────────
section "Verificación"

if [ "$QUICK" -eq 1 ]; then
  warn "pytest" "omitido (--quick) — NO declares ninguna feature como done sin esto"
elif [ "$ERRORS" -gt 0 ]; then
  warn "pytest" "omitido — hay errores estructurales que arreglar antes"
else
  # Dos suites: la del producto y la del sistema de agentes. El arnés se apoya
  # en los agentes para trabajar, así que si `agents/tests/` está roja la puerta
  # también lo está — si no, el arnés se autoverifica con las herramientas rotas.
  TEST_LOG="$(mktemp)"
  if $RUNNER pytest tests/ agents/tests/ -q --no-header >"$TEST_LOG" 2>&1; then
    SUMMARY="$(grep -Eo '[0-9]+ passed[^=]*' "$TEST_LOG" | tail -1 | sed 's/[[:space:]]*$//')"
    ok "pytest" "${SUMMARY:-suite en verde}"
  else
    fail "pytest" "$(grep -E '^(FAILED|ERROR)' "$TEST_LOG" | head -3 | tr '\n' ' ' || echo 'la suite falla')"
    [ "$JSON" -eq 1 ] || tail -15 "$TEST_LOG" | sed 's/^/      /'
  fi
  rm -f "$TEST_LOG"
fi

# ─────────────────────────────────────────────────────────────────────────────
#  Veredicto
# ─────────────────────────────────────────────────────────────────────────────
if [ "$JSON" -eq 1 ]; then
  REPORT="$REPORT" ERRORS="$ERRORS" WARNINGS="$WARNINGS" "$PY" - <<'PYEOF'
import json
import os

checks = []
for line in os.environ.get("REPORT", "").splitlines():
    if not line.strip():
        continue
    parts = line.split("\t")
    if len(parts) < 3:
        continue
    checks.append({"status": parts[0], "check": parts[1], "detail": parts[2]})

errors = int(os.environ.get("ERRORS", "0"))
print(json.dumps({
    "ready": errors == 0,
    "errors": errors,
    "warnings": int(os.environ.get("WARNINGS", "0")),
    "checks": checks,
}, indent=2, ensure_ascii=False))
PYEOF
elif [ "$ERRORS" -eq 0 ]; then
  printf '\n\033[1;32m━━ ENTORNO LISTO ━━\033[0m  %d aviso(s)\n' "$WARNINGS"
  printf 'Puedes trabajar. Siguiente paso: lee progress/current.md y elige la primera feature pendiente.\n\n'
else
  printf '\n\033[1;31m━━ ENTORNO BLOQUEADO ━━\033[0m  %d error(es), %d aviso(s)\n' "$ERRORS" "$WARNINGS"
  printf 'NO empieces a implementar. Reporta estos fallos al usuario y para.\n\n'
fi

[ "$ERRORS" -eq 0 ] || exit 1
exit 0
