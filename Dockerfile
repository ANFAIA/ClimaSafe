# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — ClimaSafeAI (PACK-002)
#
# Sirve el servicio orientado al usuario (web chat en :8080 y, como profile
# aparte en docker-compose.yml, el bot de Telegram). NO empaqueta el despliegue
# interno del sistema de agentes (arnés, progress/, featureslist.json), que se
# mantiene fuera de la imagen.
#
# Multi-stage con uv:
#   builder → resuelve dependencias del lockfile en un .venv aislado
#   runtime → copia el .venv, el paquete y los modelos ya entrenados
#
# Build (requiere permisos de docker — ver documentacion/despliegue/packaging.md):
#   docker build -t ghcr.io/cacelass/climasafeai:<version> .
# ─────────────────────────────────────────────────────────────────────────────
# syntax=docker/dockerfile:1

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# 1) Metadatos del proyecto (README.md es `readme` del paquete)
COPY pyproject.toml uv.lock README.md MANIFEST.in ./

# 2) Dependencias congeladas al lockfile, sin instalar aún el proyecto
#    (capa cacheada: solo se reconstruye si cambia el lock). Extras que el
#    servicio necesita; sin dev (mkdocs, pytest) ni extras de entrenamiento
#    (optuna, monitoring, no_supervisado, onnx): la imagen sirve, no entrena.
RUN uv sync --frozen --no-dev --no-install-project \
    --extra api \
    --extra supervisado \
    --extra redes_neuronales \
    --extra mlflow_tracking \
    --extra voice \
    --extra rag

# 3) Código fuente de los dos paquetes que instala setuptools
#    ([tool.setuptools.packages.find] include = climasafeai*, agents*)
COPY climasafeai ./climasafeai
COPY agents ./agents

# 4) El proyecto mismo, sobre las dependencias ya cacheadas
RUN uv sync --frozen --no-dev \
    --extra api \
    --extra supervisado \
    --extra redes_neuronales \
    --extra mlflow_tracking \
    --extra voice \
    --extra rag

# ── Runtime ──────────────────────────────────────────────────────────────────
FROM python:3.12-slim-bookworm AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# venv pre-resuelto desde builder
COPY --from=builder /app/.venv /app/.venv

# Servicio web (chat/app.py) y sus estáticos; entrypoint existente del contenedor
COPY chat ./chat
COPY chat/entrypoint.sh ./chat/entrypoint.sh

# Metadatos para la versión única (PACK-001): entrypoint.sh lee pyproject.toml
COPY pyproject.toml uv.lock README.md ./

# Modelos ya entrenados (*.joblib top-level + artifacts/) y BD de perfiles:
# sin esto el usuario no técnico tendría que entrenar antes del primer uso.
COPY data/climasafe.db ./data/climasafe.db
COPY models/artifacts ./models/artifacts
COPY models/*.joblib ./models/

# Usuario no root (el volumen de datos debe ser escribible por este UID)
RUN groupadd -r app && useradd -r -g app app \
 && chown -R app:app /app/data /app/models
USER app

EXPOSE 8080

# El servicio tarda en arrancar: carga ~140 MB de joblibs + import de torch
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/', timeout=4)" || exit 1

# Entrypoint existente (dskit): banner, chequeo de modelos y uvicorn :8080
ENTRYPOINT ["bash", "chat/entrypoint.sh"]
