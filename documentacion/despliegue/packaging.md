# Empaquetado y despliegue (PACK-002)

Cómo se instala ClimaSafeAI según quién lo vaya a usar:

| Quién | Vía | Esfuerzo |
|-------|-----|----------|
| No técnico | **Imagen Docker** (`docker run`) | Un comando |
| Técnico (Python) | **pip** desde el wheel/sdist | Tres comandos |
| Desarrollo del proyecto | `make setup` + `make web` | Ver [`inicio_rapido.md`](../inicio_rapido.md) |

> Estado: el fichero `Dockerfile` / `.dockerignore` / `docker-compose.yml` y los
> tests estáticos (`tests/test_packaging.py`) están en el repo. El **build y el
> push de la imagen son una acción humana pendiente**: esta máquina no tiene
> permisos de demonio Docker ni credenciales de registry. Los comandos exactos
> están más abajo.

---

## 1. Usuario no técnico — un comando

Una vez publicada la imagen en el registry:

```bash
docker run -d --name climasafe -p 8080:8080 ghcr.io/cacelass/climasafeai:latest
```

y abrir <http://localhost:8080>. La imagen lleva modelos ya entrenados y la BD
de perfiles: no hay que entrenar nada ni tocar configuración.

Alternativa con compose (clonando el repo):

```bash
git clone https://github.com/cacelass/climasafe && cd climasafe
docker compose up -d          # solo la web
```

Los datos persisten entre reinicios en los volúmenes nombrados
(`climasafe_data`, `climasafe_models`), que se siembran con lo que trae la
imagen la primera vez. Para empezar de cero:
`docker compose down -v`.

### Bot de Telegram (opcional)

El bot va en un perfil aparte porque necesita un token:

```bash
cp .env.example .env           # poner tu TELEGRAM_BOT_TOKEN de @BotFather
docker compose --profile bot up -d
```

### Qué lleva (y qué NO lleva) la imagen

- Lleva: paquete `climasafeai`, servicio web (`chat/app.py` vía el entrypoint
  `chat/entrypoint.sh`, uvicorn :8080), modelos `models/*.joblib` +
  `models/artifacts/`, BD `data/climasafe.db`. Multi-stage con `uv` y usuario
  no root; healthcheck contra `/`.
- **No** lleva: el despliegue interno del sistema de agentes (arnés,
  `progress/`, `featureslist.json`, `.opencode/`), datasets crudos, los `*.pt`
  (~2 GB, el servicio web no los carga) ni ningún secreto (`.env` está en
  `.dockerignore`). El pipeline de entrenamiento (`main.py`, `tests/`,
  `tuning/`, `monitoring/`, `ingest/`) queda fuera: la imagen sirve, no entrena.
  El despliegue interno con agentes se mantiene como está, por repo/venv.

---

## 2. Usuario técnico — pip

Con Python ≥ 3.12:

```bash
# crear entorno y actualizar pip
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

# instalar con los extras que necesites
pip install "climasafeai[api,supervisado,redes_neuronales,mlflow_tracking]" \
    git+https://github.com/cacelass/climasafe.git
```

O desde un wheel local (`make build` genera `dist/*.whl`; la versión sale de
pyproject.toml, fuente única desde PACK-001):

```bash
pip install "dist/climasafeai-<version>-py3-none-any.whl[extras...]"
```

Extras disponibles (definidos en `pyproject.toml`): `api` (FastAPI/uvicorn),
`supervisado` (XGBoost/LightGBM/shap), `redes_neuronales` (torch/LSTM),
`mlflow_tracking`, `voice` (STT/TTS del bot), `rag` (embeddings),
`no_supervisado`, `optuna`, `monitoring`, `onnx`.

Consolas instaladas:

| Comando | Qué hace |
|---------|----------|
| `climasafeai-bot` | Arranca el bot de Telegram (requiere `TELEGRAM_BOT_TOKEN`) |
| `climasafeai-mcp` | MCP server de predicción (Claude Desktop/Cursor) |

Verificado en PACK-002: el wheel `climasafeai-0.0.117-py3-none-any.whl`
se instala en un venv limpio y ambos entry points importan y existen.
Nota: el módulo del bot importa la cadena completa de modelos
(`predict_model` → mlflow/shap, `lstm_province_hybrid` → torch): sin los
extras correspondientes, `climasafeai` importará pero `climasafeai-bot`
fallará al arrancar.

npm: no aplica — el proyecto es Python puro; la demo del navegador
(`web/probar-ya/`) es estática y se publica por GitHub Pages, sin build npm.

---

## 3. Build y publicación de la imagen (acción humana)

Pendiente hasta tener permisos de docker y credenciales del registry.
Comandos exactos, con `<version>` = campo `version` de `pyproject.toml`
(hoy `0.0.117`; usa el mismo tag que el release de pip):

```bash
# build local (multi-stage, ~5 min la primera vez)
docker build -t ghcr.io/cacelass/climasafeai:<version> .
docker tag  ghcr.io/cacelass/climasafeai:<version> ghcr.io/cacelass/climasafeai:latest

# verificación antes de publicar
docker run --rm -p 8080:8080 ghcr.io/cacelass/climasafeai:<version>
# → http://localhost:8080 debe servir el chat; Ctrl+C y:

# push (requiere `docker login ghcr.io`)
docker push ghcr.io/cacelass/climasafeai:<version>
docker push ghcr.io/cacelass/climasafeai:latest
```

Sustituir `ghcr.io/cacelass/climasafeai` por el registry definitivo si cambia
(y actualizar el default de `CLIMASAFE_IMAGE` en `docker-compose.yml`).
Para CI: login con `secrets.GITHUB_TOKEN` y permisos `packages: write` en un
workflow de GitHub Actions — no se ha creado ese workflow en PACK-002.

Validación hecha sin daemon (esta máquina no puede `docker build`):
`tests/test_packaging.py` (7 checks estáticos) y `docker compose config`.

---

## 4. Formatos futuros (documentados, no implementados)

Ninguno está planificado a corto plazo; quedan como opciones evaluadas:

| Formato | Aplica si... | Notas |
|---------|--------------|-------|
| **Homebrew** | crece la base macOS del bot/MCP | Fórmula que embotella el wheel; tap propio o homebrew-core si hay demanda |
| **dmg** | app gráfica macOS | Requeriría envolver la web en Electron/Tauri; hoy la web es servida, no empaquetada |
| **Flathub** | distribución Linux sandboxed | Necesita manifest tipo Flatpak + runtime de Python; coste alto de mantenimiento |
| **Electron/Tauri** | producto desktop offline | Duplicaría la UI (ya hay web + Telegram); solo si se pide modo escritorio |

Criterio para activar cualquiera: usuarios reales pidiéndolo, no especulación.

---

## Historial de decisión

MSG-004 descartó Docker ("No existe Dockerfile ni docker-compose.yml en la
raíz") porque el programador de evaluaciones corre como worker sobre la cola,
no en contenedor. Esa decisión era sobre *cómo ejecutar el scheduler*;
PACK-002 la revierte parcialmente: Docker vuelve para **distribuir el servicio
al usuario final**, mientras el worker/scheduler sigue siendo proceso nativo.
No hay contenedor de evaluaciones programadas.
