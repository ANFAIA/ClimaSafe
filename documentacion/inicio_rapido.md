# Inicio rápido

Cómo pasar de cero a una predicción de riesgo en unos minutos.

## 1. Entorno

```bash
# Clonar y preparar (crea .venv e instala dependencias)
./init.sh                      # verificación completa del entorno
uv sync --extra dev --extra supervisado   # si prefieres sincronizar a mano
source .venv/bin/activate
```

## 2. Pipeline principal

```bash
make run          # main.py completo (datos → train → predict → reports/)
make data         # solo carga/preproceso de datos
make train        # solo entrenamiento
make predict      # solo predicciones → reports/
```

## 3. Probar una predicción

Desde la web (`make serve` → `http://localhost:8000`), el bot de Telegram, los
servidores MCP o directo:

```bash
make serve        # FastAPI en localhost:8000 (docs interactivos en /docs)
make mlflow       # UI de experimentos en http://localhost:5000
```

## 4. Calidad

```bash
make test         # pytest completo
make smoke        # tests de humo (rápidos)
make lint         # ruff check
make typecheck    # mypy / ty (informativo)
```

## 5. Documentación

```bash
make docs         # construye el sitio en site/ (MkDocs Material)
```

## Estructura del repo

```
climasafeai/       código de producto (data, db, features, ml, models, utils, visualization)
data/              raw/ + processed/
models/            modelos entrenados (.joblib)
reports/           figuras e informes
chat/              web UI y chat
tests/             tests de pytest
documentacion/     esta documentación (Markdown)
```

Detalle de comandos y estructura de outputs en [`README.md`](README.md).
