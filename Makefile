.PHONY: setup install-deps \
        data features train predict \
        test smoke lint format lock \
        lab notebook tb \
        docs \
        profile \
        mlflow \
        monitor tune serve query \
        docker-run docker-update docker-down \
        clean clean-models clean-figures clean-all \
        run info help

# ─────────────────────────────────────────────────────────────────────────────
#  Variables
# ─────────────────────────────────────────────────────────────────────────────
MODULE   = climasafeai
ML_TYPE  = supervisado
PYTHON   = python

# ─────────────────────────────────────────────────────────────────────────────
#  help  →  target por defecto
# ─────────────────────────────────────────────────────────────────────────────
.DEFAULT_GOAL := help

help:
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  ClimaSafeAI  ·  ML: $(ML_TYPE)"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@echo "  Entorno"
	@echo "    make setup          instala core + dev + extras del tipo ML"
	@echo "    make install-deps   solo dependencias del tipo ML (sin dev)"
	@echo "    make info           versiones y paquetes instalados"
	@echo ""
	@echo "  Pipeline de datos y modelos"
	@echo "    make data           descarga/preprocesa datos crudos"
	@echo "    make features       construye features desde datos procesados"
	@echo "    make train          entrena el modelo"
	@echo "    make predict        genera predicciones con el modelo entrenado"
	@echo "    make pipeline       data → features → train → predict (todo)"
	@echo ""
	@echo "  Calidad"
	@echo "    make test           pytest -v (todos los tests)"
	@echo "    make smoke          test de humo — verifica que el pipeline arranca"
	@echo "    make lint           ruff check (solo lectura, sin modificar)"
	@echo "    make format         ruff format (aplica cambios en sitio)"
	@echo ""
	@echo "  Jupyter"
	@echo "    make lab            JupyterLab  (puerto 8888)"
	@echo "    make notebook       Jupyter Notebook (puerto 8888)"
	@echo ""
	@echo "  Ejecución directa"
	@echo "    make run            ejecuta main.py"
	@echo "    make profile        cProfile de main.py → reports/profile.prof"
	@echo ""

	@echo "  Monitoring"
	@echo "    make monitor        drift detection + performance report"
	@echo ""


	@echo "  Optuna"
	@echo "    make tune           optimiza hiperparámetros (n_trials en main.py)"
	@echo ""



	@echo "  API REST"
	@echo "    make serve          FastAPI en localhost:8000  (docs en /docs)"
	@echo ""


	@echo "  Documentación"
	@echo "    make docs           sphinx-apidoc + html"
	@echo ""
	@echo "  Limpieza"
	@echo "    make clean          cachés y __pycache__"
	@echo "    make clean-models   borra .joblib y .pt de models/"
	@echo "    make clean-figures  borra figuras de reports/figures/"
	@echo "    make clean-all      todo lo anterior"
	@echo ""
	@echo "  Docker"
	@echo "    make docker-run     construye imagen y lanza el chat en http://localhost:8080"
	@echo "    make docker-update  reconstruye la imagen con los cambios mas recientes"
	@echo "    make docker-down    para y elimina los contenedores"
	@echo ""


# ─────────────────────────────────────────────────────────────────────────────
#  Entorno
# ─────────────────────────────────────────────────────────────────────────────
setup:
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  Instalando dependencias para ML tipo: $(ML_TYPE)"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	uv sync --extra dev --extra $(ML_TYPE)
	@echo ""
	@echo "  Listo. Activa el entorno con:  source .venv/bin/activate"
	@echo ""

install-deps:
	uv sync --extra $(ML_TYPE)

# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline  data → features → train → predict
#
#  Cada step llama al script correspondiente dentro del módulo.
#  Estructura esperada:
#    $(MODULE)/data/make_dataset.py        → lee data/raw/, escribe data/processed/
#    $(MODULE)/features/build_features.py  → lee data/processed/, escribe data/interim/
#    $(MODULE)/models/train_model.py       → lee data/interim/, escribe models/
#    $(MODULE)/models/predict_model.py     → lee models/ + data/interim/, escribe reports/
# ─────────────────────────────────────────────────────────────────────────────
data:
	@echo "▶  Procesando datos crudos → data/processed/"
	uv run $(PYTHON) $(MODULE)/data/make_dataset.py

features: data
	@echo "▶  Construyendo features → data/interim/"
	uv run $(PYTHON) $(MODULE)/features/build_features.py

train: features
	@echo "▶  Entrenando modelo → models/"
	uv run $(PYTHON) $(MODULE)/models/train_model.py

predict: train
	@echo "▶  Generando predicciones → reports/"
	uv run $(PYTHON) $(MODULE)/models/predict_model.py

pipeline: predict
	@echo ""
	@echo "  Pipeline completo finalizado."
	@echo ""

# ─────────────────────────────────────────────────────────────────────────────
#  Ejecución directa
# ─────────────────────────────────────────────────────────────────────────────
run:
	uv run $(PYTHON) main.py


# ─────────────────────────────────────────────────────────────────────────────
#  Monitoring — drift detection y performance tracking
# ─────────────────────────────────────────────────────────────────────────────
monitor:
	@echo "▶  Ejecutando monitorización de drift y rendimiento..."
	uv run python -m monitoring.monitor



# ─────────────────────────────────────────────────────────────────────────────
#  Optuna — optimizacion de hiperparametros
# ─────────────────────────────────────────────────────────────────────────────
tune:
	@echo "▶  Lanzando optimizacion Optuna (OPTUNA_TRIALS en main.py)"
	uv run python -m tuning.tune_model




# ─────────────────────────────────────────────────────────────────────────────
#  API REST — servir el modelo con FastAPI
# ─────────────────────────────────────────────────────────────────────────────
serve:
	@echo "▶  Lanzando API REST en http://localhost:8000"
	@echo "   Documentación interactiva: http://localhost:8000/docs"
	uv run uvicorn api.main:app --reload --port 8000


profile:
	@echo "▶  Profiling main.py → reports/profile.prof"
	uv run $(PYTHON) -m cProfile -o reports/profile.prof main.py
	@echo "   Visualiza con: uv run snakeviz reports/profile.prof"

# ─────────────────────────────────────────────────────────────────────────────
#  Calidad de código
# ─────────────────────────────────────────────────────────────────────────────
test:
	uv run pytest tests/ -v

smoke:
	@echo "▶  Test de humo — pipeline con datos sintéticos"
	uv run pytest tests/ -v -m smoke --tb=short

lint:
	uv run ruff check $(MODULE)/ tests/
	uv run ruff format --check $(MODULE)/ tests/

format:
	uv run ruff format $(MODULE)/ tests/

# ─────────────────────────────────────────────────────────────────────────────
#  Jupyter
# ─────────────────────────────────────────────────────────────────────────────
lab:
	uv run jupyter lab --ip=* --port=8888 --no-browser

notebook:
	uv run jupyter notebook --ip=* --port=8888 --no-browser



# ─────────────────────────────────────────────────────────────────────────────
#  MLflow UI
# ─────────────────────────────────────────────────────────────────────────────
mlflow:
	@echo "Lanzando MLflow UI en http://localhost:5000"
	uv run mlflow ui --port 5000



# ─────────────────────────────────────────────────────────────────────────────
#  Documentación
# ─────────────────────────────────────────────────────────────────────────────
docs:
	uv run sphinx-apidoc -o docs/source/ $(MODULE)/
	$(MAKE) html -C docs

# ─────────────────────────────────────────────────────────────────────────────
#  Limpieza
# ─────────────────────────────────────────────────────────────────────────────
clean:
	rm -rf .pytest_cache docs/build
	find $(MODULE) tests -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	@echo "  Cachés y __pycache__ eliminados."

clean-models:
	rm -f models/*.joblib models/*.pkl models/*.pt models/checkpoint-*.pt
	@echo "  Modelos eliminados."

clean-figures:
	rm -f reports/figures/*.png reports/figures/*.svg reports/figures/*.html
	@echo "  Figuras eliminadas."

clean-all: clean clean-models clean-figures
	@echo "  Limpieza completa."



# ─────────────────────────────────────────────────────────────────────────────
#  Docker — interfaz web de chat
# ─────────────────────────────────────────────────────────────────────────────
docker-run:
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  Construyendo y arrancando contenedor Docker"
	@echo "  Chat disponible en: http://localhost:8080"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	docker compose up -d --build
	@echo ""
	@echo "  Logs en tiempo real: docker compose logs -f"
	@echo ""

docker-update:
	@echo "  Reconstruyendo imagen con cambios recientes..."
	docker compose up -d --build --force-recreate
	@echo "  Imagen actualizada y contenedor reiniciado."

docker-down:
	@echo "  Parando contenedores..."
	docker compose down
	@echo "  Contenedores detenidos."


# ─────────────────────────────────────────────────────────────────────────────
#  Dependencias
# ─────────────────────────────────────────────────────────────────────────────
lock:
	uv lock
	@echo "  uv.lock actualizado."

# ─────────────────────────────────────────────────────────────────────────────
#  Info del entorno
# ─────────────────────────────────────────────────────────────────────────────
info:
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  Proyecto  : ClimaSafeAI"
	@echo "  Módulo    : $(MODULE)"
	@echo "  ML tipo   : $(ML_TYPE)"
	@echo "  Python    : $(shell uv run python --version 2>/dev/null || python --version)"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@uv pip list 2>/dev/null | head -40 || pip list | head -40
	@echo ""