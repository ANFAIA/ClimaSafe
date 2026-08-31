.PHONY: setup install-deps \
        data features train predict \
        test smoke lint format format-check typecheck lock \
        lab notebook tb \
        docs \
        pages-deploy pages-deploy-dry \
        build release \
        profile \
        mlflow \
        monitor tune serve query \
        clean clean-models clean-figures clean-all \
        run info help web \
        mcp mcp-factors mcp-token install-mcp setup-claude \
        init harness-check backlog \
        agents-list agents-run agents-doctor agents-test agents-eval \
        prompts-sync assistants-sync prompts-check \
        bot bot-start bot-daemon bot-stop bot-restart bot-logs \
        backup-bd restore-bd

# ─────────────────────────────────────────────────────────────────────────────
#  Variables
# ─────────────────────────────────────────────────────────────────────────────
MODULE   = climasafeai
ML_TYPE  = supervisado
PYTHON   = python

# `uv run` a secas sincroniza el venv contra las dependencias BASE antes de
# ejecutar, y eso desinstala todo lo que venga de un extra (torch, mlflow,
# xgboost, fastapi...). Cualquier `make test` dejaba el entorno cojo y el fallo
# aparecía después en otro sitio. Con --no-sync los targets solo ejecutan.
# Para instalar o actualizar dependencias: `make setup` (ese sí sincroniza).
UVRUN    = uv run --no-sync

# Los extras que forman el entorno real de trabajo. `uv sync` DESINSTALA lo que
# no esté en la lista, así que sincronizar con menos extras de los que usa el
# proyecto deja el venv cojo: `make setup` con solo dev+supervisado se llevaba
# torch, mlflow, fastapi, optuna y evidently por delante.
EXTRAS   = --extra dev --extra $(ML_TYPE) --extra no_supervisado \
           --extra redes_neuronales --extra mlflow_tracking --extra optuna \
           --extra api --extra monitoring --extra rag

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
	@echo "    make typecheck      chequeo de tipos con ty (informativo, no bloquea)"
	@echo ""
	@echo "  Empaquetado y release"
	@echo "    make build           uv build → dist/*.whl (wheel instalable)"
	@echo "    make release         git_agent: changelog + tag + bump (sin push)"
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
	@echo "    make docs           mkdocs build → site/ (documentacion/)"
	@echo ""
	@echo "  Arnés (ver AGENTS.md)"
	@echo "    make init           ./init.sh — la puerta: ¿se puede trabajar?"
	@echo "    make harness-check  ./init.sh --quick — sin tests"
	@echo "    make backlog        estado de featureslist.json"
	@echo ""
	@echo "  Sistema de agentes"
	@echo "    make agents-list    los 26 agentes disponibles"
	@echo "    make agents-doctor  diagnóstico integral del proyecto"
	@echo "    make agents-test    suite de tests de agents/"
	@echo "    make agents-eval    arnés + smoke + routing + contratos"
	@echo "    make prompts-sync   regenera prompts y subagentes desde el código"
	@echo ""
	@echo "  Limpieza"
	@echo "    make clean          cachés y __pycache__"
	@echo "    make clean-models   borra .joblib y .pt de models/"
	@echo "    make clean-figures  borra figuras de reports/figures/"
	@echo "    make clean-all      todo lo anterior"
	@echo ""
	@echo "  Bot Telegram (determinista)"
	@echo "    make bot            lanza el bot Telegram en foreground (alias de bot-start)"
	@echo "    make bot-start      lanza el bot Telegram en foreground (Ctrl+C)"
	@echo "    make bot-daemon     lanza en background con autoreinicio"
	@echo "    make bot-stop       para el bot"
	@echo "    make bot-restart    reinicia el bot"
	@echo "    make bot-logs       tail del log rotativo (logs/bot.log)"
	@echo ""
	@echo "  MCP (Claude Desktop / Cursor / VS Code)"
	@echo "    make mcp             MCP HTTPS autofirmado en :8101/mcp"
	@echo "    make mcp-http        MCP HTTP plano en :8101/mcp"
	@echo "    make mcp-stdio       MCP stdio (Claude Desktop local)"
	@echo "    make mcp-token ALIAS=<a> [ROL=admin]   emite el token MCP de un perfil"
	@echo "    make mcp-web         MCP + ngrok para Claude Web"
	@echo "    make install-mcp     instala symlink en ~/.local/bin/"
	@echo "    make setup-claude    genera claude_desktop_config.json"
	@echo ""


# ─────────────────────────────────────────────────────────────────────────────
#  Entorno
# ─────────────────────────────────────────────────────────────────────────────
setup:
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  Instalando dependencias para ML tipo: $(ML_TYPE)"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	uv sync $(EXTRAS)
	@echo ""
	@echo "  Listo. Activa el entorno con:  source .venv/bin/activate"
	@echo ""

install-deps:
	uv sync $(EXTRAS)

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
	$(UVRUN) $(PYTHON) $(MODULE)/data/make_dataset.py

features: data
	@echo "▶  Construyendo features → data/interim/"
	$(UVRUN) $(PYTHON) $(MODULE)/features/build_features.py

train: features
	@echo "▶  Entrenando modelo → models/"
	$(UVRUN) $(PYTHON) $(MODULE)/models/train_model.py

predict: train
	@echo "▶  Generando predicciones → reports/"
	$(UVRUN) $(PYTHON) $(MODULE)/models/predict_model.py

pipeline: predict
	@echo ""
	@echo "  Pipeline completo finalizado."
	@echo ""

# ─────────────────────────────────────────────────────────────────────────────
#  Ejecución directa
# ─────────────────────────────────────────────────────────────────────────────
run:
	$(UVRUN) $(PYTHON) main.py


# ─────────────────────────────────────────────────────────────────────────────
#  Monitoring — drift detection y performance tracking
# ─────────────────────────────────────────────────────────────────────────────
monitor:
	@echo "▶  Ejecutando monitorización de drift y rendimiento..."
	$(UVRUN) python -m monitoring.monitor



# ─────────────────────────────────────────────────────────────────────────────
#  Optuna — optimizacion de hiperparametros
# ─────────────────────────────────────────────────────────────────────────────
tune:
	@echo "▶  Lanzando optimizacion Optuna (OPTUNA_TRIALS en main.py)"
	$(UVRUN) python -m tuning.tune_model




# ─────────────────────────────────────────────────────────────────────────────
#  API REST — servir el modelo con FastAPI
# ─────────────────────────────────────────────────────────────────────────────
serve:
	@echo "▶  Lanzando API REST en http://localhost:8000"
	@echo "   Documentación interactiva: http://localhost:8000/docs"
	$(UVRUN) uvicorn api.main:app --reload --port 8000

 web:
	@echo "▶  Lanzando web chat en http://localhost:8000"
	$(UVRUN) uvicorn chat.app:app --reload --port 8000

mcp:
	@echo "▶  MCP Server — HTTPS autofirmado en https://localhost:8101/mcp"
	$(UVRUN) python -m agents.tools.prediction_mcp_tool

mcp-http:
	@echo "▶  MCP Server — HTTP plano en http://localhost:8101/mcp (Streamable HTTP)"
	$(UVRUN) python -m agents.tools.prediction_mcp_tool --insecure

mcp-token:
	@test -n "$(ALIAS)" || (echo "Uso: make mcp-token ALIAS=<alias> [ROL=admin]"; exit 1)
	$(UVRUN) python -m agents.tools.prediction_mcp_tool --emitir-token "$(ALIAS)" $(if $(ROL),--rol $(ROL),)

mcp-web: mcp-http
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  Exponiendo MCP para Claude Web via ngrok"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  1. En otra terminal: ngrok http 8101"
	@echo "  2. Copia la URL https://xxxx.ngrok.io"
	@echo "  3. Ve a claude.ai → Settings → Connectors"
	@echo "  4. Añade Custom Connector con URL: https://xxxx.ngrok.io/mcp"
	@echo "  5. Test connection → ¡Listo!"
	@echo ""
	@echo "  Si no tienes ngrok: sudo snap install ngrok"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

mcp-factors:
	@echo "▶  MCP Server — Factores en http://localhost:8100/mcp (Streamable HTTP)"
	$(UVRUN) python -m agents.tools.factors_mcp_tool

USER_BIN := $(shell echo $${HOME:-~})/.local/bin

install-mcp: $(USER_BIN)/climasafeai-mcp
	@echo "✅  climasafeai-mcp instalado en PATH: $(USER_BIN)/climasafeai-mcp"

$(USER_BIN)/climasafeai-mcp: .venv/bin/climasafeai-mcp
	@mkdir -p $(USER_BIN)
	ln -sf $(abspath $<) $(USER_BIN)/climasafeai-mcp

# ─────────────────────────────────────────────────────────────────────────────
#  Bot Telegram (determinista)
# ─────────────────────────────────────────────────────────────────────────────

bot: bot-start

bot-start:
	@echo "▶  Starting ClimaSafe Bot (foreground, Ctrl+C para parar)"
	@bash scripts/run_bot.sh

bot-daemon:
	@bash scripts/run_bot.sh --daemon

bot-stop:
	@bash scripts/run_bot.sh --stop

bot-restart:
	@bash scripts/run_bot.sh --restart

bot-logs:
	@echo "▶  ClimaSafe Bot logs"
	@tail -f logs/bot.log 2>/dev/null || echo "Todavía no hay logs (el bot no se ha iniciado)"

setup-claude:
	@$(UVRUN) python scripts/setup_claude_config.py

# ─────────────────────────────────────────────────────────────────────────────
#  Backup / restauración de la BD de perfiles (SEC-001)
# ─────────────────────────────────────────────────────────────────────────────

backup-bd:
	@echo "▶  Backup de la BD de perfiles (data/climasafe.db)"
	$(UVRUN) $(PYTHON) scripts/backup_bd.py backup

restore-bd:
	@test -n "$(ORIGEN)" || (echo "Uso: make restore-bd ORIGEN=ruta/al/backup.db"; exit 1)
	@echo "▶  Restaurando la BD desde $(ORIGEN) (con bot y web detenidos)"
	$(UVRUN) $(PYTHON) scripts/backup_bd.py restore "$(ORIGEN)"


profile:
	@echo "▶  Profiling main.py → reports/profile.prof"
	$(UVRUN) $(PYTHON) -m cProfile -o reports/profile.prof main.py
	@echo "   Visualiza con: $(UVRUN) snakeviz reports/profile.prof"

# ─────────────────────────────────────────────────────────────────────────────
#  Calidad de código
# ─────────────────────────────────────────────────────────────────────────────
test:
	@mkdir -p .pytest_tmp
	TMPDIR="$(CURDIR)/.pytest_tmp" $(UVRUN) pytest tests/ -v

smoke:
	@echo "▶  Test de humo — pipeline con datos sintéticos"
	TMPDIR="$(CURDIR)/.pytest_tmp" $(UVRUN) pytest tests/ -v -m smoke --tb=short

# ─────────────────────────────────────────────────────────────────────────────
#  Arnés — la puerta y el backlog (ver AGENTS.md)
# ─────────────────────────────────────────────────────────────────────────────
init:
	@chmod +x init.sh 2>/dev/null || true
	@./init.sh

harness-check:
	@chmod +x init.sh 2>/dev/null || true
	@./init.sh --quick

backlog:
	@$(UVRUN) $(PYTHON) -c "import json; d=json.load(open('featureslist.json')); \
	print(); print('  Backlog de ClimaSafeAI'); print(); \
	[print('  [%-11s] %-14s %s' % (f['status'], f['id'], f['title'])) for f in d['features']]; \
	print()"

# ─────────────────────────────────────────────────────────────────────────────
#  Sistema de agentes
# ─────────────────────────────────────────────────────────────────────────────
agents-list:
	$(UVRUN) python -m agents list

agents-run:
	$(UVRUN) python -m agents run $(filter-out $@,$(MAKECMDGOALS))

agents-doctor:
	$(UVRUN) python -m agents doctor

agents-test:
	$(UVRUN) pytest agents/tests/ -q

agents-eval:
	@echo "▶  Evaluación del sistema de agentes (arnés + smoke + routing + contracts)"
	$(UVRUN) python -m agents.evals.runner

prompts-sync assistants-sync:
	@echo "▶  Regenerando prompts y subagentes desde el código y los contratos..."
	$(UVRUN) python -m agents.prompts_sync --write

prompts-check:
	@$(UVRUN) python -m agents.prompts_sync

lint:
	$(UVRUN) ruff check $(MODULE)/ tests/

# El formato del repo completo tiene deuda preexistente (68 ficheros): `make
# lint` valida reglas (ruff check), no el formato. El formato es voluntario:
#   make format-check   → ruff format --check (informa, no bloquea)
#   make format         → aplica ruff format
# Ver .github/workflows/README.md — el CI lintea solo los .py cambiados.
format-check:
	$(UVRUN) ruff format --check $(MODULE)/ tests/ || true

format:
	$(UVRUN) ruff format $(MODULE)/ tests/

# ARNES-001 — `.opencode/agents/implementer.md` lleva desde el principio
# pidiendo `make typecheck` antes de devolver el control, y el target no
# existia: todas las features cerradas hasta ahora se lo saltaron.
#
# Es INFORMATIVO a proposito (`|| true`). Hoy salen 213 diagnosticos, y una
# parte son falsos positivos del checker con el scope de funciones largas
# (p. ej. `lat` en chat/app.py:1135 esta definida en la linea 837 de la misma
# funcion). Hacerlo bloqueante rompe la puerta de golpe por deuda preexistente
# y por ruido, que no es lo que se estaba pidiendo.
#
# `uvx` en vez de una dependencia nueva: no toca pyproject.toml.
typecheck:
	@echo "▶  Chequeo de tipos (informativo — no bloquea la puerta)"
	@uvx ty check $(MODULE)/ chat/ agents/ || true

# ─────────────────────────────────────────────────────────────────────────────
#  Jupyter
# ─────────────────────────────────────────────────────────────────────────────
lab:
	$(UVRUN) jupyter lab --ip=* --port=8888 --no-browser

notebook:
	$(UVRUN) jupyter notebook --ip=* --port=8888 --no-browser



# ─────────────────────────────────────────────────────────────────────────────
#  MLflow UI
# ─────────────────────────────────────────────────────────────────────────────
mlflow:
	@echo "Lanzando MLflow UI en http://localhost:5000"
	$(UVRUN) mlflow ui --port 5000



# ─────────────────────────────────────────────────────────────────────────────
#  Documentación (MkDocs Material sobre documentacion/ → site/)
# ─────────────────────────────────────────────────────────────────────────────
docs:
	$(UVRUN) mkdocs build

# ─────────────────────────────────────────────────────────────────────────────
#  Publicación a GitHub Pages personal (cacelass/cacelass.github.io)
#  Cross-repo: usa el checkout local del repo destino (o PAGES_REMOTE_URL).
#  Comparte scripts/pages_deploy.sh con .github/workflows/pages.yml.
#  - pages-deploy      copia site/ y web/probar-ya/ + commit + push
#  - pages-deploy-dry  igual pero sin push (PUSH=no), deja el commit local
#  Variables: PAGES_DIR, PAGES_REMOTE, PAGES_REMOTE_URL (ver .github/workflows/README.md)
# ─────────────────────────────────────────────────────────────────────────────
pages-deploy:
	bash scripts/pages_deploy.sh

pages-deploy-dry:
	PUSH=no bash scripts/pages_deploy.sh

# ─────────────────────────────────────────────────────────────────────────────
#  Empaquetado y release
#  - build    → `uv build` genera dist/*.whl + dist/*.tar.gz (wheel instalable)
#  - release  → delega en el git_agent del arnés: changelog + tag + bump.
#               Lee la versión de pyproject.toml (fuente única) y llama a
#               `git tag_release`, que hace bump en pyproject.toml/README.md,
#               actualiza CHANGELOG.md y crea el tag v<version> en un único
#               commit. El target NO ejecuta git por su cuenta (solo el
#               git_agent escribe el historial) y NO hace push — eso lo
#               decide el humano.
# ─────────────────────────────────────────────────────────────────────────────
build:
	uv build

release:
	@echo "▶  Release vía git_agent (changelog + tag + bump) — sin push."
	@$(UVRUN) python -m agents --json run git tag_release \
		--version "$$(python3 -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])")"

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
	@echo "  Python    : $(shell $(UVRUN) python --version 2>/dev/null || python --version)"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@uv pip list 2>/dev/null | head -40 || pip list | head -40
	@echo ""

# ─────────────────────────────────────────────────────────────────────────────
download-models:
	@echo "  Descargando/verificando modelos pre-entrenados..."
	@python -c "from main import download_models; download_models()" 2>/dev/null || echo "  Modelos en models/:" && ls models/*.joblib 2>/dev/null | wc -l && ls models/*.pt 2>/dev/null | wc -l
