# Graph Report - /home/cacelas/Documentos/anfaia/ClimaSafeAI  (2026-07-09)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 1077 nodes · 2216 edges · 71 communities (61 shown, 10 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 113 edges (avg confidence: 0.53)
- Token cost: 5,214 input · 795 output

## Graph Freshness
- Built from commit: `4c5415f4`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Git and Documentation Agent
- Dependency Management Agent
- Data Quality Agent
- Feature Engineering Pipeline
- Model Monitoring Service
- Agent System Overview
- Agent Installer Tool
- Data Acquisition and Visualization
- Model Training and Evaluation
- Shared Agent Context
- CI/CD Workflow Agent
- Notebook Analysis Agent
- Code Review Agent
- Meteorological Index Calculations
- API Validation Agent
- Secrets Scanning Agent
- Agent Orchestrator
- Model Hyperparameter Tuning
- Project Configuration Management
- Tool Registry and IO
- Chat Interface Web App
- MLflow Tracking Agent
- CLI and Context Provider
- Docker and Git Tools
- Dataset Generation Pipeline
- Machine Learning Analysis Agent
- Git Operations Tool
- Model Tuning Tests
- Base Agent Interface
- Daily Statistics Processing
- Filesystem Operations Tool
- Model Loading Tests
- REST API Tests
- Temporal Lag Features
- Model Explainability and SHAP
- Test Execution Agent
- Graph Inspection Agent
- Pytest Integration Tool
- Agent Discovery Registry
- DuckDB SQL Tool
- System Exception Hierarchy
- Dockerfile Linting Tests
- Agent Installation Inspector
- SQLite Database Tool
- Test Fixtures and Data
- Project Data Methodology
- Test Environment Configuration
- Geospatial Data Processing
- Risk Label Assignment
- Path Utility Tests
- Base Tool Interface
- Jupyter Task Automation
- Modeling Conclusions
- Core Agent Modules
- External Agent Modules
- Agent System Tests
- Docker Entrypoint Script
- Documentation Index
- Anomaly Detection Analysis
- Categorical Variable Distribution
- Numerical Variable Pairplot
- Core Project Package

## God Nodes (most connected - your core abstractions)
1. `AgentResult` - 101 edges
2. `BaseAgent` - 53 edges
3. `SharedContext` - 49 edges
4. `MissingDependencyError` - 34 edges
5. `GitAgent` - 33 edges
6. `ToolExecutionError` - 26 edges
7. `Orchestrator` - 26 edges
8. `ProcessResult` - 24 edges
9. `run_command()` - 24 edges
10. `train_models()` - 24 edges

## Surprising Connections (you probably didn't know these)
- `test_evaluate_models_signature()` --indirect_call--> `evaluate_models()`  [INFERRED]
  tests/test_proba.py → climasafeai/models/predict_model.py
- `test_train_models_signature()` --indirect_call--> `train_models()`  [INFERRED]
  tests/test_proba.py → climasafeai/models/train_model.py
- `Distribution of deaths by categorical variables` --references--> `MoMo (ISCIII) Mortality Data`  [INFERRED]
  agents/workspace/notebook/cell142_out1.png → documenatcion/diseño_modelo.md
- `test_build_models_expected_keys()` --calls--> `_build_models()`  [EXTRACTED]
  tests/test_train_model.py → climasafeai/models/train_model.py
- `test_build_models_returns_dict()` --calls--> `_build_models()`  [EXTRACTED]
  tests/test_train_model.py → climasafeai/models/train_model.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **ClimaSafe Modeling Strategy** — documenatcion_conclusiones_modelos, documenatcion_diseno_modelo, documenatcion_formulas_ml_resumen, documenatcion_formulas_riesgo_deterministico [EXTRACTED 1.00]
- **Data Sources Integration** — concept_momo_isciti, concept_era5_copernicus, documenatcion_diseno_modelo [EXTRACTED 0.95]

## Communities (71 total, 10 thin omitted)

### Community 0 - "Git and Documentation Agent"
Cohesion: 0.06
Nodes (29): DockerAgent, DocumentationAgent, Ejecuta `sphinx-apidoc` + build HTML, igual que `make docs`. Requiere         qu, Actualiza el número de versión en `pyproject.toml` (`version = "..."`)         y, Compara los targets reales de `Makefile` con las menciones de `make         <tar, Genera una entrada de changelog (vía GitAgent) e, si `dry_run=False`,         la, GitAgent, Actualiza CHANGELOG.md (delegando en `DocumentationAgent`, no         reimplemen (+21 more)

### Community 1 - "Dependency Management Agent"
Cohesion: 0.08
Nodes (31): DependencyAgent, Ejecuta `uv lock --check` (comando real de uv, verificado en su         document, Compara la versión bloqueada en `uv.lock` (o, si no existe, la         última de, Consulta advisorios OSV (vía la API de PyPI) para la versión bloqueada de cada d, Una herramienta falló al ejecutarse (proceso externo, IO, parsing...)., ToolExecutionError, test_check_outdated_without_dependencies_returns_empty(), test_check_outdated_without_pyproject_fails() (+23 more)

### Community 2 - "Data Quality Agent"
Cohesion: 0.07
Nodes (27): DataAgent, DataFrame, Path, agents.agents.data_agent — Análisis de calidad de datos para este template.  Con, Genera un informe EDA sobre `data/raw/<filename>` (o una ruta relativa         a, test_constant_columns_detected(), test_high_cardinality_columns_detected(), test_leakage_suspects_high_correlation() (+19 more)

### Community 3 - "Feature Engineering Pipeline"
Cohesion: 0.07
Nodes (43): _apply_logcols(), _feature_engineering(), preprocess_data(), process_input(), DataFrame, ndarray, Transformaciones y nuevas variables antes del modelado.     Edita esta función s, Aplica transformación logarítmica np.log1p() a las columnas indicadas.      Úsal (+35 more)

### Community 4 - "Model Monitoring Service"
Cohesion: 0.10
Nodes (34): _build_html(), check_drift(), check_performance(), DataFrame, Path, Calcula métricas actuales y las compara con el baseline guardado.      Si no exi, Genera un HTML minimalista con los resultados de monitorización., monitoring/monitor.py — Monitorización de drift y rendimiento del modelo.  Ejecu (+26 more)

### Community 5 - "Agent System Overview"
Cohesion: 0.11
Nodes (20): agents.agents.dependency_agent — Analiza pyproject.toml/uv.lock contra PyPI.  Ún, agents.agents.docker_agent — Revisión de la configuración Docker del proyecto., agents.agents.documentation_agent — Mantiene README, CHANGELOG y docs/ al día., agents.agents.git_agent — Automatización de Git para este template.  Conoce la e, agents.agents.graph_agent — Inspección de gráficos en `reports/figures/`.  Ver `, agents.agents.installer_agent — Instala agentes externos en `agents/external/`., agents.agents.ml_agent — Análisis de modelos entrenados para este template.  Con, agents.agents.mlflow_agent — Analiza experimentos MLflow del proyecto.  Solo apl (+12 more)

### Community 6 - "Agent Installer Tool"
Cohesion: 0.10
Nodes (22): InstallerAgent, Busca candidatos en `source_root`; si hay más de uno y no se dio         `subpat, _make_external_repo(), Path, A diferencia de los demás tests de este archivo, este NO usa la fixture     `con, test_install_from_git_no_candidates_fails(), test_install_from_git_valid_agent_end_to_end_via_cli(), test_install_from_git_warns_on_incomplete_structure() (+14 more)

### Community 7 - "Data Acquisition and Visualization"
Cohesion: 0.15
Nodes (28): download_era5_data(), download_momo_data(), Descarga el dataset de Momo desde la URL oficial y lo guarda en data/raw/momo_da, Descarga datos horarios de ERA5 para España desde hace 10 años.      Se genera u, plot_categorical_vs_target(), plot_class_balance(), plot_correlation_matrix(), plot_distributions() (+20 more)

### Community 8 - "Model Training and Evaluation"
Cohesion: 0.12
Nodes (30): evaluate_models(), predict_new(), predict_proba_new(), ndarray, Carga un modelo y predice sobre nuevas muestras (ya preprocesadas)., Carga un modelo y devuelve probabilidades de clase., Evalúa todos los modelos sobre train y test.     Métricas: Accuracy, F1 weighted, Entrena modelos de clasificacion y los guarda en models/.      Métrica CV: F1_we (+22 more)

### Community 9 - "Shared Agent Context"
Cohesion: 0.12
Nodes (6): Path, Solo existe cuando ml_type == 'redes_neuronales' (logs de TensorBoard)., Raíz donde los agentes guardan lo que generan (manifests, imágenes         extra, Crea (si hace falta) y devuelve `agents/workspace/<agent_name>/`., Rutas y configuración compartidas. Inmutable: si algo cambia, se crea otra insta, SharedContext

### Community 10 - "CI/CD Workflow Agent"
Cohesion: 0.12
Nodes (17): CICDAgent, agents.agents.cicd_agent — Genera y valida `.github/workflows/*.yml`.  Escribe d, test_generate_workflow_creates_valid_file(), test_generate_workflow_fails_without_project_slug(), test_generate_workflow_refuses_overwrite_without_flag(), test_list_workflows_empty_by_default(), test_validate_workflow_cross_references_makefile(), test_validate_workflow_detects_missing_runs_on() (+9 more)

### Community 11 - "Notebook Analysis Agent"
Cohesion: 0.13
Nodes (19): NotebookAgent, agents.agents.notebook_agent — Extrae salidas de notebooks e inserta interpretac, Extrae imágenes y texto de las salidas ya ejecutadas de `notebook_path`, Inserta celdas markdown con las interpretaciones ya escritas (por         quien, _make_notebook_with_image(), Path, test_extract_outputs_finds_image_and_text(), test_extract_outputs_missing_notebook_fails() (+11 more)

### Community 12 - "Code Review Agent"
Cohesion: 0.13
Nodes (18): Revisa `{{ project_slug }}/` por defecto (el paquete principal del         proye, ReviewAgent, Path, test_detects_bare_except(), test_detects_long_function(), test_detects_too_many_args(), test_finds_structurally_duplicated_functions(), test_parse_handles_syntax_error_gracefully() (+10 more)

### Community 13 - "Meteorological Index Calculations"
Cohesion: 0.11
Nodes (25): _procesar_era5_a_diario(), Convierte el xr.Dataset de ERA5 (ya filtrado a 5 puntos/provincia) en     un Dat, add_weather_index_columns(), categorize_heat_index(), celsius_to_fahrenheit(), fahrenheit_to_celsius(), heat_index(), kelvin_to_celsius() (+17 more)

### Community 14 - "API Validation Agent"
Cohesion: 0.16
Nodes (14): APIAgent, agents.agents.api_agent — Valida la API REST del proyecto (`api/main.py`).  Solo, _make_importable(), test_check_endpoints_documented_all_in_sync(), test_check_endpoints_documented_flags_undocumented_endpoint(), test_check_endpoints_documented_missing_file(), test_smoke_test_hits_real_health_endpoint(), test_smoke_test_reports_missing_app_attribute() (+6 more)

### Community 15 - "Secrets Scanning Agent"
Cohesion: 0.15
Nodes (16): SecretsAgent, test_heuristic_detects_aws_key(), test_heuristic_detects_prefixed_password_variable(), test_heuristic_detects_private_key_header(), test_heuristic_ignores_normal_low_entropy_strings(), test_heuristic_skips_git_directory(), test_shannon_entropy_high_for_random_looking_string(), test_shannon_entropy_low_for_repetitive_string() (+8 more)

### Community 16 - "Agent Orchestrator"
Cohesion: 0.15
Nodes (14): Orchestrator, Ejecuta una acción concreta de un agente concreto, sin pasar por el ruteo por ke, Rutea `query` en lenguaje natural al agente con mayor puntuación de         `can, RoutingDecision, Para cada palabra clave declarada por cada agente, usarla SOLA como     consulta, test_every_capability_keyword_routes_to_its_own_agent(), test_dispatch_auto_runs_zero_arg_action(), test_dispatch_disambiguates_commit_actions_via_aliases() (+6 more)

### Community 17 - "Model Hyperparameter Tuning"
Cohesion: 0.10
Nodes (13): load_data(), Carga el dataset desde la carpeta data/raw., _build_models(), _load_best_params(), Carga los mejores hiperparámetros cacheados si existen (de Optuna     para Rando, Busca el mejor valor de k para KNN vía GridSearchCV (5-fold,     F1_weighted --, Define los modelos a entrenar.     Tarea: clasificacion     RandomForest       →, _tune_knn_k() (+5 more)

### Community 18 - "Project Configuration Management"
Cohesion: 0.16
Nodes (18): _coerce(), env_override(), load_project_config(), _parse_flat_yaml(), ProjectConfig, Any, Path, agents.config — Configuración centralizada del sistema de agentes.  Este módulo (+10 more)

### Community 19 - "Tool Registry and IO"
Cohesion: 0.14
Nodes (12): Se pidió una herramienta que no está registrada., ToolNotFoundError, agents.tools.data_io_tool — Lectura/escritura de CSV, JSON y Parquet.  pandas, n, agents.tools — Herramientas reutilizables por cualquier agente.  Regla del siste, agents.tools.mlflow_tool — Compara runs de MLflow.  Grounding: `{{ project_slug, Any, agents.tools.registry — Registro de herramientas por nombre.  A diferencia del r, Decorador: `@register_tool("git")` sobre una clase o función factory. (+4 more)

### Community 20 - "Chat Interface Web App"
Cohesion: 0.14
Nodes (19): api_reload(), _handle_feature(), _info_message(), load_models(), predict_one(), _preprocess(), process_message(), Any (+11 more)

### Community 21 - "MLflow Tracking Agent"
Cohesion: 0.18
Nodes (12): MLflowAgent, Compara el run más reciente contra el inmediatamente anterior — no contra el his, _log_run(), mlflow_context(), test_compare_latest_detects_regression(), test_compare_latest_no_regression_has_no_warning(), test_list_runs_and_best_run(), test_list_runs_no_experiment_yet() (+4 more)

### Community 22 - "CLI and Context Provider"
Cohesion: 0.15
Nodes (13): build_parser(), main(), _parse_kwargs(), _print_result(), Any, agents.cli — CLI para usar el sistema de agentes desde línea de comandos.  Uso:, Convierte ['--target-col', 'target', '--max-count', '10'] en {'target_col': 'tar, get_context() (+5 more)

### Community 23 - "Docker and Git Tools"
Cohesion: 0.18
Nodes (12): DockerfileFinding, DockerTool, agents.tools.docker_tool — Envoltorio sobre el CLI `docker` / `docker compose` m, Valida docker-compose.yml (`docker compose config`) sin arrancar nada., agents.tools.git_tool — Envoltorio sobre el binario `git`.  Se usa `git` por sub, ProcessResult, Path, agents.tools.process_tool — Ejecución segura de comandos externos.  Todas las he (+4 more)

### Community 24 - "Dataset Generation Pipeline"
Cohesion: 0.18
Nodes (19): cargar_era5_filtrado(), dataset_calor(), dataset_frio(), download_aemet(), download_openuv(), filtrar_era5_por_puntos(), _procesar_momo_provincial(), process_data() (+11 more)

### Community 25 - "Machine Learning Analysis Agent"
Cohesion: 0.19
Nodes (7): MLAgent, Any, Path, Resumen genérico de un estimador ya entrenado: tipo, params, tamaño en memoria., Devuelve {feature: importancia} ordenado descendente, o None si el         estim, Heurística simple: si train_score - test_score supera `gap_threshold`         (e, SklearnTool

### Community 26 - "Git Operations Tool"
Cohesion: 0.21
Nodes (3): GitTool, Devuelve [(código_estado, ruta), ...] tal y como `git status --porcelain`., Devuelve una lista de commits como dicts {hash, subject, author, date}.

### Community 27 - "Model Tuning Tests"
Cohesion: 0.18
Nodes (13): _make_Xy(), test_tuning.py — Tests de la optimizacion de hiperparametros con Optuna.  Usa n_, tune_models debe devolver un dict con params por modelo., tune_models debe guardar best_params_<modelo>.joblib en artifacts/., tune_models debe guardar reports/tuning_results.csv., Tras tune_models, train_models debe cargar best_params sin error., test_train_models_usa_best_params(), test_tune_models_devuelve_dict() (+5 more)

### Community 28 - "Base Agent Interface"
Cohesion: 0.17
Nodes (9): BaseAgent, Any, Hook opcional: {nombre_de_accion: [palabras clave adicionales]}.         `best_a, Adivina qué acción de `self.actions()` encaja mejor con `query`, por         sol, True si `action_name` se puede ejecutar sin argumentos adicionales         (todo, Clase base de todos los agentes.      Subclases obligatorias a definir:, Devuelve {nombre_accion: metodo} para despacho uniforme vía run()., Despacho genérico: `agent.run("suggest_commit_message")`. (+1 more)

### Community 29 - "Daily Statistics Processing"
Cohesion: 0.16
Nodes (15): _agregar_estadisticas_diarias(), Estadísticas de la DISTRIBUCIÓN diaria (24h) del calor/frío, no solo el     pico, _df_horario(), test_make_dataset.py — Tests para climasafeai/data/make_dataset.py, load_data debe leer un CSV válido y devolver un DataFrame., Integración: _procesar_era5_a_diario debe MANTENER heat_index_c (pico) y     AÑA, load_data debe lanzar una excepción si el archivo no existe., Helper: DataFrame horario de un único (provincia, fecha). (+7 more)

### Community 30 - "Filesystem Operations Tool"
Cohesion: 0.27
Nodes (5): FilesystemTool, PathEscapesRootError, Exception, Path, agents.tools.filesystem_tool — Operaciones de archivo con la raíz del proyecto c

### Community 31 - "Model Loading Tests"
Cohesion: 0.21
Nodes (14): load_models(), Carga modelos entrenados desde disco, namespaceados por clase     (XGBoost_calor, _make_Xy(), test_train_model.py — Tests para climasafeai/models/train_model.py, Datos sintéticos pequeños para entrenamiento rápido en tests., Si no hay modelos guardados, load_models() debe devolver dict vacío., test_build_models_expected_keys(), test_build_models_returns_dict() (+6 more)

### Community 32 - "REST API Tests"
Cohesion: 0.18
Nodes (10): _inject_model(), test_api.py — Tests de la API REST de ClimaSafeAI.  Usa el cliente de test de Fa, Resetea el estado global entre tests y parchea rutas., Entrena un modelo mínimo e inyecta en _state para tests., reset_state(), test_health_con_modelo(), test_predict_clasificacion_tiene_probabilidad(), test_predict_con_modelo_ok() (+2 more)

### Community 33 - "Temporal Lag Features"
Cohesion: 0.19
Nodes (14): _agregar_rezagos_temporales(), Features de PERSISTENCIA entre días (no dentro del día): capturan que el     rie, _df_diario(), Helper: DataFrame diario (una fila por fecha) de una provincia., heat_index_c_lag1 debe ser el heat_index_c del día anterior (NaN el 1º)., wind_chill_mean_roll3 = media de los 3 días PREVIOS (hoy no cuenta)., dias_consec_bajo_umbral = nº de días fríos consecutivos ANTERIORES., El lag de una provincia no debe usar filas de otra. (+6 more)

### Community 34 - "Model Explainability and SHAP"
Cohesion: 0.18
Nodes (13): _compute_shap(), explain_models(), _plot_confusion_matrix(), DataFrame, Genera explicaciones SHAP para cada modelo entrenado.      Por cada modelo produ, Selecciona el explainer adecuado y devuelve (shap_values, X_explain)., predict_model.py — Evaluación de modelos supervisado. Tarea: clasificacion, Barra de importancia global media (|SHAP|). (+5 more)

### Community 35 - "Test Execution Agent"
Cohesion: 0.21
Nodes (7): Heurística de convención de nombres: un módulo `foo.py` se considera         "co, Equivalente a 'make smoke': solo los tests marcados @pytest.mark.smoke., TestAgent, test_coverage_report_without_project_slug_fails(), test_list_untested_modules_detects_missing_and_present(), test_list_untested_modules_excludes_init(), test_list_untested_modules_missing_package_dir_fails()

### Community 36 - "Graph Inspection Agent"
Cohesion: 0.24
Nodes (6): GraphAgent, FigureMetrics, Path, agents.tools.vision_tool — Inspección estructural de gráficos en reports/figures, Lee un PNG con `matplotlib.image.imread` (no requiere PIL) y calcula         mét, VisionTool

### Community 37 - "Pytest Integration Tool"
Cohesion: 0.27
Nodes (5): Path, PytestTool, agents.tools.pytest_tool — Ejecuta pytest y parsea sus reportes.  Formatos verif, TestFailure, TestRunSummary

### Community 38 - "Agent Discovery Registry"
Cohesion: 0.27
Nodes (4): AgentRegistry, Registro global {nombre: clase_agente}. Un único singleton: `agent_registry`., Importa todos los módulos de `agents.agents` (núcleo del template),         `age, Importa agentes expuestos como entry point `dskit.agents` por         paquetes i

### Community 39 - "DuckDB SQL Tool"
Cohesion: 0.27
Nodes (6): DuckDBTool, Any, Path, agents.tools.duckdb_tool — Consultas SQL sobre CSV/Parquet/JSON con DuckDB.  `du, Ejecuta `sql` y devuelve un DataFrame de pandas.         Ejemplo: DuckDBTool.que, Perfil rápido de un CSV/Parquet vía DESCRIBE de DuckDB (tipos inferidos, no esta

### Community 40 - "System Exception Hierarchy"
Cohesion: 0.28
Nodes (8): ActionNotSupportedError, AgentNotFoundError, AgentSystemError, Exception, agents.exceptions — Jerarquía de excepciones propia del sistema de agentes.  Usa, Excepción base de todo el sistema de agentes., Se pidió un agente que no está registrado., Se pidió una acción que el agente no expone.

### Community 41 - "Dockerfile Linting Tests"
Cohesion: 0.39
Nodes (7): Path, test_lint_does_not_flag_pinned_base_image(), test_lint_flags_missing_user(), test_lint_flags_unpinned_base_image(), test_lint_missing_file_returns_single_warning(), Path, Analiza un Dockerfile en busca de malas prácticas conocidas y bien         docum

### Community 42 - "Agent Installation Inspector"
Cohesion: 0.33
Nodes (8): _find_suspicious_literal_paths(), _inspect_agent_class(), agents.tools.agent_installer_tool — Clona/copia un agente externo, lo valida est, True si en algún sitio del cuerpo de la clase se referencia `self.ctx`., Strings literales que parecen rutas fijas a carpetas del proyecto, en vez de res, Extrae `name = "..."` del cuerpo de la clase (best-effort) y avisa si faltan atr, _uses_shared_context(), ClassDef

### Community 43 - "SQLite Database Tool"
Cohesion: 0.50
Nodes (4): Any, Path, SQLiteTool, Connection

### Community 44 - "Test Fixtures and Data"
Cohesion: 0.25
Nodes (7): df_with_target(), patch_paths(), conftest.py — Fixtures compartidas para todos los tests. Los fixtures se adaptan, DataFrame genérico con 8 columnas numéricas (200 filas)., DataFrame con features numéricas + columna target binaria., Redirige todas las constantes de ruta del proyecto a tmp_path.     Se aplica aut, sample_df()

### Community 45 - "Project Data Methodology"
Cohesion: 0.29
Nodes (7): ERA5 Meteorological Data, Hybrid Model (ML + Deterministic), MoMo (ISCIII) Mortality Data, Diseño del modelo — ClimaSafe, Fórmulas + ML vs. Aprendizaje Directo, Fórmulas de riesgo climático determinista, Distribution of deaths by categorical variables

### Community 46 - "Test Environment Configuration"
Cohesion: 0.40
Nodes (5): context(), project_root(), Path, agents.tests.conftest — Fixtures compartidas para los tests del sistema de agent, Un proyecto mínimo en un directorio temporal: estructura de carpetas del     tem

### Community 47 - "Geospatial Data Processing"
Cohesion: 0.33
Nodes (6): calcular_puntos_provincia(), cargar_provincias_unificadas(), _extremos_polygon(), Extremos N/S/E/O reales sobre el borde de un Polygon (no bounding box)., Para cada provincia, calcula 5 puntos representativos: centro + N/S/E/O     real, GeoDataFrame

### Community 48 - "Risk Label Assignment"
Cohesion: 0.47
Nodes (5): asignar_clase_riesgo_calor(), asignar_clase_riesgo_frio(), DataFrame, labels.py -- Etiquetas de riesgo (SEGURO/PRECAUCION/PELIGRO) a partir de percent, Asigna la clase de riesgo (0=SEGURO, 1=PRECAUCION, 2=PELIGRO) a partir     de lo

### Community 49 - "Path Utility Tests"
Cohesion: 0.33
Nodes (5): test_paths.py — Tests para climasafeai/utils/paths.py Común a todos los ml_type., Todas las constantes de ruta deben ser instancias de Path., make_dirs() debe crear todos los subdirectorios necesarios.     Se prueba con un, test_all_path_constants_are_path_objects(), test_project_dir_contains_pyproject()

### Community 50 - "Base Tool Interface"
Cohesion: 0.50
Nodes (4): BaseTool, ABC, agents.tools.base_tool — Contrato mínimo que deben cumplir las herramientas., Clase base opcional para herramientas con estado (p. ej. una conexión     DuckDB

### Community 52 - "Modeling Conclusions"
Cohesion: 0.67
Nodes (3): Conclusiones del modelado — ClimaSafeAI, _agregar_rezagos_temporales, evaluate_models

## Knowledge Gaps
- **16 isolated node(s):** `entrypoint.sh script`, `climasafeai`, `ClimaSafeAI Documentation Index`, `Ayuda - Recursos de Referencia`, `Fórmulas + ML vs. Aprendizaje Directo` (+11 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **10 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `AgentResult` connect `Git and Documentation Agent` to `Dependency Management Agent`, `Data Quality Agent`, `Agent System Overview`, `Agent Installer Tool`, `Shared Agent Context`, `CI/CD Workflow Agent`, `Notebook Analysis Agent`, `Code Review Agent`, `API Validation Agent`, `Secrets Scanning Agent`, `Agent Orchestrator`, `MLflow Tracking Agent`, `CLI and Context Provider`, `Machine Learning Analysis Agent`, `Base Agent Interface`, `Test Execution Agent`, `Graph Inspection Agent`, `Pytest Integration Tool`, `System Exception Hierarchy`?**
  _High betweenness centrality (0.190) - this node is a cross-community bridge._
- **Why does `SharedContext` connect `Shared Agent Context` to `Git and Documentation Agent`, `Test Execution Agent`, `Agent System Overview`, `CI/CD Workflow Agent`, `Test Environment Configuration`, `Agent Orchestrator`, `Project Configuration Management`, `MLflow Tracking Agent`, `CLI and Context Provider`, `Base Agent Interface`?**
  _High betweenness centrality (0.076) - this node is a cross-community bridge._
- **Why does `BaseAgent` connect `Base Agent Interface` to `Git and Documentation Agent`, `Dependency Management Agent`, `Data Quality Agent`, `Agent System Overview`, `Agent Installer Tool`, `Shared Agent Context`, `CI/CD Workflow Agent`, `Notebook Analysis Agent`, `Code Review Agent`, `API Validation Agent`, `Secrets Scanning Agent`, `Agent Orchestrator`, `MLflow Tracking Agent`, `CLI and Context Provider`, `Machine Learning Analysis Agent`, `Test Execution Agent`, `Graph Inspection Agent`, `Agent Discovery Registry`, `System Exception Hierarchy`?**
  _High betweenness centrality (0.062) - this node is a cross-community bridge._
- **Are the 20 inferred relationships involving `AgentResult` (e.g. with `APIAgent` and `CICDAgent`) actually correct?**
  _`AgentResult` has 20 INFERRED edges - model-reasoned connections that need verification._
- **Are the 21 inferred relationships involving `BaseAgent` (e.g. with `APIAgent` and `CICDAgent`) actually correct?**
  _`BaseAgent` has 21 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `SharedContext` (e.g. with `ProjectConfig` and `AgentResult`) actually correct?**
  _`SharedContext` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 15 inferred relationships involving `MissingDependencyError` (e.g. with `DependencyAgent` and `DockerAgent`) actually correct?**
  _`MissingDependencyError` has 15 INFERRED edges - model-reasoned connections that need verification._