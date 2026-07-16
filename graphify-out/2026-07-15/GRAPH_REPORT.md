# Graph Report - ClimaSafeAI  (2026-07-15)

## Corpus Check
- 256 files · ~421,375 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 2462 nodes · 4392 edges · 181 communities (141 shown, 40 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 170 edges (avg confidence: 0.51)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `24046753`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- TestAgent
- PlanAgent
- GStack
- ResearchTool
- DependencyAgent
- GitAgent
- preprocess_data
- InstallerAgent
- BaseAgent
- DataFrameAnalysisTool
- test_monitoring.py
- train_models
- AgentResult
- ToolExecutionError
- SharedContext
- Orchestrator
- agent_installer_tool.py
- research_agent.py
- .analyze_file
- CICDAgent
- NotebookAgent
- ScheduleAgent
- make_dataset.py
- .actions
- StatsTool
- .lint_dockerfile
- APIAgent
- .__init__
- app.py
- test_secrets_agent.py
- LSTM multi-tarea
- test_make_dataset.py
- MLAgent
- GraphifyTool
- registry.py
- Changelog
- lstm_province_hybrid.py
- DoctorAgent
- 1. Golpe de calor
- FilesystemTool
- SharedContext
- MissingDependencyError
- test_contracts.py
- ValidateTool
- test_api.py
- .load_graph
- RefactorAgent
- evaluate_lstm_province
- calibracion_umbrales.py
- Arquitectura
- GraphAgent
- personalizar_riesgo
- weather_indices.py
- orchestrator.py
- asignar_clase_riesgo_calor
- _pesos_de_clase
- ParallelTool
- train_models
- DuckDBTool
- SQLiteTool
- Conclusiones del modelado — ClimaSafeAI
- conftest.py
- agents/ — Sistema de agentes de este template
- Diseño del modelo — ClimaSafe
- SecretsAgent
- GitTool
- PlanAgent
- Features — Documentación completa
- GraphifyTool
- base_tool.py
- tasks.py
- main
- Conclusiones del modelado — ClimaSafeAI
- features_frio.py
- paths.py
- Path
- entrypoint.sh
- ClimaSafeAI Documentation Index
- Calibración de umbrales de decisión por clase
- MakeAgent
- LSTMProvinceHybridMultiTask
- test_tuning.py
- DocSearchAgent
- .detect_obsidian_vaults
- test_git_agent.py
- Coeficientes de personalización individual del riesgo
- test_proba.py
- .validate
- MLflowAgent
- StackStep
- Acciones
- NotebookAgent — Guía de análisis e inserción de comentarios
- test_plan_agent.py
- Diseño del modelo — ClimaSafe
- Features de frío con más memoria temporal (retardo epidemiológico)
- Fórmulas + ML vs. aprendizaje directo de secuencias — resumen
- Label sin fuga temporal: percentiles train-only
- Assistant (Build · DeepSeek V4 Flash Free · 47.3s)
- predict_new
- LSTM híbrida — contexto de ola para la LSTM multi-tarea
- ablacion_27v19.py
- TestAgent.md
- Acciones
- Acciones
- explain_models
- Ayuda
- MLflow — Experimentos
- Grafo de Conocimiento (Graphify)
- .list_runs
- Acciones
- Ablación de features: 27 vs 19 con el mismo label
- {{title}}
- _index.md
- Actualización del pipeline principal
- agents/external/
- Acciones
- construir_secuencias_24h
- {{title}}
- CICDAgent.md
- DependencyAgent.md
- DocumentationAgent.md
- get_context
- Assistant (Plan · DeepSeek V4 Flash Free · 14.7s)
- Assistant (Plan · DeepSeek V4 Flash Free · 18.6s)
- Assistant (Plan · DeepSeek V4 Flash Free · 26.5s)
- Plan propuesto (4 fases)
- Guía para la IA
- {{title}}
- Doctor Agent
- context
- try_model
- Work State
- Assistant (Plan · DeepSeek V4 Flash Free · 16.9s)
- tune_model.py
- Roadmap
- Agentes — Índice
- 2026-07-09 — Organización del vault + Graphify
- .event_log_path
- Assistant (Plan · DeepSeek V4 Flash Free · 7.8s)
- DataAgent.md
- GraphAgent.md
- InstallerAgent.md
- MLflowAgent.md
- __init__.py
- .can_handle
- __init__.py
- api_agent.md
- audit_agent.md
- cicd_agent.md
- data_agent.md
- dependency_agent.md
- docker_agent.md
- documentation_agent.md
- env_agent.md
- git_agent.md
- graph_agent.md
- installer_agent.md
- make_agent.md
- ml_agent.md
- mlflow_agent.md
- orchestrator.md
- plan_agent.md
- refactor_agent.md
- review_agent.md
- test_agent.md
- __init__.py
- README.md
- agentes_ia.md
- Assistant (Plan · DeepSeek V4 Flash Free · 114.6s)
- Assistant (Plan · DeepSeek V4 Flash Free · 7.6s)
- 2026-07-09_inicio_vault.md
- climasafeai

## God Nodes (most connected - your core abstractions)
1. `AgentResult` - 181 edges
2. `New session - 2026-07-14T06:34:58.880Z` - 109 edges
3. `BaseAgent` - 77 edges
4. `SharedContext` - 50 edges
5. `Orchestrator` - 45 edges
6. `MissingDependencyError` - 38 edges
7. `GraphifyTool` - 37 edges
8. `run_command()` - 35 edges
9. `GitAgent` - 33 edges
10. `register_agent()` - 33 edges

## Surprising Connections (you probably didn't know these)
- `test_evaluate_models_signature()` --indirect_call--> `evaluate_models()`  [INFERRED]
  tests/test_proba.py → climasafeai/models/predict_model.py
- `test_train_models_signature()` --indirect_call--> `train_models()`  [INFERRED]
  tests/test_proba.py → climasafeai/models/train_model.py
- `test_preprocess_data_signature()` --indirect_call--> `preprocess_data()`  [INFERRED]
  tests/test_proba.py → climasafeai/features/build_features.py
- `test_procesar_era5_a_diario_anade_estadisticas_diarias()` --calls--> `_procesar_era5_a_diario()`  [EXTRACTED]
  tests/test_make_dataset.py → climasafeai/data/make_dataset.py
- `test_load_data_signature()` --indirect_call--> `load_data()`  [INFERRED]
  tests/test_proba.py → climasafeai/data/make_dataset.py

## Import Cycles
- None detected.

## Communities (181 total, 40 thin omitted)

### Community 0 - "TestAgent"
Cohesion: 0.21
Nodes (7): Heurística de convención de nombres: un módulo `foo.py` se considera         "co, Equivalente a 'make smoke': solo los tests marcados @pytest.mark.smoke., TestAgent, test_coverage_report_without_project_slug_fails(), test_list_untested_modules_detects_missing_and_present(), test_list_untested_modules_excludes_init(), test_list_untested_modules_missing_package_dir_fails()

### Community 1 - "PlanAgent"
Cohesion: 0.16
Nodes (10): AuditAgent, Los fallos más recientes con su mensaje — por dónde empezar a mejorar., Sugerencias deterministas a partir del log. Cada una dice el síntoma         y l, Agrega el log por 'agente.acción' → runs/ok/fail/avg_ms/warnings., Informe de uso por agente/acción sobre las últimas `last` ejecuciones., Tests de la auditoría: toda ejecución vía run() queda registrada y es analizable, test_audit_agent_report_and_suggestions(), test_audit_report_on_empty_log() (+2 more)

### Community 2 - "GStack"
Cohesion: 0.11
Nodes (28): build_parser(), main(), _parse_kwargs(), _print_result(), Any, agents.cli — CLI para usar el sistema de agentes desde línea de comandos.  Uso:, Convierte ['--target-col', 'target', '--max-count', '10'] en {'target_col': 'tar, gstack — Git Stack: flujos de trabajo autónomos para el sistema de agentes.  Per (+20 more)

### Community 3 - "ResearchTool"
Cohesion: 0.07
Nodes (25): Busca papers para una consulta concreta en un backend (arxiv|openalex)., Busca papers relacionados con el PROYECTO: deriva sus keywords y consulta, Extrae las palabras clave que describen el proyecto., Métrica de una propuesta de papers: 0.5·relevancia media + 0.4·cobertura, Enfrenta candidatos arbitrarios. Cada candidato es un dict         ``{"agent": s, Heurística genérica para puntuar un AgentResult: premia el éxito, la         riq, Cada backend de `research` es un worker. Corren en paralelo, cada uno         pr, SupervisorAgent (+17 more)

### Community 4 - "DependencyAgent"
Cohesion: 0.09
Nodes (28): DependencyAgent, agents.agents.dependency_agent — Analiza pyproject.toml/uv.lock contra PyPI.  Ún, Compara la versión bloqueada en `uv.lock` (o, si no existe, la         última de, Consulta advisorios OSV (vía la API de PyPI) para la versión bloqueada de cada d, test_check_outdated_without_dependencies_returns_empty(), test_check_outdated_without_pyproject_fails(), test_check_vulnerabilities_without_lock_fails(), test_normalize_package_name_pep503() (+20 more)

### Community 5 - "GitAgent"
Cohesion: 0.09
Nodes (21): DockerAgent, DocumentationAgent, agents.agents.documentation_agent — Mantiene README, CHANGELOG y docs/ al día., Actualiza el número de versión en `pyproject.toml` (`version = "..."`)         y, Compara los targets reales de `Makefile` con las menciones de `make         <tar, Genera una entrada de changelog (vía GitAgent) e, si `dry_run=False`,         la, GitAgent, agents.agents.git_agent — Automatización de Git para este template.  Conoce la e (+13 more)

### Community 6 - "preprocess_data"
Cohesion: 0.07
Nodes (45): _apply_logcols(), _feature_engineering(), preprocess_data(), process_input(), DataFrame, ndarray, Pipeline completo de preprocesado para aprendizaje supervisado.      Pasos:, Transformaciones y nuevas variables antes del modelado.     Edita esta función s (+37 more)

### Community 7 - "InstallerAgent"
Cohesion: 0.14
Nodes (16): InstallerAgent, Busca candidatos en `source_root`; si hay más de uno y no se dio         `subpat, _make_external_repo(), Path, A diferencia de los demás tests de este archivo, este NO usa la fixture     `con, test_install_from_git_no_candidates_fails(), test_install_from_git_valid_agent_end_to_end_via_cli(), test_install_from_git_warns_on_incomplete_structure() (+8 more)

### Community 8 - "BaseAgent"
Cohesion: 0.06
Nodes (43): agents.agents.audit_agent — Auditor del equipo de agentes.  Lee `agents/workspac, agents.agents.data_agent — Análisis de calidad de datos para este template.  Con, agents.agents.docker_agent — Revisión de la configuración Docker del proyecto., agents.agents.doctor_agent — Diagnóstico integral del proyecto.  Revisa el estad, agents.agents.env_agent — Gestión del entorno de desarrollo.  Conoce `pyproject., agents.agents.installer_agent — Instala agentes externos en `agents/external/`., agents.agents.make_agent — Validación y gestión del Makefile.  Conoce los target, agents.agents.ml_agent — Análisis de modelos entrenados para este template.  Con (+35 more)

### Community 9 - "DataFrameAnalysisTool"
Cohesion: 0.09
Nodes (22): DataAgent, DataFrame, Path, Genera un informe EDA sobre `data/raw/<filename>` (o una ruta relativa         a, test_constant_columns_detected(), test_high_cardinality_columns_detected(), test_leakage_suspects_high_correlation(), test_leakage_suspects_missing_target_returns_empty() (+14 more)

### Community 10 - "test_monitoring.py"
Cohesion: 0.09
Nodes (35): _build_html(), check_drift(), check_performance(), DataFrame, Path, Calcula métricas actuales y las compara con el baseline guardado.      Si no exi, Genera un HTML minimalista con los resultados de monitorización., monitoring/monitor.py — Monitorización de drift y rendimiento del modelo.  Ejecu (+27 more)

### Community 11 - "train_models"
Cohesion: 0.19
Nodes (21): plot_categorical_vs_target(), plot_class_balance(), plot_correlation_matrix(), plot_distributions(), plot_feature_importance(), plot_pairplot(), plot_pca_variance(), DataFrame (+13 more)

### Community 12 - "AgentResult"
Cohesion: 0.14
Nodes (9): Ejecuta `uv lock --check` (comando real de uv, verificado en su         document, Ejecuta `sphinx-apidoc` + build HTML, igual que `make docs`. Requiere         qu, EnvAgent, Ejecuta `uv sync` con los extras indicados (separados por coma)., Verifica que pyproject.toml y uv.lock estén sincronizados., Añade una dependencia con `uv add`., Path, Ejecuta `args` (nunca a través de una shell — evita inyección de comandos)     y (+1 more)

### Community 13 - "ToolExecutionError"
Cohesion: 0.18
Nodes (8): DockerTool, Valida docker-compose.yml (`docker compose config`) sin arrancar nada., ProcessResult, Path, PytestTool, agents.tools.pytest_tool — Ejecuta pytest y parsea sus reportes.  Formatos verif, TestFailure, TestRunSummary

### Community 14 - "SharedContext"
Cohesion: 0.14
Nodes (20): _coerce(), env_override(), load_project_config(), _parse_flat_yaml(), ProjectConfig, Any, Path, agents.config — Configuración centralizada del sistema de agentes.  Este módulo (+12 more)

### Community 15 - "Orchestrator"
Cohesion: 0.13
Nodes (17): ActionNotSupportedError, Se pidió una acción que el agente no expone., Orchestrator, agents.orchestrator — Decide qué agente(s) usar para una tarea.  Ruteo determini, Ejecuta una acción concreta de un agente concreto, sin pasar por el ruteo por ke, Rutea `query` en lenguaje natural al agente con mayor puntuación de         `can, RoutingDecision, Para cada palabra clave declarada por cada agente, usarla SOLA como     consulta (+9 more)

### Community 16 - "agent_installer_tool.py"
Cohesion: 0.17
Nodes (14): AgentCandidate, AgentInstallerTool, _decorator_name(), _find_suspicious_literal_paths(), _inspect_agent_class(), Path, agents.tools.agent_installer_tool — Clona/copia un agente externo, lo valida est, Busca en `root` (recursivo) archivos .py que definan una clase         decorada (+6 more)

### Community 17 - "research_agent.py"
Cohesion: 0.11
Nodes (15): agents.agents.cache_agent — Agente que cachea al máximo las operaciones del graf, agents.agents.docsearch_agent — Búsqueda y navegación por el grafo de docs.  Com, agents.agents.knowledge_agent — Grafo de conocimiento + Obsidian, cacheado.  Est, audit_log_path(), Path, agents.audit — Registro de auditoría de todas las ejecuciones de agentes.  Cada, Ruta del log de auditoría (crea `agents/workspace/audit/` si no existe)., Añade una entrada al log. Nunca lanza: la auditoría no rompe lo auditado. (+7 more)

### Community 18 - ".analyze_file"
Cohesion: 0.13
Nodes (18): Revisa `{{ project_slug }}/` por defecto (el paquete principal del         proye, ReviewAgent, Path, test_detects_bare_except(), test_detects_long_function(), test_detects_too_many_args(), test_finds_structurally_duplicated_functions(), test_parse_handles_syntax_error_gracefully() (+10 more)

### Community 19 - "CICDAgent"
Cohesion: 0.29
Nodes (9): CICDAgent, test_generate_workflow_creates_valid_file(), test_generate_workflow_fails_without_project_slug(), test_generate_workflow_refuses_overwrite_without_flag(), test_list_workflows_empty_by_default(), test_validate_workflow_cross_references_makefile(), test_validate_workflow_detects_missing_runs_on(), test_validate_workflow_missing_file() (+1 more)

### Community 20 - "NotebookAgent"
Cohesion: 0.14
Nodes (18): NotebookAgent, Extrae imágenes y texto de las salidas ya ejecutadas de `notebook_path`, Inserta celdas markdown con las interpretaciones ya escritas (por         quien, _make_notebook_with_image(), Path, test_extract_outputs_finds_image_and_text(), test_extract_outputs_missing_notebook_fails(), test_insert_comments_default_does_not_touch_original() (+10 more)

### Community 21 - "ScheduleAgent"
Cohesion: 0.12
Nodes (15): Valida si una expresión cron es correcta., Convierte una expresión cron a texto legible., Calcula las próximas ejecuciones de una expresión cron., ScheduleAgent, _expand(), _parse_field(), Any, agents.tools.schedule_tool — Parseo y utilidades de cron.  Sin dependencias exte (+7 more)

### Community 22 - "make_dataset.py"
Cohesion: 0.08
Nodes (47): calcular_puntos_provincia(), cargar_era5_filtrado(), cargar_provincias_unificadas(), dataset_calor(), dataset_frio(), download_aemet(), download_era5_data(), download_momo_data() (+39 more)

### Community 23 - ".actions"
Cohesion: 0.02
Nodes (99): Assistant (Build · DeepSeek V4 Flash Free · 10.4s), Assistant (Build · DeepSeek V4 Flash Free · 10.9s), Assistant (Build · DeepSeek V4 Flash Free · 12.3s), Assistant (Build · DeepSeek V4 Flash Free · 14.0s), Assistant (Build · DeepSeek V4 Flash Free · 14.0s), Assistant (Build · DeepSeek V4 Flash Free · 1.9s), Assistant (Build · DeepSeek V4 Flash Free · 1.9s), Assistant (Build · DeepSeek V4 Flash Free · 20.6s) (+91 more)

### Community 24 - "StatsTool"
Cohesion: 0.13
Nodes (12): Any, Cohen's d: diferencia estandarizada entre dos muestras independientes., Contrasta H₀: la muestra proviene de una distribución normal.          method :, H₀: medias de dos muestras independientes iguales (t de Student)., H₀: medias de dos muestras apareadas iguales., H₀: distribuciones de dos muestras iguales (U de Mann-Whitney, no paramétrico)., H₀: frecuencias observadas = frecuencias esperadas (χ² bondad de ajuste)., H₀: dos variables categóricas son independientes (χ² sobre tabla de contingencia (+4 more)

### Community 25 - ".lint_dockerfile"
Cohesion: 0.33
Nodes (8): Path, test_lint_does_not_flag_pinned_base_image(), test_lint_flags_missing_user(), test_lint_flags_unpinned_base_image(), test_lint_missing_file_returns_single_warning(), DockerfileFinding, Path, Analiza un Dockerfile en busca de malas prácticas conocidas y bien         docum

### Community 26 - "APIAgent"
Cohesion: 0.19
Nodes (11): APIAgent, _make_importable(), test_check_endpoints_documented_all_in_sync(), test_check_endpoints_documented_flags_undocumented_endpoint(), test_check_endpoints_documented_missing_file(), test_smoke_test_hits_real_health_endpoint(), test_smoke_test_reports_missing_app_attribute(), _write_synthetic_api() (+3 more)

### Community 27 - ".__init__"
Cohesion: 0.04
Nodes (40): Datos clave sobre calor y mortalidad laboral, OIT — Ensuring safety and health at work in a changing climate (2024), Qué es, Uso en el proyecto, INSST — NTP 322: Valoración del riesgo de estrés térmico (índice WBGT), Método (índice WBGT), Nota sobre obligatoriedad, Uso en el proyecto (+32 more)

### Community 28 - "app.py"
Cohesion: 0.14
Nodes (19): api_reload(), _handle_feature(), _info_message(), load_models(), predict_one(), _preprocess(), process_message(), Any (+11 more)

### Community 29 - "test_secrets_agent.py"
Cohesion: 0.15
Nodes (16): SecretsAgent, test_heuristic_detects_aws_key(), test_heuristic_detects_prefixed_password_variable(), test_heuristic_detects_private_key_header(), test_heuristic_ignores_normal_low_entropy_strings(), test_heuristic_skips_git_directory(), test_shannon_entropy_high_for_random_looking_string(), test_shannon_entropy_low_for_repetitive_string() (+8 more)

### Community 30 - "LSTM multi-tarea"
Cohesion: 0.04
Nodes (42): Artefactos guardados, Búsquedas realizadas, Hiperparámetros (Optuna), Mejores cv_score por algoritmo, Ver también, Hiperparámetros, KNN, Resultados (+34 more)

### Community 31 - "test_make_dataset.py"
Cohesion: 0.08
Nodes (36): _agregar_estadisticas_diarias(), _agregar_rezagos_temporales(), load_data(), Series, _racha_previa(), Estadísticas de la DISTRIBUCIÓN diaria (24h) del calor/frío, no solo el     pico, Nº de días consecutivos ANTERIORES (hasta ayer) con `col > activo_si_mayor_que`., Features de PERSISTENCIA entre días (no dentro del día): capturan que el     rie (+28 more)

### Community 32 - "MLAgent"
Cohesion: 0.19
Nodes (7): MLAgent, Any, Path, Resumen genérico de un estimador ya entrenado: tipo, params, tamaño en memoria., Devuelve {feature: importancia} ordenado descendente, o None si el         estim, Heurística simple: si train_score - test_score supera `gap_threshold`         (e, SklearnTool

### Community 33 - "GraphifyTool"
Cohesion: 0.21
Nodes (14): _has_yaml(), Path, _sample_graph(), test_command_prefix_none_without_graphify(), test_command_prefix_uses_marker_python_over_binary(), test_knowledge_base_is_valid_yaml(), test_parent_summaries_groups_children_with_correlation(), test_parent_summaries_respects_min_children() (+6 more)

### Community 34 - "registry.py"
Cohesion: 0.08
Nodes (22): agents.agents.api_agent — Valida la API REST del proyecto (`api/main.py`).  Solo, agents.agents.cicd_agent — Genera y valida `.github/workflows/*.yml`.  Escribe d, Se pidió una herramienta que no está registrada., ToolNotFoundError, APITool, agents.tools.api_tool — Extrae y valida los endpoints de `api/main.py`.  Groundi, CICDTool, agents.tools.cicd_tool — Genera y valida workflows de GitHub Actions.  Grounding (+14 more)

### Community 35 - "Changelog"
Cohesion: 0.06
Nodes (34): Añadido, Añadido, Añadido, Añadido, Añadido, Añadido, Añadido, Cambiado (+26 more)

### Community 36 - "lstm_province_hybrid.py"
Cohesion: 0.15
Nodes (24): cargar_dataset_secuencias(), Carga el npz cacheado -> dict con X, y_calor, y_frio, provincias, fechas., Split TEMPORAL train/val/test con la misma regla que preprocess_data     (build_, split_secuencias_por_fecha(), escalar_diarias(), climasafeai.models.lstm_hybrid — LSTM híbrida: secuencia 24h + features diarias., StandardScaler sobre las features diarias, ajustado SOLO con train.      Scaler, escalar_secuencias() (+16 more)

### Community 37 - "DoctorAgent"
Cohesion: 0.19
Nodes (4): DoctorAgent, Ejecuta todas las verificaciones y devuelve un dict con el estado., Muestra el tamaño de los directorios principales del proyecto., Resumen ejecutivo del proyecto.

### Community 38 - "1. Golpe de calor"
Cohesion: 0.06
Nodes (30): 1.1 Heat Index — ecuación de Rothfusz (1990), 1.2 Tabla de categorías de riesgo, 1.3 Base normativa y clínica de referencia, 1.4 De Heat Index a WBGT — límites ocupacionales NIOSH, 1. Golpe de calor, 2.1 Wind Chill — ecuación NWS (2001), 2.2 Tabla de tiempo hasta riesgo significativo, 2.3 Precisión de los tiempos de congelación (+22 more)

### Community 39 - "FilesystemTool"
Cohesion: 0.27
Nodes (5): FilesystemTool, PathEscapesRootError, Exception, Path, agents.tools.filesystem_tool — Operaciones de archivo con la raíz del proyecto c

### Community 40 - "SharedContext"
Cohesion: 0.12
Nodes (6): Path, Solo existe cuando ml_type == 'redes_neuronales' (logs de TensorBoard)., Raíz donde los agentes guardan lo que generan (manifests, imágenes         extra, Crea (si hace falta) y devuelve `agents/workspace/<agent_name>/`., Rutas y configuración compartidas. Inmutable: si algo cambia, se crea otra insta, SharedContext

### Community 41 - "MissingDependencyError"
Cohesion: 0.17
Nodes (9): AgentRegistry, Registro global {nombre: clase_agente}. Un único singleton: `agent_registry`., Importa todos los módulos de `agents.agents` (núcleo del template),         `age, Importa agentes expuestos como entry point `dskit.agents` por         paquetes i, AgentNotFoundError, AgentSystemError, Exception, Excepción base de todo el sistema de agentes. (+1 more)

### Community 42 - "test_contracts.py"
Cohesion: 0.09
Nodes (18): Contract, contract_for(), agents.contracts — Contratos de rol: qué puede, qué NO puede y qué necesita cada, Contrato de un agente, o None si no lo tiene (los externos pueden no tenerlo)., Valida la coherencia del equipo. Devuelve la lista de problemas (vacía = OK)., Contrato de rol de un agente. Solo documentación estructurada + validable., validate_contracts(), Any (+10 more)

### Community 43 - "ValidateTool"
Cohesion: 0.24
Nodes (8): Any, DataFrame, Detecta outliers en datos numéricos.          method : 'iqr'    → Q1 - factor*IQ, Reporte de calidad: nulos, duplicados, constantes, cardinalidad., Perfil completo del dataset: métricas por columna + resumen., Valida que el DataFrame tenga las columnas y tipos esperados.          expected, Compara distribuciones entre dos datasets (referencia vs actual).          Para, ValidateTool

### Community 44 - "test_api.py"
Cohesion: 0.18
Nodes (10): _inject_model(), test_api.py — Tests de la API REST de ClimaSafeAI.  Usa el cliente de test de Fa, Resetea el estado global entre tests y parchea rutas., Entrena un modelo mínimo e inyecta en _state para tests., reset_state(), test_health_con_modelo(), test_predict_clasificacion_tiene_probabilidad(), test_predict_con_modelo_ok() (+2 more)

### Community 45 - ".load_graph"
Cohesion: 0.12
Nodes (13): KnowledgeAgent, Path, Actualiza el grafo con graphify y, si hay bóveda, lo exporta a Obsidian., Genera un resumen por cada nodo padre (hub) con:           - cuántos hijos y de, Preprocesa el grafo enriqueciendo cada nodo con metadatos calculados:, Limpia el grafo eliminando nodos que aportan ruido en vez de         información, Punto de entrada único para "pon el grafo y Obsidian al día". Lo usa el, Fija (y crea) el directorio de caché en graphify-out/cache/. (+5 more)

### Community 46 - "RefactorAgent"
Cohesion: 0.23
Nodes (7): Path, Reemplaza `except:` por `except Exception:` en archivos .py., Añade `-> None` a funciones/métodos sin tipo de retorno., Detecta `torch.load(..., weights_only=False)` y sugiere o aplica         un try/, Aplica un cambio y lo reporta., Corrige `def f(x=[])` → `def f(x=None)` + `if x is None: x = []`., RefactorAgent

### Community 47 - "evaluate_lstm_province"
Cohesion: 0.10
Nodes (18): evaluate_lstm_province(), load_lstm_province(), LSTMProvinceAttention, LSTMProvinceGated, LSTMProvinceMultiTask, main(), DataFrame, ndarray (+10 more)

### Community 48 - "calibracion_umbrales.py"
Cohesion: 0.14
Nodes (24): ajustar_clon(), barrer_umbrales(), calibrar_clase(), cargar_datos(), dibujar_frontera(), elegir_puntos(), frontera_pareto(), main() (+16 more)

### Community 49 - "Arquitectura"
Cohesion: 0.08
Nodes (22): Agentes, Arquitectura, Invocación, Sistema de Agentes, Ver también, Arquitectura, Componentes principales, Fuentes de datos (+14 more)

### Community 50 - "GraphAgent"
Cohesion: 0.22
Nodes (7): GraphAgent, agents.agents.graph_agent — Inspección de gráficos en `reports/figures/`.  Ver `, FigureMetrics, Path, agents.tools.vision_tool — Inspección estructural de gráficos en reports/figures, Lee un PNG con `matplotlib.image.imread` (no requiere PIL) y calcula         mét, VisionTool

### Community 51 - "personalizar_riesgo"
Cohesion: 0.14
Nodes (22): _factor_duracion_calor(), _factor_edad_calor(), _factor_edad_frio(), _factores_calor(), _factores_frio(), personalizar_riesgo(), climasafeai.features.personalizacion — modula el índice poblacional de riesgo se, Modula el índice poblacional 0-1 con los factores del perfil individual.      Pa (+14 more)

### Community 52 - "weather_indices.py"
Cohesion: 0.11
Nodes (23): add_weather_index_columns(), categorize_heat_index(), celsius_to_fahrenheit(), fahrenheit_to_celsius(), heat_index(), kelvin_to_celsius(), DataFrame, Series (+15 more)

### Community 53 - "orchestrator.py"
Cohesion: 0.40
Nodes (5): delegate_to(), _orch(), Any, agents.helpers — Funciones auxiliares para agentes.  Incluye el helper de delega, Delega una tarea a otro agente y devuelve su resultado.      Útil cuando un agen

### Community 54 - "asignar_clase_riesgo_calor"
Cohesion: 0.15
Nodes (21): asignar_clase_riesgo_calor(), asignar_clase_riesgo_frio(), _clasificar_percentil(), _mask_train_desde_corte(), DataFrame, Series, labels.py -- Etiquetas de riesgo (SEGURO/PRECAUCION/PELIGRO) a partir de percent, Clasifica una serie de mortalidad en 0/1/2 por percentil de RANGO.      Percenti (+13 more)

### Community 55 - "_pesos_de_clase"
Cohesion: 0.13
Nodes (20): evaluate_lstm(), indice_riesgo_softmax(), load_lstm(), LSTMMultiTask, _pesos_de_clase(), predict_lstm(), DataFrame, ndarray (+12 more)

### Community 56 - "ParallelTool"
Cohesion: 0.29
Nodes (5): ParallelTool, Any, Map paralelo con barra de progreso.          func     : función a aplicar a cada, Starmap paralelo para funciones que reciben tuplas.          func(*args) se apli, Divide un iterable en n_chunks aproximadamente iguales.

### Community 57 - "train_models"
Cohesion: 0.19
Nodes (21): evaluate_models(), Evalúa todos los modelos sobre train y test.     Métricas: Accuracy, F1 weighted, Entrena modelos de clasificacion y los guarda en models/.      Métrica CV: F1_we, train_models(), _make_data(), test_predict_model.py — Tests para climasafeai/models/predict_model.py, predict_proba_new debe devolver array (n_samples, n_classes)., Con threshold distinto de 0.5 debe aplicarse sobre predict_proba. (+13 more)

### Community 58 - "DuckDBTool"
Cohesion: 0.36
Nodes (5): DuckDBTool, Any, Path, Ejecuta `sql` y devuelve un DataFrame de pandas.         Ejemplo: DuckDBTool.que, Perfil rápido de un CSV/Parquet vía DESCRIBE de DuckDB (tipos inferidos, no esta

### Community 59 - "SQLiteTool"
Cohesion: 0.50
Nodes (4): Any, Path, SQLiteTool, Connection

### Community 60 - "Conclusiones del modelado — ClimaSafeAI"
Cohesion: 0.09
Nodes (19): 1. La métrica: por qué NO optimizamos accuracy ni F1 ponderado, 2. Las features: 7 → 15 → 19 (el mayor salto son los lags), 3. Los modelos y por qué XGBoost necesita pesos, 4. Modelo elegido por clase (19 features), 5. Hiperparámetros finales, 6. Techo actual y próximos pasos, 7. Estado de despliegue, Conclusiones del modelado — ClimaSafeAI (+11 more)

### Community 61 - "conftest.py"
Cohesion: 0.25
Nodes (7): df_with_target(), patch_paths(), conftest.py — Fixtures compartidas para todos los tests. Los fixtures se adaptan, DataFrame genérico con 8 columnas numéricas (200 filas)., DataFrame con features numéricas + columna target binaria., Redirige todas las constantes de ruta del proyecto a tmp_path.     Se aplica aut, sample_df()

### Community 62 - "agents/ — Sistema de agentes de este template"
Cohesion: 0.10
Nodes (20): Agentes externos, Agentes externos y lecturas recomendadas, Agentes que colaboran entre sí (no todo tiene que ser independiente), agents/ — Sistema de agentes de este template, Arquitectura, Añadir un agente nuevo, Añadir una herramienta nueva, Bucle de investigación autónoma (+12 more)

### Community 63 - "Diseño del modelo — ClimaSafe"
Cohesion: 0.14
Nodes (20): _build_models(), _load_best_params(), load_models(), Carga modelos entrenados desde disco, namespaceados por clase     (XGBoost_calor, Carga los mejores hiperparámetros cacheados si existen (de Optuna     para Rando, Busca el mejor valor de k para KNN vía GridSearchCV (5-fold,     F1_weighted --, Define los modelos a entrenar.     Tarea: clasificacion     RandomForest       →, _tune_knn_k() (+12 more)

### Community 64 - "SecretsAgent"
Cohesion: 0.12
Nodes (10): CacheAgent, _human_size(), Path, Precarga todo lo cacheable:           1. Resúmenes de nodo padre (top 20, min 2, Muestra el estado de la caché: número de entradas, tamaño, antigüedad., Limpia la caché. ``name`` opcional: solo borra entradas de una función         c, _cache_path(), Path (+2 more)

### Community 65 - "GitTool"
Cohesion: 0.18
Nodes (4): GitTool, agents.tools.git_tool — Envoltorio sobre el binario `git`.  Se usa `git` por sub, Devuelve [(código_estado, ruta), ...] tal y como `git status --porcelain`., Devuelve una lista de commits como dicts {hash, subject, author, date}.

### Community 66 - "PlanAgent"
Cohesion: 0.21
Nodes (8): PlanAgent, Path, Preguntas pendientes de una orden: una por argumento faltante + pasos sin asigna, Descompone `brief` en pasos, asigna agentes y devuelve las preguntas         nec, Responde preguntas de una orden. Claves aceptadas:           step0_filename=..., Ejecuta una orden completa delegando cada paso vía GStack.         Se niega si q, Estado de una orden concreta, o listado de todas., Parámetros sin valor por defecto de una acción (los que hay que preguntar).

### Community 67 - "Features — Documentación completa"
Cohesion: 0.11
Nodes (17): Evolución de features, Features — Documentación completa, Grupo A: Variables ERA5 crudas (4), Grupo B: Índices meteorológicos (3), Grupo C: Estadísticas distribución diaria (8), Grupo D: Persistencia temporal (12), Pipeline completo, Selección de hora de mayor riesgo (+9 more)

### Community 68 - "GraphifyTool"
Cohesion: 0.22
Nodes (9): Estadísticas rápidas del grafo (todo desde caché si está preprocesado)., GraphifyTool, Any, Normaliza links/edges a una lista única (graphify usa 'links')., Construye adyacencia no dirigida {id_nodo: set(vecinos)} desde links/edges., Agrupa los hijos por temas compartidos. Extrae palabras significativas         d, Agrupa hijos por su file_type (code/document/image/...) con conteo y labels., Para los nodos "padre" del grafo (los de mayor grado — hubs/god nodes), (+1 more)

### Community 69 - "base_tool.py"
Cohesion: 0.50
Nodes (4): BaseTool, ABC, agents.tools.base_tool — Contrato mínimo que deben cumplir las herramientas., Clase base opcional para herramientas con estado (p. ej. una conexión     DuckDB

### Community 71 - "main"
Cohesion: 0.15
Nodes (16): alinear_features_diarias(), evaluate_lstm_hybrid(), load_lstm_hybrid(), LSTMHybridMultiTask, main(), DataFrame, ndarray, StandardScaler (+8 more)

### Community 72 - "Conclusiones del modelado — ClimaSafeAI"
Cohesion: 0.23
Nodes (15): alinear_features_provincia(), crear_mapping_provincias(), escalar_features_provincia(), fetch_ine_features(), _poblar_features_embebidas(), DataFrame, ndarray, climasafeai.features.external_features — features demográficas provinciales (INE (+7 more)

### Community 73 - "features_frio.py"
Cohesion: 0.21
Nodes (16): build_candidatas(), evaluar(), _features_nocturnas(), main(), DataFrame, Path, Series, _racha_previa() (+8 more)

### Community 74 - "paths.py"
Cohesion: 0.13
Nodes (11): markov_smooth(), Experimento 5: Mejoras finales.   a) LightGBM para frío   b) Suavizado Markov (m, Aprende P(s_t | s_{t-1}) de train y corrige probabilidades en test:       P_corr, test_paths.py — Tests para climasafeai/utils/paths.py Común a todos los ml_type., Todas las constantes de ruta deben ser instancias de Path., make_dirs() debe crear todos los subdirectorios necesarios.     Se prueba con un, test_all_path_constants_are_path_objects(), test_project_dir_contains_pyproject() (+3 more)

### Community 75 - "Path"
Cohesion: 0.22
Nodes (8): Path, Lanza graphify con el prefijo correcto (intérprete + ``-m graphify`` o         e, Construye/actualiza el grafo. Si ya existe ``graph.json`` usa         ``--update, Devuelve el intérprete de Python que graphify dejó anotado en         ``graphify, Re-extrae solo los archivos nuevos o cambiados (``graphify . --update``)., Exporta el grafo como bóveda de Obsidian (``graphify export obsidian --dir``)., Consulta el grafo en lenguaje natural (``graphify query``)., CompletedProcess

### Community 78 - "Calibración de umbrales de decisión por clase"
Cohesion: 0.14
Nodes (13): 1. El problema: argmax deja la precisión de riesgo por los suelos, 2. La regla de decisión: cascada por severidad, 3. Metodología (sin mirar test ni una vez), 4. La frontera recall/precisión, 5. Resultados en test (argmax vs puntos elegidos), 6. Recomendación, 7. Integración en el código, Calibración de umbrales de decisión por clase (+5 more)

### Community 79 - "MakeAgent"
Cohesion: 0.21
Nodes (6): MakeAgent, Sugiere nuevos targets según la configuración del proyecto., Ejecuta un target de Makefile., Parsea el Makefile y devuelve {target: help_text}., Valida que el Makefile exista y que todos los targets sean sintácticamente válid, Verifica que la cadena pipeline → predict → train → features → data sea correcta

### Community 80 - "LSTMProvinceHybridMultiTask"
Cohesion: 0.19
Nodes (7): EnsembleLSTM, load_by_seed(), load_lstm_province_hybrid(), LSTMProvinceHybridMultiTask, Path, Averigua logits de N modelos LSTM province_hybrid entrenados con distintas seeds, Tronco LSTM + embedding provincia + INE + features diarias -> fusion -> 2 cabeza

### Community 81 - "test_tuning.py"
Cohesion: 0.26
Nodes (12): _make_Xy(), test_tuning.py — Tests de la optimizacion de hiperparametros con Optuna.  Usa n_, tune_models debe devolver un dict con params por modelo., tune_models debe guardar best_params_<modelo>.joblib en artifacts/., tune_models debe guardar reports/tuning_results.csv., Tras tune_models, train_models debe cargar best_params sin error., test_train_models_usa_best_params(), test_tune_models_devuelve_dict() (+4 more)

### Community 82 - "DocSearchAgent"
Cohesion: 0.23
Nodes (5): DocSearchAgent, Lista los nodos vecinos de ``node`` (por id o por label, insensible a         ma, Lista los nodos de tipo 'reference' (referencias/citas externas)., Poda nodos del grafo por tipo (p. ej. ``references``) o por id, y         opcion, Consulta la documentación en lenguaje natural vía ``graphify query`` (cacheado).

### Community 83 - ".detect_obsidian_vaults"
Cohesion: 0.17
Nodes (7): Detecta la bóveda de Obsidian del proyecto. Si no hay ninguna y         ``create, test_detect_obsidian_vaults(), test_obsidian_note_has_frontmatter_and_body(), Bloque de properties (YAML frontmatter) de una nota de Obsidian., Nota completa: frontmatter + cuerpo en Obsidian Flavored Markdown., Devuelve un archivo Obsidian Bases (`.base`, YAML) que muestra las notas, Busca bóvedas de Obsidian bajo ``root``: cualquier carpeta que         contenga

### Community 84 - "test_git_agent.py"
Cohesion: 0.20
Nodes (9): test_analyze_diff_warns_when_source_touched_without_tests(), test_guess_commit_type_defaults_to_feat_for_unknown_paths(), test_guess_commit_type_prefers_test_for_test_files(), test_parse_conventional_commit_invalid(), test_parse_conventional_commit_valid(), test_status_on_clean_repo(), test_suggest_commit_message_with_changes(), Extrae {type, scope, breaking, subject} de un mensaje Conventional Commits, o No (+1 more)

### Community 85 - "Coeficientes de personalización individual del riesgo"
Cohesion: 0.17
Nodes (11): Campos del perfil individual, Coeficientes de personalización individual del riesgo, Cómo se compone (la matemática importa), Fuentes, Implementación, Nivel y duración de la actividad — base fisiológica, Nota sobre la obesidad (el ejemplo motivador), Pendiente (mejoras futuras) (+3 more)

### Community 86 - "test_proba.py"
Cohesion: 0.17
Nodes (5): test_proba.py — Smoke tests: verifican que todos los módulos son importables y q, preprocess_data debe tener los parámetros mínimos esperados., test_evaluate_models_signature(), test_preprocess_data_signature(), test_train_models_signature()

### Community 87 - ".validate"
Cohesion: 0.22
Nodes (5): generate_ci_yaml(), Path, Extrae los targets 'make X' invocados en pasos 'run:' del workflow — para cruzar, Genera un workflow mínimo: lint + test en push/PR, usando los targets reales del, Devuelve una lista de problemas encontrados (vacía si no hay         ninguno). C

### Community 88 - "MLflowAgent"
Cohesion: 0.38
Nodes (8): MLflowAgent, _log_run(), mlflow_context(), test_compare_latest_detects_regression(), test_compare_latest_no_regression_has_no_warning(), test_list_runs_and_best_run(), test_list_runs_no_experiment_yet(), test_missing_project_slug_fails_without_mlflow_call()

### Community 89 - "StackStep"
Cohesion: 0.22
Nodes (4): Inserta un paso en una posición concreta., Resuelve referencias ``prev(key).data`` en kwargs., Añade un paso al final de la stack.          kwargs especiales (no se pasan al a, StackStep

### Community 90 - "Acciones"
Cohesion: 0.20
Nodes (9): Acciones, `build` — Actualiza el grafo y lo exporta a Obsidian, Cross-tool, Knowledge Agent — Grafo de conocimiento + Obsidian, Límite honesto, `setup_vault` — Detecta o crea la bóveda de Obsidian, `status` — Estado del grafo, caché y bóvedas, `summarize_parents` — Resume cada nodo padre con la correlación de sus hijos (+1 more)

### Community 91 - "NotebookAgent — Guía de análisis e inserción de comentarios"
Cohesion: 0.20
Nodes (9): Evidencia, Interpretación, Nivel 1 — Hecho observable, Nivel 2 — Inferencia razonable, Nivel 3 — Hipótesis, Nivel 4 — Recomendación, NotebookAgent — Guía de análisis e inserción de comentarios, Observación (+1 more)

### Community 92 - "test_plan_agent.py"
Cohesion: 0.22
Nodes (9): Tests del PlanAgent: encargo → preguntas → delegación, sin inventar nada., Un paso cuya acción necesita un argumento obligatorio (tag_release →     version, Encargo sin huecos: se ejecuta delegando en el agente dueño y queda auditado., planificar' y compañía rutean al agente plan, no a otro., test_intake_asks_for_missing_args_and_execute_refuses(), test_intake_execute_happy_path_delegates_and_records(), test_plan_routes_via_orchestrator(), test_status_lists_orders() (+1 more)

### Community 93 - "Diseño del modelo — ClimaSafe"
Cohesion: 0.20
Nodes (9): 1. Dos modelos separados (calor / frío), no uno único, 2. RiskScore artificial como selector de ventana horaria (no como feature), 3. Fuentes de datos usadas para el entrenamiento del ML, 4. Justificación de la elección de Random Forest, 5. Justificación: modelo híbrido (ML + fórmula determinista), 6. LSTM como cuarta estimación — corrigiendo el sesgo poblacional de la fórmula, Clasificación vs regresión (Parte C del notebook 0-3), Diseño del modelo — ClimaSafe (+1 more)

### Community 94 - "Features de frío con más memoria temporal (retardo epidemiológico)"
Cohesion: 0.20
Nodes (9): Features de frío con más memoria temporal (retardo epidemiológico), Importancia de las candidatas (variante todas_39), Motivación, Propuesta de integración, Qué se probó, RandomForest (baseline de referencia: 0.5256 / 0.5117), Resultados (test = último 20 % de fechas), Veredicto (+1 more)

### Community 95 - "Fórmulas + ML vs. aprendizaje directo de secuencias — resumen"
Cohesion: 0.20
Nodes (9): Ejemplo concreto, Fórmulas + ML vs. aprendizaje directo de secuencias — resumen, Input de la LSTM, La idea en una frase, Modelos totales si se implementa, Por qué interesa explorarlo: el alivio nocturno, Recomendación práctica, ¿Son suficientes 10 años de datos? (+1 more)

### Community 96 - "Label sin fuga temporal: percentiles train-only"
Cohesion: 0.20
Nodes (9): Conclusión, El problema, Experimento, La solución, Label sin fuga temporal: percentiles train-only, Magnitud de la fuga (cuánto cambia el label), Métricas honestas (test), Recomendación (+1 more)

### Community 97 - "Assistant (Build · DeepSeek V4 Flash Free · 47.3s)"
Cohesion: 0.20
Nodes (10): Assistant (Build · DeepSeek V4 Flash Free · 47.3s), Fase 1: Features por clase + nocturnas, Implementation approach:, Implementation order:, Nighttime minima for frío, Plan A: Per-class feature selection (no regeneration needed), Step 1: `build_features.py` — per-class feature selection, Step 2: `build_features.py` or `make_dataset.py` — version check (+2 more)

### Community 98 - "predict_new"
Cohesion: 0.22
Nodes (9): predict_new(), predict_proba_new(), ndarray, Carga un modelo y predice sobre nuevas muestras (ya preprocesadas).      Paramet, Carga un modelo y devuelve probabilidades de clase., predict_proba_new debe fallar si el modelo no existe., predict_new debe lanzar FileNotFoundError si el modelo no existe., test_predict_new_raises_if_missing() (+1 more)

### Community 99 - "LSTM híbrida — contexto de ola para la LSTM multi-tarea"
Cohesion: 0.22
Nodes (8): 1. Motivación: la LSTM pura no sabe en qué día de la ola está, 2. Arquitectura, 3. Resultados (test temporal, últimas ~20% de fechas), 4. Veredicto, 5. Propuesta futura: secuencias de 48-72 h (no implementada), Calor, Frío, LSTM híbrida — contexto de ola para la LSTM multi-tarea

### Community 100 - "ablacion_27v19.py"
Cohesion: 0.39
Nodes (8): cargar_datos(), clonar_modelo(), ejecutar_variante(), evaluar(), main(), Path, Ablación limpia de features: 27 vs 19 con el MISMO label.  Motivación: en la ite, Clona el modelo entrenado hoy conservando sus hiperparámetros.

### Community 101 - "TestAgent.md"
Cohesion: 0.22
Nodes (6): APIAgent, Ver también, ReviewAgent, Ver también, TestAgent, Ver también

### Community 102 - "Acciones"
Cohesion: 0.25
Nodes (7): Acciones, DocSearch Agent — Búsqueda y navegación por el grafo, `list_references` — Lista los nodos de tipo referencia/cita/enlace, `neighbors` — Vecinos de un nodo (navegación por el árbol), Nota, `prune` — Poda nodos del grafo, `search` — Consulta la documentación en lenguaje natural

### Community 103 - "Acciones"
Cohesion: 0.25
Nodes (7): Acciones, `find_papers` — Papers relacionados con el proyecto, Fuentes, Límite honesto, `project_keywords` — Palabras clave del proyecto, Research Agent — Papers relacionados con el proyecto, `search` — Búsqueda directa por consulta

### Community 104 - "explain_models"
Cohesion: 0.25
Nodes (8): _compute_shap(), explain_models(), Genera explicaciones SHAP para cada modelo entrenado.      Por cada modelo produ, Selecciona el explainer adecuado y devuelve (shap_values, X_explain)., Barra de importancia global media (|SHAP|)., Beeswarm: distribución de valores SHAP por feature (dirección + magnitud)., _shap_bar(), _shap_beeswarm()

### Community 105 - "Ayuda"
Cohesion: 0.25
Nodes (7): Ayuda, Comandos disponibles en el chat, Comandos Docker, Comandos esenciales, Estructura de outputs, MLflow, Tipo de ML: `supervisado` · Tarea: `clasificacion` · MLflow activo

### Community 106 - "MLflow — Experimentos"
Cohesion: 0.25
Nodes (8): Experimentos, MLflow — Experimentos, Model Registry (~132 versiones), Métricas registradas, Origen, Resumen, Tipos de Run, Ver también

### Community 107 - "Grafo de Conocimiento (Graphify)"
Cohesion: 0.25
Nodes (7): Archivos de salida, Comunidades destacadas, God Nodes (mayor centralidad), Grafo de Conocimiento (Graphify), Métricas, Próximos pasos, Ver también

### Community 109 - "Acciones"
Cohesion: 0.29
Nodes (6): Acciones, Formato soportado, `next_runs` — Próximas ejecuciones, Schedule Agent — Gestión de Cron, `to_human` — Traducir cron a texto legible, `validate` — Validar expresión cron

### Community 110 - "Ablación de features: 27 vs 19 con el mismo label"
Cohesion: 0.29
Nodes (6): Ablación de features: 27 vs 19 con el mismo label, Diseño, Interpretación, Motivación, Reproducción, Resultados (label nuevo fijo)

### Community 111 - "{{title}}"
Cohesion: 0.29
Nodes (6): Algoritmo, Features, Hiperparámetros, Métricas, Notas, {{title}}

### Community 112 - "_index.md"
Cohesion: 0.33
Nodes (4): MLAgent, Ver también, NotebookAgent, Ver también

### Community 113 - "Actualización del pipeline principal"
Cohesion: 0.29
Nodes (6): Actualización del pipeline principal, Cambios colaterales (2026-07-14), Cambios en `main.py`, Modelos entrenados, Pendiente para siguiente iteración, Ver también

### Community 114 - "agents/external/"
Cohesion: 0.33
Nodes (5): agents/external/, Ideas para agentes externos (no incluidos, ver `agents/README.md`), Opción 1 — un archivo suelto (la más simple), Opción 2 — un paquete pip instalado (para agentes que mantienes aparte), Qué NO cambia

### Community 115 - "Acciones"
Cohesion: 0.33
Nodes (5): Acciones, `compete` — Competición genérica, Cómo puntúa (determinista, no un juez LLM), `research` — Competición de búsqueda de papers, Supervisor Agent — Workers que compiten

### Community 116 - "construir_secuencias_24h"
Cohesion: 0.33
Nodes (6): _cargar_labels(), construir_secuencias_24h(), DataFrame, ndarray, DataFrame horario (provincia, datetime, features) -> tensor de secuencias., Une los dos parquets etiquetados -> (provincia, fecha, y_calor, y_frio,     y_ca

### Community 117 - "{{title}}"
Cohesion: 0.33
Nodes (5): Acciones, Dependencias, Entrada / Salida, Propósito, {{title}}

### Community 118 - "CICDAgent.md"
Cohesion: 0.33
Nodes (4): CICDAgent, Ver también, DockerAgent, Ver también

### Community 119 - "DependencyAgent.md"
Cohesion: 0.33
Nodes (4): DependencyAgent, Ver también, SecretsAgent, Ver también

### Community 120 - "DocumentationAgent.md"
Cohesion: 0.33
Nodes (4): DocumentationAgent, Ver también, GitAgent, Ver también

### Community 122 - "Assistant (Plan · DeepSeek V4 Flash Free · 14.7s)"
Cohesion: 0.40
Nodes (5): 1. Features de LSTM en modelos tabulares, 2. Pesos más agresivos, 3. Undersampling de días seguros, Assistant (Plan · DeepSeek V4 Flash Free · 14.7s), Plan de acción (en orden)

### Community 123 - "Assistant (Plan · DeepSeek V4 Flash Free · 18.6s)"
Cohesion: 0.40
Nodes (5): Assistant (Plan · DeepSeek V4 Flash Free · 18.6s), Lo que ya tenemos, Recomendación, Riesgo de undersampling, Tu propuesta vs alternativas

### Community 124 - "Assistant (Plan · DeepSeek V4 Flash Free · 26.5s)"
Cohesion: 0.40
Nodes (5): Assistant (Plan · DeepSeek V4 Flash Free · 26.5s), Phase 1: Per-class feature sets + nighttime minima (proven, low risk), Phase 2: INE features for tabular (exploratory), Phase 3: Optuna with Rec_riesgo objective (medium effort), Phase 4: Blending (reuse LSTM hybrid findings)

### Community 125 - "Plan propuesto (4 fases)"
Cohesion: 0.40
Nodes (5): Fase 1 — Features por clase (un cambio en `main.py`, impacto casi seguro), Fase 2 — Features INE en tabulares (exploratorio, 1 hora), Fase 3 — Optuna con Rec_riesgo (bajo esfuerzo, impacto moderado), Fase 4 — Evaluación con thresholds calibrados (ya implementado, solo activar), Plan propuesto (4 fases)

### Community 126 - "Guía para la IA"
Cohesion: 0.40
Nodes (4): Convenciones, Estructura, Flujo de trabajo recomendado, Guía para la IA

### Community 127 - "{{title}}"
Cohesion: 0.40
Nodes (4): Conclusiones / Próximos pasos, Contexto, Desarrollo, {{title}}

### Community 128 - "Doctor Agent"
Cohesion: 0.50
Nodes (3): Acciones, Doctor Agent, Uso

### Community 129 - "context"
Cohesion: 0.50
Nodes (4): context(), project_root(), Path, Un proyecto mínimo en un directorio temporal: estructura de carpetas del     tem

### Community 130 - "try_model"
Cohesion: 0.50
Nodes (4): DataFrame, Modo interactivo: introduce tus datos y el modelo predice el resultado.      Flu, try_model(), main()

### Community 131 - "Work State"
Cohesion: 0.50
Nodes (4): Active, Blocked, Completed, Work State

### Community 132 - "Assistant (Plan · DeepSeek V4 Flash Free · 16.9s)"
Cohesion: 0.50
Nodes (4): Assistant (Plan · DeepSeek V4 Flash Free · 16.9s), Contexto del problema, Evaluación realista, Qué sería "bueno"

### Community 134 - "Roadmap"
Cohesion: 0.50
Nodes (3): Completado ✅, Pendiente 🧱, Roadmap

### Community 135 - "Agentes — Índice"
Cohesion: 0.50
Nodes (4): Agentes, Agentes — Índice, Arquitectura, Ver también

### Community 136 - "2026-07-09 — Organización del vault + Graphify"
Cohesion: 0.50
Nodes (3): 2026-07-09 — Organización del vault + Graphify, Pendiente, Qué se hizo

### Community 138 - "Assistant (Plan · DeepSeek V4 Flash Free · 7.8s)"
Cohesion: 0.67
Nodes (3): Assistant (Plan · DeepSeek V4 Flash Free · 7.8s), Cambios, Lo que NO cambia

## Knowledge Gaps
- **489 isolated node(s):** `entrypoint.sh script`, `climasafeai`, `Convenciones`, `Estructura`, `Flujo de trabajo recomendado` (+484 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **40 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `AgentResult` connect `GitAgent` to `TestAgent`, `PlanAgent`, `GStack`, `ResearchTool`, `DependencyAgent`, `InstallerAgent`, `BaseAgent`, `DataFrameAnalysisTool`, `AgentResult`, `ToolExecutionError`, `Orchestrator`, `research_agent.py`, `.analyze_file`, `CICDAgent`, `NotebookAgent`, `ScheduleAgent`, `APIAgent`, `test_secrets_agent.py`, `MLAgent`, `registry.py`, `DoctorAgent`, `SharedContext`, `test_contracts.py`, `.load_graph`, `RefactorAgent`, `GraphAgent`, `SecretsAgent`, `PlanAgent`, `GraphifyTool`, `MakeAgent`, `DocSearchAgent`, `.detect_obsidian_vaults`, `.validate`, `MLflowAgent`, `StackStep`, `.list_runs`?**
  _High betweenness centrality (0.134) - this node is a cross-community bridge._
- **Why does `SharedContext` connect `SharedContext` to `TestAgent`, `context`, `GitAgent`, `BaseAgent`, `SharedContext`, `Orchestrator`, `research_agent.py`, `CICDAgent`, `MLflowAgent`, `get_context`?**
  _High betweenness centrality (0.024) - this node is a cross-community bridge._
- **Why does `BaseAgent` connect `BaseAgent` to `TestAgent`, `PlanAgent`, `ResearchTool`, `DependencyAgent`, `GitAgent`, `InstallerAgent`, `DataFrameAnalysisTool`, `AgentResult`, `Orchestrator`, `.can_handle`, `research_agent.py`, `.analyze_file`, `CICDAgent`, `NotebookAgent`, `ScheduleAgent`, `APIAgent`, `test_secrets_agent.py`, `MLAgent`, `registry.py`, `DoctorAgent`, `SharedContext`, `MissingDependencyError`, `test_contracts.py`, `.load_graph`, `RefactorAgent`, `GraphAgent`, `SecretsAgent`, `PlanAgent`, `MakeAgent`, `DocSearchAgent`, `MLflowAgent`, `get_context`?**
  _High betweenness centrality (0.022) - this node is a cross-community bridge._
- **Are the 35 inferred relationships involving `AgentResult` (e.g. with `APIAgent` and `AuditAgent`) actually correct?**
  _`AgentResult` has 35 INFERRED edges - model-reasoned connections that need verification._
- **Are the 33 inferred relationships involving `BaseAgent` (e.g. with `APIAgent` and `AuditAgent`) actually correct?**
  _`BaseAgent` has 33 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `SharedContext` (e.g. with `ProjectConfig` and `AgentResult`) actually correct?**
  _`SharedContext` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 10 inferred relationships involving `Orchestrator` (e.g. with `PlanAgent` and `SupervisorAgent`) actually correct?**
  _`Orchestrator` has 10 INFERRED edges - model-reasoned connections that need verification._