# Graph Report - /home/cacelas/Documentos/anfaia/ClimaSafeAI  (2026-07-10)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 1075 nodes · 2683 edges · 53 communities (42 shown, 11 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 157 edges (avg confidence: 0.54)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `21f7ef46`
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
- MLAgent
- GraphifyTool
- registry.py
- DoctorAgent
- FilesystemTool
- MissingDependencyError
- test_contracts.py
- ValidateTool
- test_api.py
- RefactorAgent
- GraphAgent
- orchestrator.py
- ParallelTool
- DuckDBTool
- SQLiteTool
- Diseño del modelo — ClimaSafe
- SecretsAgent
- base_tool.py
- Conclusiones del modelado — ClimaSafeAI
- entrypoint.sh
- ClimaSafeAI Documentation Index

## God Nodes (most connected - your core abstractions)
1. `AgentResult` - 163 edges
2. `BaseAgent` - 70 edges
3. `SharedContext` - 43 edges
4. `GraphifyTool` - 37 edges
5. `GitAgent` - 33 edges
6. `MissingDependencyError` - 33 edges
7. `ToolExecutionError` - 25 edges
8. `ProcessResult` - 24 edges
9. `Orchestrator` - 23 edges
10. `run_command()` - 23 edges

## Surprising Connections (you probably didn't know these)
- `test_preprocess_data_signature()` --indirect_call--> `preprocess_data()`  [INFERRED]
  tests/test_proba.py → climasafeai/features/build_features.py
- `Distribution of deaths by categorical variables` --references--> `MoMo (ISCIII) Mortality Data`  [INFERRED]
  agents/workspace/notebook/cell142_out1.png → documenatcion/diseño_modelo.md
- `test_load_data_signature()` --indirect_call--> `load_data()`  [INFERRED]
  tests/test_proba.py → climasafeai/data/make_dataset.py
- `test_evaluate_models_signature()` --indirect_call--> `evaluate_models()`  [INFERRED]
  tests/test_proba.py → climasafeai/models/predict_model.py
- `test_train_models_signature()` --indirect_call--> `train_models()`  [INFERRED]
  tests/test_proba.py → climasafeai/models/train_model.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **ClimaSafe Modeling Strategy** — documenatcion_conclusiones_modelos, documenatcion_diseno_modelo, documenatcion_formulas_ml_resumen, documenatcion_formulas_riesgo_deterministico [EXTRACTED 1.00]
- **Data Sources Integration** — concept_momo_isciti, concept_era5_copernicus, documenatcion_diseno_modelo [EXTRACTED 0.95]

## Communities (53 total, 11 thin omitted)

### Community 0 - "TestAgent"
Cohesion: 0.16
Nodes (9): TestAgent, test_coverage_report_without_project_slug_fails(), test_list_untested_modules_detects_missing_and_present(), test_list_untested_modules_excludes_init(), test_list_untested_modules_missing_package_dir_fails(), Path, PytestTool, TestFailure (+1 more)

### Community 1 - "PlanAgent"
Cohesion: 0.10
Nodes (17): AuditAgent, PlanAgent, Path, audit_log_path(), Path, read_entries(), record(), test_audit_agent_report_and_suggestions() (+9 more)

### Community 2 - "GStack"
Cohesion: 0.15
Nodes (17): build_parser(), main(), _parse_kwargs(), _print_result(), Any, auto_analyze(), auto_commit_cycle(), auto_data_pipeline() (+9 more)

### Community 3 - "ResearchTool"
Cohesion: 0.10
Nodes (13): ResearchAgent, SupervisorAgent, test_dedupe_keeps_most_cited(), test_extract_keywords_drops_short_words(), test_extract_keywords_orders_by_frequency(), test_parse_arxiv(), test_parse_openalex_reconstructs_abstract_and_doi(), test_rank_orders_by_relevance_then_citations() (+5 more)

### Community 4 - "DependencyAgent"
Cohesion: 0.12
Nodes (19): DependencyAgent, test_check_outdated_without_dependencies_returns_empty(), test_check_outdated_without_pyproject_fails(), test_check_vulnerabilities_without_lock_fails(), test_normalize_package_name_pep503(), test_parse_dependency_name_strips_version_and_extras(), test_parse_pyproject_dependencies(), test_parse_pyproject_dependencies_missing_block_returns_empty() (+11 more)

### Community 5 - "GitAgent"
Cohesion: 0.07
Nodes (18): DocumentationAgent, GitAgent, test_analyze_diff_warns_when_source_touched_without_tests(), test_guess_commit_type_defaults_to_feat_for_unknown_paths(), test_guess_commit_type_prefers_test_for_test_files(), test_parse_conventional_commit_invalid(), test_parse_conventional_commit_valid(), test_status_on_clean_repo() (+10 more)

### Community 6 - "preprocess_data"
Cohesion: 0.19
Nodes (21): _apply_logcols(), _feature_engineering(), preprocess_data(), process_input(), DataFrame, ndarray, _build_models_cv(), evaluate_models_temporal_cv() (+13 more)

### Community 7 - "InstallerAgent"
Cohesion: 0.16
Nodes (13): InstallerAgent, _make_external_repo(), Path, test_install_from_git_no_candidates_fails(), test_install_from_git_valid_agent_end_to_end_via_cli(), test_install_from_git_warns_on_incomplete_structure(), test_install_no_path_warning_when_using_self_ctx(), test_install_refuses_overwrite_without_force() (+5 more)

### Community 8 - "BaseAgent"
Cohesion: 0.21
Nodes (6): ABC, BaseAgent, register_agent(), Agentes de IA — ClimaSafe, Boxplots - Outliers marked in red, Outliers detected by method (IQR vs Z-score)

### Community 9 - "DataFrameAnalysisTool"
Cohesion: 0.11
Nodes (15): DataAgent, DataFrame, Path, test_constant_columns_detected(), test_high_cardinality_columns_detected(), test_leakage_suspects_high_correlation(), test_leakage_suspects_missing_target_returns_empty(), test_outliers_iqr_detected() (+7 more)

### Community 10 - "test_monitoring.py"
Cohesion: 0.24
Nodes (18): _build_html(), check_drift(), check_performance(), DataFrame, Path, run_monitoring(), _make_ref_curr(), test_check_drift_categorica() (+10 more)

### Community 11 - "train_models"
Cohesion: 0.05
Nodes (67): download_era5_data(), download_momo_data(), _compute_shap(), evaluate_models(), explain_models(), _plot_confusion_matrix(), predict_new(), predict_proba_new() (+59 more)

### Community 12 - "AgentResult"
Cohesion: 0.16
Nodes (4): EnvAgent, MakeAgent, TemplateAgent, AgentResult

### Community 13 - "ToolExecutionError"
Cohesion: 0.17
Nodes (8): DockerAgent, ToolExecutionError, DockerfileFinding, DockerTool, ProcessResult, Path, require_binary(), run_command()

### Community 14 - "SharedContext"
Cohesion: 0.07
Nodes (25): MLflowAgent, _coerce(), load_project_config(), _parse_flat_yaml(), ProjectConfig, Any, Path, get_context() (+17 more)

### Community 15 - "Orchestrator"
Cohesion: 0.18
Nodes (10): Orchestrator, test_every_capability_keyword_routes_to_its_own_agent(), test_dispatch_auto_runs_zero_arg_action(), test_dispatch_disambiguates_commit_actions_via_aliases(), test_dispatch_reports_required_args_instead_of_guessing_them(), test_dispatch_without_action_lists_available_actions(), test_orchestrator_returns_no_agent_for_irrelevant_query(), test_orchestrator_routes_docker_query() (+2 more)

### Community 16 - "agent_installer_tool.py"
Cohesion: 0.29
Nodes (8): AgentCandidate, _decorator_name(), _find_suspicious_literal_paths(), _inspect_agent_class(), Path, _uses_shared_context(), ClassDef, expr

### Community 18 - ".analyze_file"
Cohesion: 0.18
Nodes (13): ReviewAgent, Path, test_detects_bare_except(), test_detects_long_function(), test_detects_too_many_args(), test_finds_structurally_duplicated_functions(), test_parse_handles_syntax_error_gracefully(), CodeAnalysisTool (+5 more)

### Community 19 - "CICDAgent"
Cohesion: 0.17
Nodes (12): CICDAgent, test_generate_workflow_creates_valid_file(), test_generate_workflow_fails_without_project_slug(), test_generate_workflow_refuses_overwrite_without_flag(), test_list_workflows_empty_by_default(), test_validate_workflow_cross_references_makefile(), test_validate_workflow_detects_missing_runs_on(), test_validate_workflow_missing_file() (+4 more)

### Community 20 - "NotebookAgent"
Cohesion: 0.20
Nodes (12): NotebookAgent, _make_notebook_with_image(), Path, test_extract_outputs_finds_image_and_text(), test_extract_outputs_missing_notebook_fails(), test_insert_comments_default_does_not_touch_original(), test_insert_comments_in_place_overwrites_original(), test_notebook_tool_skips_out_of_range_cell_index() (+4 more)

### Community 21 - "ScheduleAgent"
Cohesion: 0.22
Nodes (6): ScheduleAgent, _expand(), _parse_field(), Any, ScheduleTool, date

### Community 22 - "make_dataset.py"
Cohesion: 0.09
Nodes (48): _agregar_estadisticas_diarias(), _agregar_rezagos_temporales(), calcular_puntos_provincia(), cargar_era5_filtrado(), cargar_provincias_unificadas(), dataset_calor(), dataset_frio(), download_aemet() (+40 more)

### Community 25 - ".lint_dockerfile"
Cohesion: 0.46
Nodes (6): Path, test_lint_does_not_flag_pinned_base_image(), test_lint_flags_missing_user(), test_lint_flags_unpinned_base_image(), test_lint_missing_file_returns_single_warning(), Path

### Community 26 - "APIAgent"
Cohesion: 0.19
Nodes (11): APIAgent, _make_importable(), test_check_endpoints_documented_all_in_sync(), test_check_endpoints_documented_flags_undocumented_endpoint(), test_check_endpoints_documented_missing_file(), test_smoke_test_hits_real_health_endpoint(), test_smoke_test_reports_missing_app_attribute(), _write_synthetic_api() (+3 more)

### Community 28 - "app.py"
Cohesion: 0.21
Nodes (14): api_reload(), _handle_feature(), _info_message(), load_models(), predict_one(), _preprocess(), process_message(), Any (+6 more)

### Community 29 - "test_secrets_agent.py"
Cohesion: 0.25
Nodes (12): test_heuristic_detects_aws_key(), test_heuristic_detects_prefixed_password_variable(), test_heuristic_detects_private_key_header(), test_heuristic_ignores_normal_low_entropy_strings(), test_heuristic_skips_git_directory(), test_shannon_entropy_high_for_random_looking_string(), test_shannon_entropy_low_for_repetitive_string(), detect_secrets_binary_available() (+4 more)

### Community 32 - "MLAgent"
Cohesion: 0.25
Nodes (4): MLAgent, Any, Path, SklearnTool

### Community 33 - "GraphifyTool"
Cohesion: 0.07
Nodes (28): CacheAgent, _human_size(), AgentResult, BaseAgent, Path, DocSearchAgent, KnowledgeAgent, AgentResult (+20 more)

### Community 34 - "registry.py"
Cohesion: 0.22
Nodes (4): ToolNotFoundError, Any, register_tool(), ToolRegistry

### Community 39 - "FilesystemTool"
Cohesion: 0.30
Nodes (4): FilesystemTool, PathEscapesRootError, Exception, Path

### Community 41 - "MissingDependencyError"
Cohesion: 0.27
Nodes (7): ActionNotSupportedError, AgentSystemError, MissingDependencyError, Exception, AgentInstallerTool, MLflowTool, RunSummary

### Community 42 - "test_contracts.py"
Cohesion: 0.36
Nodes (4): Contract, contract_for(), validate_contracts(), test_team_contracts_are_coherent()

### Community 43 - "ValidateTool"
Cohesion: 0.42
Nodes (3): Any, ValidateTool, DataFrame

### Community 44 - "test_api.py"
Cohesion: 0.24
Nodes (6): _inject_model(), test_health_con_modelo(), test_predict_clasificacion_tiene_probabilidad(), test_predict_con_modelo_ok(), test_predict_features_faltantes_devuelve_422(), test_predict_payload_vacio_devuelve_422()

### Community 50 - "GraphAgent"
Cohesion: 0.33
Nodes (4): GraphAgent, FigureMetrics, Path, VisionTool

### Community 53 - "orchestrator.py"
Cohesion: 0.21
Nodes (6): AgentRegistry, AgentNotFoundError, delegate_to(), _orch(), Any, RoutingDecision

### Community 58 - "DuckDBTool"
Cohesion: 0.53
Nodes (3): DuckDBTool, Any, Path

### Community 59 - "SQLiteTool"
Cohesion: 0.50
Nodes (4): Any, Path, SQLiteTool, Connection

### Community 63 - "Diseño del modelo — ClimaSafe"
Cohesion: 0.29
Nodes (7): ERA5 Meteorological Data, Hybrid Model (ML + Deterministic), MoMo (ISCIII) Mortality Data, Diseño del modelo — ClimaSafe, Fórmulas + ML vs. Aprendizaje Directo, Fórmulas de riesgo climático determinista, Distribution of deaths by categorical variables

### Community 72 - "Conclusiones del modelado — ClimaSafeAI"
Cohesion: 0.67
Nodes (3): Conclusiones del modelado — ClimaSafeAI, _agregar_rezagos_temporales, evaluate_models

## Knowledge Gaps
- **12 isolated node(s):** `entrypoint.sh script`, `ClimaSafeAI Documentation Index`, `Ayuda - Recursos de Referencia`, `Fórmulas + ML vs. Aprendizaje Directo`, `Fórmulas de riesgo climático determinista` (+7 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **11 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `AgentResult` connect `AgentResult` to `TestAgent`, `PlanAgent`, `GStack`, `ResearchTool`, `DependencyAgent`, `GitAgent`, `InstallerAgent`, `BaseAgent`, `DataFrameAnalysisTool`, `ToolExecutionError`, `SharedContext`, `Orchestrator`, `research_agent.py`, `.analyze_file`, `CICDAgent`, `NotebookAgent`, `ScheduleAgent`, `.actions`, `APIAgent`, `MLAgent`, `GraphifyTool`, `DoctorAgent`, `MissingDependencyError`, `RefactorAgent`, `GraphAgent`, `orchestrator.py`, `SecretsAgent`?**
  _High betweenness centrality (0.276) - this node is a cross-community bridge._
- **Why does `BaseAgent` connect `BaseAgent` to `TestAgent`, `PlanAgent`, `ResearchTool`, `DependencyAgent`, `GitAgent`, `InstallerAgent`, `DataFrameAnalysisTool`, `AgentResult`, `ToolExecutionError`, `SharedContext`, `Orchestrator`, `research_agent.py`, `.analyze_file`, `CICDAgent`, `NotebookAgent`, `ScheduleAgent`, `.actions`, `APIAgent`, `.__init__`, `MLAgent`, `GraphifyTool`, `DoctorAgent`, `MissingDependencyError`, `RefactorAgent`, `GraphAgent`, `orchestrator.py`, `SecretsAgent`?**
  _High betweenness centrality (0.078) - this node is a cross-community bridge._
- **Why does `SharedContext` connect `SharedContext` to `TestAgent`, `CICDAgent`, `orchestrator.py`, `Orchestrator`?**
  _High betweenness centrality (0.041) - this node is a cross-community bridge._
- **Are the 31 inferred relationships involving `AgentResult` (e.g. with `APIAgent` and `AuditAgent`) actually correct?**
  _`AgentResult` has 31 INFERRED edges - model-reasoned connections that need verification._
- **Are the 29 inferred relationships involving `BaseAgent` (e.g. with `APIAgent` and `AuditAgent`) actually correct?**
  _`BaseAgent` has 29 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `SharedContext` (e.g. with `ProjectConfig` and `Orchestrator`) actually correct?**
  _`SharedContext` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `GraphifyTool` (e.g. with `CacheAgent` and `DocSearchAgent`) actually correct?**
  _`GraphifyTool` has 4 INFERRED edges - model-reasoned connections that need verification._