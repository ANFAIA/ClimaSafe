# Documentación

Estructura general de la documentación del proyecto.

---

## Índice de contenidos

| Carpeta | Qué hay | Enlaces |
|---------|---------|---------|
| `arquitectura/` | Decisiones de diseño: modelo, fuentes externas, agentes IA, estratificación, control de acceso MCP, auditoría MCP | `README.md` interno |
| `ml/` | Experimentos, conclusiones, problemas conocidos, feature engineering, active learning, contrafactuales, retos técnicos | `README.md` interno |
| `riesgo/` | Fórmulas deterministas, coeficientes extraídos de papers, personalización | `README.md` interno |
| `modelos/` | Papers de modelos alternativos explorados + modelos actuales en producción | `actuales/`, `transformers/`, `gnn/`, `nbeats/`, `diffusion/` |
| `papers/` | Papers del dominio: factores de riesgo, índices, ocupacional, planes de acción, aclimatación | `factores-riesgo/`, `ocupacional/`, `indices-biometeorologicos/`, `planes-accion/`, `aclimatacion/` |
| `bot/` | Hosting LLM gratuito, despliegue LLM remoto | `hosting_llm_gratis.md`, `despliegue_llm_remoto.md` |
| `llm/` | Fine-tuning, dataset, QC, embeddings, LoRA/aLoRA, base vs instruct, UV, Qwen3 | Ver tabla abajo |
| `despliegue/` | Packaging, releases, verificación de deploy de cero | `packaging.md`, `releases.md`, `verificacion-deploy-cero.md` |
| `wasm/` | WebAssembly, ONNX, LLM en navegador | `estudio_wasm.md`, `llm_navegador.md` |

## Documentos raíz

| Archivo | Contenido |
|---------|-----------|
| `proximos_pasos.md` | Roadmap y tareas pendientes priorizadas (actualizado 2026-08-27) |
| `conclusion-base-conocimiento.md` | Decisión técnica sobre SQLite + MCP |
| `conformal_prediction.md` | Metodología de predicción conforme |
| `claves_api.md` | Claves de API y configuración ERA5 (`.cdsapirc`) |
| `componentes.md` | Guía de todos los componentes del sistema |
| `prd.md` | Product Requirements Document |
| `arnes_comparativa.md` | Comparativa de arneses de agentes (ADK, DeepAgents, manual) |
| `seguridad_bd.md` | Protección de la BD, logs sanitizados, backup |
| `resolucion_prediccion.md` | Resolución configurable del perfil horario |
| `mcp_apps_estudio.md` | Estudio y MVP de MCP Apps |
| `arquitectura/pipeline_prediccion.md` | Flujo completo de predicción (ensemble, override, personalización) |
| `arquitectura/base_datos.md` | Esquema SQLite, DBManager, RAG semántico |
| `arquitectura/control_acceso_mcp.md` | Control de acceso por identidad en MCP |
| `arquitectura/auditoria-mcp-espec.md` | Auditoría MCP contra spec 2025-06-18+ |
| `ml/contrafactuales.md` | Explicaciones contrafactuales (cómo reducir el riesgo) |
| `ml/active_learner.md` | Aprendizaje activo para el paper scout |
| `ml/conclusiones_modelos.md` | Métricas y comparación de modelos |
| `ml/retos_tecnicos_viabilidad.md` | Viabilidad de HMM, Bayesianas, GPs, GNNs, TFT, RL |

## Documentación LLM

| Archivo | Contenido |
|---------|-----------|
| `llm/colab-fine-tuning.md` | Guía de fine-tuning en Google Colab (GPU T4) |
| `llm/guia-fine-tuning-qwen.md` | Guía de fine-tuning LoRA/QLoRA |
| `llm/guia-revision-dataset.md` | Guía de revisión del dataset sintético |
| `llm/qc-llm-017.md` | Resultados del QC del dataset (LLM-017) |
| `llm/base-vs-instruct.md` | Estudio: modelo base vs modelo de instrucciones |
| `llm/lora-alora-multi-adapter.md` | LoRA, aLoRA y multi-adapter |
| `llm/evaluacion-granite-alora-rag.md` | Evaluación Granite aLoRA para RAG |
| `llm/diffusion-llm-fine-tuning.md` | DLMs para fine-tuning (descartado) |
| `llm/uv-aviso-riesgo.md` | Radiación UV como línea futura |
| `llm/recursos-externos.md` | Recursos externos para fine-tuning |
| `rag_006_comparativa_embeddings.md` | Comparativa de modelos de embeddings |

## Comandos esenciales

```bash
# Entorno
uv sync --extra dev --extra supervisado
source .venv/bin/activate

# Pipeline
make run          # main.py completo
make data         # solo carga/preproceso de datos
make train        # solo entrenamiento
make predict      # solo predicciones → reports/

# Calidad
make test         # pytest completo
make smoke        # tests de humo (rápidos)
make lint         # ruff check
make format       # ruff format

# MLflow
make mlflow       # UI en http://localhost:5000

# Bot
make bot-start    # arranca el bot de Telegram
make bot-stop     # para el bot

# MCP
make mcp          # servidor MCP predicción (stdio)
make mcp-factors  # servidor MCP factores (stdio)
make mcp-http     # servidor MCP predicción (HTTP)

# Documentación
make docs         # construye MkDocs en site/
```

## Comandos del chat web

| Comando | Descripción |
|---------|-------------|
| `status` | Estado del sistema y modelos cargados |
| `predict` | Predicción interactiva paso a paso |
| `info` | Detalle de features y clases |
| `train` | Lanzar entrenamiento desde el chat |
| `reload` | Recargar modelos del disco |
| `help` | Mostrar ayuda |

## Estructura de outputs

```
reports/
├── figures/
│   ├── cm_<modelo>.png        # matriz de confusión
│   └── proba_dist_*.png       # distribución de probabilidades (binario)
├── benchmark_llm019.json      # benchmark de modelos LLM gratuitos
└── resultados.csv             # métricas comparativas
```
