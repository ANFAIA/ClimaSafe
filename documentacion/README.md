# Documentación

Estructura general de la documentación del proyecto.

---

## Índice de contenidos

| Carpeta | Qué hay | Enlaces |
|---------|---------|---------|
| `arquitectura/` | Decisiones de diseño: modelo, fuentes externas, agentes IA, estratificación | `diseño_modelo.md`, `agentes_ia.md`, `evaluacion_fuentes_externas.md` |
| `ml/` | Experimentos,结论es, problemas conocidos, feature engineering | `conclusiones_modelos.md`, `problemas_conocidos.md`, `calibracion_umbrales.md` |
| `riesgo/` | Fórmulas deterministas, coeficientes extraídos de papers, personalización | `coeficientes_personalizacion_riesgo.md`, `formulas_riesgo_deterministico.md` |
| `modelos/` | Papers de modelos alternativos explorados (Transformers, GNN, N-BEATS, Diffusion) + modelos actuales en producción | `actuales/`, `transformers/`, `gnn/`, `nbeats/`, `diffusion/` |
| `papers/` | Papers del dominio: factores de riesgo, índices, ocupacional, planes de acción, aclimatación | `factores-riesgo/`, `ocupacional/`, `indices-biometeorologicos/`, `planes-accion/`, `aclimatacion/` |

## Documentos raíz

| Archivo | Contenido |
|---------|-----------|
| `proximos_pasos.md` | Roadmap y tareas pendientes priorizadas |
| `conclusion-base-conocimiento.md` | Decisión técnica sobre SQLite + MCP |
| `conformal_prediction.md` | Metodología de predicción conforme |

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

# Docker
make docker-run     # construir imagen + arrancar contenedor
make docker-update  # reconstruir con los últimos cambios
make docker-down    # parar y eliminar contenedores
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
└── resultados.csv             # métricas comparativas
```
