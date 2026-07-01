# Ayuda

Carpeta para recursos de referencia del proyecto. Papers, cheatsheets, notas
metodológicas o cualquier documentación de apoyo que no forme parte del código.

---

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

# Debug de rendimiento
make profile      # cProfile → reports/profile.prof
                  # luego: snakeviz reports/profile.prof

# Limpieza
make clean        # __pycache__ y cachés
make clean-models # borra .joblib / .pt
make clean-all    # todo

```

---

## Tipo de ML: `supervisado` · Tarea: `clasificacion` · MLflow activo

### MLflow

```bash
make mlflow        # UI en http://localhost:5000
```

Cada entrenamiento crea un run en el experimento `climasafeai`.
Los modelos se registran en el Model Registry como `climasafeai_<NombreModelo>`.
Artifacts: pesos `.joblib` / `.pt` + figuras de evaluacion.


### Comandos Docker

```bash
make docker-run     # construir imagen + arrancar contenedor
make docker-update  # reconstruir con los ultimos cambios
make docker-down    # parar y eliminar contenedores
```

### Comandos disponibles en el chat

| Comando | Descripcion |
|---|---|
| `status` | Estado del sistema y modelos cargados |
| `predict` | Prediccion interactiva paso a paso |
| `info` | Detalle de features y clases |
| `train` | Lanzar entrenamiento desde el chat |
| `reload` | Recargar modelos del disco |
| `help` | Mostrar ayuda |

---

## Estructura de outputs

```
reports/
├── figures/
│   ├── cm_<modelo>.png        # matriz de confusión
│   └── proba_dist_*.png     # distribución de probabilidades (binario)
└── resultados.csv            # métricas comparativas
```

---
