# Doctor Agent

Ejecuta un diagnóstico integral del proyecto en todas las dimensiones:
entorno, git, estructura de datos, código, tests, dependencias y configuración.

## Acciones

| Acción | Descripción |
|--------|-------------|
| `checkup` | Ejecuta todas las verificaciones y devuelve un dict con el estado de cada una. |
| `disk_usage` | Muestra el tamaño de los directorios principales (data, models, reports, notebooks). |
| `summary` | Resumen ejecutivo con nombre del proyecto, ml_type, python, nº de tests y data files. |

## Uso

```bash
uv run python -m agents run doctor checkup
uv run python -m agents run doctor disk_usage
uv run python -m agents run doctor summary
uv run python -m agents doctor              # CLI shortcut: checkup
uv run python -m agents doctor --fix        # checkup + auto_fix pipeline
```
