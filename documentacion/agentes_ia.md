# Agentes de IA — ClimaSafe

Se han añadido un conjunto de agentes para facilitar tareas de ingeniería
del repositorio (análisis, gestión de Git, documentación, revisión y
soporte para datos). Los agentes implementados en `agents/agents/` son:

- `data_agent` — herramientas para análisis y manipulación de datasets.
- `git_agent` — operaciones automatizadas de git (generar changelog, comprobar diffs).
- `documentation_agent` — sincroniza README/Makefile, genera `CHANGELOG.md` y construye docs.
- `docker_agent` — helpers para construir/ejecutar imágenes y contenedores.
- `graph_agent` — generación y análisis de grafos (depende del submódulo de gráficas).
- `ml_agent` — tareas relacionadas con entrenamiento, evaluación y tuning.
- `review_agent` — asistente para revisar código y generar sugerencias.

Estas incorporaciones permiten automatizar tareas repetitivas y mantener
la documentación y notebooks coherentes con el código. Revisa la carpeta
`agents/` para ver las acciones disponibles y su documentación.

