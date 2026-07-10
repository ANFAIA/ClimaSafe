# Changelog

## [Unreleased] — 2026-07-03

### Añadido

- implement monthly ERA5 download with spatial preprocessing and API integration

### Documentación

- documentar diseño del modelo, formulas deterministas y arquitectura de agentes
- update .env.example and add model design documentation

### Otros

- notebooks: añadir comentarios automáticos; fix: corregir preprocess_data UnboundLocalError
- añadidos agentes y sus respectivas funciones
- Estructura inicial creada

## [Unreleased] — 2026-07-10

### Añadido

- entrena por clase con XGBoost ponderado y selecciona por recall
- divide train/test por fecha en vez de aleatoriamente
- añade estadísticas diarias y persistencia temporal desde ERA5
- integrate KNN classifier with automated hyperparameter tuning and implement robust MLflow local tracking fallback
- actualizar make_dataset y añadir utilidades de procesamiento (labels, weather_indices); fix: ajustes en build_features
- expand orchestration and tooling
- implement monthly ERA5 download with spatial preprocessing and API integration

### Documentación

- documenta las decisiones de diseño y la selección por recall
- actualizar documentacion sobre agentes; chore: añadir make_dataset cambios
- documentar diseño del modelo, formulas deterministas y arquitectura de agentes
- update .env.example and add model design documentation

### Tests

- cubre las features de distribución diaria y de rezago temporal

### Mantenimiento

- añade chat/static para que la suite pueda ejecutarse

### Otros

- notebooks: actualizar comentarios automáticos (interpretaciones de gráficas)
- notebooks: añadir comentarios y actualizar ejecución (preprocesado y entrenamiento)
- Refactor code structure for improved readability and maintainability
- notebooks: añadir comentarios automáticos; fix: corregir preprocess_data UnboundLocalError
- añadidos agentes y sus respectivas funciones
- Estructura inicial creada

## [Unreleased] — 2026-07-10

### Añadido

- actualiza sistema de agentes a v0.2.0 y actualiza grafo de conocimiento
- entrena por clase con XGBoost ponderado y selecciona por recall
- divide train/test por fecha en vez de aleatoriamente
- añade estadísticas diarias y persistencia temporal desde ERA5
- integrate KNN classifier with automated hyperparameter tuning and implement robust MLflow local tracking fallback
- actualizar make_dataset y añadir utilidades de procesamiento (labels, weather_indices); fix: ajustes en build_features
- expand orchestration and tooling
- implement monthly ERA5 download with spatial preprocessing and API integration

### Documentación

- documenta las decisiones de diseño y la selección por recall
- actualizar documentacion sobre agentes; chore: añadir make_dataset cambios
- documentar diseño del modelo, formulas deterministas y arquitectura de agentes
- update .env.example and add model design documentation

### Tests

- cubre las features de distribución diaria y de rezago temporal

### Mantenimiento

- añade chat/static para que la suite pueda ejecutarse

### Otros

- notebooks: actualizar comentarios automáticos (interpretaciones de gráficas)
- notebooks: añadir comentarios y actualizar ejecución (preprocesado y entrenamiento)
- Refactor code structure for improved readability and maintainability
- notebooks: añadir comentarios automáticos; fix: corregir preprocess_data UnboundLocalError
- añadidos agentes y sus respectivas funciones
- Estructura inicial creada

## [Unreleased] — 2026-07-10

### Añadido

- mejora resúmenes de padres con tópicos y correlación explicada; añade cache agent y preprocess
- actualiza sistema de agentes a v0.2.0 y actualiza grafo de conocimiento
- entrena por clase con XGBoost ponderado y selecciona por recall
- divide train/test por fecha en vez de aleatoriamente
- añade estadísticas diarias y persistencia temporal desde ERA5
- integrate KNN classifier with automated hyperparameter tuning and implement robust MLflow local tracking fallback
- actualizar make_dataset y añadir utilidades de procesamiento (labels, weather_indices); fix: ajustes en build_features
- expand orchestration and tooling
- implement monthly ERA5 download with spatial preprocessing and API integration

### Documentación

- documenta las decisiones de diseño y la selección por recall
- actualizar documentacion sobre agentes; chore: añadir make_dataset cambios
- documentar diseño del modelo, formulas deterministas y arquitectura de agentes
- update .env.example and add model design documentation

### Tests

- cubre las features de distribución diaria y de rezago temporal

### Mantenimiento

- añade chat/static para que la suite pueda ejecutarse

### Otros

- notebooks: actualizar comentarios automáticos (interpretaciones de gráficas)
- notebooks: añadir comentarios y actualizar ejecución (preprocesado y entrenamiento)
- Refactor code structure for improved readability and maintainability
- notebooks: añadir comentarios automáticos; fix: corregir preprocess_data UnboundLocalError
- añadidos agentes y sus respectivas funciones
- Estructura inicial creada

