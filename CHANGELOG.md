# Changelog

## [Unreleased] — 2026-07-13

### Añadido

- calibración de umbrales de decisión por clase (cascada por severidad) sobre validación temporal, con puntos de operación recall/precisión y `predict_new(class_thresholds=...)` (default = argmax)

### Documentación

- documenta la frontera recall/precisión y la recomendación de umbrales (documentacion/ml/calibracion_umbrales.md)

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

## [Unreleased] — 2026-07-14

### Añadido

- features nocturnas (`t2m_min_noche`) y rachas severas (`horas_wc_severo`) para frío (+0.026 Rec_riesgo en RF)
- per-class feature selection: 27 features para calor, 19 para frío (ablación 27v19)
- pipeline completo en `main.py` reemplaza notebooks: XGBoost calor + RF frío inline + LSTM híbrida
- evaluación dual (argmax + umbrales calibrados) en main.py
- recalibración de umbrales: calor t1=0.40/t2=0.35, frío t1=0.45/t2=0.40

- LSTM híbrida con contexto de ola (secuencia 24h + features diarias INE + provincia)
- LSTM con embedding de provincia (LSTMProvince) y mecanismos de atención/gating
- HPO de atención LSTM (4 configs emb/fusión/lr probadas)
- calibración de umbrales de decisión por clase (cascada por severidad) sobre validación temporal
- baseline LightGBM como candidato a KNN
- ablación features 27 vs 19 con label fijo
- features de frío con retardo largo
- dataset a 27 features y suelo de mortalidad en labels
- `external_features.py` — datos INE por provincia para modelos híbridos
- `main.py` reescrito: pipeline de 9 pasos con verificaciones de existencia y skip automático

### Eliminado

- `experimento_label_sin_fuga.py` — script huérfano sin referencias

### Cambiado

- notebooks 0-2 (calor/frío): actualizados con modelo desplegado por clase
- notebook 0-3-LSTM: añadida Part D con experimentos province, hybrid, gated, attention HPO, ensemble
- `.vault/` actualizado: nuevo `03_MODELOS/LSTM.md`, notas de arquitectura, modelos y roadmap

### Documentación

- documenta la frontera recall/precisión y la recomendación de umbrales (`documentacion/ml/calibracion_umbrales.md`)
- documenta eliminación de fuga temporal train-test del label (`documentacion/ml/label_sin_fuga.md`)
- documenta la ablación de features (`documentacion/ml/ablacion_features_27v19.md`)
- documenta features de frío con retardo (`documentacion/ml/features_frio_retardo.md`)
- documenta LSTM híbrida (`documentacion/ml/lstm_hibrida.md`)
- conclusiones de modelos actualizadas (`documentacion/ml/conclusiones_modelos.md`)

## [Unreleased] — 2026-07-11

### Añadido

- personalización individual del índice de riesgo
- LSTM multi-tarea con comparación clasificación vs regresión

### Documentación

- corrige el nombre de la carpeta a «documentacion»

