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

## [Unreleased] — 2026-07-30

### Añadido

- Web UI grupo/volumen/curvas edad + estimacion volumetrica + fixes
- MCP server con 3 tools (predict, listar, cargar perfil) + integracion opencode
- riesgo colectivo/etiqueta, mapa de zona adaptativo y riesgo horario acumulado
- perfiles por alias, conformal prediction, frontend, docs
- personalizacion, overrides, explicabilidad, thresholds
- RAG, active learning, contrafactuales y mejora en API/chat
- calibra thresholds ML (t1=0.25/t2=0.40), elimina umbrales provincia obsoletos, personalización alineada con ML
- factores dinámicos desde JSON, personalización afecta clase final, prompt scout desambiguado
- conformal prediction para mejorar calibración de resultados
- web endpoints + MCP server + banner factores pendientes
- calidad de papers (alta/media/baja) según fuente y citas
- paper scout con arXiv/OpenAlex y clasificación LLM
- fusión LSTM, ensemble, explicabilidad y recomendaciones
- LSTM province_hybrid integrada y optimizada
- dataset a 27 features y suelo de mortalidad en labels
- experimento de features de frío con retardo largo (#11)
- ablación limpia de features 27 vs 19 con label fijo (#9)
- calibración de umbrales de decisión por clase
- LSTM híbrida con contexto de ola (secuencia 24h + features diarias)
- baseline LightGBM como candidato a sustituir a KNN
- personalización individual del índice de riesgo
- LSTM multi-tarea con comparación clasificación vs regresión
- notebooks, features pipeline y mlflow
- grafo de conocimiento y documentación
- sistema de agentes v0.2.0
- entrena por clase con XGBoost ponderado y selecciona por recall
- divide train/test por fecha en vez de aleatoriamente
- añade estadísticas diarias y persistencia temporal desde ERA5
- integrate KNN classifier with automated hyperparameter tuning and implement robust MLflow local tracking fallback
- actualizar make_dataset y añadir utilidades de procesamiento (labels, weather_indices); fix: ajustes en build_features
- expand orchestration and tooling
- implement monthly ERA5 download with spatial preprocessing and API integration

### Corrección de bugs

- EDADES_COMPARATIVA (25,55,65,75,85) evita duplicados de curva
- timeout 90s en llamada LLM del scout
- permisos de escritura para GH Actions push
- solo buscar en journals peer-review (excluir arXiv y repositorios)
- exception gitignore para factores_riesgo.json + fallback si no existe
- elimina la fuga temporal train-test del label de riesgo
- persiste modelo_desplegado_frio en 0-2-Frio

### Documentación

- reestructura documentación técnica y de riesgo
- roadmap completo con 19 tareas priorizadas y todo lo hecho marcado
- revisión 68 papers, nuevos factores documentados, próximos pasos actualizados
- reorganizar documentacion (vault + papers + problemas conocidos)
- vault Obsidian, README y CHANGELOG actualizados
- actualiza features de 19 a 27 (persistencia ampliada)
- registra LSTM, personalización y rename de documentacion
- corrige el nombre de la carpeta a «documentacion»
- documenta las decisiones de diseño y la selección por recall
- actualizar documentacion sobre agentes; chore: añadir make_dataset cambios
- documentar diseño del modelo, formulas deterministas y arquitectura de agentes
- update .env.example and add model design documentation

### Tests

- cubre las features de distribución diaria y de rezago temporal

### Mantenimiento

- pipeline de datos/features, tests y documentacion
- cambios pendientes pre-sesión
- añade chat/static para que la suite pueda ejecutarse

### Otros

- add factores_riesgo.json (force-add, gitignored)
- Merge branch 'mejoras/higiene-docs'
- Merge branch 'mejoras/label-sin-fuga'
- Merge branch 'mejoras/umbrales-decision'
- Merge branch 'mejoras/lstm-hibrida'
- Merge remote-tracking branch 'origin/main'
- notebooks: actualizar comentarios automáticos (interpretaciones de gráficas)
- notebooks: añadir comentarios y actualizar ejecución (preprocesado y entrenamiento)
- Refactor code structure for improved readability and maintainability
- notebooks: añadir comentarios automáticos; fix: corregir preprocess_data UnboundLocalError
- añadidos agentes y sus respectivas funciones
- Estructura inicial creada

## [Unreleased] — 2026-07-30

### Añadido

- orquestador de ciclo con puerta, backlog y subagentes
- Web UI grupo/volumen/curvas edad + estimacion volumetrica + fixes
- MCP server con 3 tools (predict, listar, cargar perfil) + integracion opencode
- riesgo colectivo/etiqueta, mapa de zona adaptativo y riesgo horario acumulado
- perfiles por alias, conformal prediction, frontend, docs
- personalizacion, overrides, explicabilidad, thresholds
- RAG, active learning, contrafactuales y mejora en API/chat
- calibra thresholds ML (t1=0.25/t2=0.40), elimina umbrales provincia obsoletos, personalización alineada con ML
- factores dinámicos desde JSON, personalización afecta clase final, prompt scout desambiguado
- conformal prediction para mejorar calibración de resultados
- web endpoints + MCP server + banner factores pendientes
- calidad de papers (alta/media/baja) según fuente y citas
- paper scout con arXiv/OpenAlex y clasificación LLM
- fusión LSTM, ensemble, explicabilidad y recomendaciones
- LSTM province_hybrid integrada y optimizada
- dataset a 27 features y suelo de mortalidad en labels
- experimento de features de frío con retardo largo (#11)
- ablación limpia de features 27 vs 19 con label fijo (#9)
- calibración de umbrales de decisión por clase
- LSTM híbrida con contexto de ola (secuencia 24h + features diarias)
- baseline LightGBM como candidato a sustituir a KNN
- personalización individual del índice de riesgo
- LSTM multi-tarea con comparación clasificación vs regresión
- notebooks, features pipeline y mlflow
- grafo de conocimiento y documentación
- sistema de agentes v0.2.0
- entrena por clase con XGBoost ponderado y selecciona por recall
- divide train/test por fecha en vez de aleatoriamente
- añade estadísticas diarias y persistencia temporal desde ERA5
- integrate KNN classifier with automated hyperparameter tuning and implement robust MLflow local tracking fallback
- actualizar make_dataset y añadir utilidades de procesamiento (labels, weather_indices); fix: ajustes en build_features
- expand orchestration and tooling
- implement monthly ERA5 download with spatial preprocessing and API integration

### Corrección de bugs

- EDADES_COMPARATIVA (25,55,65,75,85) evita duplicados de curva
- timeout 90s en llamada LLM del scout
- permisos de escritura para GH Actions push
- solo buscar en journals peer-review (excluir arXiv y repositorios)
- exception gitignore para factores_riesgo.json + fallback si no existe
- elimina la fuga temporal train-test del label de riesgo
- persiste modelo_desplegado_frio en 0-2-Frio

### Documentación

- reestructura documentación técnica y de riesgo
- roadmap completo con 19 tareas priorizadas y todo lo hecho marcado
- revisión 68 papers, nuevos factores documentados, próximos pasos actualizados
- reorganizar documentacion (vault + papers + problemas conocidos)
- vault Obsidian, README y CHANGELOG actualizados
- actualiza features de 19 a 27 (persistencia ampliada)
- registra LSTM, personalización y rename de documentacion
- corrige el nombre de la carpeta a «documentacion»
- documenta las decisiones de diseño y la selección por recall
- actualizar documentacion sobre agentes; chore: añadir make_dataset cambios
- documentar diseño del modelo, formulas deterministas y arquitectura de agentes
- update .env.example and add model design documentation

### Tests

- cubre las features de distribución diaria y de rezago temporal

### Mantenimiento

- pipeline de datos/features, tests y documentacion
- cambios pendientes pre-sesión
- añade chat/static para que la suite pueda ejecutarse

### Otros

- add factores_riesgo.json (force-add, gitignored)
- Merge branch 'mejoras/higiene-docs'
- Merge branch 'mejoras/label-sin-fuga'
- Merge branch 'mejoras/umbrales-decision'
- Merge branch 'mejoras/lstm-hibrida'
- Merge remote-tracking branch 'origin/main'
- notebooks: actualizar comentarios automáticos (interpretaciones de gráficas)
- notebooks: añadir comentarios y actualizar ejecución (preprocesado y entrenamiento)
- Refactor code structure for improved readability and maintainability
- notebooks: añadir comentarios automáticos; fix: corregir preprocess_data UnboundLocalError
- añadidos agentes y sus respectivas funciones
- Estructura inicial creada

## [Unreleased] — 2026-07-30

### Añadido

- pipeline de chunks, embeddings y búsqueda semántica
- orquestador de ciclo con puerta, backlog y subagentes
- Web UI grupo/volumen/curvas edad + estimacion volumetrica + fixes
- MCP server con 3 tools (predict, listar, cargar perfil) + integracion opencode
- riesgo colectivo/etiqueta, mapa de zona adaptativo y riesgo horario acumulado
- perfiles por alias, conformal prediction, frontend, docs
- personalizacion, overrides, explicabilidad, thresholds
- RAG, active learning, contrafactuales y mejora en API/chat
- calibra thresholds ML (t1=0.25/t2=0.40), elimina umbrales provincia obsoletos, personalización alineada con ML
- factores dinámicos desde JSON, personalización afecta clase final, prompt scout desambiguado
- conformal prediction para mejorar calibración de resultados
- web endpoints + MCP server + banner factores pendientes
- calidad de papers (alta/media/baja) según fuente y citas
- paper scout con arXiv/OpenAlex y clasificación LLM
- fusión LSTM, ensemble, explicabilidad y recomendaciones
- LSTM province_hybrid integrada y optimizada
- dataset a 27 features y suelo de mortalidad en labels
- experimento de features de frío con retardo largo (#11)
- ablación limpia de features 27 vs 19 con label fijo (#9)
- calibración de umbrales de decisión por clase
- LSTM híbrida con contexto de ola (secuencia 24h + features diarias)
- baseline LightGBM como candidato a sustituir a KNN
- personalización individual del índice de riesgo
- LSTM multi-tarea con comparación clasificación vs regresión
- notebooks, features pipeline y mlflow
- grafo de conocimiento y documentación
- sistema de agentes v0.2.0
- entrena por clase con XGBoost ponderado y selecciona por recall
- divide train/test por fecha en vez de aleatoriamente
- añade estadísticas diarias y persistencia temporal desde ERA5
- integrate KNN classifier with automated hyperparameter tuning and implement robust MLflow local tracking fallback
- actualizar make_dataset y añadir utilidades de procesamiento (labels, weather_indices); fix: ajustes en build_features
- expand orchestration and tooling
- implement monthly ERA5 download with spatial preprocessing and API integration

### Corrección de bugs

- EDADES_COMPARATIVA (25,55,65,75,85) evita duplicados de curva
- timeout 90s en llamada LLM del scout
- permisos de escritura para GH Actions push
- solo buscar en journals peer-review (excluir arXiv y repositorios)
- exception gitignore para factores_riesgo.json + fallback si no existe
- elimina la fuga temporal train-test del label de riesgo
- persiste modelo_desplegado_frio en 0-2-Frio

### Documentación

- reestructura documentación técnica y de riesgo
- roadmap completo con 19 tareas priorizadas y todo lo hecho marcado
- revisión 68 papers, nuevos factores documentados, próximos pasos actualizados
- reorganizar documentacion (vault + papers + problemas conocidos)
- vault Obsidian, README y CHANGELOG actualizados
- actualiza features de 19 a 27 (persistencia ampliada)
- registra LSTM, personalización y rename de documentacion
- corrige el nombre de la carpeta a «documentacion»
- documenta las decisiones de diseño y la selección por recall
- actualizar documentacion sobre agentes; chore: añadir make_dataset cambios
- documentar diseño del modelo, formulas deterministas y arquitectura de agentes
- update .env.example and add model design documentation

### Tests

- cubre las features de distribución diaria y de rezago temporal

### Mantenimiento

- pipeline de datos/features, tests y documentacion
- cambios pendientes pre-sesión
- añade chat/static para que la suite pueda ejecutarse

### Otros

- add factores_riesgo.json (force-add, gitignored)
- Merge branch 'mejoras/higiene-docs'
- Merge branch 'mejoras/label-sin-fuga'
- Merge branch 'mejoras/umbrales-decision'
- Merge branch 'mejoras/lstm-hibrida'
- Merge remote-tracking branch 'origin/main'
- notebooks: actualizar comentarios automáticos (interpretaciones de gráficas)
- notebooks: añadir comentarios y actualizar ejecución (preprocesado y entrenamiento)
- Refactor code structure for improved readability and maintainability
- notebooks: añadir comentarios automáticos; fix: corregir preprocess_data UnboundLocalError
- añadidos agentes y sus respectivas funciones
- Estructura inicial creada

## [Unreleased] — 2026-07-30

### Añadido

- tools de predicción, factores y perfiles con SQLite
- pipeline de chunks, embeddings y búsqueda semántica
- orquestador de ciclo con puerta, backlog y subagentes
- Web UI grupo/volumen/curvas edad + estimacion volumetrica + fixes
- MCP server con 3 tools (predict, listar, cargar perfil) + integracion opencode
- riesgo colectivo/etiqueta, mapa de zona adaptativo y riesgo horario acumulado
- perfiles por alias, conformal prediction, frontend, docs
- personalizacion, overrides, explicabilidad, thresholds
- RAG, active learning, contrafactuales y mejora en API/chat
- calibra thresholds ML (t1=0.25/t2=0.40), elimina umbrales provincia obsoletos, personalización alineada con ML
- factores dinámicos desde JSON, personalización afecta clase final, prompt scout desambiguado
- conformal prediction para mejorar calibración de resultados
- web endpoints + MCP server + banner factores pendientes
- calidad de papers (alta/media/baja) según fuente y citas
- paper scout con arXiv/OpenAlex y clasificación LLM
- fusión LSTM, ensemble, explicabilidad y recomendaciones
- LSTM province_hybrid integrada y optimizada
- dataset a 27 features y suelo de mortalidad en labels
- experimento de features de frío con retardo largo (#11)
- ablación limpia de features 27 vs 19 con label fijo (#9)
- calibración de umbrales de decisión por clase
- LSTM híbrida con contexto de ola (secuencia 24h + features diarias)
- baseline LightGBM como candidato a sustituir a KNN
- personalización individual del índice de riesgo
- LSTM multi-tarea con comparación clasificación vs regresión
- notebooks, features pipeline y mlflow
- grafo de conocimiento y documentación
- sistema de agentes v0.2.0
- entrena por clase con XGBoost ponderado y selecciona por recall
- divide train/test por fecha en vez de aleatoriamente
- añade estadísticas diarias y persistencia temporal desde ERA5
- integrate KNN classifier with automated hyperparameter tuning and implement robust MLflow local tracking fallback
- actualizar make_dataset y añadir utilidades de procesamiento (labels, weather_indices); fix: ajustes en build_features
- expand orchestration and tooling
- implement monthly ERA5 download with spatial preprocessing and API integration

### Corrección de bugs

- EDADES_COMPARATIVA (25,55,65,75,85) evita duplicados de curva
- timeout 90s en llamada LLM del scout
- permisos de escritura para GH Actions push
- solo buscar en journals peer-review (excluir arXiv y repositorios)
- exception gitignore para factores_riesgo.json + fallback si no existe
- elimina la fuga temporal train-test del label de riesgo
- persiste modelo_desplegado_frio en 0-2-Frio

### Documentación

- reestructura documentación técnica y de riesgo
- roadmap completo con 19 tareas priorizadas y todo lo hecho marcado
- revisión 68 papers, nuevos factores documentados, próximos pasos actualizados
- reorganizar documentacion (vault + papers + problemas conocidos)
- vault Obsidian, README y CHANGELOG actualizados
- actualiza features de 19 a 27 (persistencia ampliada)
- registra LSTM, personalización y rename de documentacion
- corrige el nombre de la carpeta a «documentacion»
- documenta las decisiones de diseño y la selección por recall
- actualizar documentacion sobre agentes; chore: añadir make_dataset cambios
- documentar diseño del modelo, formulas deterministas y arquitectura de agentes
- update .env.example and add model design documentation

### Tests

- cubre las features de distribución diaria y de rezago temporal

### Mantenimiento

- pipeline de datos/features, tests y documentacion
- cambios pendientes pre-sesión
- añade chat/static para que la suite pueda ejecutarse

### Otros

- add factores_riesgo.json (force-add, gitignored)
- Merge branch 'mejoras/higiene-docs'
- Merge branch 'mejoras/label-sin-fuga'
- Merge branch 'mejoras/umbrales-decision'
- Merge branch 'mejoras/lstm-hibrida'
- Merge remote-tracking branch 'origin/main'
- notebooks: actualizar comentarios automáticos (interpretaciones de gráficas)
- notebooks: añadir comentarios y actualizar ejecución (preprocesado y entrenamiento)
- Refactor code structure for improved readability and maintainability
- notebooks: añadir comentarios automáticos; fix: corregir preprocess_data UnboundLocalError
- añadidos agentes y sus respectivas funciones
- Estructura inicial creada

## [Unreleased] — 2026-07-31

### Añadido

- formulario Telegram con 17 estados, teclados inline, toggles multiselect y perfiles SQLite
- tools de predicción, factores y perfiles con SQLite
- pipeline de chunks, embeddings y búsqueda semántica
- orquestador de ciclo con puerta, backlog y subagentes
- Web UI grupo/volumen/curvas edad + estimacion volumetrica + fixes
- MCP server con 3 tools (predict, listar, cargar perfil) + integracion opencode
- riesgo colectivo/etiqueta, mapa de zona adaptativo y riesgo horario acumulado
- perfiles por alias, conformal prediction, frontend, docs
- personalizacion, overrides, explicabilidad, thresholds
- RAG, active learning, contrafactuales y mejora en API/chat
- calibra thresholds ML (t1=0.25/t2=0.40), elimina umbrales provincia obsoletos, personalización alineada con ML
- factores dinámicos desde JSON, personalización afecta clase final, prompt scout desambiguado
- conformal prediction para mejorar calibración de resultados
- web endpoints + MCP server + banner factores pendientes
- calidad de papers (alta/media/baja) según fuente y citas
- paper scout con arXiv/OpenAlex y clasificación LLM
- fusión LSTM, ensemble, explicabilidad y recomendaciones
- LSTM province_hybrid integrada y optimizada
- dataset a 27 features y suelo de mortalidad en labels
- experimento de features de frío con retardo largo (#11)
- ablación limpia de features 27 vs 19 con label fijo (#9)
- calibración de umbrales de decisión por clase
- LSTM híbrida con contexto de ola (secuencia 24h + features diarias)
- baseline LightGBM como candidato a sustituir a KNN
- personalización individual del índice de riesgo
- LSTM multi-tarea con comparación clasificación vs regresión
- notebooks, features pipeline y mlflow
- grafo de conocimiento y documentación
- sistema de agentes v0.2.0
- entrena por clase con XGBoost ponderado y selecciona por recall
- divide train/test por fecha en vez de aleatoriamente
- añade estadísticas diarias y persistencia temporal desde ERA5
- integrate KNN classifier with automated hyperparameter tuning and implement robust MLflow local tracking fallback
- actualizar make_dataset y añadir utilidades de procesamiento (labels, weather_indices); fix: ajustes en build_features
- expand orchestration and tooling
- implement monthly ERA5 download with spatial preprocessing and API integration

### Corrección de bugs

- EDADES_COMPARATIVA (25,55,65,75,85) evita duplicados de curva
- timeout 90s en llamada LLM del scout
- permisos de escritura para GH Actions push
- solo buscar en journals peer-review (excluir arXiv y repositorios)
- exception gitignore para factores_riesgo.json + fallback si no existe
- elimina la fuga temporal train-test del label de riesgo
- persiste modelo_desplegado_frio en 0-2-Frio

### Documentación

- reestructura documentación técnica y de riesgo
- roadmap completo con 19 tareas priorizadas y todo lo hecho marcado
- revisión 68 papers, nuevos factores documentados, próximos pasos actualizados
- reorganizar documentacion (vault + papers + problemas conocidos)
- vault Obsidian, README y CHANGELOG actualizados
- actualiza features de 19 a 27 (persistencia ampliada)
- registra LSTM, personalización y rename de documentacion
- corrige el nombre de la carpeta a «documentacion»
- documenta las decisiones de diseño y la selección por recall
- actualizar documentacion sobre agentes; chore: añadir make_dataset cambios
- documentar diseño del modelo, formulas deterministas y arquitectura de agentes
- update .env.example and add model design documentation

### Tests

- cubre las features de distribución diaria y de rezago temporal

### Mantenimiento

- pipeline de datos/features, tests y documentacion
- cambios pendientes pre-sesión
- añade chat/static para que la suite pueda ejecutarse

### Otros

- add factores_riesgo.json (force-add, gitignored)
- Merge branch 'mejoras/higiene-docs'
- Merge branch 'mejoras/label-sin-fuga'
- Merge branch 'mejoras/umbrales-decision'
- Merge branch 'mejoras/lstm-hibrida'
- Merge remote-tracking branch 'origin/main'
- notebooks: actualizar comentarios automáticos (interpretaciones de gráficas)
- notebooks: añadir comentarios y actualizar ejecución (preprocesado y entrenamiento)
- Refactor code structure for improved readability and maintainability
- notebooks: añadir comentarios automáticos; fix: corregir preprocess_data UnboundLocalError
- añadidos agentes y sus respectivas funciones
- Estructura inicial creada

## [Unreleased] — 2026-07-31

### Añadido

- cierra GIT-001
- formulario Telegram con 17 estados, teclados inline, toggles multiselect y perfiles SQLite
- tools de predicción, factores y perfiles con SQLite
- pipeline de chunks, embeddings y búsqueda semántica
- orquestador de ciclo con puerta, backlog y subagentes
- Web UI grupo/volumen/curvas edad + estimacion volumetrica + fixes
- MCP server con 3 tools (predict, listar, cargar perfil) + integracion opencode
- riesgo colectivo/etiqueta, mapa de zona adaptativo y riesgo horario acumulado
- perfiles por alias, conformal prediction, frontend, docs
- personalizacion, overrides, explicabilidad, thresholds
- RAG, active learning, contrafactuales y mejora en API/chat
- calibra thresholds ML (t1=0.25/t2=0.40), elimina umbrales provincia obsoletos, personalización alineada con ML
- factores dinámicos desde JSON, personalización afecta clase final, prompt scout desambiguado
- conformal prediction para mejorar calibración de resultados
- web endpoints + MCP server + banner factores pendientes
- calidad de papers (alta/media/baja) según fuente y citas
- paper scout con arXiv/OpenAlex y clasificación LLM
- fusión LSTM, ensemble, explicabilidad y recomendaciones
- LSTM province_hybrid integrada y optimizada
- dataset a 27 features y suelo de mortalidad en labels
- experimento de features de frío con retardo largo (#11)
- ablación limpia de features 27 vs 19 con label fijo (#9)
- calibración de umbrales de decisión por clase
- LSTM híbrida con contexto de ola (secuencia 24h + features diarias)
- baseline LightGBM como candidato a sustituir a KNN
- personalización individual del índice de riesgo
- LSTM multi-tarea con comparación clasificación vs regresión
- notebooks, features pipeline y mlflow
- grafo de conocimiento y documentación
- sistema de agentes v0.2.0
- entrena por clase con XGBoost ponderado y selecciona por recall
- divide train/test por fecha en vez de aleatoriamente
- añade estadísticas diarias y persistencia temporal desde ERA5
- integrate KNN classifier with automated hyperparameter tuning and implement robust MLflow local tracking fallback
- actualizar make_dataset y añadir utilidades de procesamiento (labels, weather_indices); fix: ajustes en build_features
- expand orchestration and tooling
- implement monthly ERA5 download with spatial preprocessing and API integration

### Corrección de bugs

- EDADES_COMPARATIVA (25,55,65,75,85) evita duplicados de curva
- timeout 90s en llamada LLM del scout
- permisos de escritura para GH Actions push
- solo buscar en journals peer-review (excluir arXiv y repositorios)
- exception gitignore para factores_riesgo.json + fallback si no existe
- elimina la fuga temporal train-test del label de riesgo
- persiste modelo_desplegado_frio en 0-2-Frio

### Documentación

- reestructura documentación técnica y de riesgo
- roadmap completo con 19 tareas priorizadas y todo lo hecho marcado
- revisión 68 papers, nuevos factores documentados, próximos pasos actualizados
- reorganizar documentacion (vault + papers + problemas conocidos)
- vault Obsidian, README y CHANGELOG actualizados
- actualiza features de 19 a 27 (persistencia ampliada)
- registra LSTM, personalización y rename de documentacion
- corrige el nombre de la carpeta a «documentacion»
- documenta las decisiones de diseño y la selección por recall
- actualizar documentacion sobre agentes; chore: añadir make_dataset cambios
- documentar diseño del modelo, formulas deterministas y arquitectura de agentes
- update .env.example and add model design documentation

### Tests

- cubre las features de distribución diaria y de rezago temporal

### Mantenimiento

- pipeline de datos/features, tests y documentacion
- cambios pendientes pre-sesión
- añade chat/static para que la suite pueda ejecutarse

### Otros

- add factores_riesgo.json (force-add, gitignored)
- Merge branch 'mejoras/higiene-docs'
- Merge branch 'mejoras/label-sin-fuga'
- Merge branch 'mejoras/umbrales-decision'
- Merge branch 'mejoras/lstm-hibrida'
- Merge remote-tracking branch 'origin/main'
- notebooks: actualizar comentarios automáticos (interpretaciones de gráficas)
- notebooks: añadir comentarios y actualizar ejecución (preprocesado y entrenamiento)
- Refactor code structure for improved readability and maintainability
- notebooks: añadir comentarios automáticos; fix: corregir preprocess_data UnboundLocalError
- añadidos agentes y sus respectivas funciones
- Estructura inicial creada

## [Unreleased] — 2026-07-31

### Añadido

- parte final descriptivo con riesgo %, temperatura, UV y recomendacion contextual
- cierra GIT-001
- formulario Telegram con 17 estados, teclados inline, toggles multiselect y perfiles SQLite
- tools de predicción, factores y perfiles con SQLite
- pipeline de chunks, embeddings y búsqueda semántica
- orquestador de ciclo con puerta, backlog y subagentes
- Web UI grupo/volumen/curvas edad + estimacion volumetrica + fixes
- MCP server con 3 tools (predict, listar, cargar perfil) + integracion opencode
- riesgo colectivo/etiqueta, mapa de zona adaptativo y riesgo horario acumulado
- perfiles por alias, conformal prediction, frontend, docs
- personalizacion, overrides, explicabilidad, thresholds
- RAG, active learning, contrafactuales y mejora en API/chat
- calibra thresholds ML (t1=0.25/t2=0.40), elimina umbrales provincia obsoletos, personalización alineada con ML
- factores dinámicos desde JSON, personalización afecta clase final, prompt scout desambiguado
- conformal prediction para mejorar calibración de resultados
- web endpoints + MCP server + banner factores pendientes
- calidad de papers (alta/media/baja) según fuente y citas
- paper scout con arXiv/OpenAlex y clasificación LLM
- fusión LSTM, ensemble, explicabilidad y recomendaciones
- LSTM province_hybrid integrada y optimizada
- dataset a 27 features y suelo de mortalidad en labels
- experimento de features de frío con retardo largo (#11)
- ablación limpia de features 27 vs 19 con label fijo (#9)
- calibración de umbrales de decisión por clase
- LSTM híbrida con contexto de ola (secuencia 24h + features diarias)
- baseline LightGBM como candidato a sustituir a KNN
- personalización individual del índice de riesgo
- LSTM multi-tarea con comparación clasificación vs regresión
- notebooks, features pipeline y mlflow
- grafo de conocimiento y documentación
- sistema de agentes v0.2.0
- entrena por clase con XGBoost ponderado y selecciona por recall
- divide train/test por fecha en vez de aleatoriamente
- añade estadísticas diarias y persistencia temporal desde ERA5
- integrate KNN classifier with automated hyperparameter tuning and implement robust MLflow local tracking fallback
- actualizar make_dataset y añadir utilidades de procesamiento (labels, weather_indices); fix: ajustes en build_features
- expand orchestration and tooling
- implement monthly ERA5 download with spatial preprocessing and API integration

### Corrección de bugs

- EDADES_COMPARATIVA (25,55,65,75,85) evita duplicados de curva
- timeout 90s en llamada LLM del scout
- permisos de escritura para GH Actions push
- solo buscar en journals peer-review (excluir arXiv y repositorios)
- exception gitignore para factores_riesgo.json + fallback si no existe
- elimina la fuga temporal train-test del label de riesgo
- persiste modelo_desplegado_frio en 0-2-Frio

### Documentación

- reestructura documentación técnica y de riesgo
- roadmap completo con 19 tareas priorizadas y todo lo hecho marcado
- revisión 68 papers, nuevos factores documentados, próximos pasos actualizados
- reorganizar documentacion (vault + papers + problemas conocidos)
- vault Obsidian, README y CHANGELOG actualizados
- actualiza features de 19 a 27 (persistencia ampliada)
- registra LSTM, personalización y rename de documentacion
- corrige el nombre de la carpeta a «documentacion»
- documenta las decisiones de diseño y la selección por recall
- actualizar documentacion sobre agentes; chore: añadir make_dataset cambios
- documentar diseño del modelo, formulas deterministas y arquitectura de agentes
- update .env.example and add model design documentation

### Tests

- cubre las features de distribución diaria y de rezago temporal

### Mantenimiento

- pipeline de datos/features, tests y documentacion
- cambios pendientes pre-sesión
- añade chat/static para que la suite pueda ejecutarse

### Otros

- add factores_riesgo.json (force-add, gitignored)
- Merge branch 'mejoras/higiene-docs'
- Merge branch 'mejoras/label-sin-fuga'
- Merge branch 'mejoras/umbrales-decision'
- Merge branch 'mejoras/lstm-hibrida'
- Merge remote-tracking branch 'origin/main'
- notebooks: actualizar comentarios automáticos (interpretaciones de gráficas)
- notebooks: añadir comentarios y actualizar ejecución (preprocesado y entrenamiento)
- Refactor code structure for improved readability and maintainability
- notebooks: añadir comentarios automáticos; fix: corregir preprocess_data UnboundLocalError
- añadidos agentes y sus respectivas funciones
- Estructura inicial creada

## [Unreleased] — 2026-07-31

### Añadido

- indexa documentacion/ y expone busqueda semantica y LLM local por MCP
- parte final descriptivo con riesgo %, temperatura, UV y recomendacion contextual
- cierra GIT-001
- formulario Telegram con 17 estados, teclados inline, toggles multiselect y perfiles SQLite
- tools de predicción, factores y perfiles con SQLite
- pipeline de chunks, embeddings y búsqueda semántica
- orquestador de ciclo con puerta, backlog y subagentes
- Web UI grupo/volumen/curvas edad + estimacion volumetrica + fixes
- MCP server con 3 tools (predict, listar, cargar perfil) + integracion opencode
- riesgo colectivo/etiqueta, mapa de zona adaptativo y riesgo horario acumulado
- perfiles por alias, conformal prediction, frontend, docs
- personalizacion, overrides, explicabilidad, thresholds
- RAG, active learning, contrafactuales y mejora en API/chat
- calibra thresholds ML (t1=0.25/t2=0.40), elimina umbrales provincia obsoletos, personalización alineada con ML
- factores dinámicos desde JSON, personalización afecta clase final, prompt scout desambiguado
- conformal prediction para mejorar calibración de resultados
- web endpoints + MCP server + banner factores pendientes
- calidad de papers (alta/media/baja) según fuente y citas
- paper scout con arXiv/OpenAlex y clasificación LLM
- fusión LSTM, ensemble, explicabilidad y recomendaciones
- LSTM province_hybrid integrada y optimizada
- dataset a 27 features y suelo de mortalidad en labels
- experimento de features de frío con retardo largo (#11)
- ablación limpia de features 27 vs 19 con label fijo (#9)
- calibración de umbrales de decisión por clase
- LSTM híbrida con contexto de ola (secuencia 24h + features diarias)
- baseline LightGBM como candidato a sustituir a KNN
- personalización individual del índice de riesgo
- LSTM multi-tarea con comparación clasificación vs regresión
- notebooks, features pipeline y mlflow
- grafo de conocimiento y documentación
- sistema de agentes v0.2.0
- entrena por clase con XGBoost ponderado y selecciona por recall
- divide train/test por fecha en vez de aleatoriamente
- añade estadísticas diarias y persistencia temporal desde ERA5
- integrate KNN classifier with automated hyperparameter tuning and implement robust MLflow local tracking fallback
- actualizar make_dataset y añadir utilidades de procesamiento (labels, weather_indices); fix: ajustes en build_features
- expand orchestration and tooling
- implement monthly ERA5 download with spatial preprocessing and API integration

### Corrección de bugs

- EDADES_COMPARATIVA (25,55,65,75,85) evita duplicados de curva
- timeout 90s en llamada LLM del scout
- permisos de escritura para GH Actions push
- solo buscar en journals peer-review (excluir arXiv y repositorios)
- exception gitignore para factores_riesgo.json + fallback si no existe
- elimina la fuga temporal train-test del label de riesgo
- persiste modelo_desplegado_frio en 0-2-Frio

### Documentación

- reestructura documentación técnica y de riesgo
- roadmap completo con 19 tareas priorizadas y todo lo hecho marcado
- revisión 68 papers, nuevos factores documentados, próximos pasos actualizados
- reorganizar documentacion (vault + papers + problemas conocidos)
- vault Obsidian, README y CHANGELOG actualizados
- actualiza features de 19 a 27 (persistencia ampliada)
- registra LSTM, personalización y rename de documentacion
- corrige el nombre de la carpeta a «documentacion»
- documenta las decisiones de diseño y la selección por recall
- actualizar documentacion sobre agentes; chore: añadir make_dataset cambios
- documentar diseño del modelo, formulas deterministas y arquitectura de agentes
- update .env.example and add model design documentation

### Tests

- cubre las features de distribución diaria y de rezago temporal

### Mantenimiento

- pipeline de datos/features, tests y documentacion
- cambios pendientes pre-sesión
- añade chat/static para que la suite pueda ejecutarse

### Otros

- add factores_riesgo.json (force-add, gitignored)
- Merge branch 'mejoras/higiene-docs'
- Merge branch 'mejoras/label-sin-fuga'
- Merge branch 'mejoras/umbrales-decision'
- Merge branch 'mejoras/lstm-hibrida'
- Merge remote-tracking branch 'origin/main'
- notebooks: actualizar comentarios automáticos (interpretaciones de gráficas)
- notebooks: añadir comentarios y actualizar ejecución (preprocesado y entrenamiento)
- Refactor code structure for improved readability and maintainability
- notebooks: añadir comentarios automáticos; fix: corregir preprocess_data UnboundLocalError
- añadidos agentes y sus respectivas funciones
- Estructura inicial creada

## [Unreleased] — 2026-07-31

### Añadido

- indexa documentacion/ y expone busqueda semantica y LLM local por MCP
- parte final descriptivo con riesgo %, temperatura, UV y recomendacion contextual
- cierra GIT-001
- formulario Telegram con 17 estados, teclados inline, toggles multiselect y perfiles SQLite
- tools de predicción, factores y perfiles con SQLite
- pipeline de chunks, embeddings y búsqueda semántica
- orquestador de ciclo con puerta, backlog y subagentes
- Web UI grupo/volumen/curvas edad + estimacion volumetrica + fixes
- MCP server con 3 tools (predict, listar, cargar perfil) + integracion opencode
- riesgo colectivo/etiqueta, mapa de zona adaptativo y riesgo horario acumulado
- perfiles por alias, conformal prediction, frontend, docs
- personalizacion, overrides, explicabilidad, thresholds
- RAG, active learning, contrafactuales y mejora en API/chat
- calibra thresholds ML (t1=0.25/t2=0.40), elimina umbrales provincia obsoletos, personalización alineada con ML
- factores dinámicos desde JSON, personalización afecta clase final, prompt scout desambiguado
- conformal prediction para mejorar calibración de resultados
- web endpoints + MCP server + banner factores pendientes
- calidad de papers (alta/media/baja) según fuente y citas
- paper scout con arXiv/OpenAlex y clasificación LLM
- fusión LSTM, ensemble, explicabilidad y recomendaciones
- LSTM province_hybrid integrada y optimizada
- dataset a 27 features y suelo de mortalidad en labels
- experimento de features de frío con retardo largo (#11)
- ablación limpia de features 27 vs 19 con label fijo (#9)
- calibración de umbrales de decisión por clase
- LSTM híbrida con contexto de ola (secuencia 24h + features diarias)
- baseline LightGBM como candidato a sustituir a KNN
- personalización individual del índice de riesgo
- LSTM multi-tarea con comparación clasificación vs regresión
- notebooks, features pipeline y mlflow
- grafo de conocimiento y documentación
- sistema de agentes v0.2.0
- entrena por clase con XGBoost ponderado y selecciona por recall
- divide train/test por fecha en vez de aleatoriamente
- añade estadísticas diarias y persistencia temporal desde ERA5
- integrate KNN classifier with automated hyperparameter tuning and implement robust MLflow local tracking fallback
- actualizar make_dataset y añadir utilidades de procesamiento (labels, weather_indices); fix: ajustes en build_features
- expand orchestration and tooling
- implement monthly ERA5 download with spatial preprocessing and API integration

### Corrección de bugs

- un campo desconocido en el perfil devuelve error explicito en vez de un 500 mudo
- EDADES_COMPARATIVA (25,55,65,75,85) evita duplicados de curva
- timeout 90s en llamada LLM del scout
- permisos de escritura para GH Actions push
- solo buscar en journals peer-review (excluir arXiv y repositorios)
- exception gitignore para factores_riesgo.json + fallback si no existe
- elimina la fuga temporal train-test del label de riesgo
- persiste modelo_desplegado_frio en 0-2-Frio

### Documentación

- reestructura documentación técnica y de riesgo
- roadmap completo con 19 tareas priorizadas y todo lo hecho marcado
- revisión 68 papers, nuevos factores documentados, próximos pasos actualizados
- reorganizar documentacion (vault + papers + problemas conocidos)
- vault Obsidian, README y CHANGELOG actualizados
- actualiza features de 19 a 27 (persistencia ampliada)
- registra LSTM, personalización y rename de documentacion
- corrige el nombre de la carpeta a «documentacion»
- documenta las decisiones de diseño y la selección por recall
- actualizar documentacion sobre agentes; chore: añadir make_dataset cambios
- documentar diseño del modelo, formulas deterministas y arquitectura de agentes
- update .env.example and add model design documentation

### Tests

- cubre las features de distribución diaria y de rezago temporal

### Mantenimiento

- pipeline de datos/features, tests y documentacion
- cambios pendientes pre-sesión
- añade chat/static para que la suite pueda ejecutarse

### Otros

- add factores_riesgo.json (force-add, gitignored)
- Merge branch 'mejoras/higiene-docs'
- Merge branch 'mejoras/label-sin-fuga'
- Merge branch 'mejoras/umbrales-decision'
- Merge branch 'mejoras/lstm-hibrida'
- Merge remote-tracking branch 'origin/main'
- notebooks: actualizar comentarios automáticos (interpretaciones de gráficas)
- notebooks: añadir comentarios y actualizar ejecución (preprocesado y entrenamiento)
- Refactor code structure for improved readability and maintainability
- notebooks: añadir comentarios automáticos; fix: corregir preprocess_data UnboundLocalError
- añadidos agentes y sus respectivas funciones
- Estructura inicial creada

## [Unreleased] — 2026-07-31

### Añadido

- la intensidad sale del MET del deporte (Compendium 2024) en vez de elegirla el usuario
- indexa documentacion/ y expone busqueda semantica y LLM local por MCP
- parte final descriptivo con riesgo %, temperatura, UV y recomendacion contextual
- cierra GIT-001
- formulario Telegram con 17 estados, teclados inline, toggles multiselect y perfiles SQLite
- tools de predicción, factores y perfiles con SQLite
- pipeline de chunks, embeddings y búsqueda semántica
- orquestador de ciclo con puerta, backlog y subagentes
- Web UI grupo/volumen/curvas edad + estimacion volumetrica + fixes
- MCP server con 3 tools (predict, listar, cargar perfil) + integracion opencode
- riesgo colectivo/etiqueta, mapa de zona adaptativo y riesgo horario acumulado
- perfiles por alias, conformal prediction, frontend, docs
- personalizacion, overrides, explicabilidad, thresholds
- RAG, active learning, contrafactuales y mejora en API/chat
- calibra thresholds ML (t1=0.25/t2=0.40), elimina umbrales provincia obsoletos, personalización alineada con ML
- factores dinámicos desde JSON, personalización afecta clase final, prompt scout desambiguado
- conformal prediction para mejorar calibración de resultados
- web endpoints + MCP server + banner factores pendientes
- calidad de papers (alta/media/baja) según fuente y citas
- paper scout con arXiv/OpenAlex y clasificación LLM
- fusión LSTM, ensemble, explicabilidad y recomendaciones
- LSTM province_hybrid integrada y optimizada
- dataset a 27 features y suelo de mortalidad en labels
- experimento de features de frío con retardo largo (#11)
- ablación limpia de features 27 vs 19 con label fijo (#9)
- calibración de umbrales de decisión por clase
- LSTM híbrida con contexto de ola (secuencia 24h + features diarias)
- baseline LightGBM como candidato a sustituir a KNN
- personalización individual del índice de riesgo
- LSTM multi-tarea con comparación clasificación vs regresión
- notebooks, features pipeline y mlflow
- grafo de conocimiento y documentación
- sistema de agentes v0.2.0
- entrena por clase con XGBoost ponderado y selecciona por recall
- divide train/test por fecha en vez de aleatoriamente
- añade estadísticas diarias y persistencia temporal desde ERA5
- integrate KNN classifier with automated hyperparameter tuning and implement robust MLflow local tracking fallback
- actualizar make_dataset y añadir utilidades de procesamiento (labels, weather_indices); fix: ajustes en build_features
- expand orchestration and tooling
- implement monthly ERA5 download with spatial preprocessing and API integration

### Corrección de bugs

- un campo desconocido en el perfil devuelve error explicito en vez de un 500 mudo
- EDADES_COMPARATIVA (25,55,65,75,85) evita duplicados de curva
- timeout 90s en llamada LLM del scout
- permisos de escritura para GH Actions push
- solo buscar en journals peer-review (excluir arXiv y repositorios)
- exception gitignore para factores_riesgo.json + fallback si no existe
- elimina la fuga temporal train-test del label de riesgo
- persiste modelo_desplegado_frio en 0-2-Frio

### Documentación

- reestructura documentación técnica y de riesgo
- roadmap completo con 19 tareas priorizadas y todo lo hecho marcado
- revisión 68 papers, nuevos factores documentados, próximos pasos actualizados
- reorganizar documentacion (vault + papers + problemas conocidos)
- vault Obsidian, README y CHANGELOG actualizados
- actualiza features de 19 a 27 (persistencia ampliada)
- registra LSTM, personalización y rename de documentacion
- corrige el nombre de la carpeta a «documentacion»
- documenta las decisiones de diseño y la selección por recall
- actualizar documentacion sobre agentes; chore: añadir make_dataset cambios
- documentar diseño del modelo, formulas deterministas y arquitectura de agentes
- update .env.example and add model design documentation

### Tests

- cubre las features de distribución diaria y de rezago temporal

### Mantenimiento

- pipeline de datos/features, tests y documentacion
- cambios pendientes pre-sesión
- añade chat/static para que la suite pueda ejecutarse

### Otros

- add factores_riesgo.json (force-add, gitignored)
- Merge branch 'mejoras/higiene-docs'
- Merge branch 'mejoras/label-sin-fuga'
- Merge branch 'mejoras/umbrales-decision'
- Merge branch 'mejoras/lstm-hibrida'
- Merge remote-tracking branch 'origin/main'
- notebooks: actualizar comentarios automáticos (interpretaciones de gráficas)
- notebooks: añadir comentarios y actualizar ejecución (preprocesado y entrenamiento)
- Refactor code structure for improved readability and maintainability
- notebooks: añadir comentarios automáticos; fix: corregir preprocess_data UnboundLocalError
- añadidos agentes y sus respectivas funciones
- Estructura inicial creada

## [Unreleased] — 2026-07-31

### Añadido

- dataset, fine-tuning LoRA de Qwen 2.5, Modelfile de Ollama y skill ClimaSafe
- la intensidad sale del MET del deporte (Compendium 2024) en vez de elegirla el usuario
- indexa documentacion/ y expone busqueda semantica y LLM local por MCP
- parte final descriptivo con riesgo %, temperatura, UV y recomendacion contextual
- cierra GIT-001
- formulario Telegram con 17 estados, teclados inline, toggles multiselect y perfiles SQLite
- tools de predicción, factores y perfiles con SQLite
- pipeline de chunks, embeddings y búsqueda semántica
- orquestador de ciclo con puerta, backlog y subagentes
- Web UI grupo/volumen/curvas edad + estimacion volumetrica + fixes
- MCP server con 3 tools (predict, listar, cargar perfil) + integracion opencode
- riesgo colectivo/etiqueta, mapa de zona adaptativo y riesgo horario acumulado
- perfiles por alias, conformal prediction, frontend, docs
- personalizacion, overrides, explicabilidad, thresholds
- RAG, active learning, contrafactuales y mejora en API/chat
- calibra thresholds ML (t1=0.25/t2=0.40), elimina umbrales provincia obsoletos, personalización alineada con ML
- factores dinámicos desde JSON, personalización afecta clase final, prompt scout desambiguado
- conformal prediction para mejorar calibración de resultados
- web endpoints + MCP server + banner factores pendientes
- calidad de papers (alta/media/baja) según fuente y citas
- paper scout con arXiv/OpenAlex y clasificación LLM
- fusión LSTM, ensemble, explicabilidad y recomendaciones
- LSTM province_hybrid integrada y optimizada
- dataset a 27 features y suelo de mortalidad en labels
- experimento de features de frío con retardo largo (#11)
- ablación limpia de features 27 vs 19 con label fijo (#9)
- calibración de umbrales de decisión por clase
- LSTM híbrida con contexto de ola (secuencia 24h + features diarias)
- baseline LightGBM como candidato a sustituir a KNN
- personalización individual del índice de riesgo
- LSTM multi-tarea con comparación clasificación vs regresión
- notebooks, features pipeline y mlflow
- grafo de conocimiento y documentación
- sistema de agentes v0.2.0
- entrena por clase con XGBoost ponderado y selecciona por recall
- divide train/test por fecha en vez de aleatoriamente
- añade estadísticas diarias y persistencia temporal desde ERA5
- integrate KNN classifier with automated hyperparameter tuning and implement robust MLflow local tracking fallback
- actualizar make_dataset y añadir utilidades de procesamiento (labels, weather_indices); fix: ajustes en build_features
- expand orchestration and tooling
- implement monthly ERA5 download with spatial preprocessing and API integration

### Corrección de bugs

- un campo desconocido en el perfil devuelve error explicito en vez de un 500 mudo
- EDADES_COMPARATIVA (25,55,65,75,85) evita duplicados de curva
- timeout 90s en llamada LLM del scout
- permisos de escritura para GH Actions push
- solo buscar en journals peer-review (excluir arXiv y repositorios)
- exception gitignore para factores_riesgo.json + fallback si no existe
- elimina la fuga temporal train-test del label de riesgo
- persiste modelo_desplegado_frio en 0-2-Frio

### Documentación

- reestructura documentación técnica y de riesgo
- roadmap completo con 19 tareas priorizadas y todo lo hecho marcado
- revisión 68 papers, nuevos factores documentados, próximos pasos actualizados
- reorganizar documentacion (vault + papers + problemas conocidos)
- vault Obsidian, README y CHANGELOG actualizados
- actualiza features de 19 a 27 (persistencia ampliada)
- registra LSTM, personalización y rename de documentacion
- corrige el nombre de la carpeta a «documentacion»
- documenta las decisiones de diseño y la selección por recall
- actualizar documentacion sobre agentes; chore: añadir make_dataset cambios
- documentar diseño del modelo, formulas deterministas y arquitectura de agentes
- update .env.example and add model design documentation

### Tests

- cubre las features de distribución diaria y de rezago temporal

### Mantenimiento

- pipeline de datos/features, tests y documentacion
- cambios pendientes pre-sesión
- añade chat/static para que la suite pueda ejecutarse

### Otros

- add factores_riesgo.json (force-add, gitignored)
- Merge branch 'mejoras/higiene-docs'
- Merge branch 'mejoras/label-sin-fuga'
- Merge branch 'mejoras/umbrales-decision'
- Merge branch 'mejoras/lstm-hibrida'
- Merge remote-tracking branch 'origin/main'
- notebooks: actualizar comentarios automáticos (interpretaciones de gráficas)
- notebooks: añadir comentarios y actualizar ejecución (preprocesado y entrenamiento)
- Refactor code structure for improved readability and maintainability
- notebooks: añadir comentarios automáticos; fix: corregir preprocess_data UnboundLocalError
- añadidos agentes y sus respectivas funciones
- Estructura inicial creada

## [Unreleased] — 2026-07-31

### Añadido

- doc_agent unifica grafo de graphify, RAG vectorial y boveda Obsidian
- dataset, fine-tuning LoRA de Qwen 2.5, Modelfile de Ollama y skill ClimaSafe
- la intensidad sale del MET del deporte (Compendium 2024) en vez de elegirla el usuario
- indexa documentacion/ y expone busqueda semantica y LLM local por MCP
- parte final descriptivo con riesgo %, temperatura, UV y recomendacion contextual
- cierra GIT-001
- formulario Telegram con 17 estados, teclados inline, toggles multiselect y perfiles SQLite
- tools de predicción, factores y perfiles con SQLite
- pipeline de chunks, embeddings y búsqueda semántica
- orquestador de ciclo con puerta, backlog y subagentes
- Web UI grupo/volumen/curvas edad + estimacion volumetrica + fixes
- MCP server con 3 tools (predict, listar, cargar perfil) + integracion opencode
- riesgo colectivo/etiqueta, mapa de zona adaptativo y riesgo horario acumulado
- perfiles por alias, conformal prediction, frontend, docs
- personalizacion, overrides, explicabilidad, thresholds
- RAG, active learning, contrafactuales y mejora en API/chat
- calibra thresholds ML (t1=0.25/t2=0.40), elimina umbrales provincia obsoletos, personalización alineada con ML
- factores dinámicos desde JSON, personalización afecta clase final, prompt scout desambiguado
- conformal prediction para mejorar calibración de resultados
- web endpoints + MCP server + banner factores pendientes
- calidad de papers (alta/media/baja) según fuente y citas
- paper scout con arXiv/OpenAlex y clasificación LLM
- fusión LSTM, ensemble, explicabilidad y recomendaciones
- LSTM province_hybrid integrada y optimizada
- dataset a 27 features y suelo de mortalidad en labels
- experimento de features de frío con retardo largo (#11)
- ablación limpia de features 27 vs 19 con label fijo (#9)
- calibración de umbrales de decisión por clase
- LSTM híbrida con contexto de ola (secuencia 24h + features diarias)
- baseline LightGBM como candidato a sustituir a KNN
- personalización individual del índice de riesgo
- LSTM multi-tarea con comparación clasificación vs regresión
- notebooks, features pipeline y mlflow
- grafo de conocimiento y documentación
- sistema de agentes v0.2.0
- entrena por clase con XGBoost ponderado y selecciona por recall
- divide train/test por fecha en vez de aleatoriamente
- añade estadísticas diarias y persistencia temporal desde ERA5
- integrate KNN classifier with automated hyperparameter tuning and implement robust MLflow local tracking fallback
- actualizar make_dataset y añadir utilidades de procesamiento (labels, weather_indices); fix: ajustes en build_features
- expand orchestration and tooling
- implement monthly ERA5 download with spatial preprocessing and API integration

### Corrección de bugs

- un campo desconocido en el perfil devuelve error explicito en vez de un 500 mudo
- EDADES_COMPARATIVA (25,55,65,75,85) evita duplicados de curva
- timeout 90s en llamada LLM del scout
- permisos de escritura para GH Actions push
- solo buscar en journals peer-review (excluir arXiv y repositorios)
- exception gitignore para factores_riesgo.json + fallback si no existe
- elimina la fuga temporal train-test del label de riesgo
- persiste modelo_desplegado_frio en 0-2-Frio

### Documentación

- reestructura documentación técnica y de riesgo
- roadmap completo con 19 tareas priorizadas y todo lo hecho marcado
- revisión 68 papers, nuevos factores documentados, próximos pasos actualizados
- reorganizar documentacion (vault + papers + problemas conocidos)
- vault Obsidian, README y CHANGELOG actualizados
- actualiza features de 19 a 27 (persistencia ampliada)
- registra LSTM, personalización y rename de documentacion
- corrige el nombre de la carpeta a «documentacion»
- documenta las decisiones de diseño y la selección por recall
- actualizar documentacion sobre agentes; chore: añadir make_dataset cambios
- documentar diseño del modelo, formulas deterministas y arquitectura de agentes
- update .env.example and add model design documentation

### Tests

- cubre las features de distribución diaria y de rezago temporal

### Mantenimiento

- pipeline de datos/features, tests y documentacion
- cambios pendientes pre-sesión
- añade chat/static para que la suite pueda ejecutarse

### Otros

- add factores_riesgo.json (force-add, gitignored)
- Merge branch 'mejoras/higiene-docs'
- Merge branch 'mejoras/label-sin-fuga'
- Merge branch 'mejoras/umbrales-decision'
- Merge branch 'mejoras/lstm-hibrida'
- Merge remote-tracking branch 'origin/main'
- notebooks: actualizar comentarios automáticos (interpretaciones de gráficas)
- notebooks: añadir comentarios y actualizar ejecución (preprocesado y entrenamiento)
- Refactor code structure for improved readability and maintainability
- notebooks: añadir comentarios automáticos; fix: corregir preprocess_data UnboundLocalError
- añadidos agentes y sus respectivas funciones
- Estructura inicial creada

## [Unreleased] — 2026-07-31

### Añadido

- doc_agent unifica grafo de graphify, RAG vectorial y boveda Obsidian
- dataset, fine-tuning LoRA de Qwen 2.5, Modelfile de Ollama y skill ClimaSafe
- la intensidad sale del MET del deporte (Compendium 2024) en vez de elegirla el usuario
- indexa documentacion/ y expone busqueda semantica y LLM local por MCP
- parte final descriptivo con riesgo %, temperatura, UV y recomendacion contextual
- cierra GIT-001
- formulario Telegram con 17 estados, teclados inline, toggles multiselect y perfiles SQLite
- tools de predicción, factores y perfiles con SQLite
- pipeline de chunks, embeddings y búsqueda semántica
- orquestador de ciclo con puerta, backlog y subagentes
- Web UI grupo/volumen/curvas edad + estimacion volumetrica + fixes
- MCP server con 3 tools (predict, listar, cargar perfil) + integracion opencode
- riesgo colectivo/etiqueta, mapa de zona adaptativo y riesgo horario acumulado
- perfiles por alias, conformal prediction, frontend, docs
- personalizacion, overrides, explicabilidad, thresholds
- RAG, active learning, contrafactuales y mejora en API/chat
- calibra thresholds ML (t1=0.25/t2=0.40), elimina umbrales provincia obsoletos, personalización alineada con ML
- factores dinámicos desde JSON, personalización afecta clase final, prompt scout desambiguado
- conformal prediction para mejorar calibración de resultados
- web endpoints + MCP server + banner factores pendientes
- calidad de papers (alta/media/baja) según fuente y citas
- paper scout con arXiv/OpenAlex y clasificación LLM
- fusión LSTM, ensemble, explicabilidad y recomendaciones
- LSTM province_hybrid integrada y optimizada
- dataset a 27 features y suelo de mortalidad en labels
- experimento de features de frío con retardo largo (#11)
- ablación limpia de features 27 vs 19 con label fijo (#9)
- calibración de umbrales de decisión por clase
- LSTM híbrida con contexto de ola (secuencia 24h + features diarias)
- baseline LightGBM como candidato a sustituir a KNN
- personalización individual del índice de riesgo
- LSTM multi-tarea con comparación clasificación vs regresión
- notebooks, features pipeline y mlflow
- grafo de conocimiento y documentación
- sistema de agentes v0.2.0
- entrena por clase con XGBoost ponderado y selecciona por recall
- divide train/test por fecha en vez de aleatoriamente
- añade estadísticas diarias y persistencia temporal desde ERA5
- integrate KNN classifier with automated hyperparameter tuning and implement robust MLflow local tracking fallback
- actualizar make_dataset y añadir utilidades de procesamiento (labels, weather_indices); fix: ajustes en build_features
- expand orchestration and tooling
- implement monthly ERA5 download with spatial preprocessing and API integration

### Corrección de bugs

- aisla el tracking URI de mlflow para que no se filtre entre tests
- un campo desconocido en el perfil devuelve error explicito en vez de un 500 mudo
- EDADES_COMPARATIVA (25,55,65,75,85) evita duplicados de curva
- timeout 90s en llamada LLM del scout
- permisos de escritura para GH Actions push
- solo buscar en journals peer-review (excluir arXiv y repositorios)
- exception gitignore para factores_riesgo.json + fallback si no existe
- elimina la fuga temporal train-test del label de riesgo
- persiste modelo_desplegado_frio en 0-2-Frio

### Documentación

- reestructura documentación técnica y de riesgo
- roadmap completo con 19 tareas priorizadas y todo lo hecho marcado
- revisión 68 papers, nuevos factores documentados, próximos pasos actualizados
- reorganizar documentacion (vault + papers + problemas conocidos)
- vault Obsidian, README y CHANGELOG actualizados
- actualiza features de 19 a 27 (persistencia ampliada)
- registra LSTM, personalización y rename de documentacion
- corrige el nombre de la carpeta a «documentacion»
- documenta las decisiones de diseño y la selección por recall
- actualizar documentacion sobre agentes; chore: añadir make_dataset cambios
- documentar diseño del modelo, formulas deterministas y arquitectura de agentes
- update .env.example and add model design documentation

### Tests

- cubre las features de distribución diaria y de rezago temporal

### Mantenimiento

- pipeline de datos/features, tests y documentacion
- cambios pendientes pre-sesión
- añade chat/static para que la suite pueda ejecutarse

### Otros

- add factores_riesgo.json (force-add, gitignored)
- Merge branch 'mejoras/higiene-docs'
- Merge branch 'mejoras/label-sin-fuga'
- Merge branch 'mejoras/umbrales-decision'
- Merge branch 'mejoras/lstm-hibrida'
- Merge remote-tracking branch 'origin/main'
- notebooks: actualizar comentarios automáticos (interpretaciones de gráficas)
- notebooks: añadir comentarios y actualizar ejecución (preprocesado y entrenamiento)
- Refactor code structure for improved readability and maintainability
- notebooks: añadir comentarios automáticos; fix: corregir preprocess_data UnboundLocalError
- añadidos agentes y sus respectivas funciones
- Estructura inicial creada

## [Unreleased] — 2026-07-31

### Añadido

- doc_agent unifica grafo de graphify, RAG vectorial y boveda Obsidian
- dataset, fine-tuning LoRA de Qwen 2.5, Modelfile de Ollama y skill ClimaSafe
- la intensidad sale del MET del deporte (Compendium 2024) en vez de elegirla el usuario
- indexa documentacion/ y expone busqueda semantica y LLM local por MCP
- parte final descriptivo con riesgo %, temperatura, UV y recomendacion contextual
- cierra GIT-001
- formulario Telegram con 17 estados, teclados inline, toggles multiselect y perfiles SQLite
- tools de predicción, factores y perfiles con SQLite
- pipeline de chunks, embeddings y búsqueda semántica
- orquestador de ciclo con puerta, backlog y subagentes
- Web UI grupo/volumen/curvas edad + estimacion volumetrica + fixes
- MCP server con 3 tools (predict, listar, cargar perfil) + integracion opencode
- riesgo colectivo/etiqueta, mapa de zona adaptativo y riesgo horario acumulado
- perfiles por alias, conformal prediction, frontend, docs
- personalizacion, overrides, explicabilidad, thresholds
- RAG, active learning, contrafactuales y mejora en API/chat
- calibra thresholds ML (t1=0.25/t2=0.40), elimina umbrales provincia obsoletos, personalización alineada con ML
- factores dinámicos desde JSON, personalización afecta clase final, prompt scout desambiguado
- conformal prediction para mejorar calibración de resultados
- web endpoints + MCP server + banner factores pendientes
- calidad de papers (alta/media/baja) según fuente y citas
- paper scout con arXiv/OpenAlex y clasificación LLM
- fusión LSTM, ensemble, explicabilidad y recomendaciones
- LSTM province_hybrid integrada y optimizada
- dataset a 27 features y suelo de mortalidad en labels
- experimento de features de frío con retardo largo (#11)
- ablación limpia de features 27 vs 19 con label fijo (#9)
- calibración de umbrales de decisión por clase
- LSTM híbrida con contexto de ola (secuencia 24h + features diarias)
- baseline LightGBM como candidato a sustituir a KNN
- personalización individual del índice de riesgo
- LSTM multi-tarea con comparación clasificación vs regresión
- notebooks, features pipeline y mlflow
- grafo de conocimiento y documentación
- sistema de agentes v0.2.0
- entrena por clase con XGBoost ponderado y selecciona por recall
- divide train/test por fecha en vez de aleatoriamente
- añade estadísticas diarias y persistencia temporal desde ERA5
- integrate KNN classifier with automated hyperparameter tuning and implement robust MLflow local tracking fallback
- actualizar make_dataset y añadir utilidades de procesamiento (labels, weather_indices); fix: ajustes en build_features
- expand orchestration and tooling
- implement monthly ERA5 download with spatial preprocessing and API integration

### Corrección de bugs

- aisla el tracking URI de mlflow para que no se filtre entre tests
- un campo desconocido en el perfil devuelve error explicito en vez de un 500 mudo
- EDADES_COMPARATIVA (25,55,65,75,85) evita duplicados de curva
- timeout 90s en llamada LLM del scout
- permisos de escritura para GH Actions push
- solo buscar en journals peer-review (excluir arXiv y repositorios)
- exception gitignore para factores_riesgo.json + fallback si no existe
- elimina la fuga temporal train-test del label de riesgo
- persiste modelo_desplegado_frio en 0-2-Frio

### Documentación

- manifiesto de deploy con cuatro perfiles, de solo RAG a Qwen fine-tuneado
- reestructura documentación técnica y de riesgo
- roadmap completo con 19 tareas priorizadas y todo lo hecho marcado
- revisión 68 papers, nuevos factores documentados, próximos pasos actualizados
- reorganizar documentacion (vault + papers + problemas conocidos)
- vault Obsidian, README y CHANGELOG actualizados
- actualiza features de 19 a 27 (persistencia ampliada)
- registra LSTM, personalización y rename de documentacion
- corrige el nombre de la carpeta a «documentacion»
- documenta las decisiones de diseño y la selección por recall
- actualizar documentacion sobre agentes; chore: añadir make_dataset cambios
- documentar diseño del modelo, formulas deterministas y arquitectura de agentes
- update .env.example and add model design documentation

### Tests

- cubre las features de distribución diaria y de rezago temporal

### Mantenimiento

- pipeline de datos/features, tests y documentacion
- cambios pendientes pre-sesión
- añade chat/static para que la suite pueda ejecutarse

### Otros

- add factores_riesgo.json (force-add, gitignored)
- Merge branch 'mejoras/higiene-docs'
- Merge branch 'mejoras/label-sin-fuga'
- Merge branch 'mejoras/umbrales-decision'
- Merge branch 'mejoras/lstm-hibrida'
- Merge remote-tracking branch 'origin/main'
- notebooks: actualizar comentarios automáticos (interpretaciones de gráficas)
- notebooks: añadir comentarios y actualizar ejecución (preprocesado y entrenamiento)
- Refactor code structure for improved readability and maintainability
- notebooks: añadir comentarios automáticos; fix: corregir preprocess_data UnboundLocalError
- añadidos agentes y sus respectivas funciones
- Estructura inicial creada

## [Unreleased] — 2026-07-31

### Añadido

- doc_agent unifica grafo de graphify, RAG vectorial y boveda Obsidian
- dataset, fine-tuning LoRA de Qwen 2.5, Modelfile de Ollama y skill ClimaSafe
- la intensidad sale del MET del deporte (Compendium 2024) en vez de elegirla el usuario
- indexa documentacion/ y expone busqueda semantica y LLM local por MCP
- parte final descriptivo con riesgo %, temperatura, UV y recomendacion contextual
- cierra GIT-001
- formulario Telegram con 17 estados, teclados inline, toggles multiselect y perfiles SQLite
- tools de predicción, factores y perfiles con SQLite
- pipeline de chunks, embeddings y búsqueda semántica
- orquestador de ciclo con puerta, backlog y subagentes
- Web UI grupo/volumen/curvas edad + estimacion volumetrica + fixes
- MCP server con 3 tools (predict, listar, cargar perfil) + integracion opencode
- riesgo colectivo/etiqueta, mapa de zona adaptativo y riesgo horario acumulado
- perfiles por alias, conformal prediction, frontend, docs
- personalizacion, overrides, explicabilidad, thresholds
- RAG, active learning, contrafactuales y mejora en API/chat
- calibra thresholds ML (t1=0.25/t2=0.40), elimina umbrales provincia obsoletos, personalización alineada con ML
- factores dinámicos desde JSON, personalización afecta clase final, prompt scout desambiguado
- conformal prediction para mejorar calibración de resultados
- web endpoints + MCP server + banner factores pendientes
- calidad de papers (alta/media/baja) según fuente y citas
- paper scout con arXiv/OpenAlex y clasificación LLM
- fusión LSTM, ensemble, explicabilidad y recomendaciones
- LSTM province_hybrid integrada y optimizada
- dataset a 27 features y suelo de mortalidad en labels
- experimento de features de frío con retardo largo (#11)
- ablación limpia de features 27 vs 19 con label fijo (#9)
- calibración de umbrales de decisión por clase
- LSTM híbrida con contexto de ola (secuencia 24h + features diarias)
- baseline LightGBM como candidato a sustituir a KNN
- personalización individual del índice de riesgo
- LSTM multi-tarea con comparación clasificación vs regresión
- notebooks, features pipeline y mlflow
- grafo de conocimiento y documentación
- sistema de agentes v0.2.0
- entrena por clase con XGBoost ponderado y selecciona por recall
- divide train/test por fecha en vez de aleatoriamente
- añade estadísticas diarias y persistencia temporal desde ERA5
- integrate KNN classifier with automated hyperparameter tuning and implement robust MLflow local tracking fallback
- actualizar make_dataset y añadir utilidades de procesamiento (labels, weather_indices); fix: ajustes en build_features
- expand orchestration and tooling
- implement monthly ERA5 download with spatial preprocessing and API integration

### Corrección de bugs

- aisla el tracking URI de mlflow para que no se filtre entre tests
- un campo desconocido en el perfil devuelve error explicito en vez de un 500 mudo
- EDADES_COMPARATIVA (25,55,65,75,85) evita duplicados de curva
- timeout 90s en llamada LLM del scout
- permisos de escritura para GH Actions push
- solo buscar en journals peer-review (excluir arXiv y repositorios)
- exception gitignore para factores_riesgo.json + fallback si no existe
- elimina la fuga temporal train-test del label de riesgo
- persiste modelo_desplegado_frio en 0-2-Frio

### Documentación

- cierra DOC-003 — README vuelve a su formato original y lo ampliado vive en documentacion/
- manifiesto de deploy con cuatro perfiles, de solo RAG a Qwen fine-tuneado
- reestructura documentación técnica y de riesgo
- roadmap completo con 19 tareas priorizadas y todo lo hecho marcado
- revisión 68 papers, nuevos factores documentados, próximos pasos actualizados
- reorganizar documentacion (vault + papers + problemas conocidos)
- vault Obsidian, README y CHANGELOG actualizados
- actualiza features de 19 a 27 (persistencia ampliada)
- registra LSTM, personalización y rename de documentacion
- corrige el nombre de la carpeta a «documentacion»
- documenta las decisiones de diseño y la selección por recall
- actualizar documentacion sobre agentes; chore: añadir make_dataset cambios
- documentar diseño del modelo, formulas deterministas y arquitectura de agentes
- update .env.example and add model design documentation

### Tests

- cubre las features de distribución diaria y de rezago temporal

### Mantenimiento

- pipeline de datos/features, tests y documentacion
- cambios pendientes pre-sesión
- añade chat/static para que la suite pueda ejecutarse

### Otros

- add factores_riesgo.json (force-add, gitignored)
- Merge branch 'mejoras/higiene-docs'
- Merge branch 'mejoras/label-sin-fuga'
- Merge branch 'mejoras/umbrales-decision'
- Merge branch 'mejoras/lstm-hibrida'
- Merge remote-tracking branch 'origin/main'
- notebooks: actualizar comentarios automáticos (interpretaciones de gráficas)
- notebooks: añadir comentarios y actualizar ejecución (preprocesado y entrenamiento)
- Refactor code structure for improved readability and maintainability
- notebooks: añadir comentarios automáticos; fix: corregir preprocess_data UnboundLocalError
- añadidos agentes y sus respectivas funciones
- Estructura inicial creada

## [Unreleased] — 2026-08-03

### Añadido

- doc_agent unifica grafo de graphify, RAG vectorial y boveda Obsidian
- dataset, fine-tuning LoRA de Qwen 2.5, Modelfile de Ollama y skill ClimaSafe
- la intensidad sale del MET del deporte (Compendium 2024) en vez de elegirla el usuario
- indexa documentacion/ y expone busqueda semantica y LLM local por MCP
- parte final descriptivo con riesgo %, temperatura, UV y recomendacion contextual
- cierra GIT-001
- formulario Telegram con 17 estados, teclados inline, toggles multiselect y perfiles SQLite
- tools de predicción, factores y perfiles con SQLite
- pipeline de chunks, embeddings y búsqueda semántica
- orquestador de ciclo con puerta, backlog y subagentes
- Web UI grupo/volumen/curvas edad + estimacion volumetrica + fixes
- MCP server con 3 tools (predict, listar, cargar perfil) + integracion opencode
- riesgo colectivo/etiqueta, mapa de zona adaptativo y riesgo horario acumulado
- perfiles por alias, conformal prediction, frontend, docs
- personalizacion, overrides, explicabilidad, thresholds
- RAG, active learning, contrafactuales y mejora en API/chat
- calibra thresholds ML (t1=0.25/t2=0.40), elimina umbrales provincia obsoletos, personalización alineada con ML
- factores dinámicos desde JSON, personalización afecta clase final, prompt scout desambiguado
- conformal prediction para mejorar calibración de resultados
- web endpoints + MCP server + banner factores pendientes
- calidad de papers (alta/media/baja) según fuente y citas
- paper scout con arXiv/OpenAlex y clasificación LLM
- fusión LSTM, ensemble, explicabilidad y recomendaciones
- LSTM province_hybrid integrada y optimizada
- dataset a 27 features y suelo de mortalidad en labels
- experimento de features de frío con retardo largo (#11)
- ablación limpia de features 27 vs 19 con label fijo (#9)
- calibración de umbrales de decisión por clase
- LSTM híbrida con contexto de ola (secuencia 24h + features diarias)
- baseline LightGBM como candidato a sustituir a KNN
- personalización individual del índice de riesgo
- LSTM multi-tarea con comparación clasificación vs regresión
- notebooks, features pipeline y mlflow
- grafo de conocimiento y documentación
- sistema de agentes v0.2.0
- entrena por clase con XGBoost ponderado y selecciona por recall
- divide train/test por fecha en vez de aleatoriamente
- añade estadísticas diarias y persistencia temporal desde ERA5
- integrate KNN classifier with automated hyperparameter tuning and implement robust MLflow local tracking fallback
- actualizar make_dataset y añadir utilidades de procesamiento (labels, weather_indices); fix: ajustes en build_features
- expand orchestration and tooling
- implement monthly ERA5 download with spatial preprocessing and API integration

### Corrección de bugs

- aisla el tracking URI de mlflow para que no se filtre entre tests
- un campo desconocido en el perfil devuelve error explicito en vez de un 500 mudo
- EDADES_COMPARATIVA (25,55,65,75,85) evita duplicados de curva
- timeout 90s en llamada LLM del scout
- permisos de escritura para GH Actions push
- solo buscar en journals peer-review (excluir arXiv y repositorios)
- exception gitignore para factores_riesgo.json + fallback si no existe
- elimina la fuga temporal train-test del label de riesgo
- persiste modelo_desplegado_frio en 0-2-Frio

### Documentación

- cierra DOC-003 — README vuelve a su formato original y lo ampliado vive en documentacion/
- manifiesto de deploy con cuatro perfiles, de solo RAG a Qwen fine-tuneado
- reestructura documentación técnica y de riesgo
- roadmap completo con 19 tareas priorizadas y todo lo hecho marcado
- revisión 68 papers, nuevos factores documentados, próximos pasos actualizados
- reorganizar documentacion (vault + papers + problemas conocidos)
- vault Obsidian, README y CHANGELOG actualizados
- actualiza features de 19 a 27 (persistencia ampliada)
- registra LSTM, personalización y rename de documentacion
- corrige el nombre de la carpeta a «documentacion»
- documenta las decisiones de diseño y la selección por recall
- actualizar documentacion sobre agentes; chore: añadir make_dataset cambios
- documentar diseño del modelo, formulas deterministas y arquitectura de agentes
- update .env.example and add model design documentation

### Tests

- cubre las features de distribución diaria y de rezago temporal

### Mantenimiento

- regenera el grafo de conocimiento del proyecto
- pipeline de datos/features, tests y documentacion
- cambios pendientes pre-sesión
- añade chat/static para que la suite pueda ejecutarse

### Otros

- add factores_riesgo.json (force-add, gitignored)
- Merge branch 'mejoras/higiene-docs'
- Merge branch 'mejoras/label-sin-fuga'
- Merge branch 'mejoras/umbrales-decision'
- Merge branch 'mejoras/lstm-hibrida'
- Merge remote-tracking branch 'origin/main'
- notebooks: actualizar comentarios automáticos (interpretaciones de gráficas)
- notebooks: añadir comentarios y actualizar ejecución (preprocesado y entrenamiento)
- Refactor code structure for improved readability and maintainability
- notebooks: añadir comentarios automáticos; fix: corregir preprocess_data UnboundLocalError
- añadidos agentes y sus respectivas funciones
- Estructura inicial creada

## [Unreleased] — 2026-08-03

### Añadido

- doc_agent unifica grafo de graphify, RAG vectorial y boveda Obsidian
- dataset, fine-tuning LoRA de Qwen 2.5, Modelfile de Ollama y skill ClimaSafe
- la intensidad sale del MET del deporte (Compendium 2024) en vez de elegirla el usuario
- indexa documentacion/ y expone busqueda semantica y LLM local por MCP
- parte final descriptivo con riesgo %, temperatura, UV y recomendacion contextual
- cierra GIT-001
- formulario Telegram con 17 estados, teclados inline, toggles multiselect y perfiles SQLite
- tools de predicción, factores y perfiles con SQLite
- pipeline de chunks, embeddings y búsqueda semántica
- orquestador de ciclo con puerta, backlog y subagentes
- Web UI grupo/volumen/curvas edad + estimacion volumetrica + fixes
- MCP server con 3 tools (predict, listar, cargar perfil) + integracion opencode
- riesgo colectivo/etiqueta, mapa de zona adaptativo y riesgo horario acumulado
- perfiles por alias, conformal prediction, frontend, docs
- personalizacion, overrides, explicabilidad, thresholds
- RAG, active learning, contrafactuales y mejora en API/chat
- calibra thresholds ML (t1=0.25/t2=0.40), elimina umbrales provincia obsoletos, personalización alineada con ML
- factores dinámicos desde JSON, personalización afecta clase final, prompt scout desambiguado
- conformal prediction para mejorar calibración de resultados
- web endpoints + MCP server + banner factores pendientes
- calidad de papers (alta/media/baja) según fuente y citas
- paper scout con arXiv/OpenAlex y clasificación LLM
- fusión LSTM, ensemble, explicabilidad y recomendaciones
- LSTM province_hybrid integrada y optimizada
- dataset a 27 features y suelo de mortalidad en labels
- experimento de features de frío con retardo largo (#11)
- ablación limpia de features 27 vs 19 con label fijo (#9)
- calibración de umbrales de decisión por clase
- LSTM híbrida con contexto de ola (secuencia 24h + features diarias)
- baseline LightGBM como candidato a sustituir a KNN
- personalización individual del índice de riesgo
- LSTM multi-tarea con comparación clasificación vs regresión
- notebooks, features pipeline y mlflow
- grafo de conocimiento y documentación
- sistema de agentes v0.2.0
- entrena por clase con XGBoost ponderado y selecciona por recall
- divide train/test por fecha en vez de aleatoriamente
- añade estadísticas diarias y persistencia temporal desde ERA5
- integrate KNN classifier with automated hyperparameter tuning and implement robust MLflow local tracking fallback
- actualizar make_dataset y añadir utilidades de procesamiento (labels, weather_indices); fix: ajustes en build_features
- expand orchestration and tooling
- implement monthly ERA5 download with spatial preprocessing and API integration

### Corrección de bugs

- aisla el tracking URI de mlflow para que no se filtre entre tests
- un campo desconocido en el perfil devuelve error explicito en vez de un 500 mudo
- EDADES_COMPARATIVA (25,55,65,75,85) evita duplicados de curva
- timeout 90s en llamada LLM del scout
- permisos de escritura para GH Actions push
- solo buscar en journals peer-review (excluir arXiv y repositorios)
- exception gitignore para factores_riesgo.json + fallback si no existe
- elimina la fuga temporal train-test del label de riesgo
- persiste modelo_desplegado_frio en 0-2-Frio

### Documentación

- cierra DOC-003 — README vuelve a su formato original y lo ampliado vive en documentacion/
- manifiesto de deploy con cuatro perfiles, de solo RAG a Qwen fine-tuneado
- reestructura documentación técnica y de riesgo
- roadmap completo con 19 tareas priorizadas y todo lo hecho marcado
- revisión 68 papers, nuevos factores documentados, próximos pasos actualizados
- reorganizar documentacion (vault + papers + problemas conocidos)
- vault Obsidian, README y CHANGELOG actualizados
- actualiza features de 19 a 27 (persistencia ampliada)
- registra LSTM, personalización y rename de documentacion
- corrige el nombre de la carpeta a «documentacion»
- documenta las decisiones de diseño y la selección por recall
- actualizar documentacion sobre agentes; chore: añadir make_dataset cambios
- documentar diseño del modelo, formulas deterministas y arquitectura de agentes
- update .env.example and add model design documentation

### Tests

- cubre las features de distribución diaria y de rezago temporal

### Mantenimiento

- no versionar el estado vivo del arnes ni los backups de datos
- regenera el grafo de conocimiento del proyecto
- pipeline de datos/features, tests y documentacion
- cambios pendientes pre-sesión
- añade chat/static para que la suite pueda ejecutarse

### Otros

- add factores_riesgo.json (force-add, gitignored)
- Merge branch 'mejoras/higiene-docs'
- Merge branch 'mejoras/label-sin-fuga'
- Merge branch 'mejoras/umbrales-decision'
- Merge branch 'mejoras/lstm-hibrida'
- Merge remote-tracking branch 'origin/main'
- notebooks: actualizar comentarios automáticos (interpretaciones de gráficas)
- notebooks: añadir comentarios y actualizar ejecución (preprocesado y entrenamiento)
- Refactor code structure for improved readability and maintainability
- notebooks: añadir comentarios automáticos; fix: corregir preprocess_data UnboundLocalError
- añadidos agentes y sus respectivas funciones
- Estructura inicial creada

## [Unreleased] — 2026-08-03

### Añadido

- doc_agent unifica grafo de graphify, RAG vectorial y boveda Obsidian
- dataset, fine-tuning LoRA de Qwen 2.5, Modelfile de Ollama y skill ClimaSafe
- la intensidad sale del MET del deporte (Compendium 2024) en vez de elegirla el usuario
- indexa documentacion/ y expone busqueda semantica y LLM local por MCP
- parte final descriptivo con riesgo %, temperatura, UV y recomendacion contextual
- cierra GIT-001
- formulario Telegram con 17 estados, teclados inline, toggles multiselect y perfiles SQLite
- tools de predicción, factores y perfiles con SQLite
- pipeline de chunks, embeddings y búsqueda semántica
- orquestador de ciclo con puerta, backlog y subagentes
- Web UI grupo/volumen/curvas edad + estimacion volumetrica + fixes
- MCP server con 3 tools (predict, listar, cargar perfil) + integracion opencode
- riesgo colectivo/etiqueta, mapa de zona adaptativo y riesgo horario acumulado
- perfiles por alias, conformal prediction, frontend, docs
- personalizacion, overrides, explicabilidad, thresholds
- RAG, active learning, contrafactuales y mejora en API/chat
- calibra thresholds ML (t1=0.25/t2=0.40), elimina umbrales provincia obsoletos, personalización alineada con ML
- factores dinámicos desde JSON, personalización afecta clase final, prompt scout desambiguado
- conformal prediction para mejorar calibración de resultados
- web endpoints + MCP server + banner factores pendientes
- calidad de papers (alta/media/baja) según fuente y citas
- paper scout con arXiv/OpenAlex y clasificación LLM
- fusión LSTM, ensemble, explicabilidad y recomendaciones
- LSTM province_hybrid integrada y optimizada
- dataset a 27 features y suelo de mortalidad en labels
- experimento de features de frío con retardo largo (#11)
- ablación limpia de features 27 vs 19 con label fijo (#9)
- calibración de umbrales de decisión por clase
- LSTM híbrida con contexto de ola (secuencia 24h + features diarias)
- baseline LightGBM como candidato a sustituir a KNN
- personalización individual del índice de riesgo
- LSTM multi-tarea con comparación clasificación vs regresión
- notebooks, features pipeline y mlflow
- grafo de conocimiento y documentación
- sistema de agentes v0.2.0
- entrena por clase con XGBoost ponderado y selecciona por recall
- divide train/test por fecha en vez de aleatoriamente
- añade estadísticas diarias y persistencia temporal desde ERA5
- integrate KNN classifier with automated hyperparameter tuning and implement robust MLflow local tracking fallback
- actualizar make_dataset y añadir utilidades de procesamiento (labels, weather_indices); fix: ajustes en build_features
- expand orchestration and tooling
- implement monthly ERA5 download with spatial preprocessing and API integration

### Corrección de bugs

- un dato meteorologico NaN ya no tumba la prediccion
- aisla el tracking URI de mlflow para que no se filtre entre tests
- un campo desconocido en el perfil devuelve error explicito en vez de un 500 mudo
- EDADES_COMPARATIVA (25,55,65,75,85) evita duplicados de curva
- timeout 90s en llamada LLM del scout
- permisos de escritura para GH Actions push
- solo buscar en journals peer-review (excluir arXiv y repositorios)
- exception gitignore para factores_riesgo.json + fallback si no existe
- elimina la fuga temporal train-test del label de riesgo
- persiste modelo_desplegado_frio en 0-2-Frio

### Documentación

- cierra DOC-003 — README vuelve a su formato original y lo ampliado vive en documentacion/
- manifiesto de deploy con cuatro perfiles, de solo RAG a Qwen fine-tuneado
- reestructura documentación técnica y de riesgo
- roadmap completo con 19 tareas priorizadas y todo lo hecho marcado
- revisión 68 papers, nuevos factores documentados, próximos pasos actualizados
- reorganizar documentacion (vault + papers + problemas conocidos)
- vault Obsidian, README y CHANGELOG actualizados
- actualiza features de 19 a 27 (persistencia ampliada)
- registra LSTM, personalización y rename de documentacion
- corrige el nombre de la carpeta a «documentacion»
- documenta las decisiones de diseño y la selección por recall
- actualizar documentacion sobre agentes; chore: añadir make_dataset cambios
- documentar diseño del modelo, formulas deterministas y arquitectura de agentes
- update .env.example and add model design documentation

### Tests

- cubre las features de distribución diaria y de rezago temporal

### Mantenimiento

- no versionar el estado vivo del arnes ni los backups de datos
- regenera el grafo de conocimiento del proyecto
- pipeline de datos/features, tests y documentacion
- cambios pendientes pre-sesión
- añade chat/static para que la suite pueda ejecutarse

### Otros

- add factores_riesgo.json (force-add, gitignored)
- Merge branch 'mejoras/higiene-docs'
- Merge branch 'mejoras/label-sin-fuga'
- Merge branch 'mejoras/umbrales-decision'
- Merge branch 'mejoras/lstm-hibrida'
- Merge remote-tracking branch 'origin/main'
- notebooks: actualizar comentarios automáticos (interpretaciones de gráficas)
- notebooks: añadir comentarios y actualizar ejecución (preprocesado y entrenamiento)
- Refactor code structure for improved readability and maintainability
- notebooks: añadir comentarios automáticos; fix: corregir preprocess_data UnboundLocalError
- añadidos agentes y sus respectivas funciones
- Estructura inicial creada

## [Unreleased] — 2026-08-04

### Añadido

- tablas de rutinas semanales y avisos diarios
- doc_agent unifica grafo de graphify, RAG vectorial y boveda Obsidian
- dataset, fine-tuning LoRA de Qwen 2.5, Modelfile de Ollama y skill ClimaSafe
- la intensidad sale del MET del deporte (Compendium 2024) en vez de elegirla el usuario
- indexa documentacion/ y expone busqueda semantica y LLM local por MCP
- parte final descriptivo con riesgo %, temperatura, UV y recomendacion contextual
- cierra GIT-001
- formulario Telegram con 17 estados, teclados inline, toggles multiselect y perfiles SQLite
- tools de predicción, factores y perfiles con SQLite
- pipeline de chunks, embeddings y búsqueda semántica
- orquestador de ciclo con puerta, backlog y subagentes
- Web UI grupo/volumen/curvas edad + estimacion volumetrica + fixes
- MCP server con 3 tools (predict, listar, cargar perfil) + integracion opencode
- riesgo colectivo/etiqueta, mapa de zona adaptativo y riesgo horario acumulado
- perfiles por alias, conformal prediction, frontend, docs
- personalizacion, overrides, explicabilidad, thresholds
- RAG, active learning, contrafactuales y mejora en API/chat
- calibra thresholds ML (t1=0.25/t2=0.40), elimina umbrales provincia obsoletos, personalización alineada con ML
- factores dinámicos desde JSON, personalización afecta clase final, prompt scout desambiguado
- conformal prediction para mejorar calibración de resultados
- web endpoints + MCP server + banner factores pendientes
- calidad de papers (alta/media/baja) según fuente y citas
- paper scout con arXiv/OpenAlex y clasificación LLM
- fusión LSTM, ensemble, explicabilidad y recomendaciones
- LSTM province_hybrid integrada y optimizada
- dataset a 27 features y suelo de mortalidad en labels
- experimento de features de frío con retardo largo (#11)
- ablación limpia de features 27 vs 19 con label fijo (#9)
- calibración de umbrales de decisión por clase
- LSTM híbrida con contexto de ola (secuencia 24h + features diarias)
- baseline LightGBM como candidato a sustituir a KNN
- personalización individual del índice de riesgo
- LSTM multi-tarea con comparación clasificación vs regresión
- notebooks, features pipeline y mlflow
- grafo de conocimiento y documentación
- sistema de agentes v0.2.0
- entrena por clase con XGBoost ponderado y selecciona por recall
- divide train/test por fecha en vez de aleatoriamente
- añade estadísticas diarias y persistencia temporal desde ERA5
- integrate KNN classifier with automated hyperparameter tuning and implement robust MLflow local tracking fallback
- actualizar make_dataset y añadir utilidades de procesamiento (labels, weather_indices); fix: ajustes en build_features
- expand orchestration and tooling
- implement monthly ERA5 download with spatial preprocessing and API integration

### Corrección de bugs

- un dato meteorologico NaN ya no tumba la prediccion
- aisla el tracking URI de mlflow para que no se filtre entre tests
- un campo desconocido en el perfil devuelve error explicito en vez de un 500 mudo
- EDADES_COMPARATIVA (25,55,65,75,85) evita duplicados de curva
- timeout 90s en llamada LLM del scout
- permisos de escritura para GH Actions push
- solo buscar en journals peer-review (excluir arXiv y repositorios)
- exception gitignore para factores_riesgo.json + fallback si no existe
- elimina la fuga temporal train-test del label de riesgo
- persiste modelo_desplegado_frio en 0-2-Frio

### Documentación

- cierra DOC-003 — README vuelve a su formato original y lo ampliado vive en documentacion/
- manifiesto de deploy con cuatro perfiles, de solo RAG a Qwen fine-tuneado
- reestructura documentación técnica y de riesgo
- roadmap completo con 19 tareas priorizadas y todo lo hecho marcado
- revisión 68 papers, nuevos factores documentados, próximos pasos actualizados
- reorganizar documentacion (vault + papers + problemas conocidos)
- vault Obsidian, README y CHANGELOG actualizados
- actualiza features de 19 a 27 (persistencia ampliada)
- registra LSTM, personalización y rename de documentacion
- corrige el nombre de la carpeta a «documentacion»
- documenta las decisiones de diseño y la selección por recall
- actualizar documentacion sobre agentes; chore: añadir make_dataset cambios
- documentar diseño del modelo, formulas deterministas y arquitectura de agentes
- update .env.example and add model design documentation

### Tests

- cubre las features de distribución diaria y de rezago temporal

### Mantenimiento

- no versionar el estado vivo del arnes ni los backups de datos
- regenera el grafo de conocimiento del proyecto
- pipeline de datos/features, tests y documentacion
- cambios pendientes pre-sesión
- añade chat/static para que la suite pueda ejecutarse

### Otros

- add factores_riesgo.json (force-add, gitignored)
- Merge branch 'mejoras/higiene-docs'
- Merge branch 'mejoras/label-sin-fuga'
- Merge branch 'mejoras/umbrales-decision'
- Merge branch 'mejoras/lstm-hibrida'
- Merge remote-tracking branch 'origin/main'
- notebooks: actualizar comentarios automáticos (interpretaciones de gráficas)
- notebooks: añadir comentarios y actualizar ejecución (preprocesado y entrenamiento)
- Refactor code structure for improved readability and maintainability
- notebooks: añadir comentarios automáticos; fix: corregir preprocess_data UnboundLocalError
- añadidos agentes y sus respectivas funciones
- Estructura inicial creada

## [Unreleased] — 2026-08-04

### Añadido

- bienvenida, respuestas mejoradas, perfil, rutinas, avisos y parte claro con porcentaje y fecha de nacimiento
- tablas de rutinas semanales y avisos diarios
- doc_agent unifica grafo de graphify, RAG vectorial y boveda Obsidian
- dataset, fine-tuning LoRA de Qwen 2.5, Modelfile de Ollama y skill ClimaSafe
- la intensidad sale del MET del deporte (Compendium 2024) en vez de elegirla el usuario
- indexa documentacion/ y expone busqueda semantica y LLM local por MCP
- parte final descriptivo con riesgo %, temperatura, UV y recomendacion contextual
- cierra GIT-001
- formulario Telegram con 17 estados, teclados inline, toggles multiselect y perfiles SQLite
- tools de predicción, factores y perfiles con SQLite
- pipeline de chunks, embeddings y búsqueda semántica
- orquestador de ciclo con puerta, backlog y subagentes
- Web UI grupo/volumen/curvas edad + estimacion volumetrica + fixes
- MCP server con 3 tools (predict, listar, cargar perfil) + integracion opencode
- riesgo colectivo/etiqueta, mapa de zona adaptativo y riesgo horario acumulado
- perfiles por alias, conformal prediction, frontend, docs
- personalizacion, overrides, explicabilidad, thresholds
- RAG, active learning, contrafactuales y mejora en API/chat
- calibra thresholds ML (t1=0.25/t2=0.40), elimina umbrales provincia obsoletos, personalización alineada con ML
- factores dinámicos desde JSON, personalización afecta clase final, prompt scout desambiguado
- conformal prediction para mejorar calibración de resultados
- web endpoints + MCP server + banner factores pendientes
- calidad de papers (alta/media/baja) según fuente y citas
- paper scout con arXiv/OpenAlex y clasificación LLM
- fusión LSTM, ensemble, explicabilidad y recomendaciones
- LSTM province_hybrid integrada y optimizada
- dataset a 27 features y suelo de mortalidad en labels
- experimento de features de frío con retardo largo (#11)
- ablación limpia de features 27 vs 19 con label fijo (#9)
- calibración de umbrales de decisión por clase
- LSTM híbrida con contexto de ola (secuencia 24h + features diarias)
- baseline LightGBM como candidato a sustituir a KNN
- personalización individual del índice de riesgo
- LSTM multi-tarea con comparación clasificación vs regresión
- notebooks, features pipeline y mlflow
- grafo de conocimiento y documentación
- sistema de agentes v0.2.0
- entrena por clase con XGBoost ponderado y selecciona por recall
- divide train/test por fecha en vez de aleatoriamente
- añade estadísticas diarias y persistencia temporal desde ERA5
- integrate KNN classifier with automated hyperparameter tuning and implement robust MLflow local tracking fallback
- actualizar make_dataset y añadir utilidades de procesamiento (labels, weather_indices); fix: ajustes en build_features
- expand orchestration and tooling
- implement monthly ERA5 download with spatial preprocessing and API integration

### Corrección de bugs

- un dato meteorologico NaN ya no tumba la prediccion
- aisla el tracking URI de mlflow para que no se filtre entre tests
- un campo desconocido en el perfil devuelve error explicito en vez de un 500 mudo
- EDADES_COMPARATIVA (25,55,65,75,85) evita duplicados de curva
- timeout 90s en llamada LLM del scout
- permisos de escritura para GH Actions push
- solo buscar en journals peer-review (excluir arXiv y repositorios)
- exception gitignore para factores_riesgo.json + fallback si no existe
- elimina la fuga temporal train-test del label de riesgo
- persiste modelo_desplegado_frio en 0-2-Frio

### Documentación

- cierra DOC-003 — README vuelve a su formato original y lo ampliado vive en documentacion/
- manifiesto de deploy con cuatro perfiles, de solo RAG a Qwen fine-tuneado
- reestructura documentación técnica y de riesgo
- roadmap completo con 19 tareas priorizadas y todo lo hecho marcado
- revisión 68 papers, nuevos factores documentados, próximos pasos actualizados
- reorganizar documentacion (vault + papers + problemas conocidos)
- vault Obsidian, README y CHANGELOG actualizados
- actualiza features de 19 a 27 (persistencia ampliada)
- registra LSTM, personalización y rename de documentacion
- corrige el nombre de la carpeta a «documentacion»
- documenta las decisiones de diseño y la selección por recall
- actualizar documentacion sobre agentes; chore: añadir make_dataset cambios
- documentar diseño del modelo, formulas deterministas y arquitectura de agentes
- update .env.example and add model design documentation

### Tests

- cubre las features de distribución diaria y de rezago temporal

### Mantenimiento

- no versionar el estado vivo del arnes ni los backups de datos
- regenera el grafo de conocimiento del proyecto
- pipeline de datos/features, tests y documentacion
- cambios pendientes pre-sesión
- añade chat/static para que la suite pueda ejecutarse

### Otros

- add factores_riesgo.json (force-add, gitignored)
- Merge branch 'mejoras/higiene-docs'
- Merge branch 'mejoras/label-sin-fuga'
- Merge branch 'mejoras/umbrales-decision'
- Merge branch 'mejoras/lstm-hibrida'
- Merge remote-tracking branch 'origin/main'
- notebooks: actualizar comentarios automáticos (interpretaciones de gráficas)
- notebooks: añadir comentarios y actualizar ejecución (preprocesado y entrenamiento)
- Refactor code structure for improved readability and maintainability
- notebooks: añadir comentarios automáticos; fix: corregir preprocess_data UnboundLocalError
- añadidos agentes y sus respectivas funciones
- Estructura inicial creada

## [Unreleased] — 2026-08-04

### Añadido

- bienvenida, respuestas mejoradas, perfil, rutinas, avisos y parte claro con porcentaje y fecha de nacimiento
- tablas de rutinas semanales y avisos diarios
- doc_agent unifica grafo de graphify, RAG vectorial y boveda Obsidian
- dataset, fine-tuning LoRA de Qwen 2.5, Modelfile de Ollama y skill ClimaSafe
- la intensidad sale del MET del deporte (Compendium 2024) en vez de elegirla el usuario
- indexa documentacion/ y expone busqueda semantica y LLM local por MCP
- parte final descriptivo con riesgo %, temperatura, UV y recomendacion contextual
- cierra GIT-001
- formulario Telegram con 17 estados, teclados inline, toggles multiselect y perfiles SQLite
- tools de predicción, factores y perfiles con SQLite
- pipeline de chunks, embeddings y búsqueda semántica
- orquestador de ciclo con puerta, backlog y subagentes
- Web UI grupo/volumen/curvas edad + estimacion volumetrica + fixes
- MCP server con 3 tools (predict, listar, cargar perfil) + integracion opencode
- riesgo colectivo/etiqueta, mapa de zona adaptativo y riesgo horario acumulado
- perfiles por alias, conformal prediction, frontend, docs
- personalizacion, overrides, explicabilidad, thresholds
- RAG, active learning, contrafactuales y mejora en API/chat
- calibra thresholds ML (t1=0.25/t2=0.40), elimina umbrales provincia obsoletos, personalización alineada con ML
- factores dinámicos desde JSON, personalización afecta clase final, prompt scout desambiguado
- conformal prediction para mejorar calibración de resultados
- web endpoints + MCP server + banner factores pendientes
- calidad de papers (alta/media/baja) según fuente y citas
- paper scout con arXiv/OpenAlex y clasificación LLM
- fusión LSTM, ensemble, explicabilidad y recomendaciones
- LSTM province_hybrid integrada y optimizada
- dataset a 27 features y suelo de mortalidad en labels
- experimento de features de frío con retardo largo (#11)
- ablación limpia de features 27 vs 19 con label fijo (#9)
- calibración de umbrales de decisión por clase
- LSTM híbrida con contexto de ola (secuencia 24h + features diarias)
- baseline LightGBM como candidato a sustituir a KNN
- personalización individual del índice de riesgo
- LSTM multi-tarea con comparación clasificación vs regresión
- notebooks, features pipeline y mlflow
- grafo de conocimiento y documentación
- sistema de agentes v0.2.0
- entrena por clase con XGBoost ponderado y selecciona por recall
- divide train/test por fecha en vez de aleatoriamente
- añade estadísticas diarias y persistencia temporal desde ERA5
- integrate KNN classifier with automated hyperparameter tuning and implement robust MLflow local tracking fallback
- actualizar make_dataset y añadir utilidades de procesamiento (labels, weather_indices); fix: ajustes en build_features
- expand orchestration and tooling
- implement monthly ERA5 download with spatial preprocessing and API integration

### Corrección de bugs

- escapa XSS en rutinas y pronostico, codigos HTTP 404/400 reales y borrado de perfil sin huerfanos
- un dato meteorologico NaN ya no tumba la prediccion
- aisla el tracking URI de mlflow para que no se filtre entre tests
- un campo desconocido en el perfil devuelve error explicito en vez de un 500 mudo
- EDADES_COMPARATIVA (25,55,65,75,85) evita duplicados de curva
- timeout 90s en llamada LLM del scout
- permisos de escritura para GH Actions push
- solo buscar en journals peer-review (excluir arXiv y repositorios)
- exception gitignore para factores_riesgo.json + fallback si no existe
- elimina la fuga temporal train-test del label de riesgo
- persiste modelo_desplegado_frio en 0-2-Frio

### Documentación

- cierra DOC-003 — README vuelve a su formato original y lo ampliado vive en documentacion/
- manifiesto de deploy con cuatro perfiles, de solo RAG a Qwen fine-tuneado
- reestructura documentación técnica y de riesgo
- roadmap completo con 19 tareas priorizadas y todo lo hecho marcado
- revisión 68 papers, nuevos factores documentados, próximos pasos actualizados
- reorganizar documentacion (vault + papers + problemas conocidos)
- vault Obsidian, README y CHANGELOG actualizados
- actualiza features de 19 a 27 (persistencia ampliada)
- registra LSTM, personalización y rename de documentacion
- corrige el nombre de la carpeta a «documentacion»
- documenta las decisiones de diseño y la selección por recall
- actualizar documentacion sobre agentes; chore: añadir make_dataset cambios
- documentar diseño del modelo, formulas deterministas y arquitectura de agentes
- update .env.example and add model design documentation

### Tests

- cubre las features de distribución diaria y de rezago temporal

### Mantenimiento

- no versionar el estado vivo del arnes ni los backups de datos
- regenera el grafo de conocimiento del proyecto
- pipeline de datos/features, tests y documentacion
- cambios pendientes pre-sesión
- añade chat/static para que la suite pueda ejecutarse

### Otros

- add factores_riesgo.json (force-add, gitignored)
- Merge branch 'mejoras/higiene-docs'
- Merge branch 'mejoras/label-sin-fuga'
- Merge branch 'mejoras/umbrales-decision'
- Merge branch 'mejoras/lstm-hibrida'
- Merge remote-tracking branch 'origin/main'
- notebooks: actualizar comentarios automáticos (interpretaciones de gráficas)
- notebooks: añadir comentarios y actualizar ejecución (preprocesado y entrenamiento)
- Refactor code structure for improved readability and maintainability
- notebooks: añadir comentarios automáticos; fix: corregir preprocess_data UnboundLocalError
- añadidos agentes y sus respectivas funciones
- Estructura inicial creada

## [Unreleased] — 2026-08-04

### Añadido

- borrar_rutina_mcp comprueba propiedad y rutinas MCP precargables con perfil
- bienvenida, respuestas mejoradas, perfil, rutinas, avisos y parte claro con porcentaje y fecha de nacimiento
- tablas de rutinas semanales y avisos diarios
- doc_agent unifica grafo de graphify, RAG vectorial y boveda Obsidian
- dataset, fine-tuning LoRA de Qwen 2.5, Modelfile de Ollama y skill ClimaSafe
- la intensidad sale del MET del deporte (Compendium 2024) en vez de elegirla el usuario
- indexa documentacion/ y expone busqueda semantica y LLM local por MCP
- parte final descriptivo con riesgo %, temperatura, UV y recomendacion contextual
- cierra GIT-001
- formulario Telegram con 17 estados, teclados inline, toggles multiselect y perfiles SQLite
- tools de predicción, factores y perfiles con SQLite
- pipeline de chunks, embeddings y búsqueda semántica
- orquestador de ciclo con puerta, backlog y subagentes
- Web UI grupo/volumen/curvas edad + estimacion volumetrica + fixes
- MCP server con 3 tools (predict, listar, cargar perfil) + integracion opencode
- riesgo colectivo/etiqueta, mapa de zona adaptativo y riesgo horario acumulado
- perfiles por alias, conformal prediction, frontend, docs
- personalizacion, overrides, explicabilidad, thresholds
- RAG, active learning, contrafactuales y mejora en API/chat
- calibra thresholds ML (t1=0.25/t2=0.40), elimina umbrales provincia obsoletos, personalización alineada con ML
- factores dinámicos desde JSON, personalización afecta clase final, prompt scout desambiguado
- conformal prediction para mejorar calibración de resultados
- web endpoints + MCP server + banner factores pendientes
- calidad de papers (alta/media/baja) según fuente y citas
- paper scout con arXiv/OpenAlex y clasificación LLM
- fusión LSTM, ensemble, explicabilidad y recomendaciones
- LSTM province_hybrid integrada y optimizada
- dataset a 27 features y suelo de mortalidad en labels
- experimento de features de frío con retardo largo (#11)
- ablación limpia de features 27 vs 19 con label fijo (#9)
- calibración de umbrales de decisión por clase
- LSTM híbrida con contexto de ola (secuencia 24h + features diarias)
- baseline LightGBM como candidato a sustituir a KNN
- personalización individual del índice de riesgo
- LSTM multi-tarea con comparación clasificación vs regresión
- notebooks, features pipeline y mlflow
- grafo de conocimiento y documentación
- sistema de agentes v0.2.0
- entrena por clase con XGBoost ponderado y selecciona por recall
- divide train/test por fecha en vez de aleatoriamente
- añade estadísticas diarias y persistencia temporal desde ERA5
- integrate KNN classifier with automated hyperparameter tuning and implement robust MLflow local tracking fallback
- actualizar make_dataset y añadir utilidades de procesamiento (labels, weather_indices); fix: ajustes en build_features
- expand orchestration and tooling
- implement monthly ERA5 download with spatial preprocessing and API integration

### Corrección de bugs

- escapa XSS en rutinas y pronostico, codigos HTTP 404/400 reales y borrado de perfil sin huerfanos
- un dato meteorologico NaN ya no tumba la prediccion
- aisla el tracking URI de mlflow para que no se filtre entre tests
- un campo desconocido en el perfil devuelve error explicito en vez de un 500 mudo
- EDADES_COMPARATIVA (25,55,65,75,85) evita duplicados de curva
- timeout 90s en llamada LLM del scout
- permisos de escritura para GH Actions push
- solo buscar en journals peer-review (excluir arXiv y repositorios)
- exception gitignore para factores_riesgo.json + fallback si no existe
- elimina la fuga temporal train-test del label de riesgo
- persiste modelo_desplegado_frio en 0-2-Frio

### Documentación

- cierra DOC-003 — README vuelve a su formato original y lo ampliado vive en documentacion/
- manifiesto de deploy con cuatro perfiles, de solo RAG a Qwen fine-tuneado
- reestructura documentación técnica y de riesgo
- roadmap completo con 19 tareas priorizadas y todo lo hecho marcado
- revisión 68 papers, nuevos factores documentados, próximos pasos actualizados
- reorganizar documentacion (vault + papers + problemas conocidos)
- vault Obsidian, README y CHANGELOG actualizados
- actualiza features de 19 a 27 (persistencia ampliada)
- registra LSTM, personalización y rename de documentacion
- corrige el nombre de la carpeta a «documentacion»
- documenta las decisiones de diseño y la selección por recall
- actualizar documentacion sobre agentes; chore: añadir make_dataset cambios
- documentar diseño del modelo, formulas deterministas y arquitectura de agentes
- update .env.example and add model design documentation

### Tests

- cubre las features de distribución diaria y de rezago temporal

### Mantenimiento

- no versionar el estado vivo del arnes ni los backups de datos
- regenera el grafo de conocimiento del proyecto
- pipeline de datos/features, tests y documentacion
- cambios pendientes pre-sesión
- añade chat/static para que la suite pueda ejecutarse

### Otros

- add factores_riesgo.json (force-add, gitignored)
- Merge branch 'mejoras/higiene-docs'
- Merge branch 'mejoras/label-sin-fuga'
- Merge branch 'mejoras/umbrales-decision'
- Merge branch 'mejoras/lstm-hibrida'
- Merge remote-tracking branch 'origin/main'
- notebooks: actualizar comentarios automáticos (interpretaciones de gráficas)
- notebooks: añadir comentarios y actualizar ejecución (preprocesado y entrenamiento)
- Refactor code structure for improved readability and maintainability
- notebooks: añadir comentarios automáticos; fix: corregir preprocess_data UnboundLocalError
- añadidos agentes y sus respectivas funciones
- Estructura inicial creada

## [Unreleased] — 2026-08-04

### Añadido

- el input del dataset de fine-tuning lleva el parte meteorologico completo de la ventana de actividad
- borrar_rutina_mcp comprueba propiedad y rutinas MCP precargables con perfil
- bienvenida, respuestas mejoradas, perfil, rutinas, avisos y parte claro con porcentaje y fecha de nacimiento
- tablas de rutinas semanales y avisos diarios
- doc_agent unifica grafo de graphify, RAG vectorial y boveda Obsidian
- dataset, fine-tuning LoRA de Qwen 2.5, Modelfile de Ollama y skill ClimaSafe
- la intensidad sale del MET del deporte (Compendium 2024) en vez de elegirla el usuario
- indexa documentacion/ y expone busqueda semantica y LLM local por MCP
- parte final descriptivo con riesgo %, temperatura, UV y recomendacion contextual
- cierra GIT-001
- formulario Telegram con 17 estados, teclados inline, toggles multiselect y perfiles SQLite
- tools de predicción, factores y perfiles con SQLite
- pipeline de chunks, embeddings y búsqueda semántica
- orquestador de ciclo con puerta, backlog y subagentes
- Web UI grupo/volumen/curvas edad + estimacion volumetrica + fixes
- MCP server con 3 tools (predict, listar, cargar perfil) + integracion opencode
- riesgo colectivo/etiqueta, mapa de zona adaptativo y riesgo horario acumulado
- perfiles por alias, conformal prediction, frontend, docs
- personalizacion, overrides, explicabilidad, thresholds
- RAG, active learning, contrafactuales y mejora en API/chat
- calibra thresholds ML (t1=0.25/t2=0.40), elimina umbrales provincia obsoletos, personalización alineada con ML
- factores dinámicos desde JSON, personalización afecta clase final, prompt scout desambiguado
- conformal prediction para mejorar calibración de resultados
- web endpoints + MCP server + banner factores pendientes
- calidad de papers (alta/media/baja) según fuente y citas
- paper scout con arXiv/OpenAlex y clasificación LLM
- fusión LSTM, ensemble, explicabilidad y recomendaciones
- LSTM province_hybrid integrada y optimizada
- dataset a 27 features y suelo de mortalidad en labels
- experimento de features de frío con retardo largo (#11)
- ablación limpia de features 27 vs 19 con label fijo (#9)
- calibración de umbrales de decisión por clase
- LSTM híbrida con contexto de ola (secuencia 24h + features diarias)
- baseline LightGBM como candidato a sustituir a KNN
- personalización individual del índice de riesgo
- LSTM multi-tarea con comparación clasificación vs regresión
- notebooks, features pipeline y mlflow
- grafo de conocimiento y documentación
- sistema de agentes v0.2.0
- entrena por clase con XGBoost ponderado y selecciona por recall
- divide train/test por fecha en vez de aleatoriamente
- añade estadísticas diarias y persistencia temporal desde ERA5
- integrate KNN classifier with automated hyperparameter tuning and implement robust MLflow local tracking fallback
- actualizar make_dataset y añadir utilidades de procesamiento (labels, weather_indices); fix: ajustes en build_features
- expand orchestration and tooling
- implement monthly ERA5 download with spatial preprocessing and API integration

### Corrección de bugs

- escapa XSS en rutinas y pronostico, codigos HTTP 404/400 reales y borrado de perfil sin huerfanos
- un dato meteorologico NaN ya no tumba la prediccion
- aisla el tracking URI de mlflow para que no se filtre entre tests
- un campo desconocido en el perfil devuelve error explicito en vez de un 500 mudo
- EDADES_COMPARATIVA (25,55,65,75,85) evita duplicados de curva
- timeout 90s en llamada LLM del scout
- permisos de escritura para GH Actions push
- solo buscar en journals peer-review (excluir arXiv y repositorios)
- exception gitignore para factores_riesgo.json + fallback si no existe
- elimina la fuga temporal train-test del label de riesgo
- persiste modelo_desplegado_frio en 0-2-Frio

### Documentación

- cierra DOC-003 — README vuelve a su formato original y lo ampliado vive en documentacion/
- manifiesto de deploy con cuatro perfiles, de solo RAG a Qwen fine-tuneado
- reestructura documentación técnica y de riesgo
- roadmap completo con 19 tareas priorizadas y todo lo hecho marcado
- revisión 68 papers, nuevos factores documentados, próximos pasos actualizados
- reorganizar documentacion (vault + papers + problemas conocidos)
- vault Obsidian, README y CHANGELOG actualizados
- actualiza features de 19 a 27 (persistencia ampliada)
- registra LSTM, personalización y rename de documentacion
- corrige el nombre de la carpeta a «documentacion»
- documenta las decisiones de diseño y la selección por recall
- actualizar documentacion sobre agentes; chore: añadir make_dataset cambios
- documentar diseño del modelo, formulas deterministas y arquitectura de agentes
- update .env.example and add model design documentation

### Tests

- cubre las features de distribución diaria y de rezago temporal

### Mantenimiento

- no versionar el estado vivo del arnes ni los backups de datos
- regenera el grafo de conocimiento del proyecto
- pipeline de datos/features, tests y documentacion
- cambios pendientes pre-sesión
- añade chat/static para que la suite pueda ejecutarse

### Otros

- add factores_riesgo.json (force-add, gitignored)
- Merge branch 'mejoras/higiene-docs'
- Merge branch 'mejoras/label-sin-fuga'
- Merge branch 'mejoras/umbrales-decision'
- Merge branch 'mejoras/lstm-hibrida'
- Merge remote-tracking branch 'origin/main'
- notebooks: actualizar comentarios automáticos (interpretaciones de gráficas)
- notebooks: añadir comentarios y actualizar ejecución (preprocesado y entrenamiento)
- Refactor code structure for improved readability and maintainability
- notebooks: añadir comentarios automáticos; fix: corregir preprocess_data UnboundLocalError
- añadidos agentes y sus respectivas funciones
- Estructura inicial creada

## [Unreleased] — 2026-08-04

### Añadido

- el input del dataset de fine-tuning lleva el parte meteorologico completo de la ventana de actividad
- borrar_rutina_mcp comprueba propiedad y rutinas MCP precargables con perfil
- bienvenida, respuestas mejoradas, perfil, rutinas, avisos y parte claro con porcentaje y fecha de nacimiento
- tablas de rutinas semanales y avisos diarios
- doc_agent unifica grafo de graphify, RAG vectorial y boveda Obsidian
- dataset, fine-tuning LoRA de Qwen 2.5, Modelfile de Ollama y skill ClimaSafe
- la intensidad sale del MET del deporte (Compendium 2024) en vez de elegirla el usuario
- indexa documentacion/ y expone busqueda semantica y LLM local por MCP
- parte final descriptivo con riesgo %, temperatura, UV y recomendacion contextual
- cierra GIT-001
- formulario Telegram con 17 estados, teclados inline, toggles multiselect y perfiles SQLite
- tools de predicción, factores y perfiles con SQLite
- pipeline de chunks, embeddings y búsqueda semántica
- orquestador de ciclo con puerta, backlog y subagentes
- Web UI grupo/volumen/curvas edad + estimacion volumetrica + fixes
- MCP server con 3 tools (predict, listar, cargar perfil) + integracion opencode
- riesgo colectivo/etiqueta, mapa de zona adaptativo y riesgo horario acumulado
- perfiles por alias, conformal prediction, frontend, docs
- personalizacion, overrides, explicabilidad, thresholds
- RAG, active learning, contrafactuales y mejora en API/chat
- calibra thresholds ML (t1=0.25/t2=0.40), elimina umbrales provincia obsoletos, personalización alineada con ML
- factores dinámicos desde JSON, personalización afecta clase final, prompt scout desambiguado
- conformal prediction para mejorar calibración de resultados
- web endpoints + MCP server + banner factores pendientes
- calidad de papers (alta/media/baja) según fuente y citas
- paper scout con arXiv/OpenAlex y clasificación LLM
- fusión LSTM, ensemble, explicabilidad y recomendaciones
- LSTM province_hybrid integrada y optimizada
- dataset a 27 features y suelo de mortalidad en labels
- experimento de features de frío con retardo largo (#11)
- ablación limpia de features 27 vs 19 con label fijo (#9)
- calibración de umbrales de decisión por clase
- LSTM híbrida con contexto de ola (secuencia 24h + features diarias)
- baseline LightGBM como candidato a sustituir a KNN
- personalización individual del índice de riesgo
- LSTM multi-tarea con comparación clasificación vs regresión
- notebooks, features pipeline y mlflow
- grafo de conocimiento y documentación
- sistema de agentes v0.2.0
- entrena por clase con XGBoost ponderado y selecciona por recall
- divide train/test por fecha en vez de aleatoriamente
- añade estadísticas diarias y persistencia temporal desde ERA5
- integrate KNN classifier with automated hyperparameter tuning and implement robust MLflow local tracking fallback
- actualizar make_dataset y añadir utilidades de procesamiento (labels, weather_indices); fix: ajustes en build_features
- expand orchestration and tooling
- implement monthly ERA5 download with spatial preprocessing and API integration

### Corrección de bugs

- escapa XSS en rutinas y pronostico, codigos HTTP 404/400 reales y borrado de perfil sin huerfanos
- un dato meteorologico NaN ya no tumba la prediccion
- aisla el tracking URI de mlflow para que no se filtre entre tests
- un campo desconocido en el perfil devuelve error explicito en vez de un 500 mudo
- EDADES_COMPARATIVA (25,55,65,75,85) evita duplicados de curva
- timeout 90s en llamada LLM del scout
- permisos de escritura para GH Actions push
- solo buscar en journals peer-review (excluir arXiv y repositorios)
- exception gitignore para factores_riesgo.json + fallback si no existe
- elimina la fuga temporal train-test del label de riesgo
- persiste modelo_desplegado_frio en 0-2-Frio

### Documentación

- cierra DOC-003 — README vuelve a su formato original y lo ampliado vive en documentacion/
- manifiesto de deploy con cuatro perfiles, de solo RAG a Qwen fine-tuneado
- reestructura documentación técnica y de riesgo
- roadmap completo con 19 tareas priorizadas y todo lo hecho marcado
- revisión 68 papers, nuevos factores documentados, próximos pasos actualizados
- reorganizar documentacion (vault + papers + problemas conocidos)
- vault Obsidian, README y CHANGELOG actualizados
- actualiza features de 19 a 27 (persistencia ampliada)
- registra LSTM, personalización y rename de documentacion
- corrige el nombre de la carpeta a «documentacion»
- documenta las decisiones de diseño y la selección por recall
- actualizar documentacion sobre agentes; chore: añadir make_dataset cambios
- documentar diseño del modelo, formulas deterministas y arquitectura de agentes
- update .env.example and add model design documentation

### Tests

- cubre las features de distribución diaria y de rezago temporal

### Mantenimiento

- make typecheck en el contrato del implementer y backlog con 6 cierres + ARNES-011 y MCP-APPS-001
- no versionar el estado vivo del arnes ni los backups de datos
- regenera el grafo de conocimiento del proyecto
- pipeline de datos/features, tests y documentacion
- cambios pendientes pre-sesión
- añade chat/static para que la suite pueda ejecutarse

### Otros

- add factores_riesgo.json (force-add, gitignored)
- Merge branch 'mejoras/higiene-docs'
- Merge branch 'mejoras/label-sin-fuga'
- Merge branch 'mejoras/umbrales-decision'
- Merge branch 'mejoras/lstm-hibrida'
- Merge remote-tracking branch 'origin/main'
- notebooks: actualizar comentarios automáticos (interpretaciones de gráficas)
- notebooks: añadir comentarios y actualizar ejecución (preprocesado y entrenamiento)
- Refactor code structure for improved readability and maintainability
- notebooks: añadir comentarios automáticos; fix: corregir preprocess_data UnboundLocalError
- añadidos agentes y sus respectivas funciones
- Estructura inicial creada

## [Unreleased] — 2026-08-04

### Añadido

- el RAG indexa el coeficiente y el DOI de cada factor, no solo el nombre
- el input del dataset de fine-tuning lleva el parte meteorologico completo de la ventana de actividad
- borrar_rutina_mcp comprueba propiedad y rutinas MCP precargables con perfil
- bienvenida, respuestas mejoradas, perfil, rutinas, avisos y parte claro con porcentaje y fecha de nacimiento
- tablas de rutinas semanales y avisos diarios
- doc_agent unifica grafo de graphify, RAG vectorial y boveda Obsidian
- dataset, fine-tuning LoRA de Qwen 2.5, Modelfile de Ollama y skill ClimaSafe
- la intensidad sale del MET del deporte (Compendium 2024) en vez de elegirla el usuario
- indexa documentacion/ y expone busqueda semantica y LLM local por MCP
- parte final descriptivo con riesgo %, temperatura, UV y recomendacion contextual
- cierra GIT-001
- formulario Telegram con 17 estados, teclados inline, toggles multiselect y perfiles SQLite
- tools de predicción, factores y perfiles con SQLite
- pipeline de chunks, embeddings y búsqueda semántica
- orquestador de ciclo con puerta, backlog y subagentes
- Web UI grupo/volumen/curvas edad + estimacion volumetrica + fixes
- MCP server con 3 tools (predict, listar, cargar perfil) + integracion opencode
- riesgo colectivo/etiqueta, mapa de zona adaptativo y riesgo horario acumulado
- perfiles por alias, conformal prediction, frontend, docs
- personalizacion, overrides, explicabilidad, thresholds
- RAG, active learning, contrafactuales y mejora en API/chat
- calibra thresholds ML (t1=0.25/t2=0.40), elimina umbrales provincia obsoletos, personalización alineada con ML
- factores dinámicos desde JSON, personalización afecta clase final, prompt scout desambiguado
- conformal prediction para mejorar calibración de resultados
- web endpoints + MCP server + banner factores pendientes
- calidad de papers (alta/media/baja) según fuente y citas
- paper scout con arXiv/OpenAlex y clasificación LLM
- fusión LSTM, ensemble, explicabilidad y recomendaciones
- LSTM province_hybrid integrada y optimizada
- dataset a 27 features y suelo de mortalidad en labels
- experimento de features de frío con retardo largo (#11)
- ablación limpia de features 27 vs 19 con label fijo (#9)
- calibración de umbrales de decisión por clase
- LSTM híbrida con contexto de ola (secuencia 24h + features diarias)
- baseline LightGBM como candidato a sustituir a KNN
- personalización individual del índice de riesgo
- LSTM multi-tarea con comparación clasificación vs regresión
- notebooks, features pipeline y mlflow
- grafo de conocimiento y documentación
- sistema de agentes v0.2.0
- entrena por clase con XGBoost ponderado y selecciona por recall
- divide train/test por fecha en vez de aleatoriamente
- añade estadísticas diarias y persistencia temporal desde ERA5
- integrate KNN classifier with automated hyperparameter tuning and implement robust MLflow local tracking fallback
- actualizar make_dataset y añadir utilidades de procesamiento (labels, weather_indices); fix: ajustes en build_features
- expand orchestration and tooling
- implement monthly ERA5 download with spatial preprocessing and API integration

### Corrección de bugs

- escapa XSS en rutinas y pronostico, codigos HTTP 404/400 reales y borrado de perfil sin huerfanos
- un dato meteorologico NaN ya no tumba la prediccion
- aisla el tracking URI de mlflow para que no se filtre entre tests
- un campo desconocido en el perfil devuelve error explicito en vez de un 500 mudo
- EDADES_COMPARATIVA (25,55,65,75,85) evita duplicados de curva
- timeout 90s en llamada LLM del scout
- permisos de escritura para GH Actions push
- solo buscar en journals peer-review (excluir arXiv y repositorios)
- exception gitignore para factores_riesgo.json + fallback si no existe
- elimina la fuga temporal train-test del label de riesgo
- persiste modelo_desplegado_frio en 0-2-Frio

### Documentación

- cierra DOC-003 — README vuelve a su formato original y lo ampliado vive en documentacion/
- manifiesto de deploy con cuatro perfiles, de solo RAG a Qwen fine-tuneado
- reestructura documentación técnica y de riesgo
- roadmap completo con 19 tareas priorizadas y todo lo hecho marcado
- revisión 68 papers, nuevos factores documentados, próximos pasos actualizados
- reorganizar documentacion (vault + papers + problemas conocidos)
- vault Obsidian, README y CHANGELOG actualizados
- actualiza features de 19 a 27 (persistencia ampliada)
- registra LSTM, personalización y rename de documentacion
- corrige el nombre de la carpeta a «documentacion»
- documenta las decisiones de diseño y la selección por recall
- actualizar documentacion sobre agentes; chore: añadir make_dataset cambios
- documentar diseño del modelo, formulas deterministas y arquitectura de agentes
- update .env.example and add model design documentation

### Tests

- cubre las features de distribución diaria y de rezago temporal

### Mantenimiento

- make typecheck en el contrato del implementer y backlog con 6 cierres + ARNES-011 y MCP-APPS-001
- no versionar el estado vivo del arnes ni los backups de datos
- regenera el grafo de conocimiento del proyecto
- pipeline de datos/features, tests y documentacion
- cambios pendientes pre-sesión
- añade chat/static para que la suite pueda ejecutarse

### Otros

- add factores_riesgo.json (force-add, gitignored)
- Merge branch 'mejoras/higiene-docs'
- Merge branch 'mejoras/label-sin-fuga'
- Merge branch 'mejoras/umbrales-decision'
- Merge branch 'mejoras/lstm-hibrida'
- Merge remote-tracking branch 'origin/main'
- notebooks: actualizar comentarios automáticos (interpretaciones de gráficas)
- notebooks: añadir comentarios y actualizar ejecución (preprocesado y entrenamiento)
- Refactor code structure for improved readability and maintainability
- notebooks: añadir comentarios automáticos; fix: corregir preprocess_data UnboundLocalError
- añadidos agentes y sus respectivas funciones
- Estructura inicial creada

## [Unreleased] — 2026-08-05

### Añadido

- el RAG indexa el coeficiente y el DOI de cada factor, no solo el nombre
- el input del dataset de fine-tuning lleva el parte meteorologico completo de la ventana de actividad
- borrar_rutina_mcp comprueba propiedad y rutinas MCP precargables con perfil
- bienvenida, respuestas mejoradas, perfil, rutinas, avisos y parte claro con porcentaje y fecha de nacimiento
- tablas de rutinas semanales y avisos diarios
- doc_agent unifica grafo de graphify, RAG vectorial y boveda Obsidian
- dataset, fine-tuning LoRA de Qwen 2.5, Modelfile de Ollama y skill ClimaSafe
- la intensidad sale del MET del deporte (Compendium 2024) en vez de elegirla el usuario
- indexa documentacion/ y expone busqueda semantica y LLM local por MCP
- parte final descriptivo con riesgo %, temperatura, UV y recomendacion contextual
- cierra GIT-001
- formulario Telegram con 17 estados, teclados inline, toggles multiselect y perfiles SQLite
- tools de predicción, factores y perfiles con SQLite
- pipeline de chunks, embeddings y búsqueda semántica
- orquestador de ciclo con puerta, backlog y subagentes
- Web UI grupo/volumen/curvas edad + estimacion volumetrica + fixes
- MCP server con 3 tools (predict, listar, cargar perfil) + integracion opencode
- riesgo colectivo/etiqueta, mapa de zona adaptativo y riesgo horario acumulado
- perfiles por alias, conformal prediction, frontend, docs
- personalizacion, overrides, explicabilidad, thresholds
- RAG, active learning, contrafactuales y mejora en API/chat
- calibra thresholds ML (t1=0.25/t2=0.40), elimina umbrales provincia obsoletos, personalización alineada con ML
- factores dinámicos desde JSON, personalización afecta clase final, prompt scout desambiguado
- conformal prediction para mejorar calibración de resultados
- web endpoints + MCP server + banner factores pendientes
- calidad de papers (alta/media/baja) según fuente y citas
- paper scout con arXiv/OpenAlex y clasificación LLM
- fusión LSTM, ensemble, explicabilidad y recomendaciones
- LSTM province_hybrid integrada y optimizada
- dataset a 27 features y suelo de mortalidad en labels
- experimento de features de frío con retardo largo (#11)
- ablación limpia de features 27 vs 19 con label fijo (#9)
- calibración de umbrales de decisión por clase
- LSTM híbrida con contexto de ola (secuencia 24h + features diarias)
- baseline LightGBM como candidato a sustituir a KNN
- personalización individual del índice de riesgo
- LSTM multi-tarea con comparación clasificación vs regresión
- notebooks, features pipeline y mlflow
- grafo de conocimiento y documentación
- sistema de agentes v0.2.0
- entrena por clase con XGBoost ponderado y selecciona por recall
- divide train/test por fecha en vez de aleatoriamente
- añade estadísticas diarias y persistencia temporal desde ERA5
- integrate KNN classifier with automated hyperparameter tuning and implement robust MLflow local tracking fallback
- actualizar make_dataset y añadir utilidades de procesamiento (labels, weather_indices); fix: ajustes en build_features
- expand orchestration and tooling
- implement monthly ERA5 download with spatial preprocessing and API integration

### Corrección de bugs

- escapa XSS en rutinas y pronostico, codigos HTTP 404/400 reales y borrado de perfil sin huerfanos
- un dato meteorologico NaN ya no tumba la prediccion
- aisla el tracking URI de mlflow para que no se filtre entre tests
- un campo desconocido en el perfil devuelve error explicito en vez de un 500 mudo
- EDADES_COMPARATIVA (25,55,65,75,85) evita duplicados de curva
- timeout 90s en llamada LLM del scout
- permisos de escritura para GH Actions push
- solo buscar en journals peer-review (excluir arXiv y repositorios)
- exception gitignore para factores_riesgo.json + fallback si no existe
- elimina la fuga temporal train-test del label de riesgo
- persiste modelo_desplegado_frio en 0-2-Frio

### Documentación

- cierra DOC-003 — README vuelve a su formato original y lo ampliado vive en documentacion/
- manifiesto de deploy con cuatro perfiles, de solo RAG a Qwen fine-tuneado
- reestructura documentación técnica y de riesgo
- roadmap completo con 19 tareas priorizadas y todo lo hecho marcado
- revisión 68 papers, nuevos factores documentados, próximos pasos actualizados
- reorganizar documentacion (vault + papers + problemas conocidos)
- vault Obsidian, README y CHANGELOG actualizados
- actualiza features de 19 a 27 (persistencia ampliada)
- registra LSTM, personalización y rename de documentacion
- corrige el nombre de la carpeta a «documentacion»
- documenta las decisiones de diseño y la selección por recall
- actualizar documentacion sobre agentes; chore: añadir make_dataset cambios
- documentar diseño del modelo, formulas deterministas y arquitectura de agentes
- update .env.example and add model design documentation

### Tests

- cubre las features de distribución diaria y de rezago temporal

### Mantenimiento

- backlog con cierre de RAG-002 y bump a 0.0.17
- make typecheck en el contrato del implementer y backlog con 6 cierres + ARNES-011 y MCP-APPS-001
- no versionar el estado vivo del arnes ni los backups de datos
- regenera el grafo de conocimiento del proyecto
- pipeline de datos/features, tests y documentacion
- cambios pendientes pre-sesión
- añade chat/static para que la suite pueda ejecutarse

### Otros

- add factores_riesgo.json (force-add, gitignored)
- Merge branch 'mejoras/higiene-docs'
- Merge branch 'mejoras/label-sin-fuga'
- Merge branch 'mejoras/umbrales-decision'
- Merge branch 'mejoras/lstm-hibrida'
- Merge remote-tracking branch 'origin/main'
- notebooks: actualizar comentarios automáticos (interpretaciones de gráficas)
- notebooks: añadir comentarios y actualizar ejecución (preprocesado y entrenamiento)
- Refactor code structure for improved readability and maintainability
- notebooks: añadir comentarios automáticos; fix: corregir preprocess_data UnboundLocalError
- añadidos agentes y sus respectivas funciones
- Estructura inicial creada

## [Unreleased] — 2026-08-05

### Añadido

- parte en lenguaje llano con confianza conformal y factores con coeficiente (BOT-013 + LLM-005)
- el RAG indexa el coeficiente y el DOI de cada factor, no solo el nombre
- el input del dataset de fine-tuning lleva el parte meteorologico completo de la ventana de actividad
- borrar_rutina_mcp comprueba propiedad y rutinas MCP precargables con perfil
- bienvenida, respuestas mejoradas, perfil, rutinas, avisos y parte claro con porcentaje y fecha de nacimiento
- tablas de rutinas semanales y avisos diarios
- doc_agent unifica grafo de graphify, RAG vectorial y boveda Obsidian
- dataset, fine-tuning LoRA de Qwen 2.5, Modelfile de Ollama y skill ClimaSafe
- la intensidad sale del MET del deporte (Compendium 2024) en vez de elegirla el usuario
- indexa documentacion/ y expone busqueda semantica y LLM local por MCP
- parte final descriptivo con riesgo %, temperatura, UV y recomendacion contextual
- cierra GIT-001
- formulario Telegram con 17 estados, teclados inline, toggles multiselect y perfiles SQLite
- tools de predicción, factores y perfiles con SQLite
- pipeline de chunks, embeddings y búsqueda semántica
- orquestador de ciclo con puerta, backlog y subagentes
- Web UI grupo/volumen/curvas edad + estimacion volumetrica + fixes
- MCP server con 3 tools (predict, listar, cargar perfil) + integracion opencode
- riesgo colectivo/etiqueta, mapa de zona adaptativo y riesgo horario acumulado
- perfiles por alias, conformal prediction, frontend, docs
- personalizacion, overrides, explicabilidad, thresholds
- RAG, active learning, contrafactuales y mejora en API/chat
- calibra thresholds ML (t1=0.25/t2=0.40), elimina umbrales provincia obsoletos, personalización alineada con ML
- factores dinámicos desde JSON, personalización afecta clase final, prompt scout desambiguado
- conformal prediction para mejorar calibración de resultados
- web endpoints + MCP server + banner factores pendientes
- calidad de papers (alta/media/baja) según fuente y citas
- paper scout con arXiv/OpenAlex y clasificación LLM
- fusión LSTM, ensemble, explicabilidad y recomendaciones
- LSTM province_hybrid integrada y optimizada
- dataset a 27 features y suelo de mortalidad en labels
- experimento de features de frío con retardo largo (#11)
- ablación limpia de features 27 vs 19 con label fijo (#9)
- calibración de umbrales de decisión por clase
- LSTM híbrida con contexto de ola (secuencia 24h + features diarias)
- baseline LightGBM como candidato a sustituir a KNN
- personalización individual del índice de riesgo
- LSTM multi-tarea con comparación clasificación vs regresión
- notebooks, features pipeline y mlflow
- grafo de conocimiento y documentación
- sistema de agentes v0.2.0
- entrena por clase con XGBoost ponderado y selecciona por recall
- divide train/test por fecha en vez de aleatoriamente
- añade estadísticas diarias y persistencia temporal desde ERA5
- integrate KNN classifier with automated hyperparameter tuning and implement robust MLflow local tracking fallback
- actualizar make_dataset y añadir utilidades de procesamiento (labels, weather_indices); fix: ajustes en build_features
- expand orchestration and tooling
- implement monthly ERA5 download with spatial preprocessing and API integration

### Corrección de bugs

- escapa XSS en rutinas y pronostico, codigos HTTP 404/400 reales y borrado de perfil sin huerfanos
- un dato meteorologico NaN ya no tumba la prediccion
- aisla el tracking URI de mlflow para que no se filtre entre tests
- un campo desconocido en el perfil devuelve error explicito en vez de un 500 mudo
- EDADES_COMPARATIVA (25,55,65,75,85) evita duplicados de curva
- timeout 90s en llamada LLM del scout
- permisos de escritura para GH Actions push
- solo buscar en journals peer-review (excluir arXiv y repositorios)
- exception gitignore para factores_riesgo.json + fallback si no existe
- elimina la fuga temporal train-test del label de riesgo
- persiste modelo_desplegado_frio en 0-2-Frio

### Documentación

- cierra DOC-003 — README vuelve a su formato original y lo ampliado vive en documentacion/
- manifiesto de deploy con cuatro perfiles, de solo RAG a Qwen fine-tuneado
- reestructura documentación técnica y de riesgo
- roadmap completo con 19 tareas priorizadas y todo lo hecho marcado
- revisión 68 papers, nuevos factores documentados, próximos pasos actualizados
- reorganizar documentacion (vault + papers + problemas conocidos)
- vault Obsidian, README y CHANGELOG actualizados
- actualiza features de 19 a 27 (persistencia ampliada)
- registra LSTM, personalización y rename de documentacion
- corrige el nombre de la carpeta a «documentacion»
- documenta las decisiones de diseño y la selección por recall
- actualizar documentacion sobre agentes; chore: añadir make_dataset cambios
- documentar diseño del modelo, formulas deterministas y arquitectura de agentes
- update .env.example and add model design documentation

### Tests

- cubre las features de distribución diaria y de rezago temporal

### Mantenimiento

- backlog con cierre de RAG-002 y bump a 0.0.17
- make typecheck en el contrato del implementer y backlog con 6 cierres + ARNES-011 y MCP-APPS-001
- no versionar el estado vivo del arnes ni los backups de datos
- regenera el grafo de conocimiento del proyecto
- pipeline de datos/features, tests y documentacion
- cambios pendientes pre-sesión
- añade chat/static para que la suite pueda ejecutarse

### Otros

- add factores_riesgo.json (force-add, gitignored)
- Merge branch 'mejoras/higiene-docs'
- Merge branch 'mejoras/label-sin-fuga'
- Merge branch 'mejoras/umbrales-decision'
- Merge branch 'mejoras/lstm-hibrida'
- Merge remote-tracking branch 'origin/main'
- notebooks: actualizar comentarios automáticos (interpretaciones de gráficas)
- notebooks: añadir comentarios y actualizar ejecución (preprocesado y entrenamiento)
- Refactor code structure for improved readability and maintainability
- notebooks: añadir comentarios automáticos; fix: corregir preprocess_data UnboundLocalError
- añadidos agentes y sus respectivas funciones
- Estructura inicial creada

## [Unreleased] — 2026-08-05

### Añadido

- cuestionario de rutina de trabajo con ocupacion e intensidad, parte con confianza conformal y chat con perfil (BOT-013 + BOT-015 + LLM-005)
- parte en lenguaje llano con confianza conformal y factores con coeficiente (BOT-013 + LLM-005)
- el RAG indexa el coeficiente y el DOI de cada factor, no solo el nombre
- el input del dataset de fine-tuning lleva el parte meteorologico completo de la ventana de actividad
- borrar_rutina_mcp comprueba propiedad y rutinas MCP precargables con perfil
- bienvenida, respuestas mejoradas, perfil, rutinas, avisos y parte claro con porcentaje y fecha de nacimiento
- tablas de rutinas semanales y avisos diarios
- doc_agent unifica grafo de graphify, RAG vectorial y boveda Obsidian
- dataset, fine-tuning LoRA de Qwen 2.5, Modelfile de Ollama y skill ClimaSafe
- la intensidad sale del MET del deporte (Compendium 2024) en vez de elegirla el usuario
- indexa documentacion/ y expone busqueda semantica y LLM local por MCP
- parte final descriptivo con riesgo %, temperatura, UV y recomendacion contextual
- cierra GIT-001
- formulario Telegram con 17 estados, teclados inline, toggles multiselect y perfiles SQLite
- tools de predicción, factores y perfiles con SQLite
- pipeline de chunks, embeddings y búsqueda semántica
- orquestador de ciclo con puerta, backlog y subagentes
- Web UI grupo/volumen/curvas edad + estimacion volumetrica + fixes
- MCP server con 3 tools (predict, listar, cargar perfil) + integracion opencode
- riesgo colectivo/etiqueta, mapa de zona adaptativo y riesgo horario acumulado
- perfiles por alias, conformal prediction, frontend, docs
- personalizacion, overrides, explicabilidad, thresholds
- RAG, active learning, contrafactuales y mejora en API/chat
- calibra thresholds ML (t1=0.25/t2=0.40), elimina umbrales provincia obsoletos, personalización alineada con ML
- factores dinámicos desde JSON, personalización afecta clase final, prompt scout desambiguado
- conformal prediction para mejorar calibración de resultados
- web endpoints + MCP server + banner factores pendientes
- calidad de papers (alta/media/baja) según fuente y citas
- paper scout con arXiv/OpenAlex y clasificación LLM
- fusión LSTM, ensemble, explicabilidad y recomendaciones
- LSTM province_hybrid integrada y optimizada
- dataset a 27 features y suelo de mortalidad en labels
- experimento de features de frío con retardo largo (#11)
- ablación limpia de features 27 vs 19 con label fijo (#9)
- calibración de umbrales de decisión por clase
- LSTM híbrida con contexto de ola (secuencia 24h + features diarias)
- baseline LightGBM como candidato a sustituir a KNN
- personalización individual del índice de riesgo
- LSTM multi-tarea con comparación clasificación vs regresión
- notebooks, features pipeline y mlflow
- grafo de conocimiento y documentación
- sistema de agentes v0.2.0
- entrena por clase con XGBoost ponderado y selecciona por recall
- divide train/test por fecha en vez de aleatoriamente
- añade estadísticas diarias y persistencia temporal desde ERA5
- integrate KNN classifier with automated hyperparameter tuning and implement robust MLflow local tracking fallback
- actualizar make_dataset y añadir utilidades de procesamiento (labels, weather_indices); fix: ajustes en build_features
- expand orchestration and tooling
- implement monthly ERA5 download with spatial preprocessing and API integration

### Corrección de bugs

- escapa XSS en rutinas y pronostico, codigos HTTP 404/400 reales y borrado de perfil sin huerfanos
- un dato meteorologico NaN ya no tumba la prediccion
- aisla el tracking URI de mlflow para que no se filtre entre tests
- un campo desconocido en el perfil devuelve error explicito en vez de un 500 mudo
- EDADES_COMPARATIVA (25,55,65,75,85) evita duplicados de curva
- timeout 90s en llamada LLM del scout
- permisos de escritura para GH Actions push
- solo buscar en journals peer-review (excluir arXiv y repositorios)
- exception gitignore para factores_riesgo.json + fallback si no existe
- elimina la fuga temporal train-test del label de riesgo
- persiste modelo_desplegado_frio en 0-2-Frio

### Documentación

- cierra DOC-003 — README vuelve a su formato original y lo ampliado vive en documentacion/
- manifiesto de deploy con cuatro perfiles, de solo RAG a Qwen fine-tuneado
- reestructura documentación técnica y de riesgo
- roadmap completo con 19 tareas priorizadas y todo lo hecho marcado
- revisión 68 papers, nuevos factores documentados, próximos pasos actualizados
- reorganizar documentacion (vault + papers + problemas conocidos)
- vault Obsidian, README y CHANGELOG actualizados
- actualiza features de 19 a 27 (persistencia ampliada)
- registra LSTM, personalización y rename de documentacion
- corrige el nombre de la carpeta a «documentacion»
- documenta las decisiones de diseño y la selección por recall
- actualizar documentacion sobre agentes; chore: añadir make_dataset cambios
- documentar diseño del modelo, formulas deterministas y arquitectura de agentes
- update .env.example and add model design documentation

### Tests

- cubre las features de distribución diaria y de rezago temporal

### Mantenimiento

- backlog con cierre de RAG-002 y bump a 0.0.17
- make typecheck en el contrato del implementer y backlog con 6 cierres + ARNES-011 y MCP-APPS-001
- no versionar el estado vivo del arnes ni los backups de datos
- regenera el grafo de conocimiento del proyecto
- pipeline de datos/features, tests y documentacion
- cambios pendientes pre-sesión
- añade chat/static para que la suite pueda ejecutarse

### Otros

- add factores_riesgo.json (force-add, gitignored)
- Merge branch 'mejoras/higiene-docs'
- Merge branch 'mejoras/label-sin-fuga'
- Merge branch 'mejoras/umbrales-decision'
- Merge branch 'mejoras/lstm-hibrida'
- Merge remote-tracking branch 'origin/main'
- notebooks: actualizar comentarios automáticos (interpretaciones de gráficas)
- notebooks: añadir comentarios y actualizar ejecución (preprocesado y entrenamiento)
- Refactor code structure for improved readability and maintainability
- notebooks: añadir comentarios automáticos; fix: corregir preprocess_data UnboundLocalError
- añadidos agentes y sus respectivas funciones
- Estructura inicial creada

## [Unreleased] — 2026-08-05

### Añadido

- tool grafica_riesgo_horario que devuelve la curva de riesgo por hora como PNG (MCP-IMG-001)
- cuestionario de rutina de trabajo con ocupacion e intensidad, parte con confianza conformal y chat con perfil (BOT-013 + BOT-015 + LLM-005)
- parte en lenguaje llano con confianza conformal y factores con coeficiente (BOT-013 + LLM-005)
- el RAG indexa el coeficiente y el DOI de cada factor, no solo el nombre
- el input del dataset de fine-tuning lleva el parte meteorologico completo de la ventana de actividad
- borrar_rutina_mcp comprueba propiedad y rutinas MCP precargables con perfil
- bienvenida, respuestas mejoradas, perfil, rutinas, avisos y parte claro con porcentaje y fecha de nacimiento
- tablas de rutinas semanales y avisos diarios
- doc_agent unifica grafo de graphify, RAG vectorial y boveda Obsidian
- dataset, fine-tuning LoRA de Qwen 2.5, Modelfile de Ollama y skill ClimaSafe
- la intensidad sale del MET del deporte (Compendium 2024) en vez de elegirla el usuario
- indexa documentacion/ y expone busqueda semantica y LLM local por MCP
- parte final descriptivo con riesgo %, temperatura, UV y recomendacion contextual
- cierra GIT-001
- formulario Telegram con 17 estados, teclados inline, toggles multiselect y perfiles SQLite
- tools de predicción, factores y perfiles con SQLite
- pipeline de chunks, embeddings y búsqueda semántica
- orquestador de ciclo con puerta, backlog y subagentes
- Web UI grupo/volumen/curvas edad + estimacion volumetrica + fixes
- MCP server con 3 tools (predict, listar, cargar perfil) + integracion opencode
- riesgo colectivo/etiqueta, mapa de zona adaptativo y riesgo horario acumulado
- perfiles por alias, conformal prediction, frontend, docs
- personalizacion, overrides, explicabilidad, thresholds
- RAG, active learning, contrafactuales y mejora en API/chat
- calibra thresholds ML (t1=0.25/t2=0.40), elimina umbrales provincia obsoletos, personalización alineada con ML
- factores dinámicos desde JSON, personalización afecta clase final, prompt scout desambiguado
- conformal prediction para mejorar calibración de resultados
- web endpoints + MCP server + banner factores pendientes
- calidad de papers (alta/media/baja) según fuente y citas
- paper scout con arXiv/OpenAlex y clasificación LLM
- fusión LSTM, ensemble, explicabilidad y recomendaciones
- LSTM province_hybrid integrada y optimizada
- dataset a 27 features y suelo de mortalidad en labels
- experimento de features de frío con retardo largo (#11)
- ablación limpia de features 27 vs 19 con label fijo (#9)
- calibración de umbrales de decisión por clase
- LSTM híbrida con contexto de ola (secuencia 24h + features diarias)
- baseline LightGBM como candidato a sustituir a KNN
- personalización individual del índice de riesgo
- LSTM multi-tarea con comparación clasificación vs regresión
- notebooks, features pipeline y mlflow
- grafo de conocimiento y documentación
- sistema de agentes v0.2.0
- entrena por clase con XGBoost ponderado y selecciona por recall
- divide train/test por fecha en vez de aleatoriamente
- añade estadísticas diarias y persistencia temporal desde ERA5
- integrate KNN classifier with automated hyperparameter tuning and implement robust MLflow local tracking fallback
- actualizar make_dataset y añadir utilidades de procesamiento (labels, weather_indices); fix: ajustes en build_features
- expand orchestration and tooling
- implement monthly ERA5 download with spatial preprocessing and API integration

### Corrección de bugs

- escapa XSS en rutinas y pronostico, codigos HTTP 404/400 reales y borrado de perfil sin huerfanos
- un dato meteorologico NaN ya no tumba la prediccion
- aisla el tracking URI de mlflow para que no se filtre entre tests
- un campo desconocido en el perfil devuelve error explicito en vez de un 500 mudo
- EDADES_COMPARATIVA (25,55,65,75,85) evita duplicados de curva
- timeout 90s en llamada LLM del scout
- permisos de escritura para GH Actions push
- solo buscar en journals peer-review (excluir arXiv y repositorios)
- exception gitignore para factores_riesgo.json + fallback si no existe
- elimina la fuga temporal train-test del label de riesgo
- persiste modelo_desplegado_frio en 0-2-Frio

### Documentación

- cierra DOC-003 — README vuelve a su formato original y lo ampliado vive en documentacion/
- manifiesto de deploy con cuatro perfiles, de solo RAG a Qwen fine-tuneado
- reestructura documentación técnica y de riesgo
- roadmap completo con 19 tareas priorizadas y todo lo hecho marcado
- revisión 68 papers, nuevos factores documentados, próximos pasos actualizados
- reorganizar documentacion (vault + papers + problemas conocidos)
- vault Obsidian, README y CHANGELOG actualizados
- actualiza features de 19 a 27 (persistencia ampliada)
- registra LSTM, personalización y rename de documentacion
- corrige el nombre de la carpeta a «documentacion»
- documenta las decisiones de diseño y la selección por recall
- actualizar documentacion sobre agentes; chore: añadir make_dataset cambios
- documentar diseño del modelo, formulas deterministas y arquitectura de agentes
- update .env.example and add model design documentation

### Tests

- cubre las features de distribución diaria y de rezago temporal

### Mantenimiento

- backlog con cierre de RAG-002 y bump a 0.0.17
- make typecheck en el contrato del implementer y backlog con 6 cierres + ARNES-011 y MCP-APPS-001
- no versionar el estado vivo del arnes ni los backups de datos
- regenera el grafo de conocimiento del proyecto
- pipeline de datos/features, tests y documentacion
- cambios pendientes pre-sesión
- añade chat/static para que la suite pueda ejecutarse

### Otros

- add factores_riesgo.json (force-add, gitignored)
- Merge branch 'mejoras/higiene-docs'
- Merge branch 'mejoras/label-sin-fuga'
- Merge branch 'mejoras/umbrales-decision'
- Merge branch 'mejoras/lstm-hibrida'
- Merge remote-tracking branch 'origin/main'
- notebooks: actualizar comentarios automáticos (interpretaciones de gráficas)
- notebooks: añadir comentarios y actualizar ejecución (preprocesado y entrenamiento)
- Refactor code structure for improved readability and maintainability
- notebooks: añadir comentarios automáticos; fix: corregir preprocess_data UnboundLocalError
- añadidos agentes y sus respectivas funciones
- Estructura inicial creada

## [Unreleased] — 2026-08-05

### Añadido

- tool grafica_riesgo_horario que devuelve la curva de riesgo por hora como PNG (MCP-IMG-001)
- cuestionario de rutina de trabajo con ocupacion e intensidad, parte con confianza conformal y chat con perfil (BOT-013 + BOT-015 + LLM-005)
- parte en lenguaje llano con confianza conformal y factores con coeficiente (BOT-013 + LLM-005)
- el RAG indexa el coeficiente y el DOI de cada factor, no solo el nombre
- el input del dataset de fine-tuning lleva el parte meteorologico completo de la ventana de actividad
- borrar_rutina_mcp comprueba propiedad y rutinas MCP precargables con perfil
- bienvenida, respuestas mejoradas, perfil, rutinas, avisos y parte claro con porcentaje y fecha de nacimiento
- tablas de rutinas semanales y avisos diarios
- doc_agent unifica grafo de graphify, RAG vectorial y boveda Obsidian
- dataset, fine-tuning LoRA de Qwen 2.5, Modelfile de Ollama y skill ClimaSafe
- la intensidad sale del MET del deporte (Compendium 2024) en vez de elegirla el usuario
- indexa documentacion/ y expone busqueda semantica y LLM local por MCP
- parte final descriptivo con riesgo %, temperatura, UV y recomendacion contextual
- cierra GIT-001
- formulario Telegram con 17 estados, teclados inline, toggles multiselect y perfiles SQLite
- tools de predicción, factores y perfiles con SQLite
- pipeline de chunks, embeddings y búsqueda semántica
- orquestador de ciclo con puerta, backlog y subagentes
- Web UI grupo/volumen/curvas edad + estimacion volumetrica + fixes
- MCP server con 3 tools (predict, listar, cargar perfil) + integracion opencode
- riesgo colectivo/etiqueta, mapa de zona adaptativo y riesgo horario acumulado
- perfiles por alias, conformal prediction, frontend, docs
- personalizacion, overrides, explicabilidad, thresholds
- RAG, active learning, contrafactuales y mejora en API/chat
- calibra thresholds ML (t1=0.25/t2=0.40), elimina umbrales provincia obsoletos, personalización alineada con ML
- factores dinámicos desde JSON, personalización afecta clase final, prompt scout desambiguado
- conformal prediction para mejorar calibración de resultados
- web endpoints + MCP server + banner factores pendientes
- calidad de papers (alta/media/baja) según fuente y citas
- paper scout con arXiv/OpenAlex y clasificación LLM
- fusión LSTM, ensemble, explicabilidad y recomendaciones
- LSTM province_hybrid integrada y optimizada
- dataset a 27 features y suelo de mortalidad en labels
- experimento de features de frío con retardo largo (#11)
- ablación limpia de features 27 vs 19 con label fijo (#9)
- calibración de umbrales de decisión por clase
- LSTM híbrida con contexto de ola (secuencia 24h + features diarias)
- baseline LightGBM como candidato a sustituir a KNN
- personalización individual del índice de riesgo
- LSTM multi-tarea con comparación clasificación vs regresión
- notebooks, features pipeline y mlflow
- grafo de conocimiento y documentación
- sistema de agentes v0.2.0
- entrena por clase con XGBoost ponderado y selecciona por recall
- divide train/test por fecha en vez de aleatoriamente
- añade estadísticas diarias y persistencia temporal desde ERA5
- integrate KNN classifier with automated hyperparameter tuning and implement robust MLflow local tracking fallback
- actualizar make_dataset y añadir utilidades de procesamiento (labels, weather_indices); fix: ajustes en build_features
- expand orchestration and tooling
- implement monthly ERA5 download with spatial preprocessing and API integration

### Corrección de bugs

- escapa XSS en rutinas y pronostico, codigos HTTP 404/400 reales y borrado de perfil sin huerfanos
- un dato meteorologico NaN ya no tumba la prediccion
- aisla el tracking URI de mlflow para que no se filtre entre tests
- un campo desconocido en el perfil devuelve error explicito en vez de un 500 mudo
- EDADES_COMPARATIVA (25,55,65,75,85) evita duplicados de curva
- timeout 90s en llamada LLM del scout
- permisos de escritura para GH Actions push
- solo buscar en journals peer-review (excluir arXiv y repositorios)
- exception gitignore para factores_riesgo.json + fallback si no existe
- elimina la fuga temporal train-test del label de riesgo
- persiste modelo_desplegado_frio en 0-2-Frio

### Documentación

- cierra DOC-003 — README vuelve a su formato original y lo ampliado vive en documentacion/
- manifiesto de deploy con cuatro perfiles, de solo RAG a Qwen fine-tuneado
- reestructura documentación técnica y de riesgo
- roadmap completo con 19 tareas priorizadas y todo lo hecho marcado
- revisión 68 papers, nuevos factores documentados, próximos pasos actualizados
- reorganizar documentacion (vault + papers + problemas conocidos)
- vault Obsidian, README y CHANGELOG actualizados
- actualiza features de 19 a 27 (persistencia ampliada)
- registra LSTM, personalización y rename de documentacion
- corrige el nombre de la carpeta a «documentacion»
- documenta las decisiones de diseño y la selección por recall
- actualizar documentacion sobre agentes; chore: añadir make_dataset cambios
- documentar diseño del modelo, formulas deterministas y arquitectura de agentes
- update .env.example and add model design documentation

### Tests

- cubre las features de distribución diaria y de rezago temporal

### Mantenimiento

- cierre de BOT-013, MCP-IMG-001 y LLM-005, backlog con BOT-015/016 y DATA-003, bump a 0.0.20
- backlog con cierre de RAG-002 y bump a 0.0.17
- make typecheck en el contrato del implementer y backlog con 6 cierres + ARNES-011 y MCP-APPS-001
- no versionar el estado vivo del arnes ni los backups de datos
- regenera el grafo de conocimiento del proyecto
- pipeline de datos/features, tests y documentacion
- cambios pendientes pre-sesión
- añade chat/static para que la suite pueda ejecutarse

### Otros

- add factores_riesgo.json (force-add, gitignored)
- Merge branch 'mejoras/higiene-docs'
- Merge branch 'mejoras/label-sin-fuga'
- Merge branch 'mejoras/umbrales-decision'
- Merge branch 'mejoras/lstm-hibrida'
- Merge remote-tracking branch 'origin/main'
- notebooks: actualizar comentarios automáticos (interpretaciones de gráficas)
- notebooks: añadir comentarios y actualizar ejecución (preprocesado y entrenamiento)
- Refactor code structure for improved readability and maintainability
- notebooks: añadir comentarios automáticos; fix: corregir preprocess_data UnboundLocalError
- añadidos agentes y sus respectivas funciones
- Estructura inicial creada

## [Unreleased] — 2026-08-05

### Añadido

- modulo chat_flow para el cuestionario conversacional de /chat (CHAT-001)
- tool grafica_riesgo_horario que devuelve la curva de riesgo por hora como PNG (MCP-IMG-001)
- cuestionario de rutina de trabajo con ocupacion e intensidad, parte con confianza conformal y chat con perfil (BOT-013 + BOT-015 + LLM-005)
- parte en lenguaje llano con confianza conformal y factores con coeficiente (BOT-013 + LLM-005)
- el RAG indexa el coeficiente y el DOI de cada factor, no solo el nombre
- el input del dataset de fine-tuning lleva el parte meteorologico completo de la ventana de actividad
- borrar_rutina_mcp comprueba propiedad y rutinas MCP precargables con perfil
- bienvenida, respuestas mejoradas, perfil, rutinas, avisos y parte claro con porcentaje y fecha de nacimiento
- tablas de rutinas semanales y avisos diarios
- doc_agent unifica grafo de graphify, RAG vectorial y boveda Obsidian
- dataset, fine-tuning LoRA de Qwen 2.5, Modelfile de Ollama y skill ClimaSafe
- la intensidad sale del MET del deporte (Compendium 2024) en vez de elegirla el usuario
- indexa documentacion/ y expone busqueda semantica y LLM local por MCP
- parte final descriptivo con riesgo %, temperatura, UV y recomendacion contextual
- cierra GIT-001
- formulario Telegram con 17 estados, teclados inline, toggles multiselect y perfiles SQLite
- tools de predicción, factores y perfiles con SQLite
- pipeline de chunks, embeddings y búsqueda semántica
- orquestador de ciclo con puerta, backlog y subagentes
- Web UI grupo/volumen/curvas edad + estimacion volumetrica + fixes
- MCP server con 3 tools (predict, listar, cargar perfil) + integracion opencode
- riesgo colectivo/etiqueta, mapa de zona adaptativo y riesgo horario acumulado
- perfiles por alias, conformal prediction, frontend, docs
- personalizacion, overrides, explicabilidad, thresholds
- RAG, active learning, contrafactuales y mejora en API/chat
- calibra thresholds ML (t1=0.25/t2=0.40), elimina umbrales provincia obsoletos, personalización alineada con ML
- factores dinámicos desde JSON, personalización afecta clase final, prompt scout desambiguado
- conformal prediction para mejorar calibración de resultados
- web endpoints + MCP server + banner factores pendientes
- calidad de papers (alta/media/baja) según fuente y citas
- paper scout con arXiv/OpenAlex y clasificación LLM
- fusión LSTM, ensemble, explicabilidad y recomendaciones
- LSTM province_hybrid integrada y optimizada
- dataset a 27 features y suelo de mortalidad en labels
- experimento de features de frío con retardo largo (#11)
- ablación limpia de features 27 vs 19 con label fijo (#9)
- calibración de umbrales de decisión por clase
- LSTM híbrida con contexto de ola (secuencia 24h + features diarias)
- baseline LightGBM como candidato a sustituir a KNN
- personalización individual del índice de riesgo
- LSTM multi-tarea con comparación clasificación vs regresión
- notebooks, features pipeline y mlflow
- grafo de conocimiento y documentación
- sistema de agentes v0.2.0
- entrena por clase con XGBoost ponderado y selecciona por recall
- divide train/test por fecha en vez de aleatoriamente
- añade estadísticas diarias y persistencia temporal desde ERA5
- integrate KNN classifier with automated hyperparameter tuning and implement robust MLflow local tracking fallback
- actualizar make_dataset y añadir utilidades de procesamiento (labels, weather_indices); fix: ajustes en build_features
- expand orchestration and tooling
- implement monthly ERA5 download with spatial preprocessing and API integration

### Corrección de bugs

- escapa XSS en rutinas y pronostico, codigos HTTP 404/400 reales y borrado de perfil sin huerfanos
- un dato meteorologico NaN ya no tumba la prediccion
- aisla el tracking URI de mlflow para que no se filtre entre tests
- un campo desconocido en el perfil devuelve error explicito en vez de un 500 mudo
- EDADES_COMPARATIVA (25,55,65,75,85) evita duplicados de curva
- timeout 90s en llamada LLM del scout
- permisos de escritura para GH Actions push
- solo buscar en journals peer-review (excluir arXiv y repositorios)
- exception gitignore para factores_riesgo.json + fallback si no existe
- elimina la fuga temporal train-test del label de riesgo
- persiste modelo_desplegado_frio en 0-2-Frio

### Documentación

- cierra DOC-003 — README vuelve a su formato original y lo ampliado vive en documentacion/
- manifiesto de deploy con cuatro perfiles, de solo RAG a Qwen fine-tuneado
- reestructura documentación técnica y de riesgo
- roadmap completo con 19 tareas priorizadas y todo lo hecho marcado
- revisión 68 papers, nuevos factores documentados, próximos pasos actualizados
- reorganizar documentacion (vault + papers + problemas conocidos)
- vault Obsidian, README y CHANGELOG actualizados
- actualiza features de 19 a 27 (persistencia ampliada)
- registra LSTM, personalización y rename de documentacion
- corrige el nombre de la carpeta a «documentacion»
- documenta las decisiones de diseño y la selección por recall
- actualizar documentacion sobre agentes; chore: añadir make_dataset cambios
- documentar diseño del modelo, formulas deterministas y arquitectura de agentes
- update .env.example and add model design documentation

### Tests

- cubre las features de distribución diaria y de rezago temporal

### Mantenimiento

- cierre de BOT-013, MCP-IMG-001 y LLM-005, backlog con BOT-015/016 y DATA-003, bump a 0.0.20
- backlog con cierre de RAG-002 y bump a 0.0.17
- make typecheck en el contrato del implementer y backlog con 6 cierres + ARNES-011 y MCP-APPS-001
- no versionar el estado vivo del arnes ni los backups de datos
- regenera el grafo de conocimiento del proyecto
- pipeline de datos/features, tests y documentacion
- cambios pendientes pre-sesión
- añade chat/static para que la suite pueda ejecutarse

### Otros

- add factores_riesgo.json (force-add, gitignored)
- Merge branch 'mejoras/higiene-docs'
- Merge branch 'mejoras/label-sin-fuga'
- Merge branch 'mejoras/umbrales-decision'
- Merge branch 'mejoras/lstm-hibrida'
- Merge remote-tracking branch 'origin/main'
- notebooks: actualizar comentarios automáticos (interpretaciones de gráficas)
- notebooks: añadir comentarios y actualizar ejecución (preprocesado y entrenamiento)
- Refactor code structure for improved readability and maintainability
- notebooks: añadir comentarios automáticos; fix: corregir preprocess_data UnboundLocalError
- añadidos agentes y sus respectivas funciones
- Estructura inicial creada

## [Unreleased] — 2026-08-06

### Añadido

- benchmark de modelos LLM con val.jsonl (LLM-003)
- modulo chat_flow para el cuestionario conversacional de /chat (CHAT-001)
- tool grafica_riesgo_horario que devuelve la curva de riesgo por hora como PNG (MCP-IMG-001)
- cuestionario de rutina de trabajo con ocupacion e intensidad, parte con confianza conformal y chat con perfil (BOT-013 + BOT-015 + LLM-005)
- parte en lenguaje llano con confianza conformal y factores con coeficiente (BOT-013 + LLM-005)
- el RAG indexa el coeficiente y el DOI de cada factor, no solo el nombre
- el input del dataset de fine-tuning lleva el parte meteorologico completo de la ventana de actividad
- borrar_rutina_mcp comprueba propiedad y rutinas MCP precargables con perfil
- bienvenida, respuestas mejoradas, perfil, rutinas, avisos y parte claro con porcentaje y fecha de nacimiento
- tablas de rutinas semanales y avisos diarios
- doc_agent unifica grafo de graphify, RAG vectorial y boveda Obsidian
- dataset, fine-tuning LoRA de Qwen 2.5, Modelfile de Ollama y skill ClimaSafe
- la intensidad sale del MET del deporte (Compendium 2024) en vez de elegirla el usuario
- indexa documentacion/ y expone busqueda semantica y LLM local por MCP
- parte final descriptivo con riesgo %, temperatura, UV y recomendacion contextual
- cierra GIT-001
- formulario Telegram con 17 estados, teclados inline, toggles multiselect y perfiles SQLite
- tools de predicción, factores y perfiles con SQLite
- pipeline de chunks, embeddings y búsqueda semántica
- orquestador de ciclo con puerta, backlog y subagentes
- Web UI grupo/volumen/curvas edad + estimacion volumetrica + fixes
- MCP server con 3 tools (predict, listar, cargar perfil) + integracion opencode
- riesgo colectivo/etiqueta, mapa de zona adaptativo y riesgo horario acumulado
- perfiles por alias, conformal prediction, frontend, docs
- personalizacion, overrides, explicabilidad, thresholds
- RAG, active learning, contrafactuales y mejora en API/chat
- calibra thresholds ML (t1=0.25/t2=0.40), elimina umbrales provincia obsoletos, personalización alineada con ML
- factores dinámicos desde JSON, personalización afecta clase final, prompt scout desambiguado
- conformal prediction para mejorar calibración de resultados
- web endpoints + MCP server + banner factores pendientes
- calidad de papers (alta/media/baja) según fuente y citas
- paper scout con arXiv/OpenAlex y clasificación LLM
- fusión LSTM, ensemble, explicabilidad y recomendaciones
- LSTM province_hybrid integrada y optimizada
- dataset a 27 features y suelo de mortalidad en labels
- experimento de features de frío con retardo largo (#11)
- ablación limpia de features 27 vs 19 con label fijo (#9)
- calibración de umbrales de decisión por clase
- LSTM híbrida con contexto de ola (secuencia 24h + features diarias)
- baseline LightGBM como candidato a sustituir a KNN
- personalización individual del índice de riesgo
- LSTM multi-tarea con comparación clasificación vs regresión
- notebooks, features pipeline y mlflow
- grafo de conocimiento y documentación
- sistema de agentes v0.2.0
- entrena por clase con XGBoost ponderado y selecciona por recall
- divide train/test por fecha en vez de aleatoriamente
- añade estadísticas diarias y persistencia temporal desde ERA5
- integrate KNN classifier with automated hyperparameter tuning and implement robust MLflow local tracking fallback
- actualizar make_dataset y añadir utilidades de procesamiento (labels, weather_indices); fix: ajustes en build_features
- expand orchestration and tooling
- implement monthly ERA5 download with spatial preprocessing and API integration

### Corrección de bugs

- escapa XSS en rutinas y pronostico, codigos HTTP 404/400 reales y borrado de perfil sin huerfanos
- un dato meteorologico NaN ya no tumba la prediccion
- aisla el tracking URI de mlflow para que no se filtre entre tests
- un campo desconocido en el perfil devuelve error explicito en vez de un 500 mudo
- EDADES_COMPARATIVA (25,55,65,75,85) evita duplicados de curva
- timeout 90s en llamada LLM del scout
- permisos de escritura para GH Actions push
- solo buscar en journals peer-review (excluir arXiv y repositorios)
- exception gitignore para factores_riesgo.json + fallback si no existe
- elimina la fuga temporal train-test del label de riesgo
- persiste modelo_desplegado_frio en 0-2-Frio

### Documentación

- cierra DOC-003 — README vuelve a su formato original y lo ampliado vive en documentacion/
- manifiesto de deploy con cuatro perfiles, de solo RAG a Qwen fine-tuneado
- reestructura documentación técnica y de riesgo
- roadmap completo con 19 tareas priorizadas y todo lo hecho marcado
- revisión 68 papers, nuevos factores documentados, próximos pasos actualizados
- reorganizar documentacion (vault + papers + problemas conocidos)
- vault Obsidian, README y CHANGELOG actualizados
- actualiza features de 19 a 27 (persistencia ampliada)
- registra LSTM, personalización y rename de documentacion
- corrige el nombre de la carpeta a «documentacion»
- documenta las decisiones de diseño y la selección por recall
- actualizar documentacion sobre agentes; chore: añadir make_dataset cambios
- documentar diseño del modelo, formulas deterministas y arquitectura de agentes
- update .env.example and add model design documentation

### Tests

- cubre las features de distribución diaria y de rezago temporal

### Mantenimiento

- cierre de BOT-013, MCP-IMG-001 y LLM-005, backlog con BOT-015/016 y DATA-003, bump a 0.0.20
- backlog con cierre de RAG-002 y bump a 0.0.17
- make typecheck en el contrato del implementer y backlog con 6 cierres + ARNES-011 y MCP-APPS-001
- no versionar el estado vivo del arnes ni los backups de datos
- regenera el grafo de conocimiento del proyecto
- pipeline de datos/features, tests y documentacion
- cambios pendientes pre-sesión
- añade chat/static para que la suite pueda ejecutarse

### Otros

- add factores_riesgo.json (force-add, gitignored)
- Merge branch 'mejoras/higiene-docs'
- Merge branch 'mejoras/label-sin-fuga'
- Merge branch 'mejoras/umbrales-decision'
- Merge branch 'mejoras/lstm-hibrida'
- Merge remote-tracking branch 'origin/main'
- notebooks: actualizar comentarios automáticos (interpretaciones de gráficas)
- notebooks: añadir comentarios y actualizar ejecución (preprocesado y entrenamiento)
- Refactor code structure for improved readability and maintainability
- notebooks: añadir comentarios automáticos; fix: corregir preprocess_data UnboundLocalError
- añadidos agentes y sus respectivas funciones
- Estructura inicial creada

## [Unreleased] — 2026-08-06

### Añadido

- benchmark de modelos LLM con val.jsonl (LLM-003)
- modulo chat_flow para el cuestionario conversacional de /chat (CHAT-001)
- tool grafica_riesgo_horario que devuelve la curva de riesgo por hora como PNG (MCP-IMG-001)
- cuestionario de rutina de trabajo con ocupacion e intensidad, parte con confianza conformal y chat con perfil (BOT-013 + BOT-015 + LLM-005)
- parte en lenguaje llano con confianza conformal y factores con coeficiente (BOT-013 + LLM-005)
- el RAG indexa el coeficiente y el DOI de cada factor, no solo el nombre
- el input del dataset de fine-tuning lleva el parte meteorologico completo de la ventana de actividad
- borrar_rutina_mcp comprueba propiedad y rutinas MCP precargables con perfil
- bienvenida, respuestas mejoradas, perfil, rutinas, avisos y parte claro con porcentaje y fecha de nacimiento
- tablas de rutinas semanales y avisos diarios
- doc_agent unifica grafo de graphify, RAG vectorial y boveda Obsidian
- dataset, fine-tuning LoRA de Qwen 2.5, Modelfile de Ollama y skill ClimaSafe
- la intensidad sale del MET del deporte (Compendium 2024) en vez de elegirla el usuario
- indexa documentacion/ y expone busqueda semantica y LLM local por MCP
- parte final descriptivo con riesgo %, temperatura, UV y recomendacion contextual
- cierra GIT-001
- formulario Telegram con 17 estados, teclados inline, toggles multiselect y perfiles SQLite
- tools de predicción, factores y perfiles con SQLite
- pipeline de chunks, embeddings y búsqueda semántica
- orquestador de ciclo con puerta, backlog y subagentes
- Web UI grupo/volumen/curvas edad + estimacion volumetrica + fixes
- MCP server con 3 tools (predict, listar, cargar perfil) + integracion opencode
- riesgo colectivo/etiqueta, mapa de zona adaptativo y riesgo horario acumulado
- perfiles por alias, conformal prediction, frontend, docs
- personalizacion, overrides, explicabilidad, thresholds
- RAG, active learning, contrafactuales y mejora en API/chat
- calibra thresholds ML (t1=0.25/t2=0.40), elimina umbrales provincia obsoletos, personalización alineada con ML
- factores dinámicos desde JSON, personalización afecta clase final, prompt scout desambiguado
- conformal prediction para mejorar calibración de resultados
- web endpoints + MCP server + banner factores pendientes
- calidad de papers (alta/media/baja) según fuente y citas
- paper scout con arXiv/OpenAlex y clasificación LLM
- fusión LSTM, ensemble, explicabilidad y recomendaciones
- LSTM province_hybrid integrada y optimizada
- dataset a 27 features y suelo de mortalidad en labels
- experimento de features de frío con retardo largo (#11)
- ablación limpia de features 27 vs 19 con label fijo (#9)
- calibración de umbrales de decisión por clase
- LSTM híbrida con contexto de ola (secuencia 24h + features diarias)
- baseline LightGBM como candidato a sustituir a KNN
- personalización individual del índice de riesgo
- LSTM multi-tarea con comparación clasificación vs regresión
- notebooks, features pipeline y mlflow
- grafo de conocimiento y documentación
- sistema de agentes v0.2.0
- entrena por clase con XGBoost ponderado y selecciona por recall
- divide train/test por fecha en vez de aleatoriamente
- añade estadísticas diarias y persistencia temporal desde ERA5
- integrate KNN classifier with automated hyperparameter tuning and implement robust MLflow local tracking fallback
- actualizar make_dataset y añadir utilidades de procesamiento (labels, weather_indices); fix: ajustes en build_features
- expand orchestration and tooling
- implement monthly ERA5 download with spatial preprocessing and API integration

### Corrección de bugs

- escapa XSS en rutinas y pronostico, codigos HTTP 404/400 reales y borrado de perfil sin huerfanos
- un dato meteorologico NaN ya no tumba la prediccion
- aisla el tracking URI de mlflow para que no se filtre entre tests
- un campo desconocido en el perfil devuelve error explicito en vez de un 500 mudo
- EDADES_COMPARATIVA (25,55,65,75,85) evita duplicados de curva
- timeout 90s en llamada LLM del scout
- permisos de escritura para GH Actions push
- solo buscar en journals peer-review (excluir arXiv y repositorios)
- exception gitignore para factores_riesgo.json + fallback si no existe
- elimina la fuga temporal train-test del label de riesgo
- persiste modelo_desplegado_frio en 0-2-Frio

### Documentación

- cierra DOC-003 — README vuelve a su formato original y lo ampliado vive en documentacion/
- manifiesto de deploy con cuatro perfiles, de solo RAG a Qwen fine-tuneado
- reestructura documentación técnica y de riesgo
- roadmap completo con 19 tareas priorizadas y todo lo hecho marcado
- revisión 68 papers, nuevos factores documentados, próximos pasos actualizados
- reorganizar documentacion (vault + papers + problemas conocidos)
- vault Obsidian, README y CHANGELOG actualizados
- actualiza features de 19 a 27 (persistencia ampliada)
- registra LSTM, personalización y rename de documentacion
- corrige el nombre de la carpeta a «documentacion»
- documenta las decisiones de diseño y la selección por recall
- actualizar documentacion sobre agentes; chore: añadir make_dataset cambios
- documentar diseño del modelo, formulas deterministas y arquitectura de agentes
- update .env.example and add model design documentation

### Tests

- cubre las features de distribución diaria y de rezago temporal

### Mantenimiento

- ignora y limpia la exportacion del vault, solo notas curadas
- cierre de BOT-013, MCP-IMG-001 y LLM-005, backlog con BOT-015/016 y DATA-003, bump a 0.0.20
- backlog con cierre de RAG-002 y bump a 0.0.17
- make typecheck en el contrato del implementer y backlog con 6 cierres + ARNES-011 y MCP-APPS-001
- no versionar el estado vivo del arnes ni los backups de datos
- regenera el grafo de conocimiento del proyecto
- pipeline de datos/features, tests y documentacion
- cambios pendientes pre-sesión
- añade chat/static para que la suite pueda ejecutarse

### Otros

- add factores_riesgo.json (force-add, gitignored)
- Merge branch 'mejoras/higiene-docs'
- Merge branch 'mejoras/label-sin-fuga'
- Merge branch 'mejoras/umbrales-decision'
- Merge branch 'mejoras/lstm-hibrida'
- Merge remote-tracking branch 'origin/main'
- notebooks: actualizar comentarios automáticos (interpretaciones de gráficas)
- notebooks: añadir comentarios y actualizar ejecución (preprocesado y entrenamiento)
- Refactor code structure for improved readability and maintainability
- notebooks: añadir comentarios automáticos; fix: corregir preprocess_data UnboundLocalError
- añadidos agentes y sus respectivas funciones
- Estructura inicial creada

## [Unreleased] — 2026-08-06

### Añadido

- fine-tuning de Qwen en Google Colab con dataset verificado (LLM-006)
- benchmark de modelos LLM con val.jsonl (LLM-003)
- modulo chat_flow para el cuestionario conversacional de /chat (CHAT-001)
- tool grafica_riesgo_horario que devuelve la curva de riesgo por hora como PNG (MCP-IMG-001)
- cuestionario de rutina de trabajo con ocupacion e intensidad, parte con confianza conformal y chat con perfil (BOT-013 + BOT-015 + LLM-005)
- parte en lenguaje llano con confianza conformal y factores con coeficiente (BOT-013 + LLM-005)
- el RAG indexa el coeficiente y el DOI de cada factor, no solo el nombre
- el input del dataset de fine-tuning lleva el parte meteorologico completo de la ventana de actividad
- borrar_rutina_mcp comprueba propiedad y rutinas MCP precargables con perfil
- bienvenida, respuestas mejoradas, perfil, rutinas, avisos y parte claro con porcentaje y fecha de nacimiento
- tablas de rutinas semanales y avisos diarios
- doc_agent unifica grafo de graphify, RAG vectorial y boveda Obsidian
- dataset, fine-tuning LoRA de Qwen 2.5, Modelfile de Ollama y skill ClimaSafe
- la intensidad sale del MET del deporte (Compendium 2024) en vez de elegirla el usuario
- indexa documentacion/ y expone busqueda semantica y LLM local por MCP
- parte final descriptivo con riesgo %, temperatura, UV y recomendacion contextual
- cierra GIT-001
- formulario Telegram con 17 estados, teclados inline, toggles multiselect y perfiles SQLite
- tools de predicción, factores y perfiles con SQLite
- pipeline de chunks, embeddings y búsqueda semántica
- orquestador de ciclo con puerta, backlog y subagentes
- Web UI grupo/volumen/curvas edad + estimacion volumetrica + fixes
- MCP server con 3 tools (predict, listar, cargar perfil) + integracion opencode
- riesgo colectivo/etiqueta, mapa de zona adaptativo y riesgo horario acumulado
- perfiles por alias, conformal prediction, frontend, docs
- personalizacion, overrides, explicabilidad, thresholds
- RAG, active learning, contrafactuales y mejora en API/chat
- calibra thresholds ML (t1=0.25/t2=0.40), elimina umbrales provincia obsoletos, personalización alineada con ML
- factores dinámicos desde JSON, personalización afecta clase final, prompt scout desambiguado
- conformal prediction para mejorar calibración de resultados
- web endpoints + MCP server + banner factores pendientes
- calidad de papers (alta/media/baja) según fuente y citas
- paper scout con arXiv/OpenAlex y clasificación LLM
- fusión LSTM, ensemble, explicabilidad y recomendaciones
- LSTM province_hybrid integrada y optimizada
- dataset a 27 features y suelo de mortalidad en labels
- experimento de features de frío con retardo largo (#11)
- ablación limpia de features 27 vs 19 con label fijo (#9)
- calibración de umbrales de decisión por clase
- LSTM híbrida con contexto de ola (secuencia 24h + features diarias)
- baseline LightGBM como candidato a sustituir a KNN
- personalización individual del índice de riesgo
- LSTM multi-tarea con comparación clasificación vs regresión
- notebooks, features pipeline y mlflow
- grafo de conocimiento y documentación
- sistema de agentes v0.2.0
- entrena por clase con XGBoost ponderado y selecciona por recall
- divide train/test por fecha en vez de aleatoriamente
- añade estadísticas diarias y persistencia temporal desde ERA5
- integrate KNN classifier with automated hyperparameter tuning and implement robust MLflow local tracking fallback
- actualizar make_dataset y añadir utilidades de procesamiento (labels, weather_indices); fix: ajustes en build_features
- expand orchestration and tooling
- implement monthly ERA5 download with spatial preprocessing and API integration

### Corrección de bugs

- escapa XSS en rutinas y pronostico, codigos HTTP 404/400 reales y borrado de perfil sin huerfanos
- un dato meteorologico NaN ya no tumba la prediccion
- aisla el tracking URI de mlflow para que no se filtre entre tests
- un campo desconocido en el perfil devuelve error explicito en vez de un 500 mudo
- EDADES_COMPARATIVA (25,55,65,75,85) evita duplicados de curva
- timeout 90s en llamada LLM del scout
- permisos de escritura para GH Actions push
- solo buscar en journals peer-review (excluir arXiv y repositorios)
- exception gitignore para factores_riesgo.json + fallback si no existe
- elimina la fuga temporal train-test del label de riesgo
- persiste modelo_desplegado_frio en 0-2-Frio

### Documentación

- cierra DOC-003 — README vuelve a su formato original y lo ampliado vive en documentacion/
- manifiesto de deploy con cuatro perfiles, de solo RAG a Qwen fine-tuneado
- reestructura documentación técnica y de riesgo
- roadmap completo con 19 tareas priorizadas y todo lo hecho marcado
- revisión 68 papers, nuevos factores documentados, próximos pasos actualizados
- reorganizar documentacion (vault + papers + problemas conocidos)
- vault Obsidian, README y CHANGELOG actualizados
- actualiza features de 19 a 27 (persistencia ampliada)
- registra LSTM, personalización y rename de documentacion
- corrige el nombre de la carpeta a «documentacion»
- documenta las decisiones de diseño y la selección por recall
- actualizar documentacion sobre agentes; chore: añadir make_dataset cambios
- documentar diseño del modelo, formulas deterministas y arquitectura de agentes
- update .env.example and add model design documentation

### Tests

- cubre las features de distribución diaria y de rezago temporal

### Mantenimiento

- ignora y limpia la exportacion del vault, solo notas curadas
- cierre de BOT-013, MCP-IMG-001 y LLM-005, backlog con BOT-015/016 y DATA-003, bump a 0.0.20
- backlog con cierre de RAG-002 y bump a 0.0.17
- make typecheck en el contrato del implementer y backlog con 6 cierres + ARNES-011 y MCP-APPS-001
- no versionar el estado vivo del arnes ni los backups de datos
- regenera el grafo de conocimiento del proyecto
- pipeline de datos/features, tests y documentacion
- cambios pendientes pre-sesión
- añade chat/static para que la suite pueda ejecutarse

### Otros

- add factores_riesgo.json (force-add, gitignored)
- Merge branch 'mejoras/higiene-docs'
- Merge branch 'mejoras/label-sin-fuga'
- Merge branch 'mejoras/umbrales-decision'
- Merge branch 'mejoras/lstm-hibrida'
- Merge remote-tracking branch 'origin/main'
- notebooks: actualizar comentarios automáticos (interpretaciones de gráficas)
- notebooks: añadir comentarios y actualizar ejecución (preprocesado y entrenamiento)
- Refactor code structure for improved readability and maintainability
- notebooks: añadir comentarios automáticos; fix: corregir preprocess_data UnboundLocalError
- añadidos agentes y sus respectivas funciones
- Estructura inicial creada

## [Unreleased] — 2026-08-06

### Añadido

- reindexa por hash de contenido — editar el cuerpo sin tocar la clave ya refresca el fragmento (RAG-005)
- fine-tuning de Qwen en Google Colab con dataset verificado (LLM-006)
- benchmark de modelos LLM con val.jsonl (LLM-003)
- modulo chat_flow para el cuestionario conversacional de /chat (CHAT-001)
- tool grafica_riesgo_horario que devuelve la curva de riesgo por hora como PNG (MCP-IMG-001)
- cuestionario de rutina de trabajo con ocupacion e intensidad, parte con confianza conformal y chat con perfil (BOT-013 + BOT-015 + LLM-005)
- parte en lenguaje llano con confianza conformal y factores con coeficiente (BOT-013 + LLM-005)
- el RAG indexa el coeficiente y el DOI de cada factor, no solo el nombre
- el input del dataset de fine-tuning lleva el parte meteorologico completo de la ventana de actividad
- borrar_rutina_mcp comprueba propiedad y rutinas MCP precargables con perfil
- bienvenida, respuestas mejoradas, perfil, rutinas, avisos y parte claro con porcentaje y fecha de nacimiento
- tablas de rutinas semanales y avisos diarios
- doc_agent unifica grafo de graphify, RAG vectorial y boveda Obsidian
- dataset, fine-tuning LoRA de Qwen 2.5, Modelfile de Ollama y skill ClimaSafe
- la intensidad sale del MET del deporte (Compendium 2024) en vez de elegirla el usuario
- indexa documentacion/ y expone busqueda semantica y LLM local por MCP
- parte final descriptivo con riesgo %, temperatura, UV y recomendacion contextual
- cierra GIT-001
- formulario Telegram con 17 estados, teclados inline, toggles multiselect y perfiles SQLite
- tools de predicción, factores y perfiles con SQLite
- pipeline de chunks, embeddings y búsqueda semántica
- orquestador de ciclo con puerta, backlog y subagentes
- Web UI grupo/volumen/curvas edad + estimacion volumetrica + fixes
- MCP server con 3 tools (predict, listar, cargar perfil) + integracion opencode
- riesgo colectivo/etiqueta, mapa de zona adaptativo y riesgo horario acumulado
- perfiles por alias, conformal prediction, frontend, docs
- personalizacion, overrides, explicabilidad, thresholds
- RAG, active learning, contrafactuales y mejora en API/chat
- calibra thresholds ML (t1=0.25/t2=0.40), elimina umbrales provincia obsoletos, personalización alineada con ML
- factores dinámicos desde JSON, personalización afecta clase final, prompt scout desambiguado
- conformal prediction para mejorar calibración de resultados
- web endpoints + MCP server + banner factores pendientes
- calidad de papers (alta/media/baja) según fuente y citas
- paper scout con arXiv/OpenAlex y clasificación LLM
- fusión LSTM, ensemble, explicabilidad y recomendaciones
- LSTM province_hybrid integrada y optimizada
- dataset a 27 features y suelo de mortalidad en labels
- experimento de features de frío con retardo largo (#11)
- ablación limpia de features 27 vs 19 con label fijo (#9)
- calibración de umbrales de decisión por clase
- LSTM híbrida con contexto de ola (secuencia 24h + features diarias)
- baseline LightGBM como candidato a sustituir a KNN
- personalización individual del índice de riesgo
- LSTM multi-tarea con comparación clasificación vs regresión
- notebooks, features pipeline y mlflow
- grafo de conocimiento y documentación
- sistema de agentes v0.2.0
- entrena por clase con XGBoost ponderado y selecciona por recall
- divide train/test por fecha en vez de aleatoriamente
- añade estadísticas diarias y persistencia temporal desde ERA5
- integrate KNN classifier with automated hyperparameter tuning and implement robust MLflow local tracking fallback
- actualizar make_dataset y añadir utilidades de procesamiento (labels, weather_indices); fix: ajustes en build_features
- expand orchestration and tooling
- implement monthly ERA5 download with spatial preprocessing and API integration

### Corrección de bugs

- escapa XSS en rutinas y pronostico, codigos HTTP 404/400 reales y borrado de perfil sin huerfanos
- un dato meteorologico NaN ya no tumba la prediccion
- aisla el tracking URI de mlflow para que no se filtre entre tests
- un campo desconocido en el perfil devuelve error explicito en vez de un 500 mudo
- EDADES_COMPARATIVA (25,55,65,75,85) evita duplicados de curva
- timeout 90s en llamada LLM del scout
- permisos de escritura para GH Actions push
- solo buscar en journals peer-review (excluir arXiv y repositorios)
- exception gitignore para factores_riesgo.json + fallback si no existe
- elimina la fuga temporal train-test del label de riesgo
- persiste modelo_desplegado_frio en 0-2-Frio

### Documentación

- cierra DOC-003 — README vuelve a su formato original y lo ampliado vive en documentacion/
- manifiesto de deploy con cuatro perfiles, de solo RAG a Qwen fine-tuneado
- reestructura documentación técnica y de riesgo
- roadmap completo con 19 tareas priorizadas y todo lo hecho marcado
- revisión 68 papers, nuevos factores documentados, próximos pasos actualizados
- reorganizar documentacion (vault + papers + problemas conocidos)
- vault Obsidian, README y CHANGELOG actualizados
- actualiza features de 19 a 27 (persistencia ampliada)
- registra LSTM, personalización y rename de documentacion
- corrige el nombre de la carpeta a «documentacion»
- documenta las decisiones de diseño y la selección por recall
- actualizar documentacion sobre agentes; chore: añadir make_dataset cambios
- documentar diseño del modelo, formulas deterministas y arquitectura de agentes
- update .env.example and add model design documentation

### Tests

- cubre las features de distribución diaria y de rezago temporal

### Mantenimiento

- ignora y limpia la exportacion del vault, solo notas curadas
- cierre de BOT-013, MCP-IMG-001 y LLM-005, backlog con BOT-015/016 y DATA-003, bump a 0.0.20
- backlog con cierre de RAG-002 y bump a 0.0.17
- make typecheck en el contrato del implementer y backlog con 6 cierres + ARNES-011 y MCP-APPS-001
- no versionar el estado vivo del arnes ni los backups de datos
- regenera el grafo de conocimiento del proyecto
- pipeline de datos/features, tests y documentacion
- cambios pendientes pre-sesión
- añade chat/static para que la suite pueda ejecutarse

### Otros

- add factores_riesgo.json (force-add, gitignored)
- Merge branch 'mejoras/higiene-docs'
- Merge branch 'mejoras/label-sin-fuga'
- Merge branch 'mejoras/umbrales-decision'
- Merge branch 'mejoras/lstm-hibrida'
- Merge remote-tracking branch 'origin/main'
- notebooks: actualizar comentarios automáticos (interpretaciones de gráficas)
- notebooks: añadir comentarios y actualizar ejecución (preprocesado y entrenamiento)
- Refactor code structure for improved readability and maintainability
- notebooks: añadir comentarios automáticos; fix: corregir preprocess_data UnboundLocalError
- añadidos agentes y sus respectivas funciones

## [Unreleased] — 2026-08-06

### Añadido

- reindexa por hash de contenido — editar el cuerpo sin tocar la clave ya refresca el fragmento (RAG-005)
- fine-tuning de Qwen en Google Colab con dataset verificado (LLM-006)
- benchmark de modelos LLM con val.jsonl (LLM-003)
- modulo chat_flow para el cuestionario conversacional de /chat (CHAT-001)
- tool grafica_riesgo_horario que devuelve la curva de riesgo por hora como PNG (MCP-IMG-001)
- cuestionario de rutina de trabajo con ocupacion e intensidad, parte con confianza conformal y chat con perfil (BOT-013 + BOT-015 + LLM-005)
- parte en lenguaje llano con confianza conformal y factores con coeficiente (BOT-013 + LLM-005)
- el RAG indexa el coeficiente y el DOI de cada factor, no solo el nombre
- el input del dataset de fine-tuning lleva el parte meteorologico completo de la ventana de actividad
- borrar_rutina_mcp comprueba propiedad y rutinas MCP precargables con perfil
- bienvenida, respuestas mejoradas, perfil, rutinas, avisos y parte claro con porcentaje y fecha de nacimiento
- tablas de rutinas semanales y avisos diarios
- doc_agent unifica grafo de graphify, RAG vectorial y boveda Obsidian
- dataset, fine-tuning LoRA de Qwen 2.5, Modelfile de Ollama y skill ClimaSafe
- la intensidad sale del MET del deporte (Compendium 2024) en vez de elegirla el usuario
- indexa documentacion/ y expone busqueda semantica y LLM local por MCP
- parte final descriptivo con riesgo %, temperatura, UV y recomendacion contextual
- cierra GIT-001
- formulario Telegram con 17 estados, teclados inline, toggles multiselect y perfiles SQLite
- tools de predicción, factores y perfiles con SQLite
- pipeline de chunks, embeddings y búsqueda semántica
- orquestador de ciclo con puerta, backlog y subagentes
- Web UI grupo/volumen/curvas edad + estimacion volumetrica + fixes
- MCP server con 3 tools (predict, listar, cargar perfil) + integracion opencode
- riesgo colectivo/etiqueta, mapa de zona adaptativo y riesgo horario acumulado
- perfiles por alias, conformal prediction, frontend, docs
- personalizacion, overrides, explicabilidad, thresholds
- RAG, active learning, contrafactuales y mejora en API/chat
- calibra thresholds ML (t1=0.25/t2=0.40), elimina umbrales provincia obsoletos, personalización alineada con ML
- factores dinámicos desde JSON, personalización afecta clase final, prompt scout desambiguado
- conformal prediction para mejorar calibración de resultados
- web endpoints + MCP server + banner factores pendientes
- calidad de papers (alta/media/baja) según fuente y citas
- paper scout con arXiv/OpenAlex y clasificación LLM
- fusión LSTM, ensemble, explicabilidad y recomendaciones
- LSTM province_hybrid integrada y optimizada
- dataset a 27 features y suelo de mortalidad en labels
- experimento de features de frío con retardo largo (#11)
- ablación limpia de features 27 vs 19 con label fijo (#9)
- calibración de umbrales de decisión por clase
- LSTM híbrida con contexto de ola (secuencia 24h + features diarias)
- baseline LightGBM como candidato a sustituir a KNN
- personalización individual del índice de riesgo
- LSTM multi-tarea con comparación clasificación vs regresión
- notebooks, features pipeline y mlflow
- grafo de conocimiento y documentación
- sistema de agentes v0.2.0
- entrena por clase con XGBoost ponderado y selecciona por recall
- divide train/test por fecha en vez de aleatoriamente
- añade estadísticas diarias y persistencia temporal desde ERA5
- integrate KNN classifier with automated hyperparameter tuning and implement robust MLflow local tracking fallback
- actualizar make_dataset y añadir utilidades de procesamiento (labels, weather_indices); fix: ajustes en build_features
- expand orchestration and tooling
- implement monthly ERA5 download with spatial preprocessing and API integration

### Corrección de bugs

- escapa XSS en rutinas y pronostico, codigos HTTP 404/400 reales y borrado de perfil sin huerfanos
- un dato meteorologico NaN ya no tumba la prediccion
- aisla el tracking URI de mlflow para que no se filtre entre tests
- un campo desconocido en el perfil devuelve error explicito en vez de un 500 mudo
- EDADES_COMPARATIVA (25,55,65,75,85) evita duplicados de curva
- timeout 90s en llamada LLM del scout
- permisos de escritura para GH Actions push
- solo buscar en journals peer-review (excluir arXiv y repositorios)
- exception gitignore para factores_riesgo.json + fallback si no existe
- elimina la fuga temporal train-test del label de riesgo
- persiste modelo_desplegado_frio en 0-2-Frio

### Documentación

- cierra DOC-003 — README vuelve a su formato original y lo ampliado vive en documentacion/
- manifiesto de deploy con cuatro perfiles, de solo RAG a Qwen fine-tuneado
- reestructura documentación técnica y de riesgo
- roadmap completo con 19 tareas priorizadas y todo lo hecho marcado
- revisión 68 papers, nuevos factores documentados, próximos pasos actualizados
- reorganizar documentacion (vault + papers + problemas conocidos)
- vault Obsidian, README y CHANGELOG actualizados
- actualiza features de 19 a 27 (persistencia ampliada)
- registra LSTM, personalización y rename de documentacion
- corrige el nombre de la carpeta a «documentacion»
- documenta las decisiones de diseño y la selección por recall
- actualizar documentacion sobre agentes; chore: añadir make_dataset cambios
- documentar diseño del modelo, formulas deterministas y arquitectura de agentes

### Tests

- cubre las features de distribución diaria y de rezago temporal

### Mantenimiento

- grafo de conocimiento actualizado (graphify-out 2026-08-06)
- ignora y limpia la exportacion del vault, solo notas curadas
- cierre de BOT-013, MCP-IMG-001 y LLM-005, backlog con BOT-015/016 y DATA-003, bump a 0.0.20
- backlog con cierre de RAG-002 y bump a 0.0.17
- make typecheck en el contrato del implementer y backlog con 6 cierres + ARNES-011 y MCP-APPS-001
- no versionar el estado vivo del arnes ni los backups de datos
- regenera el grafo de conocimiento del proyecto
- pipeline de datos/features, tests y documentacion
- cambios pendientes pre-sesión
- añade chat/static para que la suite pueda ejecutarse

### Otros

- add factores_riesgo.json (force-add, gitignored)
- Merge branch 'mejoras/higiene-docs'
- Merge branch 'mejoras/label-sin-fuga'
- Merge branch 'mejoras/umbrales-decision'
- Merge branch 'mejoras/lstm-hibrida'
- Merge remote-tracking branch 'origin/main'
- notebooks: actualizar comentarios automáticos (interpretaciones de gráficas)
- notebooks: añadir comentarios y actualizar ejecución (preprocesado y entrenamiento)
- Refactor code structure for improved readability and maintainability
- notebooks: añadir comentarios automáticos; fix: corregir preprocess_data UnboundLocalError
- añadidos agentes y sus respectivas funciones

## [Unreleased] — 2026-08-06

### Añadido

- reindexa por hash de contenido — editar el cuerpo sin tocar la clave ya refresca el fragmento (RAG-005)
- fine-tuning de Qwen en Google Colab con dataset verificado (LLM-006)
- benchmark de modelos LLM con val.jsonl (LLM-003)
- modulo chat_flow para el cuestionario conversacional de /chat (CHAT-001)
- tool grafica_riesgo_horario que devuelve la curva de riesgo por hora como PNG (MCP-IMG-001)
- cuestionario de rutina de trabajo con ocupacion e intensidad, parte con confianza conformal y chat con perfil (BOT-013 + BOT-015 + LLM-005)
- parte en lenguaje llano con confianza conformal y factores con coeficiente (BOT-013 + LLM-005)
- el RAG indexa el coeficiente y el DOI de cada factor, no solo el nombre
- el input del dataset de fine-tuning lleva el parte meteorologico completo de la ventana de actividad
- borrar_rutina_mcp comprueba propiedad y rutinas MCP precargables con perfil
- bienvenida, respuestas mejoradas, perfil, rutinas, avisos y parte claro con porcentaje y fecha de nacimiento
- tablas de rutinas semanales y avisos diarios
- doc_agent unifica grafo de graphify, RAG vectorial y boveda Obsidian
- dataset, fine-tuning LoRA de Qwen 2.5, Modelfile de Ollama y skill ClimaSafe
- la intensidad sale del MET del deporte (Compendium 2024) en vez de elegirla el usuario
- indexa documentacion/ y expone busqueda semantica y LLM local por MCP
- parte final descriptivo con riesgo %, temperatura, UV y recomendacion contextual
- cierra GIT-001
- formulario Telegram con 17 estados, teclados inline, toggles multiselect y perfiles SQLite
- tools de predicción, factores y perfiles con SQLite
- pipeline de chunks, embeddings y búsqueda semántica
- orquestador de ciclo con puerta, backlog y subagentes
- Web UI grupo/volumen/curvas edad + estimacion volumetrica + fixes
- MCP server con 3 tools (predict, listar, cargar perfil) + integracion opencode
- riesgo colectivo/etiqueta, mapa de zona adaptativo y riesgo horario acumulado
- perfiles por alias, conformal prediction, frontend, docs
- personalizacion, overrides, explicabilidad, thresholds
- RAG, active learning, contrafactuales y mejora en API/chat
- calibra thresholds ML (t1=0.25/t2=0.40), elimina umbrales provincia obsoletos, personalización alineada con ML
- factores dinámicos desde JSON, personalización afecta clase final, prompt scout desambiguado
- conformal prediction para mejorar calibración de resultados
- web endpoints + MCP server + banner factores pendientes
- calidad de papers (alta/media/baja) según fuente y citas
- paper scout con arXiv/OpenAlex y clasificación LLM
- fusión LSTM, ensemble, explicabilidad y recomendaciones
- LSTM province_hybrid integrada y optimizada
- dataset a 27 features y suelo de mortalidad en labels
- experimento de features de frío con retardo largo (#11)
- ablación limpia de features 27 vs 19 con label fijo (#9)
- calibración de umbrales de decisión por clase
- LSTM híbrida con contexto de ola (secuencia 24h + features diarias)
- baseline LightGBM como candidato a sustituir a KNN
- personalización individual del índice de riesgo
- LSTM multi-tarea con comparación clasificación vs regresión
- notebooks, features pipeline y mlflow
- grafo de conocimiento y documentación
- sistema de agentes v0.2.0
- entrena por clase con XGBoost ponderado y selecciona por recall
- divide train/test por fecha en vez de aleatoriamente
- añade estadísticas diarias y persistencia temporal desde ERA5
- integrate KNN classifier with automated hyperparameter tuning and implement robust MLflow local tracking fallback
- actualizar make_dataset y añadir utilidades de procesamiento (labels, weather_indices); fix: ajustes en build_features
- expand orchestration and tooling
- implement monthly ERA5 download with spatial preprocessing and API integration

### Corrección de bugs

- escapa XSS en rutinas y pronostico, codigos HTTP 404/400 reales y borrado de perfil sin huerfanos
- un dato meteorologico NaN ya no tumba la prediccion
- aisla el tracking URI de mlflow para que no se filtre entre tests
- un campo desconocido en el perfil devuelve error explicito en vez de un 500 mudo
- EDADES_COMPARATIVA (25,55,65,75,85) evita duplicados de curva
- timeout 90s en llamada LLM del scout
- permisos de escritura para GH Actions push
- solo buscar en journals peer-review (excluir arXiv y repositorios)
- exception gitignore para factores_riesgo.json + fallback si no existe
- elimina la fuga temporal train-test del label de riesgo
- persiste modelo_desplegado_frio en 0-2-Frio

### Documentación

- cierra DOC-003 — README vuelve a su formato original y lo ampliado vive en documentacion/
- manifiesto de deploy con cuatro perfiles, de solo RAG a Qwen fine-tuneado
- reestructura documentación técnica y de riesgo
- roadmap completo con 19 tareas priorizadas y todo lo hecho marcado
- revisión 68 papers, nuevos factores documentados, próximos pasos actualizados
- reorganizar documentacion (vault + papers + problemas conocidos)
- vault Obsidian, README y CHANGELOG actualizados
- actualiza features de 19 a 27 (persistencia ampliada)
- registra LSTM, personalización y rename de documentacion
- corrige el nombre de la carpeta a «documentacion»
- documenta las decisiones de diseño y la selección por recall
- actualizar documentacion sobre agentes; chore: añadir make_dataset cambios

### Tests

- cubre las features de distribución diaria y de rezago temporal

### Mantenimiento

- ARNES-001 y DATA-003 bloqueadas (prioridad LLM-006 / opencode en paralelo); LLM-006 añadida e in_progress
- grafo de conocimiento actualizado (graphify-out 2026-08-06)
- ignora y limpia la exportacion del vault, solo notas curadas
- cierre de BOT-013, MCP-IMG-001 y LLM-005, backlog con BOT-015/016 y DATA-003, bump a 0.0.20
- backlog con cierre de RAG-002 y bump a 0.0.17
- make typecheck en el contrato del implementer y backlog con 6 cierres + ARNES-011 y MCP-APPS-001
- no versionar el estado vivo del arnes ni los backups de datos
- regenera el grafo de conocimiento del proyecto
- pipeline de datos/features, tests y documentacion
- cambios pendientes pre-sesión
- añade chat/static para que la suite pueda ejecutarse

### Otros

- add factores_riesgo.json (force-add, gitignored)
- Merge branch 'mejoras/higiene-docs'
- Merge branch 'mejoras/label-sin-fuga'
- Merge branch 'mejoras/umbrales-decision'
- Merge branch 'mejoras/lstm-hibrida'
- Merge remote-tracking branch 'origin/main'
- notebooks: actualizar comentarios automáticos (interpretaciones de gráficas)
- notebooks: añadir comentarios y actualizar ejecución (preprocesado y entrenamiento)
- Refactor code structure for improved readability and maintainability
- notebooks: añadir comentarios automáticos; fix: corregir preprocess_data UnboundLocalError
- añadidos agentes y sus respectivas funciones

## [Unreleased] — 2026-08-07

### Añadido

- reindexa por hash de contenido — editar el cuerpo sin tocar la clave ya refresca el fragmento (RAG-005)
- fine-tuning de Qwen en Google Colab con dataset verificado (LLM-006)
- benchmark de modelos LLM con val.jsonl (LLM-003)
- modulo chat_flow para el cuestionario conversacional de /chat (CHAT-001)
- tool grafica_riesgo_horario que devuelve la curva de riesgo por hora como PNG (MCP-IMG-001)
- cuestionario de rutina de trabajo con ocupacion e intensidad, parte con confianza conformal y chat con perfil (BOT-013 + BOT-015 + LLM-005)
- parte en lenguaje llano con confianza conformal y factores con coeficiente (BOT-013 + LLM-005)
- el RAG indexa el coeficiente y el DOI de cada factor, no solo el nombre
- el input del dataset de fine-tuning lleva el parte meteorologico completo de la ventana de actividad
- borrar_rutina_mcp comprueba propiedad y rutinas MCP precargables con perfil
- bienvenida, respuestas mejoradas, perfil, rutinas, avisos y parte claro con porcentaje y fecha de nacimiento
- tablas de rutinas semanales y avisos diarios
- doc_agent unifica grafo de graphify, RAG vectorial y boveda Obsidian
- dataset, fine-tuning LoRA de Qwen 2.5, Modelfile de Ollama y skill ClimaSafe
- la intensidad sale del MET del deporte (Compendium 2024) en vez de elegirla el usuario
- indexa documentacion/ y expone busqueda semantica y LLM local por MCP
- parte final descriptivo con riesgo %, temperatura, UV y recomendacion contextual
- cierra GIT-001
- formulario Telegram con 17 estados, teclados inline, toggles multiselect y perfiles SQLite
- tools de predicción, factores y perfiles con SQLite
- pipeline de chunks, embeddings y búsqueda semántica
- orquestador de ciclo con puerta, backlog y subagentes
- Web UI grupo/volumen/curvas edad + estimacion volumetrica + fixes
- MCP server con 3 tools (predict, listar, cargar perfil) + integracion opencode
- riesgo colectivo/etiqueta, mapa de zona adaptativo y riesgo horario acumulado
- perfiles por alias, conformal prediction, frontend, docs
- personalizacion, overrides, explicabilidad, thresholds
- RAG, active learning, contrafactuales y mejora en API/chat
- calibra thresholds ML (t1=0.25/t2=0.40), elimina umbrales provincia obsoletos, personalización alineada con ML
- factores dinámicos desde JSON, personalización afecta clase final, prompt scout desambiguado
- conformal prediction para mejorar calibración de resultados
- web endpoints + MCP server + banner factores pendientes
- calidad de papers (alta/media/baja) según fuente y citas
- paper scout con arXiv/OpenAlex y clasificación LLM
- fusión LSTM, ensemble, explicabilidad y recomendaciones
- LSTM province_hybrid integrada y optimizada
- dataset a 27 features y suelo de mortalidad en labels
- experimento de features de frío con retardo largo (#11)
- ablación limpia de features 27 vs 19 con label fijo (#9)
- calibración de umbrales de decisión por clase
- LSTM híbrida con contexto de ola (secuencia 24h + features diarias)
- baseline LightGBM como candidato a sustituir a KNN
- personalización individual del índice de riesgo
- LSTM multi-tarea con comparación clasificación vs regresión
- notebooks, features pipeline y mlflow
- grafo de conocimiento y documentación
- sistema de agentes v0.2.0
- entrena por clase con XGBoost ponderado y selecciona por recall
- divide train/test por fecha en vez de aleatoriamente
- añade estadísticas diarias y persistencia temporal desde ERA5
- integrate KNN classifier with automated hyperparameter tuning and implement robust MLflow local tracking fallback
- actualizar make_dataset y añadir utilidades de procesamiento (labels, weather_indices); fix: ajustes en build_features
- expand orchestration and tooling

### Corrección de bugs

- el perfil horario usa solo el dia objetivo (DATA-003); arreglado riesgo colectivo por etiqueta (BUG-002)
- escapa XSS en rutinas y pronostico, codigos HTTP 404/400 reales y borrado de perfil sin huerfanos
- un dato meteorologico NaN ya no tumba la prediccion
- aisla el tracking URI de mlflow para que no se filtre entre tests
- un campo desconocido en el perfil devuelve error explicito en vez de un 500 mudo
- EDADES_COMPARATIVA (25,55,65,75,85) evita duplicados de curva
- timeout 90s en llamada LLM del scout
- permisos de escritura para GH Actions push
- solo buscar en journals peer-review (excluir arXiv y repositorios)
- exception gitignore para factores_riesgo.json + fallback si no existe
- elimina la fuga temporal train-test del label de riesgo
- persiste modelo_desplegado_frio en 0-2-Frio

### Documentación

- cierra DOC-003 — README vuelve a su formato original y lo ampliado vive en documentacion/
- manifiesto de deploy con cuatro perfiles, de solo RAG a Qwen fine-tuneado
- reestructura documentación técnica y de riesgo
- roadmap completo con 19 tareas priorizadas y todo lo hecho marcado
- revisión 68 papers, nuevos factores documentados, próximos pasos actualizados
- reorganizar documentacion (vault + papers + problemas conocidos)
- vault Obsidian, README y CHANGELOG actualizados
- actualiza features de 19 a 27 (persistencia ampliada)
- registra LSTM, personalización y rename de documentacion
- corrige el nombre de la carpeta a «documentacion»
- documenta las decisiones de diseño y la selección por recall
- actualizar documentacion sobre agentes; chore: añadir make_dataset cambios

### Tests

- cubre las features de distribución diaria y de rezago temporal

### Mantenimiento

- ARNES-001 y DATA-003 bloqueadas (prioridad LLM-006 / opencode en paralelo); LLM-006 añadida e in_progress
- grafo de conocimiento actualizado (graphify-out 2026-08-06)
- ignora y limpia la exportacion del vault, solo notas curadas
- cierre de BOT-013, MCP-IMG-001 y LLM-005, backlog con BOT-015/016 y DATA-003, bump a 0.0.20
- backlog con cierre de RAG-002 y bump a 0.0.17
- make typecheck en el contrato del implementer y backlog con 6 cierres + ARNES-011 y MCP-APPS-001
- no versionar el estado vivo del arnes ni los backups de datos
- regenera el grafo de conocimiento del proyecto
- pipeline de datos/features, tests y documentacion
- cambios pendientes pre-sesión
- añade chat/static para que la suite pueda ejecutarse

### Otros

- add factores_riesgo.json (force-add, gitignored)
- Merge branch 'mejoras/higiene-docs'
- Merge branch 'mejoras/label-sin-fuga'
- Merge branch 'mejoras/umbrales-decision'
- Merge branch 'mejoras/lstm-hibrida'
- Merge remote-tracking branch 'origin/main'
- notebooks: actualizar comentarios automáticos (interpretaciones de gráficas)
- notebooks: añadir comentarios y actualizar ejecución (preprocesado y entrenamiento)
- Refactor code structure for improved readability and maintainability
- notebooks: añadir comentarios automáticos; fix: corregir preprocess_data UnboundLocalError
- añadidos agentes y sus respectivas funciones

## [Unreleased] — 2026-08-10

### Añadido

- cierra FORECAST-001 — tendencia semanal con banda conformal y fin del fallback silencioso del forecast
- cierra BOT-019 — /start deduce la salida del perfil y la rutina: 8 pasos a 2, sin perder informacion
- cierran BOT-012 (franja de mayor riesgo y recomendada en el parte) y BOT-014 (chat con solo canal dominante y factores xN)
- cierra MCP-002 — MCP solo lectura por defecto, escritura por token de arranque
- cierra BOT-020 — parte del bot facil de entender con clasificacion, factores xN, tabla horaria y recomendaciones
- cierra DATA-007 — resolucion sub-horaria elegible en predict_ensemble, API y MCP
- cierra ARNES-014 — commit automatico de cierre con flag --commit, acotado a rutas del ticket
- dueno por feature y commit automatico de cierre acotado a rutas (ARNES-013/ARNES-014)
- resolucion sub-horaria de la prediccion por puntos (DATA-004/DATA-007)
- control de acceso del MCP por identidad y token (MCP-003)
- reindexa por hash de contenido — editar el cuerpo sin tocar la clave ya refresca el fragmento (RAG-005)
- fine-tuning de Qwen en Google Colab con dataset verificado (LLM-006)
- benchmark de modelos LLM con val.jsonl (LLM-003)
- modulo chat_flow para el cuestionario conversacional de /chat (CHAT-001)
- tool grafica_riesgo_horario que devuelve la curva de riesgo por hora como PNG (MCP-IMG-001)
- cuestionario de rutina de trabajo con ocupacion e intensidad, parte con confianza conformal y chat con perfil (BOT-013 + BOT-015 + LLM-005)
- parte en lenguaje llano con confianza conformal y factores con coeficiente (BOT-013 + LLM-005)
- el RAG indexa el coeficiente y el DOI de cada factor, no solo el nombre
- el input del dataset de fine-tuning lleva el parte meteorologico completo de la ventana de actividad
- borrar_rutina_mcp comprueba propiedad y rutinas MCP precargables con perfil
- bienvenida, respuestas mejoradas, perfil, rutinas, avisos y parte claro con porcentaje y fecha de nacimiento
- tablas de rutinas semanales y avisos diarios
- doc_agent unifica grafo de graphify, RAG vectorial y boveda Obsidian
- dataset, fine-tuning LoRA de Qwen 2.5, Modelfile de Ollama y skill ClimaSafe
- la intensidad sale del MET del deporte (Compendium 2024) en vez de elegirla el usuario
- indexa documentacion/ y expone busqueda semantica y LLM local por MCP
- parte final descriptivo con riesgo %, temperatura, UV y recomendacion contextual
- cierra GIT-001
- formulario Telegram con 17 estados, teclados inline, toggles multiselect y perfiles SQLite
- tools de predicción, factores y perfiles con SQLite
- pipeline de chunks, embeddings y búsqueda semántica
- orquestador de ciclo con puerta, backlog y subagentes
- Web UI grupo/volumen/curvas edad + estimacion volumetrica + fixes
- MCP server con 3 tools (predict, listar, cargar perfil) + integracion opencode
- riesgo colectivo/etiqueta, mapa de zona adaptativo y riesgo horario acumulado
- perfiles por alias, conformal prediction, frontend, docs
- personalizacion, overrides, explicabilidad, thresholds
- RAG, active learning, contrafactuales y mejora en API/chat
- calibra thresholds ML (t1=0.25/t2=0.40), elimina umbrales provincia obsoletos, personalización alineada con ML
- factores dinámicos desde JSON, personalización afecta clase final, prompt scout desambiguado
- conformal prediction para mejorar calibración de resultados
- web endpoints + MCP server + banner factores pendientes
- calidad de papers (alta/media/baja) según fuente y citas
- paper scout con arXiv/OpenAlex y clasificación LLM
- fusión LSTM, ensemble, explicabilidad y recomendaciones
- LSTM province_hybrid integrada y optimizada
- dataset a 27 features y suelo de mortalidad en labels
- experimento de features de frío con retardo largo (#11)
- ablación limpia de features 27 vs 19 con label fijo (#9)
- calibración de umbrales de decisión por clase
- LSTM híbrida con contexto de ola (secuencia 24h + features diarias)
- baseline LightGBM como candidato a sustituir a KNN
- personalización individual del índice de riesgo
- LSTM multi-tarea con comparación clasificación vs regresión
- notebooks, features pipeline y mlflow
- grafo de conocimiento y documentación
- sistema de agentes v0.2.0

### Corrección de bugs

- fine_tune compatible con transformers 5.5 (BUG-004)
- el perfil horario usa solo el dia objetivo (DATA-003); arreglado riesgo colectivo por etiqueta (BUG-002)
- escapa XSS en rutinas y pronostico, codigos HTTP 404/400 reales y borrado de perfil sin huerfanos
- un dato meteorologico NaN ya no tumba la prediccion
- aisla el tracking URI de mlflow para que no se filtre entre tests
- un campo desconocido en el perfil devuelve error explicito en vez de un 500 mudo
- EDADES_COMPARATIVA (25,55,65,75,85) evita duplicados de curva
- timeout 90s en llamada LLM del scout
- permisos de escritura para GH Actions push
- solo buscar en journals peer-review (excluir arXiv y repositorios)
- exception gitignore para factores_riesgo.json + fallback si no existe
- elimina la fuga temporal train-test del label de riesgo
- persiste modelo_desplegado_frio en 0-2-Frio

### Documentación

- cierra DOC-003 — README vuelve a su formato original y lo ampliado vive en documentacion/
- manifiesto de deploy con cuatro perfiles, de solo RAG a Qwen fine-tuneado
- reestructura documentación técnica y de riesgo
- roadmap completo con 19 tareas priorizadas y todo lo hecho marcado
- revisión 68 papers, nuevos factores documentados, próximos pasos actualizados
- reorganizar documentacion (vault + papers + problemas conocidos)
- vault Obsidian, README y CHANGELOG actualizados
- actualiza features de 19 a 27 (persistencia ampliada)
- registra LSTM, personalización y rename de documentacion
- corrige el nombre de la carpeta a «documentacion»
- documenta las decisiones de diseño y la selección por recall

### Tests

- cubre las features de distribución diaria y de rezago temporal

### Mantenimiento

- CHANGELOG y bump de version a 0.0.26
- ARNES-001 y DATA-003 bloqueadas (prioridad LLM-006 / opencode en paralelo); LLM-006 añadida e in_progress
- grafo de conocimiento actualizado (graphify-out 2026-08-06)
- ignora y limpia la exportacion del vault, solo notas curadas
- cierre de BOT-013, MCP-IMG-001 y LLM-005, backlog con BOT-015/016 y DATA-003, bump a 0.0.20
- backlog con cierre de RAG-002 y bump a 0.0.17
- make typecheck en el contrato del implementer y backlog con 6 cierres + ARNES-011 y MCP-APPS-001
- no versionar el estado vivo del arnes ni los backups de datos
- regenera el grafo de conocimiento del proyecto
- pipeline de datos/features, tests y documentacion
- cambios pendientes pre-sesión
- añade chat/static para que la suite pueda ejecutarse

### Otros

- add factores_riesgo.json (force-add, gitignored)
- Merge branch 'mejoras/higiene-docs'
- Merge branch 'mejoras/label-sin-fuga'
- Merge branch 'mejoras/umbrales-decision'
- Merge branch 'mejoras/lstm-hibrida'
- Merge remote-tracking branch 'origin/main'

## [Unreleased] — 2026-08-12

### Añadido

- predecir acepta weather opcional y reporta el canal que mueve la clase (DATA-008)
- guarda la ultima salida del chat para ofrecer repetirla (BOT-017)
- cierra MCP-APPS-001
- cierra UX-001
- cierra RAG-003
- cierra BOT-021
- cierra BUG-006
- cierra DATA-008
- cierra ARNES-014
- cierra ARNES-003 — modo debug del payload hacia el LLM (CLIMASAFE_DEBUG_LLM)
- cierra MAPA-001 — exportacion del mapa de riesgo en PNG y GeoJSON
- cierra CSV-001 — riesgo colectivo desde CSV con validacion por fila/campo y orgullo colectivo solo en competicion/deporte
- cierra FORECAST-001 — tendencia semanal con banda conformal y fin del fallback silencioso del forecast
- cierra BOT-019 — /start deduce la salida del perfil y la rutina: 8 pasos a 2, sin perder informacion
- cierran BOT-012 (franja de mayor riesgo y recomendada en el parte) y BOT-014 (chat con solo canal dominante y factores xN)
- cierra MCP-002 — MCP solo lectura por defecto, escritura por token de arranque
- cierra BOT-020 — parte del bot facil de entender con clasificacion, factores xN, tabla horaria y recomendaciones
- cierra DATA-007 — resolucion sub-horaria elegible en predict_ensemble, API y MCP
- cierra ARNES-014 — commit automatico de cierre con flag --commit, acotado a rutas del ticket
- dueno por feature y commit automatico de cierre acotado a rutas (ARNES-013/ARNES-014)
- resolucion sub-horaria de la prediccion por puntos (DATA-004/DATA-007)
- control de acceso del MCP por identidad y token (MCP-003)
- reindexa por hash de contenido — editar el cuerpo sin tocar la clave ya refresca el fragmento (RAG-005)
- fine-tuning de Qwen en Google Colab con dataset verificado (LLM-006)
- benchmark de modelos LLM con val.jsonl (LLM-003)
- modulo chat_flow para el cuestionario conversacional de /chat (CHAT-001)
- tool grafica_riesgo_horario que devuelve la curva de riesgo por hora como PNG (MCP-IMG-001)
- cuestionario de rutina de trabajo con ocupacion e intensidad, parte con confianza conformal y chat con perfil (BOT-013 + BOT-015 + LLM-005)
- parte en lenguaje llano con confianza conformal y factores con coeficiente (BOT-013 + LLM-005)
- el RAG indexa el coeficiente y el DOI de cada factor, no solo el nombre
- el input del dataset de fine-tuning lleva el parte meteorologico completo de la ventana de actividad
- borrar_rutina_mcp comprueba propiedad y rutinas MCP precargables con perfil
- bienvenida, respuestas mejoradas, perfil, rutinas, avisos y parte claro con porcentaje y fecha de nacimiento
- tablas de rutinas semanales y avisos diarios
- doc_agent unifica grafo de graphify, RAG vectorial y boveda Obsidian
- dataset, fine-tuning LoRA de Qwen 2.5, Modelfile de Ollama y skill ClimaSafe
- la intensidad sale del MET del deporte (Compendium 2024) en vez de elegirla el usuario
- indexa documentacion/ y expone busqueda semantica y LLM local por MCP
- parte final descriptivo con riesgo %, temperatura, UV y recomendacion contextual
- cierra GIT-001
- formulario Telegram con 17 estados, teclados inline, toggles multiselect y perfiles SQLite
- tools de predicción, factores y perfiles con SQLite
- pipeline de chunks, embeddings y búsqueda semántica
- orquestador de ciclo con puerta, backlog y subagentes
- Web UI grupo/volumen/curvas edad + estimacion volumetrica + fixes
- MCP server con 3 tools (predict, listar, cargar perfil) + integracion opencode
- riesgo colectivo/etiqueta, mapa de zona adaptativo y riesgo horario acumulado
- perfiles por alias, conformal prediction, frontend, docs
- personalizacion, overrides, explicabilidad, thresholds
- RAG, active learning, contrafactuales y mejora en API/chat
- calibra thresholds ML (t1=0.25/t2=0.40), elimina umbrales provincia obsoletos, personalización alineada con ML
- factores dinámicos desde JSON, personalización afecta clase final, prompt scout desambiguado
- conformal prediction para mejorar calibración de resultados
- web endpoints + MCP server + banner factores pendientes
- calidad de papers (alta/media/baja) según fuente y citas
- paper scout con arXiv/OpenAlex y clasificación LLM
- fusión LSTM, ensemble, explicabilidad y recomendaciones
- LSTM province_hybrid integrada y optimizada

### Corrección de bugs

- BUG-005 — fine_tune.py sin checkpoints intermedios (save_strategy no) y tests del fix
- BUG-005 — sin checkpoints intermedios en entrenar: SFTConfig de trl no se picklea
- fine_tune compatible con transformers 5.5 (BUG-004)
- el perfil horario usa solo el dia objetivo (DATA-003); arreglado riesgo colectivo por etiqueta (BUG-002)
- escapa XSS en rutinas y pronostico, codigos HTTP 404/400 reales y borrado de perfil sin huerfanos
- un dato meteorologico NaN ya no tumba la prediccion
- aisla el tracking URI de mlflow para que no se filtre entre tests
- un campo desconocido en el perfil devuelve error explicito en vez de un 500 mudo
- EDADES_COMPARATIVA (25,55,65,75,85) evita duplicados de curva
- timeout 90s en llamada LLM del scout
- permisos de escritura para GH Actions push
- solo buscar en journals peer-review (excluir arXiv y repositorios)
- exception gitignore para factores_riesgo.json + fallback si no existe

### Documentación

- comparativa ADK/DeepAgents/manual y sync de uv.lock tras bump de version (ARNES-009)
- desinstala timm y fastai en el notebook de Colab y lo documenta (BUG-006)
- cierra TG-002
- cierra MCP-003
- cierra BOT-016
- enlaza los gaps detectados (META-001 metricas, UV-001 linea UV) y registra los nuevos tickets de UX en el backlog
- cierra DOC-004
- cierra DOC-004 — que es un PRD y el PRD de ClimaSafeAI escrito
- cierra BUG-005
- cierra DOC-003 — README vuelve a su formato original y lo ampliado vive en documentacion/
- manifiesto de deploy con cuatro perfiles, de solo RAG a Qwen fine-tuneado
- reestructura documentación técnica y de riesgo
- roadmap completo con 19 tareas priorizadas y todo lo hecho marcado
- revisión 68 papers, nuevos factores documentados, próximos pasos actualizados
- reorganizar documentacion (vault + papers + problemas conocidos)

### Tests

- HOY dinamico en vez de fecha hardcodeada que caducaba

### Mantenimiento

- ignora el directorio de modelos LLM (gguf y Modelfile son artefactos del fine-tuning)
- CHANGELOG y bump de version a 0.0.26
- ARNES-001 y DATA-003 bloqueadas (prioridad LLM-006 / opencode en paralelo); LLM-006 añadida e in_progress
- grafo de conocimiento actualizado (graphify-out 2026-08-06)
- ignora y limpia la exportacion del vault, solo notas curadas
- cierre de BOT-013, MCP-IMG-001 y LLM-005, backlog con BOT-015/016 y DATA-003, bump a 0.0.20
- backlog con cierre de RAG-002 y bump a 0.0.17
- make typecheck en el contrato del implementer y backlog con 6 cierres + ARNES-011 y MCP-APPS-001
- no versionar el estado vivo del arnes ni los backups de datos
- regenera el grafo de conocimiento del proyecto
- pipeline de datos/features, tests y documentacion
- cambios pendientes pre-sesión

### Otros

- add factores_riesgo.json (force-add, gitignored)

## [Unreleased] — 2026-08-18

### Añadido

- cierra MCP-004
- cierra LLM-016
- cierra LLM-015
- cierra PACK-001
- aviso medico-legal emergente + disclaimer permanente en demo y web del proyecto

### Corrección de bugs

- basetemp de pytest fuera de /tmp (se llenaba y rompia la puerta de forma no determinista)
- trackea overrides/main.html (el *.html del gitignore lo ignoraba — rompia mkdocs en CI)

### Mantenimiento

- arregla make lint (ruff check verde, format separado), suite tests de CI/CD, boton a riesgo-personalizacion y movil en la demo

