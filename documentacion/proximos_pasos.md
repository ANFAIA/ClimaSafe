# Próximos pasos — hoja de ruta

**Última revisión:** 2026-07-22 (contrafactuales + frontend)

---

## Lo hecho ✅

### Web
| # | Qué | Archivos |
|---|-----|----------|
| ✅ | `GET /api/pending-factors` — devuelve factores con `implementado: false` | `chat/app.py:470` |
| ✅ | `GET /api/factores` — devuelve factores implementados agrupados por tipo/categoría | `chat/app.py:453` |
| ✅ | `POST /api/predict` — predicción completa con normalización de perfil y factores personalizados | `chat/app.py:478` |
| ✅ | `GET /api/status` ahora incluye `has_pending_factors` | `chat/app.py:407` |
| ✅ | Checkboxes dinámicos desde `/api/factores` | `chat/static/index.html` |
| ✅ | Sección medicación (antipsicóticos, diuréticos de asa) | `chat/static/index.html` |
| ✅ | Sección exposición laboral (estrés térmico, esfuerzo térmico) | `chat/static/index.html` |
| ✅ | Checkbox alcohol reciente | `chat/static/index.html` |
| ✅ | Modal de aprobación de factores pendientes | `chat/static/index.html` |
| ✅ | POST `/api/approve-factor` | `chat/app.py` |
| ✅ | Toggle Hoy/Mañana en formulario | `chat/static/index.html` |
| ✅ | Tooltip "% Grasa (solo aplica en esfuerzo)" | `chat/static/index.html:290` |
| ✅ | Labels de modelo legibles (XGBoost, RandomForest, LSTM híbrida) | `explicabilidad.py` + `index.html` |

### API — Forecast
| # | Qué | Archivos |
|---|-----|----------|
| ✅ | `POST /api/predict?date=YYYY-MM-DD` — predicción para hoy o mañana | `chat/app.py` |
| ✅ | Validación: no pasado, no >2 días, formato ISO | `chat/app.py` |
| ✅ | `fetch_weather_data(target_date)` — filtrado de forecast por día objetivo | `climasafeai/data/weather_fetcher.py` |
| ✅ | `predict_ensemble(target_date)` — parámetro propagado | `climasafeai/models/ensemble.py` |

### Agentes
| # | Qué | Archivos |
|---|-----|----------|
| ✅ | Paper scout v1 — busca papers, clasifica con LLM, guarda en SQLite | `agents/paper_scout.py` |
| ✅ | Prompt del scout desambiguado + extracción estructurada | `agents/paper_scout.py` |
| ✅ | Auto-aprobación de factores calidad=alta+DOI | `agents/paper_scout.py` |
| ✅ | MCP server — 6 tools para gestionar factores | `agents/tools/factors_mcp_tool.py` |
| ✅ | Scout conectado a graphify (GraphifyTool.build al finalizar) | `agents/paper_scout.py:888` |
| ✅ | Aprendizaje activo — clasificador ligero (LogisticRegression + all-MiniLM) filtra irrelevantes antes del LLM, mejora con cada approve/reject | `climasafeai/ml/active_learner.py`, `agents/paper_scout.py` |

### Personalización / ML
| # | Qué | Archivos |
|---|-----|----------|
| ✅ | Factor UV según fototipo — usa `uv_max` de OpenUV, modula por fototipo (I→×1.25, VI→sin efecto) | `personalizacion.py` + `weather_fetcher.py` |
| ✅ | Override HI condicional — solo fuerza PRECAUCION si HI≥32 o UV>3 | `climasafeai/models/ensemble.py:342` |
| ✅ | Calibración isotónica post-hoc RF frío | `climasafeai/models/calibrate.py` |
| ✅ | Clase final desde probabilidad personalizada | `climasafeai/models/ensemble.py` |
| ✅ | Threshold personalización t2 como constante `PERS_THRESHOLD_PELIGRO=0.55` | `climasafeai/models/ensemble.py:421` |
| ✅ | Eliminado legacy `alcohol_reciente` — solo via `situacion_social` dinámico | `climasafeai/features/personalizacion.py:249` |
| ✅ | Limpieza `_normalize_perfil` — mapeos obsoletos removidos | `chat/app.py:425` |

### Documentación
| # | Qué | Archivos |
|---|-----|----------|
| ✅ | Reorganización `documentacion/` completa | `documentacion/` |
| ✅ | `riesgo/formulas_ml_resumen.md` → `ml/formulas_vs_secuencias.md` | movido |
| ✅ | Renombrados: `coeficientes_extraidos`→`coeficientes_literatura`, `coeficientes_personalizacion_riesgo`→`personalizacion_individual`, `formulas_riesgo_deterministico`→`formulas_deterministas` | `documentacion/riesgo/` |
| ✅ | READMEs creados en `arquitectura/`, `ml/`, `riesgo/` | `documentacion/*/README.md` |
| ✅ | Eliminados 6 papers duplicados en `modelos/` | `documentacion/modelos/` |
| ✅ | Referencias crossdoc actualizadas (10 archivos) | varios |

### SQLite + RAG
| # | Qué | Archivos |
|---|-----|----------|
| ✅ | Esquema completo (perfiles, factores, historial, relaciones MxM) | `data/schema.sql` |
| ✅ | `DBManager` — CRUD + migración desde JSON | `climasafeai/db/manager.py` |
| ✅ | MCP server migrado a SQLite | `agents/tools/factors_mcp_tool.py` |
| ✅ | Endpoints migrados a SQLite | `chat/app.py` |
| ✅ | Guardado automático perfil+consulta en `/api/predict` | `chat/app.py` |
| ✅ | sqlite-vec RAG — embeddings semánticos sobre factores de riesgo con search por similitud coseno | `climasafeai/db/rag.py`, `data/schema.sql`, `pyproject.toml` |
| ✅ | `POST /api/rag-search` + MCP tool `search_factors_mcp` | `chat/app.py`, `agents/tools/factors_mcp_tool.py` |
| ✅ | `init_rag()`, `search_factores()`, auto-sync en DBManager | `climasafeai/db/manager.py` |

### Contrafactuales
| # | Qué | Archivos |
|---|-----|----------|
| ✅ | `generar_contrafactuales()` — 6 cambios accionables, impacto con/sin cap, rankeados por pp de mejora | `climasafeai/models/explicabilidad.py` |
| ✅ | `POST /api/contrafactuales` — endpoint con clase final (con override HI) vs personalizada | `chat/app.py` |
| ✅ | Frontend integrado — tarjeta "Como reducir tu riesgo" se muestra automáticamente tras predecir | `chat/static/index.html` |

---

## Lo que queda por hacer

### Alta prioridad

| # | Tarea | Archivos/detalle | Depende de |
|---|-------|-----------------|------------|
| 1 | **Probar scout con prompt nuevo** — cuando se restablezca cuota Gemini, ejecutar `uv run python -m agents scout --dry-run --query exertional` | `agents/paper_scout.py` | Cuota Gemini |
| 2 | **Verificar auto-aprobación** — factores calidad=alta+DOI en SQLite | `agents/paper_scout.py` | #1 |

### Media prioridad

| # | Tarea | Archivos/detalle | Depende de |
|---|-------|-----------------|------------|
| — | *(completadas en sesión 2026-07-22)* | | |

### Baja prioridad

| # | Tarea | Archivos/detalle | Depende de |
|---|-------|-----------------|------------|
| 13 | **Dockerizar stack completo** | `Dockerfile`, `docker-compose.yml` | — |

---

## Resumen visual

```
Alta ──┬── 1. Probar scout (cuota Gemini)
       └── 2. Verificar auto-aprobación

Baja ───└── 13. Docker

Retos ──┬── HMM / Cadenas Markov
         ├── Redes Bayesianas
         ├── Teoría causal (Pearl)
         ├── Aprendizaje activo
         ├── Procesos Gaussianos
         ├── Graph Neural Networks
         ├── Series temporales (TFT / N-BEATS)
         └── Aprendizaje por refuerzo
```

---

## Retos técnicos / Aprendizaje

Ideas para aprender conceptos nuevos que aporten valor al proyecto y sean transferibles a futuros proyectos. No hay orden de prioridad fijo — cada uno abre un camino distinto.

### Cadenas de Markov / HMM (Hidden Markov Models)

**Qué es:** Modelo probabilístico donde el estado futuro depende solo del presente (propiedad de Markov). HMM añade que el estado real es "oculto" y solo vemos observaciones ruidosas.

**Por qué es interesante:** Hoy predecimos riesgo punto a punto (foto fija). Con HMM modelaríamos la *trayectoria*: dado que hoy hay precaución, ¿qué probabilidad hay de que mañana sea peligro? ¿y en 3 días? También permite detectar regímenes ocultos (ej. "ola de calor que se intensifica" vs "pico aislado").

**Aplica en:** `climasafeai/models/ensemble.py` — capa temporal sobre las predicciones diarias.

**Qué aprenderías:** Algoritmo forward-backward, Viterbi, Baum-Welch, inferencia en secuencias.

---

### Redes Bayesianas

**Qué es:** Grafo acíclico dirigido donde cada nodo es una variable y cada arista una dependencia condicional. Aprendes la estructura (qué conecta con qué) y los parámetros (tablas de probabilidad condicional) desde datos o desde conocimiento experto.

**Por qué es interesante:** Tus factores de riesgo ya forman un grafo natural: `diabetes → cardiovascular`, `antipsicóticos → riesgo_calor`, `edad → todas`. Con una red bayesiana podrías:
- Inferir factores latentes: "riesgo alto sin causa obvia → quizás hay factor desconocido"
- Hacer diagnóstico inverso: "dado riesgo peligro, ¿qué factor es más probable que lo esté causando?"
- Integrar conocimiento experto (literatura) con datos empíricos
- El modelo resultante es totalmente interpretable (nada de caja negra)

**Aplica en:** Nuevo módulo `climasafeai/models/bayes.py` — como alternativa o complemento al ML actual.

**Qué aprenderías:** d-separación, inferencia exacta (variable elimination) vs aproximada (MCMC, loopy BP), estimación de parámetros (MLE, MAP), aprendizaje de estructura (PC, Hill-Climbing, BIC).

---

### Teoría causal (Pearl, Do-calculus)

**Qué es:** Marco formal para distinguir correlación de causalidad. Introduce tres niveles: 1) ver (asociación), 2) hacer (intervención), 3) imaginar (contrafactuales). Herramientas clave: DAGs causales, criterio back-door/front-door, do-calculus.

**Por qué es interesante:** En salud, esto es crítico. Que `diabetes` y `riesgo_calor` estén correlacionados no significa que uno cause el otro — quizás hay una variable confusora (edad). Con DAGs causales puedes responder preguntas como:
- "Si eliminamos el factor X (con una intervención), ¿cuánto cambia el riesgo?"
- "¿Qué pasaría si todos los usuarios tomaran una pastilla de aclimatación?"
- Es el siguiente salto cualitativo respecto a "solo predecir".

**Aplica en:** Diseño de features, interpretación de resultados, integración con redes bayesianas.

**Qué aprenderías:** DAGs causales, criterios de identificación, do-calculus, contrafactuales de Pearl, librerías como DoWhy, CausalNex.

---

### Aprendizaje activo ✅ (implementado)

**Qué es:** Clasificador ligero (LogisticRegression + all-MiniLM) que filtra papers irrelevantes antes de llamar al LLM. Mejora con cada approve/reject del usuario sin necesidad de fine-tune del LLM.

**Estado:** Implementado en `climasafeai/ml/active_learner.py` e integrado en `agents/paper_scout.py`.

**Cómo funciona:**
1. Cada paper pasa por el clasificador antes del LLM
2. Si confianza > 0.75 y es "irrelevante" → se descarta sin llamar a Gemini
3. Si es "aceptable" o incierto → pasa al LLM para extracción estructurada
4. Cada veredicto del LLM se almacena como ejemplo de entrenamiento
5. Al final de la ronda, retrain() mejora el clasificador

**Entrenamiento inicial:** 27 factores implementados como positivos + 15 negativos sintéticos (cocina, deportes, finanzas, etc.). El clasificador se entrena automáticamente al importar `agents.paper_scout`.

**Ver estado actual:**
```python
from climasafeai.ml.active_learner import ActiveLearner
al = ActiveLearner()
print(al.stats())
```

**Re-entrenar desde cero:**
```bash
.venv/bin/python3 scripts/entrenar_active_learner.py
```
(el script limpia la tabla y la re-puebla desde factores_riesgo)

**Ver arquitectura:** `climasafeai/ml/active_learner.py`

**Aplica en:** `agents/paper_scout.py` — filtro previo a la clasificación por LLM.

**Qué aprenderías:** Logistic Regression sobre embeddings, uncertainty sampling, pool-based active learning, cold-start con sintéticos.

---

### Procesos Gaussianos (GP)

**Qué es:** Modelo no paramétrico que define una distribución sobre funciones. Cada predicción viene con una media y una varianza (incertidumbre). La covarianza entre puntos la define un kernel (RBF, Matern, etc.).

**Por qué es interesante:** A diferencia de XGB/LSTM, los GPs te dan incertidumbre *de forma nativa* — no necesitas ensembles ni Monte Carlo Dropout. Sabes cuándo el modelo está "adivinando" (alta varianza) vs cuándo está seguro. Además funcionan bien con pocos datos, lo que los hace ideales para factores nuevos que el scout descubre pero aún tienen pocas observaciones. Limitación: escalan mal (O(n³)), pero con aproximaciones (SVGP, KISS-GP) son viables.

**Aplica en:** `climasafeai/models/predict_model.py` — como modelo complementario para factores con pocos datos, o en el módulo de calibración.

**Experimento:** `notebooks/0-5-experimentos.ipynb` — prototipo GP sobre datos sintéticos de calor, compara con XGBoost, visualiza incertidumbre.

**Qué aprenderías:** Kernels, función de covarianza, optimización de verosimilitud marginal, sparse GPs, librerías como GPyTorch, scikit-learn.

---

### Graph Neural Networks (GNN)

**Qué es:** Redes neuronales que operan sobre datos en grafo. Cada nodo tiene un vector de características que se actualiza agregando información de sus vecinos (message passing). Tipos: GCN, GAT, GraphSAGE.

**Por qué es interesante:** Tus factores no viven en un vector plano — forman un grafo. `diabetes` es comorbilidad de calor y frío, `antipsicóticos` se relaciona con `mental`, `respiratoria` se agrava en frío. Con GNNs aprenderías representaciones de cada factor en el contexto de sus relaciones, lo que permitiría:
- Predecir el coef de un factor nuevo basado en su posición en el grafo
- Agrupar factores por "perfil de riesgo compartido"
- Generar embeddings para el RAG (mucho mejores que all-MiniLM)

**Aplica en:** Nuevo módulo `climasafeai/models/gnn.py` — embedding de factores para predicción y RAG.

**Qué aprenderías:** Message passing, atención sobre grafos (GAT), pooling, inductive vs transductive learning, PyTorch Geometric (que ya tenéis como dependencia).

---

### Series temporales avanzadas (TFT / N-BEATS)

**Qué es:** Modelos de forecast multi-horizonte más modernos que LSTM vanilla. Temporal Fusion Transformer (TFT) añade atención interpretable y handling de features estáticas/conocidas/futuras. N-BEATS usa bloques fully-connected con residuales y descomposición tendencia+estacionalidad.

**Por qué es interesante:** Tu LSTM híbrido es un primer paso. TFT te daría forecast a 3-5 días con intervalos de confianza y atención que muestra qué features importan en cada paso de tiempo. N-BEATS es más simple pero muy competitivo en benchmarks. Ambos reemplazarían o complementarían el LSTM actual.

**Aplica en:** `climasafeai/models/lstm/` — alternativa a los modelos tabulares para forecast multi-día.

**Qué aprenderías:** Mecanismos de atención, variable selection networks, quantile outputs, librerías como PyTorch Forecasting, Darts.

---

### Aprendizaje por refuerzo (RL)

**Qué es:** Un agente interactúa con un entorno, toma acciones, recibe recompensas, y aprende una política que maximiza la recompensa acumulada. Tipos: value-based (DQN), policy-based (PPO), model-based (MCTS).

**Por qué es interesante:** Tu app da un riesgo, pero el usuario quiere *qué hacer*. Con RL podrías entrenar un agente que:
- Recomiende acciones óptimas: "hidratarse cada 20 min" o "reducir ejercicio 30%" o "buscar sombra ahora"
- Aprenda de las consecuencias de sus recomendaciones (el usuario siguió el consejo → el riesgo bajó → recompensa positiva)
- Se adapte a cada usuario individualmente (personalización dinámica)

El salto cualitativo es pasar de "predecir riesgo" a "reducir riesgo activamente".

**Aplica en:** Nuevo agente `agents/recommender.py` — sistema de recomendación de acciones basado en RL.

**Qué aprenderías:** MDPs, Bellman equations, exploration vs exploitation, policy gradients, librerías como Stable-Baselines3, Gymnasium.

---

## Referencias

- `conclusion-base-conocimiento.md` — decisión técnica de base de conocimiento
- `arquitectura/agentes_ia.md` — arquitectura de agentes existente
- `ml/conclusiones_modelos.md` — métricas y comparación de modelos actuales
- `arquitectura/diseño_modelo.md` — diseño general del sistema de predicción
- `README.md` — índice general de toda la documentación
