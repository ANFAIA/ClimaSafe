# ClimaSafe — Documentación

> Sistema de **aviso** de riesgo por temperatura (calor / frío) por provincia y
> día, con ML. Esta página sintetiza el núcleo técnico: los modelos y cómo se
> combinan, la base científica y los factores individuales de riesgo. Cada
> sección enlaza a la página completa donde vive el detalle.

---

## Modelos y sus pesos

Cuatro modelos alimentan la predicción, combinados en un **ensemble
conformal-weighted** (implementación: `climasafeai/models/ensemble.py` →
`_conformal_weighted_ensemble`):

| Modelo | Rol | Rec_riesgo (umbrales calibrados) |
|---|---|---|
| **XGBoost** | Modelo tabular de **calor** (pesos balanceados) | **0.668** |
| **RandomForest** | Modelo tabular de **frío** (`class_weight="balanced"`) | **0.612** |
| **LSTM province_hybrid** | LSTM + embedding de provincia + INE + features diarias, tarea multi-tarea calor/frío | **0.737** calor / **0.708** frío |
| **Fórmula determinista** | Índices térmicos clásicos (Heat Index, Wind Chill, WBGT) sobre datos actuales | — |

### Cómo se combinan

- Cada modelo ML (XGBoost, RandomForest) lleva un **prediction set conformal**
  (`SplitConformalCalibrator`, alpha=0.1): el set tiene tamaño 1, 2 o 3 y el
  modelo pesa **`1 / set_size`** en la media.
- El **LSTM** y la **Fórmula** no tienen conformal → pesan **`1/2`** cada uno.
- La probabilidad del ensemble es la media ponderada de las probabilidades de
  riesgo de los modelos disponibles; la clase sale de la cascada de umbrales
  (PELIGRO si `prob >= PERS_THRESHOLD_PELIGRO`).

### Umbrales y calibración

- **Calibración isotónica post-hoc en frío** (en calor reduce sensibilidad en
  provincias frías): las probabilidades del RF_frio se recalibran antes de
  umbralizar.
- **`CLASS_THRESHOLDS_RECOMENDADOS`** (`climasafeai/models/predict_model.py`):
  calor `t1=0.25, t2=0.10`; frío `t1=0.21, t2=0.20` (en frío, sobre
  probabilidades ya calibradas con isotónica).
- **`PERS_THRESHOLD_PELIGRO = 0.55`**: umbral de PELIGRO sobre la probabilidad
  **personalizada** (es `P(riesgo) = P(1)+P(2)`, no `P(peligro) = P(2)`, por eso
  es más exigente que el `t2` del ML).
- La métrica de selección es `Rec_riesgo` (recall medio de las clases de riesgo
  1 y 2), **no** accuracy ni F1 ponderado: es un sistema de aviso y vale más
  sobre-avisar que perderse un día de riesgo. El techo actual es la precisión
  de las clases de riesgo (~0.08–0.27).

### Detalle

- [`ml/conclusiones_modelos.md`](ml/conclusiones_modelos.md) — comparativa,
  features (lags de persistencia) y por qué XGBoost necesita pesos.
- [`ml/calibracion_umbrales.md`](ml/calibracion_umbrales.md) — cascada por
  severidad y puntos de operación (argmax vs umbrales).
- [`conformal_prediction.md`](conformal_prediction.md) — metodología conforme y
  el peso `1/set_size`.
- [`modelos/actuales/README.md`](modelos/actuales/README.md) — ficha de cada
  modelo desplegado.

---

## Estudios y papers considerados

La base científica del sistema (resumen; el detalle y las citas completas en
[`papers/README.md`](papers/README.md); el README del repositorio, sección
**Base científica**, se refleja en el índice de [`README.md`](README.md)):

| Fuente | Uso en el sistema |
|---|---|
| **NIOSH** (2016-106) | Umbrales de exposición laboral (WBGT), ciclos trabajo/descanso, aclimatación ×1.6 |
| **Rothfusz Heat Index (1990)** | Ecuación de referencia del Heat Index |
| **NWS Wind Chill** | Ecuación de referencia del Wind Chill |
| **WHO Heat Health Action Plans** | Recomendaciones a ciudades y sistemas de salud |
| **OIT 2024** (Seguridad Climática) | Mortalidad laboral por calor |
| **INSST NTP-322** | Estrés térmico y normativa española (WBGT) |
| **Ministerio de Sanidad — Plan Calor** | Episodios extremos y mortalidad en España |
| **MoMo / ISCIII** | Monitorización de Mortalidad Diaria — **fuente del target** (X30 calor / X31 frío) |

- [`papers/README.md`](papers/README.md) — catálogo de papers por temática
  (aclimatación, factores de riesgo, índices, ocupacional, planes de acción).
- [`riesgo/coeficientes_literatura.md`](riesgo/coeficientes_literatura.md) —
  coeficientes extraídos de la literatura.

---

## Factores individuales y sus pesos

La predicción poblacional se personaliza **en odds** con factores
multiplicativos por perfil (origen: literatura epidemiológica — los OR/RR
publicados, redondeados). Un campo ausente = factor neutro ×1.0; el **producto
total se capa a 3.0** (los factores no son independientes). Fuente completa:
[`riesgo/personalizacion_individual.md`](riesgo/personalizacion_individual.md)
y `data/factores_riesgo.json`.

### Principales factores — CALOR

| Factor | Coef. | Fuente |
|---|---|---|
| Enf. cardiovascular | ×1.4 | Semenza 1996 (NEJM); Circ. Research 2024 |
| Diabetes | ×1.2 | Rev. sistemática LMIC 2025 |
| Salud mental / antipsicóticos | ×1.8 | Sci. Reports 2025 (OR 2.43) |
| Enf. respiratoria | ×1.3 | eBioMedicine 2016 (Bunker) |
| Diuréticos de asa | ×1.3 | PLOS One 2020 (Medicare) |
| No aclimatado | ×1.6 | NIOSH 2016-106 (RAL vs REL) |
| Situación social (máx., no producto) | vive_solo ×1.5 · no_sale ×2.0 · encamado ×2.0 · sin aire acond. ×2.5 · alcohol ×1.8 | Semenza 1996 (Chicago) |
| Edad | 65a ×1.2 · 75a ×1.5 · 85a ×2.0 | literatura (vulnerabilidad directa) |
| Grasa corporal | ×0.85–×1.15 (continua, relativa a edad+sexo) | curvas CUN-BAE/ENPE |
| Nivel de actividad | ligera ×1.1 · moderada ×1.3 · intensa ×1.6 · muy intensa ×2.0 | NIOSH (carga metabólica, MET) |
| Entrenado (actividad ≥ moderada) | **×0.5 sobre el exceso** del factor de actividad | fisiología del ejercicio |
| Hora del día (ventana 12–18 h) | solapa ≥75 % ×1.3 · 50–75 % ×1.2 · <50 % ×1.1 | NIOSH; fisiología circadiana |
| Duración actividad | 1–2 h ×1.1 · 2–4 h ×1.25 · >4 h ×1.4 | NIOSH (exposición continua) |
| Fatiga acumulada (≥4 h a HI≥27 °C) | 4–5 h ×1.2 · ≥6 h ×1.3 | NIOSH 2017-127; Flouris, Lancet PH 2018 |

### Principales factores — FRÍO

| Factor | Coef. | Fuente |
|---|---|---|
| Enf. cardiovascular | ×1.5 | CVD = hasta 70 % del exceso invernal |
| Enf. respiratoria | ×1.4 | 2ª causa del exceso invernal |
| Vivienda fría / aislamiento | ×1.5 | exceso invernal en viviendas mal aisladas |
| Edad | 65a ×1.2 · 75a ×1.4 · 85a ×1.7 | literatura |
| Actividad | ligera/mod ×0.95/×0.9 (protectora) · intensa con sudor+viento ×1.2 | mecanismo fisiológico |
| Grasa alta | ×0.9 (protectora, aísla) | mecanismo fisiológico |

Reglas de composición: **máximo (no producto)** entre factores sociales
situacionales; los factores fisiológicos/médicos sí se multiplican **en odds**
`odds_ind = odds × f1 × ... × fn`, y el resultado se capa a **3.0**.

- [`riesgo/personalizacion_individual.md`](riesgo/personalizacion_individual.md)
  — tabla completa con RR/OR publicados, intervalos y confianza por factor.
- [`riesgo/formulas_deterministas.md`](riesgo/formulas_deterministas.md) —
  índices térmicos (Heat Index, Wind Chill, WBGT).
- `data/factores_riesgo.json` — base de datos versionada de factores (coef,
  DOI, calidad, implementado).

---

## Componentes

Consulta el sistema desde el [bot de Telegram](componentes.md), la
[web UI](componentes.md), los [servidores MCP](componentes.md) para asistentes
o el [RAG vectorial](componentes.md) (sqlite-vec) — resumen en
[`componentes.md`](componentes.md).

## Índice estructural

La estructura completa de carpetas y el índice de documentos está en
[`README.md`](README.md).
