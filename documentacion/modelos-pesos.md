# Modelos y pesos

## Los cuatro modelos

| Modelo | Rol | Rec_riesgo (umbrales calibrados) |
|---|---|---|
| **XGBoost** | Modelo tabular de **calor** (pesos balanceados) | **0.668** |
| **RandomForest** | Modelo tabular de **frío** (`class_weight="balanced"`) | **0.612** |
| **LSTM province_hybrid** | LSTM + embedding de provincia + INE + features diarias, tarea multi-tarea calor/frío | **0.737** calor / **0.708** frío |
| **Fórmula determinista** | Índices térmicos clásicos (Heat Index, Wind Chill, WBGT) sobre datos actuales | — |

La métrica de selección es **Rec_riesgo** (recall medio de las clases de riesgo
1 y 2), **no** accuracy: es un sistema de aviso y vale más sobre-avisar que
perderse un día de riesgo.

## Cómo se combinan (ensemble conformal-weighted)

Cada modelo no vota igual: pesa según la **confianza conformal** de su
predicción.

- XGBoost y RandomForest llevan un **prediction set conformal**
  (`SplitConformalCalibrator`, alpha=0.1): el set tiene tamaño 1 (confianza
  alta), 2 (media) o 3 (baja), y el modelo pesa **`1 / set_size`** en la media.
- El LSTM y la Fórmula no tienen conformal → pesan **`1/2`** cada uno.
- La probabilidad del ensemble es la media ponderada de las probabilidades de
  riesgo de los modelos disponibles (para calor: XGBoost + LSTM + Fórmula; para
  frío: RandomForest + LSTM + Fórmula).

## Umbrales y calibración

- **Calibración isotónica post-hoc en frío** (en calor reduce sensibilidad en
  provincias frías): las probabilidades del RandomForest_frío se recalibran
  antes de umbralizar.
- **Umbrales de clase** (`CLASS_THRESHOLDS_RECOMENDADOS`): calor
  `t1=0.25, t2=0.10`; frío `t1=0.21, t2=0.20` (sobre probabilidades ya
  calibradas).
- **`PERS_THRESHOLD_PELIGRO = 0.55`**: umbral de PELIGRO sobre la probabilidad
  **personalizada** (es `P(riesgo) = P(1)+P(2)`, por eso es más exigente).
- **Overrides físicos**: si el Heat Index ≥ 39 °C o el Wind Chill ≤ −25 °C, la
  clase sube a PELIGRO con independencia del ML; hay overrides intermedios por
  vulnerabilidad y por UV.

## Factores individuales y sus pesos

La probabilidad poblacional se personaliza **en odds** con factores
multiplicativos por perfil (origen: OR/RR publicados en la literatura,
redondeados). Un campo ausente = factor neutro ×1.0; el producto total se capa
a **3.0** (los factores no son independientes).

### Calor

| Factor | Coef. | Fuente |
|---|---|---|
| Enf. cardiovascular | ×1.4 | Semenza 1996 (NEJM); Circ. Research 2024 |
| Diabetes | ×1.2 | Rev. sistemática LMIC 2025 |
| Salud mental / antipsicóticos | ×1.8 | Sci. Reports 2025 (OR 2.43) |
| Enf. respiratoria | ×1.3 | eBioMedicine 2016 (Bunker) |
| Diuréticos de asa | ×1.3 | PLOS One 2020 (Medicare) |
| No aclimatado | ×1.6 | NIOSH 2016-106 (RAL vs REL) |
| Situación social (máx., no producto) | vive_solo ×1.5 · no_sale ×2.0 · sin aire acond. ×2.5 · alcohol ×1.8 | Semenza 1996 (Chicago) |
| Edad | 65a ×1.2 · 75a ×1.5 · 85a ×2.0 | literatura |
| Grasa corporal | ×0.85–×1.15 (continua) | curvas CUN-BAE/ENPE |
| Nivel de actividad | ligera ×1.1 · moderada ×1.3 · intensa ×1.6 · muy intensa ×2.0 | NIOSH (MET) |
| Entrenado (actividad ≥ moderada) | **×0.5 sobre el exceso** | fisiología del ejercicio |
| Hora del día (12–18 h) | solape ≥75 % ×1.3 · 50–75 % ×1.2 · <50 % ×1.1 | NIOSH |
| Duración actividad | 1–2 h ×1.1 · 2–4 h ×1.25 · >4 h ×1.4 | NIOSH |
| Fatiga acumulada (≥4 h a HI≥27 °C) | 4–5 h ×1.2 · ≥6 h ×1.3 | NIOSH 2017-127; Flouris, Lancet PH 2018 |
| Ocupación laboral exterior | oficina ×1.0 · reparto ×1.35 · mantenimiento ×1.7 · construcción ×2.2 · campo ×2.7 | carga metabólica |

### Frío

| Factor | Coef. | Fuente |
|---|---|---|
| Enf. cardiovascular | ×1.5 | CVD = hasta 70 % del exceso invernal |
| Enf. respiratoria | ×1.4 | 2ª causa del exceso invernal |
| Vivienda fría / aislamiento | ×1.5 | exceso invernal en viviendas mal aisladas |
| Edad | 65a ×1.2 · 75a ×1.4 · 85a ×1.7 | literatura |
| Actividad | ligera/mod ×0.95/×0.9 (protectora) · intensa con sudor+viento ×1.2 | mecanismo fisiológico |
| Grasa alta | ×0.9 (protectora) | mecanismo fisiológico |

Reglas de composición: **máximo (no producto)** entre factores sociales
situacionales; los factores fisiológicos/médicos se multiplican **en odds**
`odds_ind = odds × f1 × ... × fn`, y el resultado se capa a **3.0**.

## Detalle técnico

La implementación vive en el repositorio:

- `climasafeai/models/ensemble.py` — ensemble conformal-weighted, overrides.
- `climasafeai/models/predict_model.py` — umbrales de clase.
- `climasafeai/models/conformal.py` — SplitConformalCalibrator.
- `climasafeai/models/calibrate.py` — isotónica en frío.
- `climasafeai/features/personalizacion.py` — factores individuales.
- `data/factores_riesgo.json` — base versionada de factores (coef, DOI, calidad).
