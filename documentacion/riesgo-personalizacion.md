# Riesgo y personalización

## Índices de sensación térmica

El riesgo se ancla en índices fisiológicos clásicos, calculados a partir de
temperatura, humedad y viento:

| Índice | Fórmula de referencia | Uso |
|---|---|---|
| **Heat Index (HI)** | Rothfusz (1990), NWS | Sensación de calor con humedad; umbrales 27 °C (PRECAUCIÓN) y 39 °C (PELIGRO) |
| **Wind Chill (WC)** | NWS Wind Chill Advisory | Sensación de frío con viento; umbrales 0 °C (PRECAUCIÓN) y −25 °C (PELIGRO) |
| **WBGT** | NIOSH / INSST NTP-322 | Estrés térmico laboral (exposición ocupacional) |

Estos índices alimentan la Fórmula determinista del ensemble y los **overrides
físicos**: si el HI ≥ 39 °C o el WC ≤ −25 °C, la clase final sube a PELIGRO con
independencia del ML.

## De probabilidad poblacional a riesgo personal

La probabilidad de riesgo del ensemble es **poblacional** (una persona media en
esa provincia y día). El riesgo personal se obtiene multiplicando **en odds**
los factores individuales:

```
odds_ind = odds_poblacional × f1 × f2 × ... × fn
prob_personal = odds_ind / (1 + odds_ind)     (con cap de producto a 3.0)
```

Un campo ausente = factor neutro ×1.0. El **producto total se capa a 3.0** (los
factores no son independientes: mayor + obeso + cardiópata solapan mecanismos).
Los factores sociales situacionales se combinan con **máximo**, no producto.

## Factores individuales — CALOR

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

## Factores individuales — FRÍO

| Factor | Coef. | Fuente |
|---|---|---|
| Enf. cardiovascular | ×1.5 | CVD = hasta 70 % del exceso invernal |
| Enf. respiratoria | ×1.4 | 2ª causa del exceso invernal |
| Vivienda fría / aislamiento | ×1.5 | exceso invernal en viviendas mal aisladas |
| Edad | 65a ×1.2 · 75a ×1.4 · 85a ×1.7 | literatura |
| Actividad | ligera/mod ×0.95/×0.9 (protectora) · intensa con sudor+viento ×1.2 | mecanismo fisiológico |
| Grasa alta | ×0.9 (protectora) | mecanismo fisiológico |

## Clase final

La clase se decide con una cascada de umbrales sobre la probabilidad
personalizada (`PERS_THRESHOLD_PELIGRO = 0.55` para PELIGRO; `t1` de
`CLASS_THRESHOLDS_RECOMENDADOS` para PRECAUCIÓN), con overrides físicos por
Heat Index / Wind Chill / UV descritos arriba.

## Detalle técnico

- `climasafeai/features/personalizacion.py` — factores individuales.
- `climasafeai/features/weather_indices.py` — Heat Index, Wind Chill, WBGT.
- `data/factores_riesgo.json` — base versionada de factores (coef, DOI, calidad).
