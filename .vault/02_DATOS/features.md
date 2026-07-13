---
type: datos
created: 2026-07-09
tags:
  - datos
  - features
status: active
---

# Features — Documentación completa

27 features idénticas para ambos modelos (calor y frío). La diferencia está en el target y el modelo final.

## Grupo A: Variables ERA5 crudas (4)

| # | Columna | Descripción | Unidad |
|---|---------|-------------|--------|
| 1 | `t2m_c` | Temperatura del aire a 2m | °C |
| 2 | `rh` | Humedad relativa | % (0-100) |
| 3 | `wind_speed_kmh` | Velocidad del viento a 10m | km/h |
| 4 | `sp` | Presión superficial | Pa |

**Conversiones:**
- `t2m_c = t_k - 273.15` (Kelvin → °C)
- `rh = 100 × exp((a×Td)/(b+Td)) / exp((a×T)/(b+T))` con `a=17.625`, `b=243.04` (Magnus-Tetens, Alduchov & Eskridge 1996)
- `wind_speed_kmh = √(u10² + v10²) × 3.6` (m/s → km/h)

## Grupo B: Índices meteorológicos (3)

| # | Columna | Descripción | Unidad |
|---|---------|-------------|--------|
| 5 | `heat_index_c` | Heat Index (Rothfusz 1990) | °C |
| 6 | `wbgt_c` | WBGT desde HI (Bernard & Iheanacho 2015) | °C |
| 7 | `wind_chill_c` | Wind Chill (NWS 2001) | °C |

**Heat Index** — Válido para T≥26.7°C y RH≥40%. Fuera de ese rango se usa fórmula simplificada.

**WBGT** — `WBGT = -0.0034×HI°F² + 0.96×HI°F - 34`. Precisión ±2°C.

**Wind Chill** — Válido para T≤10°C y V>4.8 km/h. Fuera se devuelve T sin modificar.

## Grupo C: Estadísticas distribución diaria (8)

Calculadas sobre las 24h del día antes de colapsar a la hora pico.

**Umbrales:** `HEAT_INDEX_UMBRAL_C = 32.0` | `WIND_CHILL_UMBRAL_C = 0.0`

| # | Columna | Descripción |
|---|---------|-------------|
| 8 | `heat_index_mean` | Media de HI de las 24h — acumulación de calor |
| 9 | `heat_index_std` | Desviación estándar — sostenido vs puntual |
| 10 | `heat_index_min` | Mínimo — proxy de alivio nocturno |
| 11 | `horas_sobre_umbral` | Horas con HI > 32°C — exposición prolongada |
| 12 | `wind_chill_mean` | Media de WC de las 24h — frío sostenido |
| 13 | `wind_chill_std` | Desviación estándar del WC |
| 14 | `wind_chill_max` | Máximo (menos frío del día) |
| 15 | `horas_bajo_umbral` | Horas con WC < 0°C — exposición a frío |

## Grupo D: Persistencia temporal (12)

Calculadas estrictamente sobre el pasado (shift(1) por provincia) — sin fuga.

| # | Columna | Descripción |
|---|---------|-------------|
| 16 | `heat_index_c_lag1` | HI pico del día anterior |
| 17 | `heat_index_c_roll3` | Media móvil del HI pico de 3 días previos |
| 18 | `heat_index_c_roll7` | Media móvil del HI pico de 7 días previos |
| 19 | `dias_consec_sobre_umbral` | Racha de días consecutivos con HI > 32°C |
| 20 | `grados_dia_calor_roll7` | Grados-día de calor acumulados en 7 días |
| 21 | `grados_dia_calor_roll14` | Grados-día de calor acumulados en 14 días |
| 22 | `wind_chill_mean_roll3` | Media móvil WC de 3 días previos |
| 23 | `wind_chill_mean_roll7` | Media móvil WC de 7 días previos |
| 24 | `wind_chill_mean_roll14` | Media móvil WC de 14 días previos |
| 25 | `grados_dia_frio_roll7` | Grados-día de frío acumulados en 7 días |
| 26 | `grados_dia_frio_roll14` | Grados-día de frío acumulados en 14 días |
| 27 | `dias_consec_bajo_umbral` | Rachas de días fríos consecutivos |

**Nota:** Los lags fueron la mejora más importante (+12% F1 macro en calor, +7% en frío).

## Target

Generado en `climasafeai/features/labels.py` a partir de percentiles de mortalidad MoMo por provincia.

- **Calor:** `clase_riesgo_calor` desde `defunciones_atrib_exc_temp`
- **Frío:** `clase_riesgo_frio` desde `defunciones_atrib_def_temp`
- Cortes: percentil 75 (precaución) y percentil 95 (peligro) por provincia

## Selección de hora de mayor riesgo

En `weather_indices.py`: `RiskScore = max(HeatIndex, -WindChill)` para cada hora. Se extraen las features de ERA5 de esa hora exacta. El RiskScore nunca entra como feature.

## Pipeline completo

```
ERA5 horario → filtro 5pt/provincia → media espacial
→ conversiones → índices meteorológicos → stats 24h
→ selección hora pico → merge stats + pico → lags temporales
→ merge con MoMo → asignar clase riesgo → preprocess_data()
```

## Evolución de features

| Iteración | Features | Mejora |
|-----------|----------|--------|
| Base | 7 (solo hora pico) | — |
| +Distribución diaria | +8 = 15 | Ayuda a calor |
| +Persistencia (lags) | +4 = 19 | Mayor mejora, ambas clases |
| +Persistencia ampliada | +8 = **27** | Calor mejora (Rec_riesgo XGB 0.614 → 0.633); frío queda igual (RF 0.527 → 0.526) |

**Nota (iteración 27):** junto a las 8 features nuevas cambió también el label (suelo de mortalidad: ≥2 muertes para `PELIGRO`), así que la atribución de la mejora no es pura features.

## Ver también

- [[fuentes]]
- [[01_PROYECTO/modelos]]
- [[07_REFERENCIAS/papers]]
- [[03_MODELOS/hiperparametros]]