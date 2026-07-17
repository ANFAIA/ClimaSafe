---
type: observacion
created: 2026-07-16
tags:
  - observacion
  - ensemble
  - explicabilidad
  - recomendaciones
  - lstm
  - documentacion
status: active
---

# Fusión LSTM, ensemble, explicabilidad y recomendaciones

## Cambios estructurales

### Fusión de archivos LSTM
- `lstm_hybrid.py` + `lstm_province.py` → `lstm_province_hybrid.py`
- Reducción de 22 a 19 archivos .py
- Todos los imports actualizados

### Nuevos módulos

| Módulo | Propósito |
|--------|-----------|
| `ensemble.py` | Orquesta XGBoost, RF, LSTM, fórmula con guardarraíl físico y override |
| `explicabilidad.py` | SHAP con FEATURE_NAME_MAP (31 entradas), explicación fórmula con HI ventana actividad |
| `recomendaciones.py` | `_riesgo_dominante()` calor/frío/ambos, filtrado por tags |
| `weather_fetcher.py` | Open-Meteo inlined (reemplaza a `openmeteo_client.py`), perfil_horario |

### Mejoras en predict_model.py
- Perfil horario: HI/WC para cada hora en ventana de actividad
- Override degrada solo 1 nivel (PELIGRO→PRECAUCION), nunca anula
- Barra actividad compacta en 1 línea
- Inputs normalizados (`nivel_actividad`, comorbilidades, situacion_social)
- Explicaciones ML filtradas por `prob_riesgo >= 0.35`

### Personalización
- `producto_bruto` y `capado` visibles en output como "Factor total calculado: xN" / "Aplicado (cap x3.0): xN"

## Documentación

Sección 7 añadida a `documentacion/arquitectura/diseño_modelo.md` con la exploración de modelos avanzados:

| Modelo | Motivo descarte |
|--------|----------------|
| WeatherNext 2 (Google) | Pago por API, resolución 6h |
| TimesFM 2.5 (Google) | 200M params, GPU necesaria |
| Granite TTM-R3 (IBM) | Licencia CC-BY-NC-SA |
| Prithvi-EO-2.0 (IBM/NASA) | GPU, pipeline complejo |

**Principio:** solo open-source gratuito, sin API keys, sin GPU en producción.

## Vault Obsidian actualizado
- `01_PROYECTO/arquitectura.md` — nuevos módulos, ensemble, fuentes simplificadas
- `01_PROYECTO/modelos.md` — ensemble, explicabilidad, recomendaciones, modelos descartados
- `03_MODELOS/LSTM.md` — fusión de archivos reflejada
- `01_PROYECTO/roadmap.md` — todos los cambios marcados como completados

## Ver también
- [[01_PROYECTO/arquitectura]]
- [[01_PROYECTO/modelos]]
- [[03_MODELOS/LSTM]]
- [[2026-07-15_optimizacion_lstm]]
