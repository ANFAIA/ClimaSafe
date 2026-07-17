---
type: proyecto
created: 2026-07-09
tags:
  - proyecto
  - arquitectura
status: active
---

# Arquitectura

ClimaSafeAI es un sistema de aviso de riesgo climático personalizado (calor/frío/UV) con ML.

## Componentes principales

- `climasafeai/` — paquete Python con módulos data, features, models, utils, visualization
- `agents/` — 15 agentes auto-registrados con orquestador, registry y CLI
- `main.py` — pipeline principal (entry point)
- `.vault/` — documentación del proyecto en Obsidian

## Fuentes de datos

| Fuente | Datos |
|--------|-------|
| Open-Meteo | API meteorológica (gratis, sin API key) |
| MoMo (ISCIII) | Mortalidad diaria España |

> El sistema solo usa fuentes gratuitas y open-source. Open-Meteo reemplazó a ERA5/AEMET/OpenUV como fuente única de datos meteorológicos en producción.

## Pipeline de inferencia (`predict_model.py`)

En producción, cada consulta ejecuta:

```
1. Clima actual → Open-Meteo (Tª, humedad, viento, UV, HI, WC)
2. Perfil horario → HI y WC para cada hora en la ventana de actividad
3. Ensemble de 4 modelos en paralelo:
   ├── XGBoost (calor) → prob_riesgo_calor
   ├── RandomForest (frío) → prob_riesgo_frio
   ├── LSTM → prob_riesgo_calor + prob_riesgo_frio
   └── Fórmula determinista → HI/WC/UV con tiempo de exposición
4. Guardarraíl físico: degrada PELIGRO→PRECAUCION si HI<27°C y WC>0°C y UV<6
5. Personalización: factores edad, grasa, actividad, comorbilidades, fototipo
6. Explicabilidad: SHAP + explicación fórmula con HI ventana actividad
7. Recomendaciones filtradas por riesgo dominante (calor/frío/ambos)
```

## Modelos

| Modelo | Captura | Fuente de calibración | Estado |
|--------|---------|----------------------|--------|
| XGBoost | Riesgo poblacional calor | MoMo (España) | Producción |
| RandomForest | Riesgo poblacional frío | MoMo (España) | Producción |
| LSTM province_hybrid | Correlación temporal HI/WC ↔ mortalidad | MoMo (España) | Producción |
| Fórmula determinista | Riesgo individual (fototipo, exposición) | NWS/OMS | Producción |

Las 4 estimaciones se combinan con el criterio **más restrictivo**. La LSTM está en `lstm_province_hybrid.py` (fusión de `lstm_hybrid.py` + `lstm_province.py`).

## Módulos clave (nuevos)

| Módulo | Función |
|--------|---------|
| `ensemble.py` | Orquesta 4 modelos, guardarraíl físico, perfil_horario, override (degradación PELIGRO→PRECAUCION) |
| `explicabilidad.py` | SHAP con FEATURE_NAME_MAP (31 entradas), explicación fórmula con HI ventana actividad |
| `recomendaciones.py` | `_riesgo_dominante()` determina si el riesgo es calor/frío/ambos y filtra recomendaciones |
| `weather_fetcher.py` | Open-Meteo inlined, perfil_horario con HI/WC por hora |

## Ver también

- [[modelos]] — detalle de experimentos y métricas

## Ver también

- [[modelos]] — detalle de experimentos y métricas
- [[agentes]] — sistema de agentes