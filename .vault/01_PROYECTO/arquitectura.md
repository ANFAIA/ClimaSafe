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
| ERA5 | Reanálisis climático |
| AEMET | Estaciones meteorológicas España |
| Open-Meteo | API meteorológica |
| OpenUV | Radiación UV |
| MoMo (ISCIII) | Mortalidad diaria |

## Pipeline ML (`main.py:51`)

El pipeline completo se ejecuta con `python main.py` (opción 0). Cada paso verifica si su output ya existe y salta automáticamente.

```
1. Descarga de datos crudos (MoMo + ERA5) — skip si existen
2. Preprocesado → parquets etiquetados — skip si existen y actualizados
3. Secuencias LSTM 24h (secuencias_24h.npz) — skip si existe
4. Preprocesado ML: train/test split por fecha + escalado
5. Entrenamiento: XGBoost (calor) + RandomForest (frío) inline
6. Entrenamiento: LSTM híbrida (secuencia 24h + features diarias)
7. Evaluación tabulares (argmax + umbrales calibrados)
8. Evaluación LSTM híbrida
9. Tabla comparativa final (Rec_riesgo como métrica principal)
```

## Modelos

| Modelo | Clase | Métrica guía | Estado |
|--------|-------|-------------|--------|
| XGBoost | Calor | Recall riesgo | Producción (27 features) |
| RandomForest | Frío | Recall riesgo | Producción (19 features) |
| LSTM híbrida | Calor + Frío | Rec_riesgo | En pipeline |

La LSTM híbrida (`climasafeai/models/lstm_hybrid.py`) combina un tronco LSTM sobre secuencias de 24h con features diarias (31 columnas: 27 clásicas + 4 nocturnas/rachas severas), superando a la LSTM base. Ver [[03_MODELOS/LSTM]].

Selección de features por clase: calor usa 27 features completas (grupos A-D), frío usa 19 (sin persistencia avanzada, que dañaba -0.020 Rec_riesgo). Ver `documentacion/ablacion_features_27v19.md`.

Umbrales de decisión calibrados (cascada por severidad): calor t1=0.40/t2=0.35, frío t1=0.45/t2=0.40. Mejoran Rec_riesgo +0.035 calor, +0.095 frío vs argmax.

## Ver también

- [[modelos]] — detalle de experimentos y métricas
- [[agentes]] — sistema de agentes