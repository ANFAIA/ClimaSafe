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

## Pipeline ML

```
Descarga → Feature Engineering (stats 24h, lags, medias móviles)
→ Entrenamiento separado por clase de riesgo → Predicción
```

## Modelos en producción

| Modelo | Clase | Métrica guía |
|--------|-------|-------------|
| XGBoost | Calor | Recall riesgo |
| RandomForest | Frío | Recall riesgo |

## Ver también

- [[modelos]] — detalle de experimentos y métricas
- [[agentes]] — sistema de agentes