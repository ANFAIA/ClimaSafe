---
type: referencia
created: 2026-07-09
tags:
  - referencias
  - papers
status: active
---

# Papers y referencias

## Papers científicos

| Referencia | Aporte | Uso en el proyecto |
|------------|--------|-------------------|
| **Rothfusz (1990)** — *NWS regression on Steadman tables* | Ecuación del Heat Index | `weather_indices.py` — `heat_index()` |
| **Bernard & Iheanacho (2015)** — *J. Occup. Environ. Hyg.* | WBGT desde Heat Index | `weather_indices.py` — `wbgt_from_heat_index()` |
| **NWS (2001)** — *Wind Chill Temperature Index* | Ecuación oficial Wind Chill | `weather_indices.py` — `wind_chill()` |
| **Alduchov & Eskridge (1996)** — Magnus-Tetens constants | Humedad relativa desde punto de rocío | `weather_indices.py` — `relative_humidity_from_dewpoint()` |
| **NIOSH [2016-106]** — *Occupational Heat Exposure Guidelines* | Límites WBGT | Referencia clínica |
| **OMS** — *Global Solar UV Index: A Practical Guide* | Tiempo hasta eritema por fototipo | Fórmula determinista de quemadura solar |

## Guías y normativas

| Documento | Contexto |
|-----------|----------|
| NIOSH Occupational Heat Exposure Guidelines | Límites de exposición ocupacional al calor |
| INSST NTP-322 | Estrés térmico, normativa española |
| OIT, Informe Seguridad Climática (2024) | Mortalidad laboral por calor global |
| WHO Heat Health Action Plans | Recomendaciones ante calor extremo |
| Ministerio de Sanidad, Plan Calor 2026 | Datos de mortalidad España |
| NOAA Wind Chill Chart | Rejilla temperatura×viento |

## Fuentes de datos

| Fuente | URL |
|--------|-----|
| ERA5 (Copernicus CDS) | [cds.climate.copernicus.eu](https://cds.climate.copernicus.eu/) |
| AEMET OpenData | [opendata.aemet.es](https://opendata.aemet.es/) |
| Open-Meteo API | [open-meteo.com](https://open-meteo.com/en/docs) |
| OpenUV | [openuv.io](https://www.openuv.io/) |
| MoMo (ISCIII) | [momo.isciii.es](https://momo.isciii.es/public/momo/dashboard) |
| CNIG (shapefile provincias) | [centrodedescargas.cnig.es](https://centrodedescargas.cnig.es/CentroDescargas/limites-municipales-provinciales-autonomicos) |

## Ver también

- [[02_DATOS/fuentes]]
- [[02_DATOS/features]] — implementación de las fórmulas
- [[documenatcion/formulas_riesgo_deterministico]] — fórmulas detalladas