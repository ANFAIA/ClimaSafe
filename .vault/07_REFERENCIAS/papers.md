---
type: referencia
created: 2026-07-09
tags:
  - referencias
  - papers
status: active
---

# Papers y referencias

## 1. Índices meteorológicos (fórmulas base del proyecto)

### Rothfusz (1990) — Heat Index
- **Referencia:** Rothfusz LP. *The Heat Index Equation.* NWS Technical Attachment SR 90-23, NOAA/NWS, 1990.
- **Aporte:** Ecuación de regresión polinómica que estima la temperatura aparente (sensación térmica) combinando temperatura del aire y humedad relativa. Válida para T ≥ 26.7°C y RH ≥ 40%; fuera de ese rango usa fórmulas simplificadas.
- **Uso:** Base de la columna `heat_index_c` en `weather_indices.py`. El proyecto convierte °C ↔ °F para aplicar la ecuación y viceversa.
- **Documentación completa:** `documentacion/papers/indices-biometeorologicos/rothfusz-1990-heat-index.md`

### NWS (2001) — Wind Chill
- **Referencia:** National Weather Service. *Wind Chill Temperature Index*, vigente desde 01/11/2001.
- **Aporte:** Ecuación oficial que calcula la sensación térmica por frío combinando temperatura y velocidad del viento. Válida para T ≤ 10°C y V > 4.8 km/h; fuera de ese rango se usa la temperatura real.
- **Uso:** Base de la columna `wind_chill_c` en `weather_indices.py`.
- **Documentación completa:** `documentacion/papers/indices-biometeorologicos/nws-wind-chill-2001.md`

### Alduchov & Eskridge (1996) — Humedad relativa desde punto de rocío
- **Referencia:** Alduchov OA, Eskridge RE. *Magnus-Tetens constants*.
- **Aporte:** Ecuación de Magnus-Tetens con constantes mejoradas para estimar la humedad relativa a partir de la temperatura del aire y la temperatura del punto de rocío.
- **Uso:** Base de `relative_humidity_from_dewpoint()` en `weather_indices.py`.
- **Documentación completa:** `documentacion/papers/indices-biometeorologicos/rothfusz-1990-heat-index.md` (incluida como conversión auxiliar)

### Bernard & Iheanacho (2015) — WBGT desde Heat Index
- **Referencia:** Bernard TE, Iheanacho IA. *J. Occup. Environ. Hyg.*, 2015.
- **Aporte:** Aproximación simplificada del WBGT a partir del Heat Index: `WBGT = -0.0034·HI°F² + 0.96·HI°F - 34`. Precisión ±2°C respecto al WBGT instrumental.
- **Uso:** Base de la columna `wbgt_c` en `weather_indices.py`. Permite aplicar los límites ocupacionales NIOSH/INSST sin tener WBGT instrumental.
- **Documentación completa:** incluida en `documentacion/riesgo/formulas_deterministas.md` §1.4

---

## 2. Límites de exposición (estándares ocupacionales)

### NIOSH [2016-106] — Occupational Heat Exposure
- **Referencia:** DHHS (NIOSH) Publication No. 2016-106, 2016.
- **Aporte:** Define los límites de exposición al calor por WBGT según **carga de trabajo** (ligera/moderada/pesada/muy pesada) y **aclimatación**. Dos umbrales: REL (aclimatados) y RAL (no aclimatados), más restrictivo. La diferencia RAL/REL (factor ~1.6) sustenta el coeficiente de aclimatación del proyecto.
- **Uso:** Base de los coeficientes de nivel de actividad y aclimatación en la tabla de personalización del riesgo.
- **Documentación completa:** `documentacion/papers/ocupacional/niosh-2016-106-heat.md`

### INSST NTP 322 — Valoración del riesgo WBGT (España)
- **Referencia:** Instituto Nacional de Seguridad y Salud en el Trabajo. NTP 322, 1994.
- **Aporte:** Normativa técnica española para valoración de estrés térmico mediante WBGT. Distingue exterior con/sin carga solar. Mismo esquema que NIOSH — carga de trabajo + aclimatación — pero en versión española.
- **Nota:** Actualizada por la NTP 1189 (2023, UNE-EN ISO 7243:2017). Se conserva la NTP 322 por continuidad histórica.
- **Uso:** Referencia legal española del método WBGT, complementaria a NIOSH.
- **Documentación completa:** `documentacion/papers/indices-biometeorologicos/insst-ntp-322-wbgt.md`

---

## 3. Medicación y vulnerabilidad personal (factores de personalización)

### Layton et al. (2020) — Heatwaves, medications and hospitalization (PLOS One)
- **Referencia:** Layton JB, et al. *PLOS One.* 2020;15(12):e0243665. DOI: 10.1371/journal.pone.0243665.
- **Resumen:** Estudio observacional sobre 9.721 pacientes Medicare (2007–2012) hospitalizados por calor. Examina cómo los medicamentos sensibilizadores al calor (diuréticos de asa, antipsicóticos, anticolinérgicos) afectan al riesgo de hospitalización.
- **Hallazgos clave:**
  - Diuréticos de asa: RR 1.37 (IC 95% 1.14–1.66) → sustenta el coeficiente **×1.3**
  - Antipsicóticos: RR 1.37 (IC 95% 1.14–1.66) → respalda parcialmente el factor de enfermedad mental
  - Anticolinérgicos: RR 1.16 (IC 95% 1.00–1.35)
  - En comorbilidad cardíaca específica (IC, infarto, demencia): RR hasta 1.84
- **Uso:** Base empírica del coeficiente de **diuréticos de asa ×1.3** en la tabla de personalización de calor.
- **Documentación completa:** `documentacion/papers/factores-riesgo/layton-2020-medications-heat.md`

### Wong et al. (2024) — Psychotropics and extreme temperature (Psychological Medicine)
- **Referencia:** Wong AYS, et al. *Psychological Medicine.* 2024. DOI: 10.1017/S0033291724002824.
- **Resumen:** Self-controlled case series con datos de seguros japoneses (2014–2021). Analiza si los psicofármacos modifican la asociación entre olas de calor y enfermedad por calor, infarto y delirium en personas con enfermedad mental grave (SMI) y depresión.
- **Hallazgos clave:**
  - Enfermedad mental grave: IRR enfermedad por calor 1.48 (1.40–1.56)
  - Depresión: IRR enfermedad por calor 1.54 (1.46–1.64)
  - **No se halló** diferencia significativa entre quienes tomaban y no tomaban psicofármacos → el riesgo lo marca el **diagnóstico**, no el fármaco aislado.
- **Uso:** Base del factor **enfermedad mental ×1.8** en calor. El factor se asocia al diagnóstico (vulnerabilidad, conducta, aislamiento) y no a la medicación específica.
- **Documentación completa:** `documentacion/papers/factores-riesgo/wong-2024-psychotropics-heat.md`

---

## 4. Guías institucionales y planes de acción

### OIT (2024) — Ensuring Safety and Health at Work in a Changing Climate
- **Referencia:** International Labour Organization. Ginebra: ILO, abril 2024.
- **Resumen:** Informe global sobre cambio climático y seguridad laboral. Cubre 6 riesgos: calor excesivo, radiación UV, eventos extremos, contaminación del aire, vectores y agroquímicos.
- **Dato clave:** 2.410 millones de trabajadores expuestos a calor excesivo anualmente; ~18.970 muertes/año atribuibles.
- **Uso:** Contexto de magnitud del problema. Respalda la relevancia del riesgo por **calor en población trabajadora** (conexión con nivel_actividad).
- **Documentación completa:** `documentacion/papers/ocupacional/ilo-2024-safety-health-climate.md`

### OMS — Heat–Health Action Plans (Guidance)
- **Referencia:** WHO Regional Office for Europe. ISBN 978 92 890 7191 8.
- **Resumen:** Guía para diseñar planes de acción frente al calor (HHAP). Define 8 elementos: coordinación, sistemas de alerta, comunicación, reducción de exposición, grupos vulnerables, preparación sanitaria, vigilancia y medidas a largo plazo.
- **Uso:** Marco de diseño del sistema como herramienta de aviso. Los grupos vulnerables de la OMS coinciden con los factores de personalización del proyecto.
- **Documentación completa:** `documentacion/papers/planes-accion/who-heat-health-action-plans.md`

### Ministerio de Sanidad (España) — Plan Nacional frente al exceso de temperaturas
- **Referencia:** Ministerio de Sanidad. Ediciones anuales desde 2004 (2024/2025/2026).
- **Resumen:** Plan activado cada verano (mayo–septiembre) desde 2004. En 2024 introduce la zona de meteoalerta de AEMET (182 zonas de meteosalud). Datos de 2025: 3.832 muertes atribuibles al calor (+87.6% vs 2024), 96% mayores de 65 años.
- **Uso:** Los datos de mortalidad MoMo del plan son la misma fuente del label del modelo. Las cifras confirman el perfil de vulnerabilidad por edad que refuerza los coeficientes de personalización.
- **Documentación completa:** `documentacion/papers/planes-accion/plan-nacional-calor-sanidad.md`

---

## 5. Fuentes de datos

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
- [formulas_deterministas.md](../../documentacion/riesgo/formulas_deterministas.md) — fórmulas detalladas (en `documentacion/`)
