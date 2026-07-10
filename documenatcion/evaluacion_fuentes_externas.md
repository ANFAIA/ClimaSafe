# Evaluación de fuentes externas: WeatherNext 2, Prithvi EO 2.0 y AlphaEarth Foundations

**Fecha:** 2026-07-10
**Decisión:** las tres se descartan como fuentes del proyecto. Los entrenamientos
ya realizados (XGBoost calor / RandomForest frío, seleccionados por recall) no
se retocan: ninguna de las tres aportaba motivo para reentrenar. Este documento
deja constancia de por qué, y de las condiciones bajo las que merecería la pena
reabrir cada una.

---

## 1. WeatherNext 2 (Google DeepMind) — descartado para producción actual

**Uso propuesto:** sustituir las observaciones por pronósticos en inferencia,
para estimar el riesgo con días de antelación.

**Por qué la idea era buena:** es la única de las tres que aportaba una
capacidad nueva real (pasar de "riesgo de hoy" a "riesgo a N días") y no tocaba
el entrenamiento — solo la entrada en inferencia. Resolución 0,25° (~25 km),
horizonte 15 días, histórico 2022-presente en CC BY 4.0.

**Por qué se descarta:**

1. **El dataset público (Earth Engine / BigQuery) es 6-horario** (00z, 06z,
   12z, 18z) y el pipeline de features es horario por diseño: el Grupo C
   (estadísticas de distribución diaria: `heat_index_std`,
   `horas_sobre_umbral`, `heat_index_min`…) se calcula sobre las 24 horas del
   día, y `select_risk_hour_row` elige la hora pico entre 24 candidatas. Con 4
   muestras/día esas features pierden su significado, y hay un sesgo
   sistemático grave: en España el pico de calor es hacia las 15-16h local, y
   las muestras 12z/18z (~13h/19h local) se lo saltan casi siempre → Heat
   Index pico subestimado → menos alertas justo en el caso crítico. La salida
   horaria del modelo existe pero solo en el early access de Vertex AI
   (formulario, términos experimentales, dependencia de Google Cloud).
2. **Discrepancias tren-inferencia adicionales**: `rh` se deriva del punto de
   rocío a 2m (Magnus-Tetens) y no está confirmado que WeatherNext publique
   `d2m`; `sp` es presión superficial y muchos modelos publican `msl` (no
   intercambiables con la meseta a 600-700 m). Los pronósticos además suavizan
   los extremos respecto a ERA5, que es justo donde vive el modelo.
3. **Open-Meteo ya cubre el caso**: su API de pronóstico es horaria, llega a
   16 días, es gratuita y ya está en el stack — encaja en el pipeline actual
   sin cambios de features.

**Cuándo reabrirlo:** si se consigue acceso a la salida horaria (Vertex AI) y
un benchmark demuestra que mejora el recall de las clases de riesgo frente a
Open-Meteo al horizonte objetivo (sugerido: 7 días). El benchmark es barato:
histórico CC BY 2022-presente contra ERA5, midiendo la degradación del recall
por día de antelación con ambas fuentes.

**Pendiente en cualquier caso (aplica también a Open-Meteo):** el paso a
inferencia con pronósticos exige un pipeline híbrido para el Grupo D
(persistencia temporal): para el día t+k, los lags/rachas mezclan pasado
observado y futuro pronosticado, y hay que validar cuánto degrada por día de
antelación. Además WeatherNext no produce UV: la fórmula determinista y la
LSTM necesitarían una fuente de UV pronosticado (p. ej. CAMS de Copernicus);
Open UV solo cubre tiempo real.

---

## 2. Prithvi EO 2.0 (IBM & NASA) — descartado: sobreingeniería

**Uso propuesto:** extraer variables geoespaciales históricas (NDVI, NDBI, uso
del suelo, proximidad a masas de agua, evolución de la vegetación).

**Por qué se descarta:** Prithvi es un foundation model de 600M de parámetros
que **no produce esas variables out of the box** — exige fine-tuning con datos
etiquetados, decoder de segmentación, GPU y el toolkit TerraTorch. Y todas las
variables propuestas ya existen sin ningún modelo:

- NDVI y NDBI son aritmética de bandas sobre Sentinel-2/Landsat (dos líneas en
  Earth Engine).
- El uso del suelo ya es producto terminado: **SIOSE** (IGN, específico de
  España), CORINE Land Cover, ESA WorldCover (10 m).
- Proximidad a masas de agua: capas vectoriales públicas (IGN, OSM).
- Cambios en el territorio: comparar esos productos entre años.

Prithvi tiene sentido para tareas de segmentación a medida sin producto
existente y con etiquetas propias. No es el caso.

---

## 3. AlphaEarth Foundations (Google DeepMind) — descartado: contradice el diseño anti-sesgo geográfico

**Uso propuesto:** añadir el embedding geoespacial (64 dimensiones, anual,
10 m, Earth Engine) de cada provincia como features del modelo.

**Por qué se descarta:** `build_features.py` excluye deliberadamente
`provincia`, `lat` y `lon` como features para que el modelo aprenda "38°C +
poca humedad → riesgo alto" y no "Madrid → riesgo alto". Un embedding de 64
dimensiones **constante por provincia** es una huella dactilar de la provincia:
con ~50 provincias, XGBoost/RandomForest pueden partir por dimensiones del
embedding y memorizar la provincia, reintroduciendo por la puerta de atrás
exactamente el sesgo que esa decisión evita. Además el embedding es opaco
(inexplicable ante salud pública) y cubre 2017+, un año menos que el histórico.

**La versión legítima de la idea (trabajo futuro, no comprometido):** el hueco
que AlphaEarth quería llenar es real — las 19 features actuales son 100%
meteorológicas y dos provincias con el mismo tiempo son indistinguibles para el
modelo (isla de calor urbana, envejecimiento, vivienda). Si se aborda, la forma
compatible con el diseño es:

- **3-5 variables estáticas interpretables**, no 64 opacas: % de superficie
  urbana ponderado por población (GHS-POP), NDVI medio, altitud media,
  costa/interior — derivadas de Sentinel-2, SIOSE y WorldCover (ver sección 2:
  es la "versión buena" de Prithvi y AlphaEarth a la vez).
- **Validación leave-province-out**: la mejora de recall debe sostenerse sobre
  provincias no vistas en entrenamiento; si solo mejora dentro de las
  conocidas, es memorización y se retira.
- Nota: es probable que las diferencias reales entre provincias con clima
  similar las expliquen mejor la estructura de edad, renta o calidad de la
  vivienda (INE) que el satélite — si se abre esta línea, comparar contra ese
  baseline demográfico primero.

---

## Referencias

- [WeatherNext 2 — Earth Engine Data Catalog](https://developers.google.com/earth-engine/datasets/catalog/projects_gcp-public-data-weathernext_assets_weathernext_2_0_0) (resolución 6-horaria del dataset público)
- [WeatherNext 2 — anuncio de Google](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/weathernext-2/) (salida hasta 1h solo vía early access de Vertex AI)
- [Accessing WeatherNext forecasts](https://developers.google.com/weathernext/guides/access-forecast)
- [Prithvi-EO-2.0 — GitHub NASA-IMPACT](https://github.com/NASA-IMPACT/Prithvi-EO-2.0) · [paper (arXiv 2412.02732)](https://arxiv.org/abs/2412.02732)
- [Satellite Embedding dataset (AlphaEarth) — tutorial de Earth Engine](https://developers.google.com/earth-engine/tutorials/community/satellite-embedding-01-introduction)
- [Open-Meteo Forecast API](https://open-meteo.com/en/docs) (horaria, 16 días, sin clave)
