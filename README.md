# ClimaSafe

![Python](https://img.shields.io/badge/Python-3.12+-blue?logo=python&logoColor=white)
![ML](https://img.shields.io/badge/ML-XGBoost%20%2F%20RandomForest-orange)
![Tracking](https://img.shields.io/badge/Experiment%20Tracking-MLflow-blue?logo=mlflow)
![Version](https://img.shields.io/badge/Version-0.0.1-green)
![Author](https://img.shields.io/badge/Author-Alejandro%20Cancelas%20Chapela-blueviolet)
![Template](https://img.shields.io/badge/Generado%20con-dskit-58a6ff?logo=github)

> Sistema de **aviso** de riesgo por temperatura (calor / frío) por provincia y día, con ML

**Tipo de ML:** `supervisado`  
**Autor:** Alejandro Cancelas Chapela  
**Versión:** 0.0.1 · XGBoost (calor) + RandomForest (frío) + LSTM province_hybrid


ClimaSafe estima, para cada **provincia y día**, el nivel de riesgo por temperatura
(`0` seguro / `1` precaución / `2` peligro) a partir de variables meteorológicas de
ERA5, para anticipar días peligrosos antes de que ocurran. Es un sistema de **aviso**:
se prioriza **no perderse días de riesgo** (recall), asumiendo más falsas alarmas antes
que un aviso de menos. (La radiación UV queda como línea futura; hoy cubre calor y frío.)

### Enfoque de modelado

- **Target**: percentiles de mortalidad atribuida de MoMo (X30 calor / X31 frío),
  calculados **por provincia** para no penalizar a las provincias pequeñas.
- **Features**: índices de sensación térmica (Heat Index, WBGT, Wind Chill) de la hora de
  mayor riesgo del día, + **distribución diaria** de las 24 h (media/desv/mín-máx, horas
  sobre/bajo umbral), + **persistencia temporal** (lags y medias móviles del pasado, p. ej.
  `wind_chill_mean_roll7`, `dias_consec_bajo_umbral`) — el frío es acumulativo, así que la
  *racha* de días fríos pesa más que el día suelto.
- **Split por fecha** (no aleatorio) para no filtrar días de la misma ola entre train y test.
- **Tres modelos**: **XGBoost (calor)**, **RandomForest (frío)** y **LSTM province_hybrid**
  (LSTM + embedding provincia + INE + features diarias, tarea multi-tarea calor/frío).
  Elegidos por **recall de las clases de riesgo** (`Rec_riesgo`), no por accuracy.
- Seguimiento con **MLflow** y validación cruzada **temporal por años**.
- Rec_riesgo actual (con umbrales calibrados): XGBoost **0.668** (calor), RF **0.612** (frío),
  LSTM **0.737** calor / **0.708** frío.
- Detalle y justificación de cada decisión en
  [`documentacion/ml/conclusiones_modelos.md`](documentacion/ml/conclusiones_modelos.md).

---


### Fuentes de datos abiertas

- **ERA5 (ECMWF / Copernicus)** — histórico para entrenamiento del modelo, España minimo 10 años.

- **AEMET OpenData** — datos oficiales para España.

- **Open-Meteo API** — datos meteorológicos sin clave de API. En producción, su
  API de pronóstico (horaria, hasta 16 días) es también la fuente prevista para
  estimar el riesgo con días de antelación: encaja con el pipeline de features
  actual (que requiere resolución horaria) sin cambios.

- **Open UV** — índice UV en tiempo real por coordenada GPS, complementa a Open-Meteo en producción.

- **MoMo** (ISCIII) — Monitorización de la Mortalidad Diaria; muertes atribuibles a calor (X30) y frío (X31), por provincia y día. Fuente del target/label del modelo.

Se evaluaron y descartaron otras tres fuentes (WeatherNext 2, Prithvi EO 2.0 y
AlphaEarth Foundations); el análisis y los motivos están en
[`documentacion/arquitectura/evaluacion_fuentes_externas.md`](documentacion/arquitectura/evaluacion_fuentes_externas.md).

---

### Base científica
  
- NIOSH Occupational Heat Exposure Guidelines y documentos conexos sobre patologías por calor.

- Rothfusz Heat Index (1990), referencia para el cálculo del Heat Index.

- NWS Wind Chill Advisory, referencia para el cálculo del Wind Chill.

- WHO Heat Health Action Plans — recomendaciones a ciudades y sistemas de salud.

- OIT Informe Seguridad Climática (2024) y datos de mortalidad laboral por calor.

- INSST NTP-322 sobre estrés térmico y normativa española aplicable.

- Ministerio de Sanidad de España — Plan Calor 2026, con datos de mortalidad y episodios extremos en España.


---


## Estructura del proyecto

```
climasafeai/
├── data/
│   ├── raw/            ← datos originales (nunca modificar)
│   ├── interim/        ← datos en proceso
│   └── processed/      ← datos listos para modelar
├── models/             ← modelos por clase ({Modelo}_{calor,frio}.joblib, modelo_desplegado_*)
│   └── artifacts/      ← scalers, encoders, feature_names_{clase}.joblib
├── notebooks/
│   ├── 0-0-...-Descargadatos.ipynb
│   ├── 0-1-...-ProcesamientoDatos.ipynb
│   └── 0-2-...-Ejecucion.ipynb
├── reports/figures/    ← gráficos generados
├── climasafeai/
│   ├── data/           make_dataset.py
│   ├── features/       build_features.py
│   ├── models/         train_model.py · predict_model.py · temporal_cv.py
│   ├── visualization/  visualize.py
│   └── utils/          paths.py
├── documentacion/      arquitectura/ · ml/ · riesgo/ · modelos/ · papers/
├── tests/
├── main.py             ← pipeline completo
├── Makefile
└── pyproject.toml
```

## Inicio rápido

> **Nota sobre ERA5:** El cliente oficial de Copernicus (`cdsapi`) requiere un archivo de configuración llamado `.cdsapirc`, ubicado en el directorio personal del usuario (`~/.cdsapirc` en Linux/macOS o `C:\Users\<usuario>\.cdsapirc` en Windows). Cada usuario debe generar su propio **Personal Access Token** desde su cuenta del **Copernicus Climate Data Store (CDS)** y crear este archivo siguiendo la documentación oficial. Este archivo es personal, **no debe incluirse en el repositorio ni compartirse con otros usuarios**.

**Documentación oficial:** https://cds.climate.copernicus.eu/how-to-api

Crear el archivo:

```bash
nano ~/.cdsapirc
```

Contenido del archivo:

```yaml
url: https://cds.climate.copernicus.eu/api
key: TU_PERSONAL_ACCESS_TOKEN
```

Guardar el archivo y, opcionalmente, restringir sus permisos:

```bash
chmod 600 ~/.cdsapirc
```
descargar el shapefile de https://centrodedescargas.cnig.es/CentroDescargas/limites-municipales-provinciales-autonomicos y aañdirlo a data/raw

Consulta el archivo `documentacion` para más detalles.

---

Template generado con https://github.com/cacelass/dskit

---

> **Early-stage project.** Architecture, stack and scope may evolve during development.
>
> Built as part of the **ANFAIA Summer Grants 2026**.