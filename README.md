# ClimaSafe

![Python](https://img.shields.io/badge/Python-3.12+-blue?logo=python&logoColor=white)
![ML Type](https://img.shields.io/badge/ML-Random%20Forest-orange)
![Tracking](https://img.shields.io/badge/Experiment%20Tracking-MLflow-blue?logo=mlflow)
![Version](https://img.shields.io/badge/Version-0.0.1-green)
![Author](https://img.shields.io/badge/Author-Alejandro%20Cancelas%20Chapela-blueviolet)
![Template](https://img.shields.io/badge/Generado%20con-dskit-58a6ff?logo=github)

> Prediccion de riesgo climatico personalizado (calor/frio/uv) con ML

**Tipo de ML:** `supervisado`  
**Autor:** Alejandro Cancelas Chapela  
**Versión:** 0.0.1 · RandomForest


ClimaSafe predice el nivel de riesgo climático (calor, frío y radiación UV)
a partir de variables meteorológicas y de ubicación, con el objetivo de anticipar
condiciones potencialmente peligrosas para las personas antes de que ocurran.
El modelo (Random Forest, clasificación supervisada) se entrena con datos históricos
de clima y devuelve una categoría de riesgo interpretable, pensada para
integrarse en alertas o recomendaciones personalizadas.

---


### Fuentes de datos abiertas

- ERA5 (ECMWF / Copernicus) — histórico para entrenamiento del modelo, España minimo 10 años.

- AEMET OpenData — datos oficiales para España.

- Open-Meteo API — datos meteorológicos sin clave de API.

- Open UV — índice UV en tiempo real por coordenada GPS, complementa a Open-Meteo en producción.

- MoMo (ISCIII) — Monitorización de la Mortalidad Diaria; muertes atribuibles a calor (X30) y frío (X31), por provincia y día. Fuente del target/label del modelo.


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
├── models/             ← modelos entrenados (.joblib / .pt)
│   └── artifacts/      ← encoders, scalers, etc.
├── notebooks/
│   ├── 0-0-...-Descargadatos.ipynb
│   ├── 0-1-...-ProcesamientoDatos.ipynb
│   └── 0-2-...-Ejecucion.ipynb
├── reports/figures/    ← gráficos generados
├── climasafeai/
│   ├── data/           make_dataset.py
│   ├── features/       build_features.py
│   ├── models/         train_model.py · predict_model.py
│   ├── visualization/  visualize.py
│   └── utils/          paths.py
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


Consulta el archivo `documentacion` para más detalles.

---

Template generado con https://github.com/cacelass/dskit