# ClimaSafe

![Python](https://img.shields.io/badge/Python-3.12+-blue?logo=python&logoColor=white)
![ML](https://img.shields.io/badge/ML-XGBoost%20%2F%20RandomForest-orange)
![Tracking](https://img.shields.io/badge/Experiment%20Tracking-MLflow-blue?logo=mlflow)
![Version](https://img.shields.io/badge/Version-0.0.53-green)
![Author](https://img.shields.io/badge/Author-Alejandro%20Cancelas%20Chapela-blueviolet)
![Template](https://img.shields.io/badge/Generado%20con-dskit-58a6ff?logo=github)

> Sistema de **aviso** de riesgo por temperatura (calor / frío) por provincia y día, con ML

**Tipo de ML:** `supervisado`  
**Autor:** Alejandro Cancelas Chapela  
**Versión:** 0.0.53 · XGBoost (calor) + RandomForest (frío) + LSTM province_hybrid


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

Consulta la [documentación del proyecto](documentacion/README.md) para más
detalles: [pipeline de predicción](documentacion/arquitectura/pipeline_prediccion.md),
[personalización individual](documentacion/riesgo/personalizacion_individual.md),
[predicción conforme](documentacion/conformal_prediction.md) y
[componentes (bot, web, MCP, RAG)](documentacion/componentes.md).

## Claves de API

Todas van en **`.env`**, en la raíz del repo. Está en `.gitignore`, así que no se
sube. Es el único sitio: `make spacebot-start` lo carga, y el resto de comandos lo
leen desde ahí.

```bash
# Datos climáticos
ERA5S_API_KEY=...          # https://cds.climate.copernicus.eu  (además de ~/.cdsapirc)
AEMET_API_KEY=...          # https://opendata.aemet.es/centrodedescargas/altaUsuario
OpenUV_API_KEY=...         # https://www.openuv.io

# Bot de Telegram
TELEGRAM_BOT_TOKEN=...     # te lo da @BotFather — formato 1234567890:AA...
GEMINI_API_KEY=...         # https://aistudio.google.com/api-keys
GROQ_API_KEY=...           # https://console.groq.com/keys — formato gsk_...
```

> **No las exportes también en `~/.bashrc`.** Una copia vieja ahí pisa la de `.env`
> y el bot falla con un 401 que no dice de dónde viene. Si ya la tienes puesta,
> bórrala de `.bashrc` y déjala solo en `.env`.

Para comprobar que las tres claves del bot valen — no el formato, sino que el
proveedor las acepta:

```bash
make spacebot     # las prueba contra Groq, Google y Telegram y responde ✓ o ✗
```

### Formatos que despistan

- **Google emite dos formatos de clave**: el clásico `AIzaSy…` de 39 caracteres y
  el nuevo `AQ.…` de 53. Los dos son válidos. Que una clave tenga buena pinta no
  significa que sirva: una revocada tiene el formato perfecto, por eso `make
  spacebot` la prueba de verdad.
- Si Google responde `401 ACCESS_TOKEN_TYPE_UNSUPPORTED`, la clave no vale —
  cópiala otra vez desde AI Studio con el botón de copiar de su fila.

## El bot de Telegram (spacebot)

```bash
make spacebot         # instala binario + config + skill, y valida las claves
make spacebot-start   # arranca (carga .env por ti)
make spacebot-logs    # tail de los logs
make spacebot-stop
```

El config vive en `agents/spacebot/config.toml` y se instala en
`~/.spacebot/config.toml`. **Edita siempre el del repo** y vuelve a lanzar `make
spacebot`: el instalado se regenera desde él (guardando copia del anterior).

### Por qué el enrutado de modelos es el que es

Los tres roles que invocan herramientas (`channel`, `branch`, `worker`) van por
**Gemini**; `cortex` y `compactor` por **Groq**. No es una preferencia estética:

- **Groq valida las tool calls en su servidor**, y los modelos llama se dejan el
  campo `content` al llamar a la herramienta `reply`. El error es
  `tool call validation failed: ... missing properties: 'content'` y **no dispara
  el fallback** (spacebot solo reintenta ante errores de cuota), así que la
  conversación se muere en el primer mensaje. Le pasa al 8b y también al 70b.
- El free tier de Groq da 100k tokens al día en el 70b. Una petición del canal pesa
  ~9k (system prompt + skill + esquemas MCP), o sea unos **11 mensajes al día**.
  Y el 8b, con 6000 TPM, devuelve `413 Payload Too Large` con cualquier
  conversación real.
- Los ids de modelo se escriben **tal cual los devuelve el proveedor**. `GET
  https://api.groq.com/openai/v1/models` con tu clave te da la lista buena:
  `qwen-qwen3.6-27b` no existe, el id real lleva barra (`qwen/qwen3.6-27b`).
- **Gemini no sirve para los roles que usan herramientas** con spacebot 0.5.0. Los
  Gemini 3.x razonan, y al emitir una function call devuelven un
  `thought_signature` que hay que reenviarles en la petición siguiente. Spacebot es
  anterior a eso, así que el segundo turno —cuando hay que devolverle el resultado
  de la herramienta— muere con `400: Function call is missing a thought_signature`.
  Los Gemini 2.x no razonan y valdrían, pero esta cuenta ya no los tiene: `2.5`
  responde `404 no longer available to new projects` y toda la familia `2.0`
  responde `429`. Gemini se queda para `cortex` y `compactor`, que no llaman a
  herramientas.

Para comprobar cualquiera de estas cosas antes de tocar el config, la API se prueba
en dos líneas — no hace falta reiniciar el bot ni leer logs:

```bash
curl -s https://generativelanguage.googleapis.com/v1beta/openai/models \
  -H "Authorization: Bearer $GEMINI_API_KEY" | head
curl -s https://api.groq.com/openai/v1/models \
  -H "Authorization: Bearer $GROQ_API_KEY" | head
```

---

Template generado con https://github.com/cacelass/dskit

---

> **Early-stage project.** Architecture, stack and scope may evolve during development.
>
> Built as part of the **ANFAIA Summer Grants 2026**.