# ClimaSafe

![Python](https://img.shields.io/badge/Python-3.12+-blue?logo=python&logoColor=white)
![ML](https://img.shields.io/badge/ML-XGBoost%20%2F%20RandomForest-orange)
![Tracking](https://img.shields.io/badge/Experiment%20Tracking-MLflow-blue?logo=mlflow)
![Version](https://img.shields.io/badge/Version-0.0.72-green)
![Author](https://img.shields.io/badge/Author-Alejandro%20Cancelas%20Chapela-blueviolet)
![Template](https://img.shields.io/badge/Generado%20con-dskit-58a6ff?logo=github)

> Sistema de **aviso** de riesgo por temperatura (calor / frío) por provincia y día, con ML

**Tipo de ML:** `supervisado`  
**Autor:** Alejandro Cancelas Chapela  
**Versión:** 0.0.72 · XGBoost (calor) + RandomForest (frío) + LSTM province_hybrid


ClimaSafe estima, para cada **provincia y día**, el nivel de riesgo por temperatura
(`0` seguro / `1` precaución / `2` peligro) a partir de variables meteorológicas de
ERA5, para anticipar días peligrosos antes de que ocurran. Es un sistema de **aviso**:
se prioriza **no perderse días de riesgo** (recall), asumiendo más falsas alarmas antes
que un aviso de menos. (La radiación UV queda como línea futura; hoy cubre calor y frío.)

### Probar online

Sin instalar nada:

- **Demo interactiva** — [probar-ya](https://cacelass.github.io/climasafe/probar-ya/):
  elige provincia y perfil y mira el riesgo.
- **Documentación** — [climasafe/documentacion](https://cacelass.github.io/climasafe/documentacion/).
- **Home del proyecto** — [cacelass.github.io](https://cacelass.github.io/).

### Enfoque de modelado

Riesgo diario **por provincia** a partir de ERA5: el target son percentiles de
mortalidad atribuida de MoMo (X30 calor / X31 frío), calculados por provincia
para no penalizar a las pequeñas. Las features combinan sensación térmica
(Heat Index, WBGT, Wind Chill) de la hora de mayor riesgo, distribución diaria
de las 24 h (media/desv/mín-máx, horas sobre/bajo umbral) y **persistencia
temporal** (lags y medias móviles) — el frío es acumulativo, así que la *racha*
de días fríos pesa más que el día suelto. Split **por fecha** (no aleatorio)
para no filtrar días de la misma ola entre train y test, seguimiento con
**MLflow**, validación cruzada temporal por años y modelos elegidos por
**recall de las clases de riesgo** (`Rec_riesgo`), no por accuracy.

| Modelo | Rol | Rec_riesgo (umbrales calibrados) |
|--------|-----|----------------------------------|
| XGBoost | calor | **0.668** |
| RandomForest | frío | **0.612** |
| LSTM province_hybrid | multi-tarea calor/frío (LSTM + embedding provincia + INE + features diarias) | **0.737** calor / **0.708** frío |

Detalle y justificación de cada decisión en
[`documentacion/ml/conclusiones_modelos.md`](documentacion/ml/conclusiones_modelos.md)
y en [`documentacion/ml/lstm_hibrida.md`](documentacion/ml/lstm_hibrida.md).

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
sube. Es el único sitio: `make bot-start` lo carga, y el resto de comandos lo
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
curl -s https://api.groq.com/openai/v1/models \
  -H "Authorization: Bearer $GROQ_API_KEY" | head
curl -s https://generativelanguage.googleapis.com/v1beta/openai/models \
  -H "Authorization: Bearer $GEMINI_API_KEY" | head
```

### Formatos que despistan

- **Google emite dos formatos de clave**: el clásico `AIzaSy…` de 39 caracteres y
  el nuevo `AQ.…` de 53. Los dos son válidos. Que una clave tenga buena pinta no
  significa que sirva: una revocada tiene el formato perfecto, por eso se prueba
  contra el proveedor.
- Si Google responde `401 ACCESS_TOKEN_TYPE_UNSUPPORTED`, la clave no vale —
  cópiala otra vez desde AI Studio con el botón de copiar de su fila.

## El bot de Telegram

Bot **determinista** (sin LLM externo): ejecuta el pipeline real
(`predict_ensemble`) y responde con la clase de riesgo y recomendaciones.
El antiguo bot conversacional (`spacebot`) se eliminó en BOT-002; la capa
conversacional hoy la sirven las **MCP tools** y el LLM local opcional
(Qwen 2.5 + RAG). Setup completo en `skills/climasafeai/SKILL.md` y detalle
de todos los componentes en [`documentacion/componentes.md`](documentacion/componentes.md).

| Comando | Qué hace |
|---------|----------|
| `make bot-start` | Arranca el bot (carga `.env` por ti) |
| `uv run python -m climasafeai.bot.telegram_bot` | Arranque directo |

Requiere `TELEGRAM_BOT_TOKEN` en `.env` (te lo da @BotFather).

---

## CI/CD y publicación

Tres workflows de GitHub Actions en `.github/workflows/`:

| Workflow | Trigger | Qué hace |
|----------|---------|----------|
| `ci.yml` | PR + push a main | `make test` + lint de los ficheros Python cambiados |
| `release.yml` | push a main | Tag `v0.0.x` + CHANGELOG + GitHub Release (idempotente) |
| `pages.yml` | push a main | Publica docs y demo en `cacelass.github.io` |

- **`ci.yml`**: test con `uv sync --all-extras`. El lint es *solo de los `.py`
  cambiados*: `make lint` completo arrastra deuda preexistente (23 errores en
  ficheros ajenos). Detalle de la decisión en `.github/workflows/README.md`.
- **`release.yml`**: el bump de versión lo hace `harness finish` en local antes
  del push; el workflow publica la versión que ya está en `pyproject.toml`
  (tag + CHANGELOG + release notes). No commitea código de producto.
- **`pages.yml`**: publica `site/` (MkDocs) y `web/probar-ya/` en
  `cacelass/cacelass.github.io` bajo `climasafe/documentacion/` y
  `climasafe/probar-ya/`.

### Publicar a GitHub Pages (docs + demo)

La publicación es **cross-repo** y necesita un PAT sobre `cacelass/cacelass.github.io`:

1. Crea un PAT (classic o fine-grained) con scope **`repo`** solo para
   `cacelass/cacelass.github.io`.
2. Añádelo como secreto **`PAGES_DEPLOY_TOKEN`** en
   *Settings → Secrets and variables → Actions* del repo ANFAIA/ClimaSafe.
3. El workflow `pages.yml` lo usa solo. Si el secreto no existe, el job se
   salta con un aviso (no falla).

Sin token (o para probar), el mismo deploy en local:

```bash
make pages-deploy        # copia site/ y web/probar-ya/ + commit + push al repo local
make pages-deploy-dry    # igual pero sin push (PUSH=no) — deja el commit hecho y avisa
```

Variables: `PAGES_DIR` (default `~/Documentos/migithub/cacelass.github.io`),
`PAGES_REMOTE` (default `origin`), `PAGES_REMOTE_URL` (clona el destino si el
checkout no existe).

---

Template generado con https://github.com/cacelass/dskit

---

> **Early-stage project.** Architecture, stack and scope may evolve during development.
>
> Built as part of the **ANFAIA Summer Grants 2026**.