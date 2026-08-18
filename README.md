# ClimaSafe

![Python](https://img.shields.io/badge/Python-3.12+-blue?logo=python&logoColor=white)
![ML](https://img.shields.io/badge/ML-XGBoost%20%2F%20RandomForest-orange)
![Tracking](https://img.shields.io/badge/Experiment%20Tracking-MLflow-blue?logo=mlflow)
![Version](https://img.shields.io/badge/Version-0.0.79-green)
![Author](https://img.shields.io/badge/Author-Alejandro%20Cancelas%20Chapela-blueviolet)
![Template](https://img.shields.io/badge/Generado%20con-dskit-58a6ff?logo=github)

> Sistema de **aviso** de riesgo por temperatura (calor / frío) por provincia y día, con ML

**Tipo de ML:** `supervisado`  
**Autor:** Alejandro Cancelas Chapela  
**Versión:** 0.0.79 · XGBoost (calor) + RandomForest (frío) + LSTM province_hybrid

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

- **ERA5** (entrenamiento), **AEMET OpenData**, **Open-Meteo** (producción, sin clave),
  **Open UV** (índice UV) y **MoMo** (target/label).
- Se evaluaron y descartaron WeatherNext 2, Prithvi EO 2.0 y AlphaEarth Foundations:
  análisis y motivos en
  [`documentacion/arquitectura/evaluacion_fuentes_externas.md`](documentacion/arquitectura/evaluacion_fuentes_externas.md).

---

### Base científica

Rothfusz Heat Index (1990), NWS Wind Chill, NIOSH, WHO Heat Health Action Plans,
OIT (2024), INSST NTP-322 y el Plan Calor del Ministerio de Sanidad — fichas, citas
y coeficientes en [`documentacion/papers/`](documentacion/papers/README.md).

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

Ponerse a trabajar en unos minutos: entorno, pipeline y calidad en
[`documentacion/inicio_rapido.md`](documentacion/inicio_rapido.md). Claves de API,
`.cdsapirc` (ERA5) y solución de problemas de claves en
[`documentacion/claves_api.md`](documentacion/claves_api.md).

## El bot de Telegram

Bot **determinista** (sin LLM externo): ejecuta el pipeline real
(`predict_ensemble`) y responde con la clase de riesgo y recomendaciones.
El antiguo bot conversacional (`spacebot`) se eliminó en BOT-002; la capa
conversacional hoy la sirven las **MCP tools** y el LLM local opcional
(Qwen 2.5 + RAG).

| Comando | Qué hace |
|---------|----------|
| `make bot-start` | Arranca el bot (carga `.env` por ti) |
| `uv run python -m climasafeai.bot.telegram_bot` | Arranque directo |

Requiere `TELEGRAM_BOT_TOKEN` en `.env` (te lo da @BotFather). Setup completo en
`skills/climasafeai/SKILL.md` y detalle de todos los componentes en
[`documentacion/componentes.md`](documentacion/componentes.md).

## CI/CD y publicación

Tres workflows de GitHub Actions (CI, release y GitHub Pages) con su detalle —
decisión del lint, secreto `PAGES_DEPLOY_TOKEN` y deploy local con
`make pages-deploy` — en [`.github/workflows/README.md`](.github/workflows/README.md).

---

Template generado con https://github.com/cacelass/dskit

---

> **Early-stage project.** Architecture, stack and scope may evolve during development.
>
> Built as part of the **ANFAIA Summer Grants 2026**.
