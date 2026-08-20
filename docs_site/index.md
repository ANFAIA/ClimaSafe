# ClimaSafe

> Sistema de **aviso** de riesgo cardiovascular por temperatura (calor / frío),
> personalizado por persona, ubicación, día y hora.

ClimaSafe estima tu nivel de riesgo — **SEGURO / PRECAUCIÓN / PELIGRO** — ante el
calor y el frío, para cualquier punto de España y para el día (u hora) que
elijas. Combina tres modelos de ML con índices térmicos clásicos y ajusta el
riesgo poblacional con tus factores individuales (edad, comorbilidades,
medicación, actividad, aclimatación…).

## Pruébalo

- **[Demo en el navegador](https://cacelass.github.io/climasafe/probar-ya/)** —
  el pipeline completo ejecutándose con WebAssembly, sin servidor.
- **[Repositorio](https://github.com/ANFAIA/ClimaSafe)** — código, modelos y
  datos.

## Cómo funciona, en una frase

Los datos meteorológicos (Open-Meteo, entrenamiento con ERA5) se convierten en
características de sensación térmica y persistencia temporal; un ensemble de
XGBoost (calor), RandomForest (frío) y una LSTM con embedding de provincia
produce la probabilidad de riesgo poblacional; esa probabilidad se personaliza
con factores de la literatura epidemiológica y se traduce a clase.

## Qué hay en esta documentación

| Página | Qué explica |
|---|---|
| [Modelos y pesos](modelos-pesos.md) | Los 4 modelos + la fórmula, cómo se combinan (ensemble conformal), métricas, umbrales, el modelo bayesiano de contraste y los factores individuales con su peso. |
| [Riesgo y personalización](riesgo-personalizacion.md) | Índices de sensación térmica y la tabla completa de factores individuales con su peso y su fuente. |
| [Arquitectura](arquitectura.md) | El flujo completo: datos → features → modelos → ensemble → personalización → canales (web, demo, bot, MCP, RAG). |
| [Papers](papers.md) | La base científica: guías y estudios que sustentan índices, factores y umbrales. |
| [LLM](llm.md) | El papel del LLM (redacción, no predicción), modelos soportados, fine-tuning, RAG y hosting remoto gratuito. |

La documentación completa del proyecto (notas internas, decisiones y actas)
vive en el repositorio: [`documentacion/`](https://github.com/ANFAIA/ClimaSafe/tree/main/documentacion).

---

ClimaSafe se desarrolló como parte de las **ANFAIA Summer Grants 2026**.
