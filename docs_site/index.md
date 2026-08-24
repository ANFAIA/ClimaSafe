# ClimaSafe — Documentación técnica

> **Para desarrolladores e ingenieros.** Si no eres técnico, usa la
> [guía de usuario](guia-usuario.md).

Sistema de **aviso** de riesgo cardiovascular por temperatura (calor / frío),
personalizado por persona, ubicación, día y hora.

## Pruébalo

- **[Demo en el navegador](https://cacelass.github.io/climasafe/probar-ya/)** —
  el pipeline completo ejecutándose con WebAssembly, sin servidor.
- **[Bot de Telegram](https://cacelass.github.io/climasafe/telegram.html)** —
  alertas diarias y consultas de riesgo.
- **[Servidor MCP](https://cacelass.github.io/climasafe/mcp.html)** —
  integración con asistentes de IA.
- **[Repositorio](https://github.com/ANFAIA/ClimaSafe)** — código, modelos y
  datos.

## Cómo funciona, en una frase

Los datos meteorológicos (Open-Meteo, entrenamiento con ERA5) se convierten en
características de sensación térmica y persistencia temporal; un ensemble de
XGBoost (calor), RandomForest (frío) y una LSTM con embedding de provincia
produce la probabilidad de riesgo poblacional; esa probabilidad se personaliza
con factores de la literatura epidemiológica y se traduce a clase.

## Documentación

### Para todos

| Página | Qué explica |
|---|---|
| [Guía de usuario](guia-usuario.md) | Cómo usar ClimaSafe sin ser técnico: la demo, el bot, los niveles de riesgo. |

### Para desarrolladores

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
