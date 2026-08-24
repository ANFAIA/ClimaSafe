# ClimaSafe — Documentación técnica

> **Para desarrolladores e ingenieros.** Documentación de integración, arquitectura
> y modelos. Si no eres técnico, usa la [guía de usuario](guia-usuario.md).

ClimaSafe es un sistema de **aviso** de riesgo cardiovascular por temperatura
(calor / frío), personalizado por persona, ubicación, día y hora. Esta página
es el punto de entrada a la documentación técnica: explica qué hace el sistema,
cómo está construido y cómo se integra.

## Qué hace el sistema

ClimaSafe estima un nivel de riesgo — **SEGURO / PRECAUCIÓN / PELIGRO** — ante el
calor y el frío, para cualquier punto de España y para el día (u hora) elegido.
El resultado combina:

- Un **riesgo poblacional** calculado por un ensemble de modelos de ML a partir
  de variables meteorológicas.
- Un **riesgo individual** que ajusta ese riesgo poblacional con los factores
  personales del usuario (edad, comorbilidades, medicación, aclimatación…).

## Arquitectura

El flujo completo es:

```
datos meteorológicos → features → modelos → ensemble conformal → personalización → canales
```

1. **Datos**: Open-Meteo en tiempo real, ERA5 para entrenamiento, AEMET/OpenUV
   y mortalidad MoMo como target.
2. **Features**: índices de sensación térmica (Heat Index, WBGT, Wind Chill) y
   persistencia temporal.
3. **Modelos**: ensemble de XGBoost (calor), RandomForest (frío) y una LSTM con
   embedding de provincia.
4. **Ensemble conformal**: se combinan ponderando por el tamaño del conjunto de
   predicción conformal, con cobertura garantizada.
5. **Personalización**: factores multiplicativos de la literatura epidemiológica.
6. **Canales**: web, demo WebAssembly, bot de Telegram, servidor MCP y RAG.

## Componentes y cómo se integran

| Canal | Qué ofrece | Enlace |
|-------|-----------|--------|
| **Demo (WASM)** | Pipeline completo en el navegador, sin servidor | [probar-ya](https://cacelass.github.io/climasafe/probar-ya/) |
| **Bot de Telegram** | Alertas diarias y consultas de riesgo | [telegram.html](https://cacelass.github.io/climasafe/telegram.html) |
| **Servidor MCP** | Acceso al motor de riesgo desde asistentes de IA | [mcp.html](https://cacelass.github.io/climasafe/mcp.html) |
| **Repositorio** | Código, modelos, datos y tests | [ANFAIA/ClimaSafe](https://github.com/ANFAIA/ClimaSafe) |

## Documentación técnica por tema

| Página | Qué explica |
|---|---|
| [Modelos y pesos](modelos-pesos.md) | Los 4 modelos + la fórmula, cómo se combinan (ensemble conformal), métricas, umbrales, el modelo bayesiano de contraste y los factores individuales con su peso. |
| [Riesgo y personalización](riesgo-personalizacion.md) | Índices de sensación térmica y la tabla completa de factores individuales con su peso y su fuente. |
| [Arquitectura](arquitectura.md) | El flujo completo: datos → features → modelos → ensemble → personalización → canales (web, demo, bot, MCP, RAG). |
| [Papers](papers.md) | La base científica: guías y estudios que sustentan índices, factores y umbrales. |
| [LLM](llm.md) | El papel del LLM (redacción, no predicción), modelos soportados, fine-tuning, RAG y hosting remoto gratuito. |

## Documentación interna del proyecto

La documentación completa (notas internas, decisiones, actas) vive en el repositorio:
[`documentacion/`](https://github.com/ANFAIA/ClimaSafe/tree/main/documentacion).

## Licencia

ClimaSafe está publicado bajo **Apache 2.0**: se puede usar, modificar y
distribuir libremente, incluso comercialmente, conservando el aviso de licencia.

---

ClimaSafe se desarrolló como parte de las **ANFAIA Summer Grants 2026**.
