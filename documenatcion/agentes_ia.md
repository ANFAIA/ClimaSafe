# Agentes de IA — ClimaSafe

Arquitectura de agentes en tiempo de ejecución del sistema: framework
de orquestación y modelo de lenguaje usados por los nodos de
percepción, decisión y actuación del sistema.

---

## 1. Framework de orquestación: LangGraph

### 1.1 Alternativas consideradas

Se evalúan tres alternativas de framework de orquestación de agentes:

| Framework | Modelo mental | Encaja con ClimaSafe |
|---|---|---|
| **LangGraph** | Grafo de nodos y flechas — cada nodo puede ser código determinista o un LLM; el flujo entre nodos es explícito y controlable por el desarrollador. | Sí — flujo fijo y predecible con ramas condicionales. |
| **CrewAI** | Equipo de agentes con rol y objetivo propio que negocian entre sí cómo dividirse el trabajo, con alta autonomía sobre el camino a seguir. | No — el sistema no necesita que los agentes decidan el orden de ejecución. |
| **MCP** | Protocolo para que un agente acceda a herramientas/datos externos de forma estándar. No orquesta agentes entre sí. | Complementario, no sustituto — puede usarse *dentro* de un nodo LangGraph para acceder a APIs externas de forma estandarizada. |

### 1.2 Por qué LangGraph

- El flujo del sistema es fijo y conocido de antemano: ingesta → cálculo
  de índices físicos → modelo ML + fórmula determinista → recomendación.
  No se necesita que los agentes negocien el orden (a diferencia de
  CrewAI).
- La mayoría de los nodos son **código determinista**, no requieren LLM
  (cálculo de Heat Index/Wind Chill/UV, inferencia de los
  `RandomForestClassifier`, la fórmula de respaldo). Solo el nodo final
  de recomendación necesita lenguaje natural. LangGraph permite mezclar
  ambos tipos de nodo en el mismo grafo sin forzar que todo pase por un
  LLM.
- La lógica del máximo entre ML y fórmula (ver `diseno_modelo.md`,
  sección 5) es una rama condicional explícita del grafo — el caso de
  uso natural de LangGraph.
- No es excluyente con MCP: si en el futuro se quiere dar acceso
  estandarizado a herramientas externas dentro de un nodo, MCP puede
  usarse como capa de acceso a esas herramientas sin cambiar el
  orquestador.

---

## 2. Modelo de lenguaje: Groq API (Llama)

### 2.1 Justificación según coste, latencia y privacidad

| Criterio | Groq (remoto) |
|---|---|
| Coste | Tier gratuito sin tarjeta, limitado por rate limit (no por créditos). Suficiente para el MVP de la beca. |
| Latencia | Muy baja — hardware LPU dedicado a inferencia, uno de los proveedores más rápidos del mercado. |
| Privacidad | Los datos de la consulta (perfil, ubicación) salen a un servicio de terceros. |

**Decisión:** Groq API. El sistema no maneja datos personales sensibles
más allá de coordenadas GPS y perfil físico básico (ver
`diseno_modelo.md` — el sistema no requiere identidad del usuario), por
lo que la privacidad no es un bloqueante decisivo. La combinación de
coste cero y latencia mínima encaja con el objetivo de un MVP funcional
en 8 semanas, usando modelos open-weight (Llama) accesibles vía API sin
necesidad de infraestructura propia.


### 2.2 Rate limits del free tier (verificado en console.groq.com)

| Modelo | RPM | RPD | TPM | TPD |
|---|--:|--:|--:|--:|
| `llama-3.1-8b-instant` | 30 | 14.400 | 6.000 | 500.000 |
| `llama-3.3-70b-versatile` | 30 | 1.000 | 12.000 | 100.000 |
| `meta-llama/llama-4-scout-17b-16e-instruct` | 30 | 1.000 | 30.000 | 500.000 |
| `qwen/qwen3-32b` | 60 | 1.000 | 6.000 | 500.000 |
| `openai/gpt-oss-120b` | 30 | 1.000 | 8.000 | 200.000 |
| `openai/gpt-oss-20b` | 30 | 1.000 | 8.000 | 200.000 |
| `groq/compound` | 30 | 250 | 70.000 | — |
| `groq/compound-mini` | 30 | 250 | 70.000 | — |

RPM = peticiones/minuto · RPD = peticiones/día · TPM = tokens/minuto ·
TPD = tokens/día. Fuente: [console.groq.com/docs/rate-limits](https://console.groq.com/docs/rate-limits)
— revisar antes de implementación por si han cambiado.

Para el nodo de recomendación (único que usa LLM), `llama-3.1-8b-instant`
tiene el RPD más alto (14.400) — el más holgado para el volumen de
consultas esperado en el MVP, si el modelo 8B da calidad suficiente para
redactar la recomendación final. Si se necesita más capacidad de
razonamiento, `llama-3.3-70b-versatile` baja a 1.000 peticiones/día —
suficiente para demo y desarrollo, pero a vigilar si el proyecto escala.

---

## 3. Mapeo de agentes/nodos del grafo

Correspondencia entre los roles funcionales del sistema (percepción,
decisión, actuación) y las piezas ya diseñadas en `diseno_modelo.md` y
`formulas_riesgo_deterministico.md`:

| Rol funcional | Nodo LangGraph | Tipo | Referencia |
|---|---|---|---|
| Percepción / recopilación de datos | Ingesta Open-Meteo / AEMET / ERA5 / OpenUV | Código determinista | `diseno_modelo.md` |
| Cálculo de índices físicos | Heat Index, Wind Chill, UV | Código determinista | `formulas_riesgo_deterministico.md` |
| Análisis / toma de decisiones | Modelo ML (calor + frío) + fórmula de respaldo, lógica del máximo | Código determinista | `diseno_modelo.md` |
| Actuación / respuesta | Clasificación final + recomendación horaria en lenguaje natural | LLM (Groq/Llama) | — |

Solo el último nodo requiere LLM. Todos los anteriores son funciones
puras que LangGraph encadena como parte del mismo grafo.

---

## 4. Supervisión humana (human-in-the-loop)

Se definen umbrales críticos que requieren posible supervisión antes
de emitir la recomendación final, en vez de automatizar la respuesta
por completo:

- Heat Index > 41°C (clase PELIGRO extremo, ver
  `formulas_riesgo_deterministico.md` §1.2)
- Índice UV ≥ 7
