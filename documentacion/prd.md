# Qué es un PRD, cómo se hace, y el PRD de ClimaSafeAI

**Creado:** 2026-08-11
**Última revisión:** 2026-08-11
**Estado:** v1 (feature DOC-004)

---

## Parte 1 — Qué es un PRD (y qué no es)

Un PRD (*Product Requirements Document*) responde **para qué existe un producto**: qué problema resuelve, a quién, qué incluye (y qué no) y cómo se sabe que funciona. No dice *cómo* se construye — eso es diseño técnico — ni *cuándo* — eso es el roadmap.

### Secciones que lleva

1. **Problema** — contexto y dolor concreto, no una wishlist.
2. **A quién sirve** — usuarios y casos de uso.
3. **Alcance** — qué entrega esta versión del producto.
4. **Fuera de alcance** — qué NO se hace y por qué (evita que cada idea nueva intente colarse).
5. **Métricas de éxito** — cómo se sabe que el producto funciona.
6. **Mantenimiento** — quién lo mantiene y cuándo se revisa.

(Algunos PRDs añaden requisitos funcionales detallados; aquí esos ya viven en el backlog, así que no se duplican.)

### En qué se diferencia de lo que ClimaSafeAI ya tiene

| Documento | Pregunta que responde | Cadencia |
|---|---|---|
| **PRD** (`prd.md`) | ¿Para qué existe? ¿Para quién? ¿Qué NO hacemos? ¿Cómo medimos el éxito? | Estable; cambia poco |
| **Roadmap** (`proximos_pasos.md`) | ¿Qué construimos y en qué orden, y qué está hecho? | Viva; cambia en cada sesión |
| **Backlog** (`featureslist.json`) | ¿Qué tareas concretas, con qué criterios de aceptación? | Muy viva; cada feature |
| **README** (`README.md`) | ¿Qué es el sistema y cómo se usa? | Técnica; va con el código |

El PRD es la capa de *por qué*; el roadmap decide *qué y cuándo*; el backlog lo descompone en tareas verificables; el README describe la implementación. Un PRD mal planteado sería una cuarta copia del roadmap que se desincroniza sola — por eso este documento enlaza en vez de copiar (Parte 2, sección «Relación»).

Este documento es además la respuesta por escrito a la duda de la reunión del 06/08/2026 («¿qué es un PRD, cómo se hace?»); la Parte 2 la aplica al proyecto.

---

## Parte 2 — PRD de ClimaSafeAI

### Problema

Las olas de calor y frío matan, y la mortalidad atribuible por temperatura (X30/X31 de MoMo) se concentra en pocos días al año. Los sistemas oficiales avisan a nivel de zona, no de persona. ClimaSafeAI existe para **anticipar los días peligrosos por provincia** y traducirlos en un aviso **personalizado** según factores individuales (edad, sexo, grasa, entrenamiento, medicación, actividad...), priorizando **no perderse ningún día de riesgo** (recall) aunque cueste más falsas alarmas. Definición técnica completa: [`../README.md`](../README.md).

### A quién sirve

- **Personas vulnerables o con factores de riesgo** (edad ≥60, comorbilidades, trabajo al aire libre, deportistas en competición) que quieren saber si el día de mañana es peligroso *para ellas*, no solo para su provincia.
- **Organizadores de eventos y responsables de grupos** que necesitan el riesgo de un colectivo (competiciones, eventos deportivos, excursiones) — roadmap Fase 1 y [`CSV-001`](../featureslist.json).
- **Asistentes** (bot de Telegram, servidor MCP, chat web) que consultan el riesgo en su flujo cotidiano.

### Alcance (v1)

| Bloque | Qué cubre | Dónde está detallado |
|---|---|---|
| Aviso por temperatura (calor/frío) por provincia y día | Modelos ML + pipeline + umbrales calibrados | [`../README.md`](../README.md), [`ml/conclusiones_modelos.md`](ml/conclusiones_modelos.md) |
| Personalización individual | Factores de riesgo con coeficientes de la literatura | [`riesgo/personalizacion_individual.md`](riesgo/personalizacion_individual.md), [`riesgo/formulas_deterministas.md`](riesgo/formulas_deterministas.md) |
| Riesgo colectivo y demográfico | Modo grupo por etiquetas, CSV (`CSV-001`), estimación volumétrica | Roadmap Fase 1: [`proximos_pasos.md`](proximos_pasos.md) |
| Mapa de riesgo por zona | Grid por km², perfiles de vulnerabilidad, export (`MAPA-001`) | Roadmap Fase 2 |
| Bot de Telegram determinista | Flujo guiado por teclados, perfiles SQLite, geocodificación | Roadmap Fase 3 |
| MCP para asistentes | Tools de predicción y factores, control de acceso (`MCP-002`, `MCP-003`) | Roadmap Fase 4 |
| RAG + LLM | Base de conocimiento, explicaciones, contrafactuales | [`conclusion-base-conocimiento.md`](conclusion-base-conocimiento.md), [`componentes.md`](componentes.md) |

Cada bloque enlaza al roadmap y al backlog que lo desglosan en tareas con criterios de aceptación; este PRD no repite esos criterios.

### Fuera de alcance (y por qué)

- **No sustituye a los avisos oficiales** (AEMET, Protección Civil): es una capa complementaria de anticipación y personalización.
- **No es diagnóstico ni consejo médico**: los factores de personalización son riesgo relativo de la literatura, no evaluación clínica.
- **Radiación UV**: queda declarada como línea futura (README). Ticket de evaluación: [`UV-001`](../featureslist.json).
- **Wearables y datos en streaming** (`DATA-005`, `DATA-006`): opcional de prioridad baja; sin fuente de datos viable no se aborda.
- **Cobertura no global**: España por provincias (fuentes MoMo/ERA5 y modelo calibrados para España).

Las métricas de éxito de abajo ("Uso real", "Cobertura") no se medían en producción al redactar este PRD; su medición es el ticket [`META-001`](../featureslist.json).

Si una propuesta nueva cae en estas zonas, no entra directa al backlog: primero se discute si el PRD debe cambiar.

### Métricas de éxito

| Métrica | Define éxito | Referencia actual |
|---|---|---|
| Recall de clases de riesgo (no perderse días peligrosos) | Que ningún día de riesgo quede sin aviso | `Rec_riesgo`: XGBoost 0.668 (calor), RF 0.612 (frío), LSTM 0.737/0.708 — [`ml/conclusiones_modelos.md`](ml/conclusiones_modelos.md) |
| Precisión de avisos | Que las falsas alarmas tengan coste asumible | Umbrales calibrados: [`ml/calibracion_umbrales.md`](ml/calibracion_umbrales.md) |
| Anticipación | Que el aviso llegue con días de antelación útil (forecast, no solo histórico) | Fuente prevista: Open-Meteo ([`../README.md`](../README.md)) |
| Cobertura | Provincias × días con aviso fiable | — |
| Uso real | Consultas en bot/web/API, % con perfil personalizado | — |

Un criterio de éxito se cumple con dato medido, no con intuición. Cualquier feature nueva debe mejorar estas métricas sin romper las anteriores; los valores medidos viven en los ficheros enlazados, no aquí.

### Relación con el roadmap y el backlog

El PRD **no copia**: los ficheros siguientes son la fuente del detalle y donde se verifican los criterios. Rutas verificadas el 2026-08-11:

- Roadmap: [`proximos_pasos.md`](proximos_pasos.md) — fases, bloque por bloque, y estado de cada uno.
- Backlog: [`../featureslist.json`](../featureslist.json) — 97 features con criterios de aceptación; IDs citados aquí y presentes: `CSV-001`, `MAPA-001`, `ENS-001`, `BOT-001/002`, `MCP-001/002/003`, `DATA-005/006`, `LLM-006`, `RAG-001`, `DOC-004`.
- README: [`../README.md`](../README.md) — definición técnica, fuentes y claves.
- Documentación técnica: [`ml/conclusiones_modelos.md`](ml/conclusiones_modelos.md), [`riesgo/personalizacion_individual.md`](riesgo/personalizacion_individual.md), [`arquitectura/pipeline_prediccion.md`](arquitectura/pipeline_prediccion.md).

### Mantenimiento (quién y cuándo)

- **Dueño:** Alejandro Cancelas Chapela, autor y mantenedor del proyecto. Este es un proyecto de una sola persona: el dueño del PRD es el humano, no un agente.
- **Se actualiza cuando:**
  1. Cambia el problema, los usuarios o el alcance (nueva línea de producto, nuevo colectivo).
  2. Una propuesta de feature contradice «fuera de alcance»: primero se revisa el PRD, luego entra al backlog.
  3. Se cierra una fase del roadmap o cambia una métrica de éxito.
- **Cadencia mínima:** revisión al inicio de cada fase del roadmap y, como mucho, cada 3 meses. La fecha de la última revisión va en la cabecera de este documento.
- **Regla de oro:** este documento no contiene listas de tareas — esas viven en el roadmap y en `featureslist.json` (editado vía `harness`). Si lo obsoleto está allí, se edita allí, no aquí.