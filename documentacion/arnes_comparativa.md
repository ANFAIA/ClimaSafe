# Comparativa de arneses: Google ADK, LangChain DeepAgents o hacerlo a mano

**Fecha:** 2026-08-11
**Feature:** ARNES-009 (entregable de análisis, sin código)
**Decide:** qué implementa ARNES-006 (bucle de agente propio en Python sobre LiteLLM)

## Decisión técnica

| Aspecto | Elección | Por qué |
|---------|----------|---------|
| Bucle agéntico | **Hacerlo a mano sobre LiteLLM**, siguiendo byoharness.dev | El harness de 26 agentes ya existe y LiteLLM ya está en el stack; un framework añade una capa de adaptación y una dependencia que no compran nada de lo que el backlog ya planea construir |
| Capa de proveedor | **LiteLLM** (no se reescribe) | Ya abstrae Ollama, Groq y OpenAI; es lo que usa `climasafeai/llm/rag_qwen.py` hoy |
| Catálogo de tools | El registro existente (`@register_agent`, `agents/tools/`) expuesto como esquema de herramientas | Cumple ARNES-006 tal cual: auto-registro y catálogo cerrado |
| Seguridad | Frontera determinista propia (ARNES-005/011) | El gateway de permisos es una función Python entre el bucle y las tools, no un middleware ajeno |

## Contexto: el caso concreto

ClimaSafeAI ya tiene un arnés **hecho a mano y funcionando**:

- 26 agentes Python en `agents/agents/` + `agents/external/`, auto-registrados con
  `@register_agent` (discovery por decorador, sin tocar el núcleo al añadir uno).
- ~30 tools deterministas en `agents/tools/` (git, filesystem, docker, mlflow,
  sqlite, duckdb, pytest, MCP...).
- Una puerta (`init.sh` / `harness gate`) que decide si se puede trabajar, y un
  backlog con criterios de aceptación verificables.
- **LiteLLM ya es dependencia del proyecto** (`litellm>=1.93.0` en
  `pyproject.toml`) y se usa en producción en `climasafeai/llm/rag_qwen.py`.

Lo que ARNES-009 decide no es «construir un arnés desde cero»: es si el bucle
**LLM-driven** que falta (el agente que elige tools de un catálogo cerrado para
ejecutar una feature) se escribe a mano, o se compra montándolo sobre Google ADK
o LangChain DeepAgents. La referencia de coste que motivó todo el bloque es la
de spacebot: **~7.900 tokens por mensaje** (ARNES-002, ARNES-010).

## Opción A — Hacerlo a mano sobre LiteLLM (byoharness.dev)

byoharness.dev (*Build Your Own Coding Agent*, MIT) es un currículo de 20
lecciones que recorre exactamente las piezas que el backlog de ARNES ya pide:
bucle agéntico (lección 01), **puerta de permisos** (02), **interfaz de
proveedor** (03), estado de conversación (06), **compactación** (07), tools
plug-and-play (09), **subagentes** (11), AGENTS.md como contexto (15),
**visor de tokens** (16), **prompt caching** (17) y memoria (19). Es un
blueprint, no una dependencia: se lee, se adapta al catálogo de tools propio y
queda como código del repositorio, testeable con pytest.

## Opción B — Google ADK (adk.dev)

Framework open source de Google (v2.0 en 2026), Python, agentes como clases con
tools. Puntos relevantes para este caso:

- **Tools:** funciones Python decoradas o clases `LlmAgent`/`Agent`; soporta
  MCP. Exige migrar el registro propio a su convención de agentes.
- **Guardrails:** callbacks en tres puntos (`before_model_callback`,
  `after_model_callback`, `before_tool_callback`) y plugins de seguridad.
- **Tokens:** emite `usage_metadata` por llamada y hay plugin de analítica a
  BigQuery; pero por defecto reenvía el prompt estático + definiciones de tools
  en cada turno (el propio proyecto discute caché para mitigarlo).
- **Modelos:** agnóstico vía wrapper `LiteLlm`, pero optimizado para Gemini.

## Opción C — LangChain DeepAgents (langchain-ai/deepagents)

Harness «batteries-included» de LangChain sobre LangGraph, pensado para tareas
largas de coding/research.

- **Tools:** middleware con filesystem virtual, shell con sandbox, skills,
  memoria vía AGENTS.md; subagentes con aislamiento de contexto por defecto.
- **Guardrails:** human-in-the-loop (`interrupt_on`), allowlists por tool,
  middleware; dependen de LangGraph.
- **Tokens:** **alto consumo por diseño** (planning, subagentes y filesystem
  añaden tokens; la propia documentación lo reconoce); prompt caching solo
  automático para Anthropic/Bedrock. Es justo lo contrario de la restricción de
  ARNES-010 (~7.900 tokens de referencia).
- **Modelos:** agnóstico (cualquier modelo con tool calling, incluido Ollama).

## Comparativa por criterios

| Criterio | A mano sobre LiteLLM | Google ADK | DeepAgents |
|----------|----------------------|------------|------------|
| **Encaje con los 26 agentes** | Directo: el registro existente se expone como catálogo cerrado; un adaptador fino lee `@register_agent` y genera el JSON schema | Reescritura: los 26 agentes hay que envolverlos en la convención de agentes/tools de ADK | Reescritura: envolver en tools LangChain + decidir qué middlewares (filesystem, subagentes) se dejan fuera |
| **Tools** | Catálogo cerrado por construcción: solo se registra lo que se decora | Catálogo abierto por defecto; hay que cerrarlo a mano y probarlo | Catálogo ancho (filesystem+shell+subagentes) que hay que **podar** activamente; más superficie que proteger |
| **Guardrails / control** | Gateway de permisos = una función Python entre bucle y tools (ARNES-005/011); 100 % determinista | Callbacks en 3 puntos + plugins: potentes pero ajenos; el control de ejecución lo orquesta ADK | HITL y middleware de LangGraph: potentes, pero toda tool pasa por la maquinaria del framework |
| **Coste de tokens** | Mínimo por construcción: el prompt es lo que tú metes; visor y caché = lecciones 16-17 de byoharness | `usage_metadata` gratis, pero reenvía prompt+tools en cada turno; caché hay que añadirla con plugins | Alto por diseño (planning, subagentes, filesystem); caché solo automática en Anthropic/Bedrock — choca con la referencia de ~7.900 tokens |
| **Coste de mantenimiento** | Bajo: solo se mantiene código propio, sin ciclos de release ajenos | Medio: dependencia viva (ADK 1.x→2.0 ya obligó a migrar) + capa de adaptación | Medio-alto: `deepagents` + `langchain` + `langgraph` + LangSmith si se usa |
| **Testing** | pytest directo: el bucle es una función pura pequeña, sin mocks de framework | Unit tests con callbacks internos que hay que parchear (los propios usuarios documentan parches para leer usage) | Tests sobre LangGraph (checkpointing, streaming); más superficie a simular |
| **Portabilidad entre proveedores** | LiteLLM (ya en el stack): Ollama, Groq, OpenAI con una llamada | `LiteLlm` lo permite; Gemini es el ciudadano de primera | Cualquier modelo con tool calling; pero arrastra el ecosistema LangChain |
| **Curva de aprendizaje** | Baja: es el propio código + currículo gratuito; el equipo ya conoce el arnés | Media: convenciones de ADK + migración del registro | Media-alta: LangGraph, middlewares, profiles, subagentes |

## Qué aporta cada framework por encima del baseline a mano

Para ser justos, cada framework resuelve piezas que el backlog planea hacer a
mano. Esto es lo que se compraría con cada uno:

| Pieza | Lo que aportaría ADK | Lo que aportaría DeepAgents | ¿Cuánto cuesta a mano? |
|-------|----------------------|-----------------------------|------------------------|
| Guardrails | Callbacks en 3 puntos del ciclo | HITL + middleware de permisos | Lección 02 de byoharness + ARNES-005/011; ~100 líneas de función de aprobación |
| Observabilidad de tokens | `usage_metadata` + plugin BigQuery | LangSmith tracing | Lección 16 (visor de tokens) + ARNES-004/010; contador en el propio bucle |
| Subagentes | Orquestación multi-agente | Aislamiento de contexto por defecto | Lección 11 + ARNES-007 (markdown, ya planeado) |
| Compactación | — | Summarization/offloading | Lección 07 + ARNES-007 (2 estrategias intercambiables) |
| Memoria / AGENTS.md | — | Carga de AGENTS.md + skills | Lección 15 + ARNES-007 (subagentes en markdown reutilizando `.opencode/agents/`) |
| Debug / UI | `adk web`, CLI | Deep Agents CLI, LangSmith | ARNES-003 (modo debug con payload exacto) — suficiente para este proyecto |
| Caché de prompt | Plugin propio | Automática solo en Anthropic/Bedrock | Lección 17 + ARNES-010 |

Ninguna de esas piezas justifica reescribir el registro de 26 agentes ni
arrastrar una dependencia de framework: todas están ya presupuestadas como
tickets (ARNES-003, 004, 005, 007, 010, 011) sobre el bucle propio.

## Elección justificada

**Hacerlo a mano sobre LiteLLM, siguiendo byoharness.dev.** Razones:

1. **El arnés ya existe.** Los 26 agentes, el registro por decorador y la puerta
   son código propio, documentado y testeado. Un framework obliga a envolver ese
   código en su convención (clases de ADK o tools LangChain) para obtener
   prestaciones que el backlog ya tiene en tickets. Es coste de adaptación sin
   ganancia neta.
2. **El requisito dominante es el control, y el control se paga con
   determinismo.** ARNES-005/010/011 piden una frontera de permisos, un tope de
   tokens y una auditoría *antes* de ejecutar cualquier tool. En un bucle propio
   esa frontera es una función Python entre el bucle y las tools; en un
   framework es middleware que vive dentro de su maquinaria y hay que aprender y
   vigilar su orden de ejecución.
3. **La restricción de coste es dura y los frameworks van en dirección
   contraria.** ARNES-010 compara contra ~7.900 tokens por mensaje. DeepAgents
   admite que planning+subagentes+filesystem aumentan el consumo; ADK reenvía el
   prompt estático en cada turno salvo que se añada caché. A mano, el prompt es
   exactamente lo que se construye, y el visor de tokens (lección 16) sale casi
   gratis.
4. **La portabilidad ya está resuelta en la capa correcta.** LiteLLM está en
   `pyproject.toml` y en producción (`rag_qwen.py`). ARNES-006 lo dice
   explícitamente: «NO se reescribe la capa de proveedor». Montar el bucle sobre
   LiteLLM hereda esa portabilidad sin arrastrar un framework por encima.
5. **Principio del proyecto: cero dependencias innecesarias.** AGENTS.md lo
   declara. `litellm` es la única dependencia nueva que el bucle necesita, y ya
   está.
6. **El riesgo del «reinventar la rueda» es acotado** porque byoharness.dev es
   un blueprint público y probado, y cada pieza que se construya tiene su ticket
   con criterios de aceptación verificables (ARNES-006, 007, 010, 011).

**Coste asumido:** implementar y mantener a mano lo que ADK/DeepAgents regalan
(guardrails, compactación, subagentes, caché). Se mitiga con el currículo de
byoharness, con tests por pieza y con el roadmap ya escrito en el backlog.

## Próximos pasos

| Ticket | Pieza que implementa sobre el bucle propio |
|--------|---------------------------------------------|
| ARNES-006 | Bucle REPL + bucle interno, catálogo cerrado de tools, capítulo 03 de byoharness cubierto con LiteLLM |
| ARNES-007 | Subagentes en markdown + compactación intercambiable |
| ARNES-003 | Modo debug (payload exacto al LLM) |
| ARNES-004 / ARNES-010 | Contador de tokens y tope de presupuesto por petición |
| ARNES-005 / ARNES-011 | Gateway de permisos y Security/Policy Layer |
| ARNES-008 | Evaluación del arnés propio contra el flujo determinista |

## Documentación relacionada

- `AGENTS.md` — protocolo del arnés actual (26 agentes, puerta, harness).
- `documentacion/arquitectura/agentes_ia.md` — arquitectura de los agentes.
- `featureslist.json` — ARNES-002, 005, 006, 007, 010, 011 (mismos criterios).
- Fuentes: [byoharness.dev](https://byoharness.dev), [adk.dev](https://adk.dev),
  [langchain-ai/deepagents](https://github.com/langchain-ai/deepagents),
  `pyproject.toml` (`litellm>=1.93.0`), `climasafeai/llm/rag_qwen.py`.
