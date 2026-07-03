# agents/ — Sistema de agentes de este template

Sistema de agentes especializados, tipo plugin, integrado en el template
`dskit`. Cada proyecto generado con Copier a partir de este template incluye
esta carpeta completa y funcional desde el primer commit.

## Filosofía

- **No es un chatbot.** Cada agente ejecuta tareas reales con herramientas
  (git, docker, pandas, sklearn...), no conversa. El punto de entrada en
  lenguaje natural (`Orchestrator.dispatch`) es una capa fina de ruteo por
  encima de acciones deterministas, no un envoltorio de un LLM.
- **Agnóstico de proveedor de IA.** `agents/` es Python puro, invocado por
  CLI (`python -m agents ...`). No importa el SDK de Anthropic, ni el de
  ningún otro proveedor, ni depende de ningún formato propio de una
  herramienta concreta (Claude Code, Cursor, opencode...). Cualquier agente
  de codificación que pueda ejecutar comandos de shell puede usar este
  sistema exactamente igual.
- **Una responsabilidad por agente.** `GitAgent` no toca datos, `DataAgent`
  no toca Docker. Si una tarea necesita dos agentes, se orquestan en
  secuencia (ver `DocumentationAgent.update_changelog`, que llama a
  `GitAgent` internamente) — no se crea un agente todopoderoso.
- **Cero dependencias innecesarias.** Todo lo que puede hacerse con la
  librería estándar de Python, se hace ahí (`subprocess`, `ast`, `sqlite3`,
  `urllib`, `importlib.metadata`...). Las dependencias reales del template
  (pandas, numpy, scikit-learn, matplotlib, joblib) se reutilizan; nada
  nuevo se añade a `pyproject.toml`.
- **Los agentes conocen este template**, no son genéricos: saben que el
  código vive en `{{ project_slug }}/`, que los datasets viven en
  `data/raw|interim|processed/`, que los modelos se guardan en
  `models/*.joblib`, etc. (ver `agents/context.py`).

## Arquitectura

```
agents/
├── __init__.py            # API pública: Orchestrator, BaseAgent, AgentResult...
├── __main__.py             # permite `python -m agents ...`
├── cli.py                  # CLI (list / describe / run / ask / tools)
├── config.py                # lee .copier-answers.yml -> ProjectConfig
├── context.py                # SharedContext: rutas + config, calculado una vez
├── orchestrator.py            # rutea lenguaje natural -> agente por capabilities
├── exceptions.py               # jerarquía de excepciones propia
├── core/
│   ├── base_agent.py            # BaseAgent, AgentResult
│   └── registry.py               # @register_agent + auto-descubrimiento
├── tools/                          # herramientas reutilizables, una responsabilidad cada una
│   ├── git_tool.py · docker_tool.py · process_tool.py · filesystem_tool.py
│   ├── data_io_tool.py · dataframe_analysis_tool.py · sklearn_tool.py
│   ├── vision_tool.py · duckdb_tool.py · sqlite_tool.py · rest_tool.py
│   └── code_analysis_tool.py
├── agents/                          # los 7 agentes iniciales (+ plantilla de ejemplo)
│   ├── git_agent.py · data_agent.py · graph_agent.py · docker_agent.py
│   ├── ml_agent.py · review_agent.py · documentation_agent.py
│   └── _template_agent.py            # plantilla — no se auto-registra (prefijo `_`)
├── external/                          # agentes de terceros / tuyos, fuera del núcleo
│   ├── README.md
│   └── __init__.py
└── prompts/                            # una ficha markdown por agente (rol, cuándo usarlo)
```

### Flujo de una petición

```
Orchestrator.dispatch("genera el changelog desde el último tag")
  -> select_agent(): puntúa cada agente registrado con can_handle(query)
  -> agente ganador: GitAgent (keywords: "changelog", "git"...)
  -> agent.run("generate_changelog", since_tag=None)
  -> devuelve AgentResult(success, data, warnings)
```

Si prefieres evitar el ruteo por palabras clave, invoca el agente
directamente — es la vía recomendada cuando ya sabes qué necesitas:

```python
from agents.agents.git_agent import GitAgent
result = GitAgent().suggest_commit_message()
print(result.data["suggested_message"])
```

## Uso desde la CLI

```bash
uv run python -m agents list
uv run python -m agents describe git
uv run python -m agents run git suggest_commit_message
uv run python -m agents run data eda_report --filename dataset.csv --target-col target
uv run python -m agents ask "revisa el Dockerfile"
uv run python -m agents tools
```

## Instalar este mismo sistema en otro proyecto

Existe una Skill de Claude (`dskit-agents-installer`, fuera de este
template) que empaqueta esta carpeta y la instala en cualquier proyecto
Python, resolviendo `project_slug` automáticamente. Es un artefacto
separado de este template — pregunta por ella si la necesitas, no vive
dentro de `agents/` porque no tiene sentido que un proyecto ya generado
cargue con su propio instalador.

## Los 7 agentes iniciales

| Agente | Responsabilidad | Herramientas que usa |
|---|---|---|
| `git` | Conventional Commits, changelog, release notes, breaking changes, resumen de PR | `git_tool` |
| `data` | EDA: constantes, cardinalidad, missing, outliers, fuga de información, correlaciones | `data_io_tool`, `dataframe_analysis_tool` |
| `graph` | Audita `reports/figures/`: figuras vacías, aspect ratio inusual | `vision_tool` |
| `docker` | Lint de Dockerfile, validación de docker-compose | `docker_tool` |
| `ml` | Inspección de modelos `.joblib`, importancia de variables, overfitting | `sklearn_tool` |
| `review` | Funciones largas, demasiados argumentos, `except` desnudos, duplicación estructural | `code_analysis_tool` |
| `documentation` | README ↔ Makefile desincronizados, actualiza CHANGELOG.md, genera docs Sphinx | `filesystem_tool`, `process_tool`, agente `git` |

Cada agente documenta en su propio docstring qué responsabilidades de la
lista original están implementadas y cuáles quedan como extensión (p. ej.
`GitAgent.detect_breaking_changes` solo mira mensajes de commit, no el diff
de la API pública — está señalado explícitamente en su `AgentResult.warnings`).

## Cómo extender el sistema

### Añadir un agente nuevo

1. Copia `agents/agents/_template_agent.py` a `agents/agents/mi_agente.py`
   (quita el `_` inicial).
2. Define `name`, `description`, `capabilities` y los métodos de acción.
3. Nada más — el registro y la CLI lo descubren solos.

### Añadir una herramienta nueva

Créala en `agents/tools/`, decórala con `@register_tool("nombre")` si quieres
que aparezca en `python -m agents tools`, e impórtala desde el agente que la
necesite. Nunca dupliques una herramienta existente entre agentes.

### Agentes externos

`agents/external/` acepta dos vías (ver `agents/external/README.md`):
1. **Un archivo suelto**: mismo patrón `@register_agent`, se auto-descubre.
2. **Un paquete pip instalado** que expone un entry point del grupo
   `dskit.agents` en su propio `pyproject.toml` — útil si el agente externo
   tiene dependencias propias o lo compartes entre varios proyectos.

Ninguna de las dos vías requiere tocar `orchestrator.py`, `cli.py` ni ningún
otro agente.

## Extender el ruteo del Orchestrator

`Orchestrator.select_agent` usa una heurística de palabras clave
(`BaseAgent.can_handle`), determinista a propósito (ver Filosofía). Si más
adelante quieres un ruteo basado en un LLM (de cualquier proveedor —
Anthropic, OpenAI, un modelo local...) decidiendo qué agente usar, ese único
método es el punto de extensión — no hace falta tocar ningún agente ni el
resto del `Orchestrator`.

---

## Agentes externos y lecturas recomendadas

Nada de lo que sigue viene integrado por defecto: son punteros a proyectos
de terceros que pueden servir de inspiración, o como base para un agente en
`agents/external/`. Las descripciones están verificadas contra la
documentación/README de cada proyecto en el momento de escribir esto — aun
así, estos repos cambian rápido, conviene revisar el estado actual antes de
depender de alguno.

### Frameworks de skills para agentes de codificación

- **[obra/superpowers](https://github.com/obra/superpowers)** — metodología
  de desarrollo compuesta por skills encadenadas (brainstorming → writing-plans
  → ejecución con revisión en dos fases). Se instala como plugin de Claude
  Code (`/plugin install superpowers@claude-plugins-official`) o vía
  `npx skills add`. La skill
  [`brainstorming`](https://www.skills.sh/obra/superpowers/brainstorming) en
  concreto fuerza una fase de diseño con aprobación explícita antes de
  escribir código — encaja bien como disciplina previa a pedirle a un agente
  de este sistema (p. ej. `review`) que toque código.
- **[mattpocock/skills](https://github.com/mattpocock/skills)** — colección
  de skills más pequeñas y componibles (no una metodología monolítica). La
  skill `teach` convierte el directorio actual en un espacio de aprendizaje
  con estado (`MISSION.md`, lecciones HTML numeradas, registro de progreso) —
  pensada para aprender un tema a lo largo de varias sesiones, no directamente
  relacionada con `dskit`, pero reutilizable en cualquier proyecto.
- **skill-creator** (Anthropic,
  [skills.sh/anthropics/skills/skill-creator](https://www.skills.sh/anthropics/skills/skill-creator)) —
  la skill oficial de Anthropic para crear otras skills (estructura de
  `SKILL.md`, cuándo separar en `scripts/`/`references/`/`assets/`, cómo
  redactar la descripción para que el auto-descubrimiento la dispare bien).
  Es la misma que usa este asistente internamente; te la he dejado como
  archivo aparte junto a este proyecto (ver el mensaje de chat) para que la
  tengas disponible sin depender de que seas usuario de Claude Code.

### Memoria a largo plazo para agentes

Ninguno de estos es "memoria para `agents/`" tal cual — son piezas de
infraestructura que podrías conectar si construyeras un agente conversacional
por encima de este sistema (algo que, recuerda, este template evita a
propósito, ver Filosofía).

- **[topoteretes/cognee](https://github.com/topoteretes/cognee)** — motor de
  memoria basado en un grafo de conocimiento auto-alojado (pipeline
  "Extract, Cognify, Load"); combina búsqueda vectorial y de grafo. Se usa en
  ~6 líneas de Python (`cognee.add` / `cognee.cognify` / `cognee.search`) o
  como plugin de Claude Code.
- **[redis/agent-memory-server](https://github.com/redis/agent-memory-server)** —
  servidor de memoria (API REST + servidor MCP) sobre Redis, con memoria de
  trabajo (por sesión) y memoria a largo plazo con extracción automática de
  temas/entidades vía LLM. Soporta OpenAI, Anthropic y otros proveedores vía
  LiteLLM.
- **[FalkorDB/FalkorDB](https://github.com/FalkorDB/FalkorDB)** — base de
  datos de grafos (OpenCypher) optimizada para GraphRAG, usa matrices dispersas
  (GraphBLAS) en vez del modelo de almacenamiento habitual de un grafo. Sirve
  como backend de grafo para sistemas de memoria tipo cognee/Mem0 (hay un
  plugin oficial `FalkorDB/mem0-falkordb`).
- **[supermemoryai/supermemory](https://github.com/supermemoryai/supermemory)** —
  API de memoria/contexto para agentes, con integraciones ya hechas para
  LangChain, LangGraph, OpenAI Agents SDK y la herramienta de memoria de
  Claude, entre otras.
- **[EverMind-AI/EverOS](https://github.com/EverMind-AI/EverOS)** — capa de
  memoria local-first y "Markdown-nativa" (los datos se guardan como
  `.md` + SQLite + LanceDB, sin servicios externos obligatorios), con
  recuperación híbrida (BM25 + vectorial). **Aviso explícito**: en el
  momento de escribir esto, la actividad del repositorio es de apenas unos
  días — es un proyecto extremadamente nuevo, sin trayectoria que evaluar
  todavía. No lo trates como una pieza probada en producción sin revisar tú
  mismo su estado actual.

### Deep research

- **[langchain-ai/local-deep-researcher](https://github.com/langchain-ai/local-deep-researcher)** —
  asistente de investigación web completamente local (LangGraph + Ollama),
  itera búsqueda → resumen → reflexión hasta producir un informe con fuentes.
  LangChain mantiene también
  [`open_deep_research`](https://github.com/langchain-ai/open_deep_research),
  su versión no local con más proveedores de LLM/búsqueda — no estoy seguro
  de cuál de las dos está más activa en este momento, mejor revisar ambos
  repos antes de elegir.
- Los artículos que enlazaste (*Building agent memory with knowledge
  graphs* en theneuralmaze.substack.com, la comparativa de deep research en
  trilogyai.substack.com, y la lista
  [DavidZWZ/Awesome-Deep-Research](https://github.com/DavidZWZ/Awesome-Deep-Research))
  no los he podido leer en profundidad en esta sesión — te los enlazo tal
  cual, sin resumir contenido que no he verificado.

### Bucle de investigación autónoma

- **[karpathy/autoresearch](https://github.com/karpathy/autoresearch)** —
  de Andrej Karpathy: un agente modifica el código de entrenamiento
  (`train.py`), entrena 5 minutos en una única GPU, mide una métrica, hace
  `git commit` si mejora o revierte si empeora, y repite en bucle sin
  intervención humana. El patrón de fondo (proponer → medir → conservar si
  mejora, descartar si no, con git como historial verificable) es
  trasladable fuera de ML: es, básicamente, la misma idea que
  `MLAgent.check_overfitting` podría usar como bucle automático en vez de
  una comprobación puntual — no está implementado así en este template, es
  una dirección de extensión razonable si te interesa.

### Sobre `IBM/drop-agent`

No he podido encontrar un repositorio público con ese nombre exacto en
GitHub — es posible que el nombre haya cambiado, que sea un repo privado, o
que me esté fallando la búsqueda. Antes de construir nada sobre él, confirma
la URL exacta.
