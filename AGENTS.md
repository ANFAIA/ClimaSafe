# Sistema de Agentes — ClimaSafeAI

Este proyecto incluye un sistema de 26 agentes que automatizan todo el ciclo de
desarrollo: desde análisis de datos hasta release, pasando por revisión de
código, tests, dependencias y despliegue.

> **Git en este repo:** los commits los decide el humano y los hace el agente
> `git` — Conventional Commits, **sin** línea de co-autoría ni pie de ninguna
> herramienta de IA:
>
> ```bash
> uv run python -m agents --json run git suggest_commit_message   # propone el mensaje
> uv run python -m agents --json run git commit_with_changelog --message "<msg>"
> ```
>
> Ningún asistente ejecuta `git` ni `gh` por su cuenta ni por iniciativa propia
> — ni siquiera para aplicar o deshacer un cambio. Se editan los ficheros, se
> dejan sin commitear, y el humano decide cuándo se commitea.

---

# Protocolo del arnés — LEE ESTO PRIMERO

Este fichero es el punto de entrada. Todo agente que trabaje en este proyecto
sigue estos pasos **en orden**, antes de escribir una sola línea de código.

```
1. ./init.sh                    ¿el entorno está sano?   si no → PARA
2. progress/current.md          ¿hay trabajo a medias?   si sí → retómalo
3. featureslist.json            primera feature pendiente con deps en done
4. marcar in_progress           en featureslist.json + rellenar current.md
5. delegar                      explorer → implementer → reviewer
6. verificar                    ./init.sh en verde + criterios uno a uno
7. done                         featureslist.json + resumen en history.md
```

## La regla que no se salta

**Ninguna feature se marca `done` sin que `./init.sh` pase en verde.**

Y no es una instrucción: es código. `harness finish` ejecuta la puerta antes de
tocar el backlog y devuelve `success=false` si está en rojo o si no le pasas
evidencia. No hay forma de rodearlo pidiéndoselo amablemente al modelo — la
única sería editar el JSON a mano, y eso está prohibido explícitamente.

`./init.sh` verifica el entorno, los ficheros del arnés, el formato del backlog
y ejecuta la suite de tests. Si sale `ENTORNO BLOQUEADO`, el agente para y lo
reporta — no implementa encima de un proyecto roto ni arregla el arnés por su
cuenta.

```bash
./init.sh            # verificación completa
./init.sh --quick    # solo estructura, sin tests (no vale para cerrar features)
./init.sh --json     # salida estructurada para consumo por agentes

uv run python -m agents --json run harness gate    # lo mismo, vía agente
```

## Piezas del arnés

| Fichero | Qué es |
|---------|--------|
| `AGENTS.md` | Este fichero. Punto de entrada y reglas del juego |
| `CLAUDE.md` | Puntero a este fichero para Claude Code. No duplica nada |
| `init.sh` | La puerta: decide si se puede trabajar. Exit != 0 → parar |
| `featureslist.json` | Backlog: qué hay que hacer, con criterios de aceptación |
| `progress/current.md` | Estado vivo de la feature en curso |
| `progress/history.md` | Append-only: lo cerrado y con qué evidencia |
| `progress/<agente>-<ID>.md` | Resultado de cada subagente |
| `.opencode/agents/*.md` | Definición de cada agente del arnés |

## Los agentes del arnés

| Agente | Capa | Hace |
|--------|------|------|
| `lider` | razona (primary) | Orquesta el ciclo. No escribe código de producto |
| `explorer` | razona (subagent) | Investiga en **solo lectura** y responde una pregunta |
| `implementer` | razona (subagent) | Implementa **una** feature con sus tests |
| `reviewer` | razona (subagent) | Aprueba o rechaza tras ejecutar la puerta |
| `harness` | ejecuta (Python) | **Único** que escribe `featureslist.json` y `progress/` |

Un recurso, un dueño: nadie edita el backlog ni el progreso a mano; todo pasa
por `harness`. El `implementer` es el único que toca el código de producto.

```bash
uv run python -m agents --json run harness next          # ¿qué toca?
uv run python -m agents --json run harness start --id DATA-001
uv run python -m agents --json run harness record --agent explorer --id DATA-001 --content "..."
uv run python -m agents --json run harness finish --id DATA-001 --evidence "$(make test 2>&1 | tail -5)"
uv run python -m agents --json run harness block --id DATA-001 --reason "falta el dataset"
uv run python -m agents --json run harness add --id API-002 --title "..." --criteria "a;b"
```

## Memoria externa: por qué existe `progress/`

La ventana de contexto se degrada mucho antes de llenarse. Por eso el estado
del trabajo vive en ficheros, no en la conversación:

- **Al lanzar un subagente, no le heredes contexto.** Dale el ID de la feature,
  sus criterios y las rutas que necesita. Nada más.
- **Todo subagente registra su resultado con `harness record` antes de devolver
  el control.** Si solo lo dice en su respuesta, se pierde.
- **El siguiente agente lee `progress/`, no el repositorio entero.**

Las memorias del proyecto no se pisan:

| Dónde | Dueño | Plazo |
|-------|-------|-------|
| `progress/` | `harness` | La feature en curso y el histórico de features |
| `agents/workspace/audit/` | `audit` | Log de ejecuciones de agentes |
| `graphify-out/` + bóveda Obsidian | `knowledge` | Conocimiento estable del proyecto y sus datos |
| `documentacion/` | humano | Notas de trabajo y próximos pasos |

Detalles del formato en `progress/README.md`.

## Evidencia, no afirmaciones

Un agente no declara que algo funciona: lo demuestra. Cada criterio de
aceptación se cierra pegando la **salida real** del comando que lo prueba.
«Los tests pasan» sin la salida de `pytest` es motivo de rechazo automático.

## El arnés se automejora

Estos ficheros son parte del repositorio, así que se corrigen como cualquier
otro código. Si un fallo se cuela dos veces:

- ¿Es una comprobación automatizable? → a `init.sh`, y deja de depender de que
  alguien se acuerde.
- ¿Es una regla del proyecto? → a este fichero.
- ¿Es un criterio de revisión? → a `.opencode/agents/reviewer.md`.

Deja constancia del cambio en `progress/history.md`.

## Arranque

```bash
./init.sh                                    # verifica que se puede trabajar
make init                                    # lo mismo, vía Makefile
make harness-check                           # solo estructura del arnés
```

Y en el asistente, para arrancar el ciclo:

> Lee `AGENTS.md` y sigue el protocolo: ejecuta `./init.sh`, lee `progress/` y
> elige la primera feature pendiente.

---

## Filosofía

- **No es un chatbot.** Cada agente ejecuta tareas reales (git, docker, tests...),
  no conversa. El ruteo por lenguaje natural es una capa fina sobre acciones
  deterministas.
- **Agnóstico de proveedor de IA.** Python puro, sin SDK de ningún proveedor.
  Cualquier agente de codificación que ejecute comandos de shell puede usarlo.
- **Una responsabilidad por agente.** `GitAgent` no toca datos, `DataAgent` no
  toca Docker. Si una tarea necesita dos agentes, se orquestan en secuencia.
- **Cero dependencias innecesarias.** Usa la stdlib donde puede; reutiliza las
  dependencias del proyecto (pandas, sklearn, etc.).
- **Conoce este template.** Los agentes saben que el código vive en
  `climasafeai/`, los datasets en `data/`, los modelos en `models/`.

## Principios de comportamiento

Estos principios aplican a cualquier agente de codificación (Claude Code,
Codex, Cursor, Gemini, Cline, Copilot, opencode...) que trabaje en este
proyecto. No dependen de un proveedor ni de una herramienta concreta.

### Piensa antes de codear

No asumas. No escondas dudas. Superficia los trade-offs.

- Si una instrucción es ambigua, presenta múltiples interpretaciones en vez
  de elegir una en silencio
- Si algo no está claro, pregunta — no inventes
- Si existe un enfoque más simple, dilo
- Para cuando estés confuso: nombra qué no entiendes y pide aclaración

### Simplicidad primero

El mínimo código que resuelve el problema. Nada especulativo.

- Nada de funcionalidades que no se pidieron
- Nada de abstracciones para código que se usa una vez
- Nada de "flexibilidad" o "configurabilidad" no solicitada
- Nada de manejo de errores para escenarios imposibles
- Si 200 líneas pueden ser 50, reescríbelas

### Cambios quirúrgicos

Toca solo lo que debes. No mejoren código ajeno.

- No "mejores" código, comentarios o formato adyacente
- No refactorices cosas que no están rotas
- Respeta el estilo existente, aunque lo harías diferente
- Si ves código muerto no relacionado, menciónalo — no lo borres
- Al borrar código tuyo, elimina imports/variables/funciones que tus cambios
  dejaron sin usar. No toques código muerto preexistente

### Ejecución guiada por objetivos

Define criterios de éxito. Itera hasta verificarlos.

En vez de decir "añade validación", escribe "escribe tests para entradas
inválidas, luego haz que pasen". En vez de "arregla el bug", escribe
"escribe un test que lo reproduzca, luego haz que pase".

Para tareas multi-paso, usa un plan con verificación por paso:

```
1. [Paso] → verificar: [cómo]
2. [Paso] → verificar: [cómo]
```

### Concisión

Sé breve. Di lo mismo con la mitad de palabras.

- Elimina relleno ("I'd be happy to help", "Sure!", "Let me take a look")
- Preserva el contenido técnico: código, comandos, rutas, errores
- Usa frases cortas y directas. Un fragmento vale si es claro
- No repitas lo que el usuario ya sabe
- Una línea vale más que un párrafo

El test: si un ingeniero senior diría "esto es demasiado complicado",
simplifícalo. Si una respuesta puede perder la mitad de palabras sin perder
información, hazlo.

---

## Referencia (bajo demanda)

El catálogo — los 26 agentes con su responsabilidad, sus acciones y sus
parámetros — vive en `.opencode/orchestrator.md` para no ocupar contexto en
cada sesión. Es la **única** copia del catálogo: no lo dupliques aquí.

```bash
uv run python -m agents list                # los agentes, desde la CLI
uv run python -m agents describe <agente>   # acciones y contrato de uno
uv run python -m agents --json ask "<lo que sea>"   # ruteo automático
```

El ciclo detallado del arnés está en `agents/prompts/harness_workflow.md`.
