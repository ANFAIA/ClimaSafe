# progress/ — memoria externa del arnés

Esta carpeta existe para resolver un problema concreto: **el teléfono
descompuesto entre agentes.** Cuando el líder lanza un subagente, ese subagente
arranca con el contexto vacío. Si el resultado de su trabajo solo vive en su
ventana de contexto, se pierde en cuanto termina.

La regla es: **todo subagente escribe su resultado en un fichero de esta
carpeta antes de devolver el control.** El siguiente agente lee `progress/` en
vez de releer el repositorio entero — menos tokens, menos degradación.

## Ficheros

**Nadie escribe aquí a mano.** El dueño de esta carpeta es el agente Python
`harness`; los agentes markdown le piden que escriba:

| Fichero | Se escribe con | Qué contiene |
|---------|----------------|--------------|
| `current.md` | `harness start` / `finish` / `block` | Estado **derivado**: la tarea en curso, o el índice si hay varias |
| `current-<dueño>.md` | `harness start` | Detalle de una tarea en curso: criterios, bitácora y bloqueos |
| `history.md` | `harness finish` | Append-only: features cerradas con su evidencia |
| `<AGENTE>-<FEATURE-ID>.md` | `harness record` | Resultado de una ejecución concreta |

```bash
uv run python -m agents --json run harness record \
  --agent explorer --id DATA-001 --verdict ok --content "<informe>"
```

## Varios asistentes a la vez: el candado es por dueño

Dos asistentes pueden trabajar en paralelo (uno en opencode, otro en Claude
Code) si cada uno se identifica al abrir su feature:

```bash
uv run python -m agents --json run harness start --id ARNES-013 --owner claude
uv run python -m agents --json run harness start --id DATA-004  --owner opencode
```

La regla: **un dueño, una feature abierta.** Se pueden tener varias features
`in_progress` a la vez si son de dueños distintos; abrir una segunda con el
mismo dueño se rechaza. Todas las features **sin** campo `owner` comparten un
mismo dueño implícito —el legado—, así que quien no usa `--owner` ve el
comportamiento de siempre: una sola tarea abierta.

Y por eso `current.md` dejó de ser un fichero que escribe quien pasa por ahí:

- **`current.md` es estado derivado** de `featureslist.json`. `harness` lo
  regenera entero en cada `start`, `finish` y `block`, mirando qué hay
  `in_progress` — no quién llamó. Con 0 tareas queda en idle; con 1, es la
  ficha completa de siempre (mismo formato, mismos campos); con 2 o más, es un
  índice con feature, dueño, fecha de inicio y ruta al detalle de cada una.
- **Cada tarea en curso tiene su ficha.** Ahí va el detalle. El fichero se
  llama por el dueño si lo hay (`current-claude.md`) y por el id de la feature
  si no (`current-data-004.md`), así que una tarea abierta sin `--owner`
  tampoco pierde su objetivo ni sus criterios cuando el índice se activa.
  `finish` y `block` retiran **solo** la ficha de esa feature; las demás no se
  tocan. Antes `finish` reescribía `current.md` en idle a lo bruto y borraba el
  trabajo del otro asistente.
- Una ficha que ya existe no se sobrescribe: la bitácora que haya escrito su
  dueño se queda.
- El nombre del dueño se normaliza (minúsculas, sin espacios alrededor) para el
  candado, y se sanea a `[a-z0-9_-]` para el nombre de fichero.

`current.md` existe siempre —`init.sh` lo comprueba—, tenga o no tareas
abiertas.

## Formato de los ficheros de subagente

Nombre: `explorer-DATA-001.md`, `implementer-FEAT-001.md`, `reviewer-FEAT-001.md`.
La cabecera (fecha y veredicto) la pone `harness record`; tú aportas el cuerpo:

```markdown
## Qué hice
## Qué encontré / qué cambié
## Evidencia
(comandos ejecutados y su salida — no "los tests pasan", sino la salida real)
## Qué falta
```

## Reglas

1. **Evidencia, no afirmaciones.** «Los tests pasan» no vale; pega la salida de
   `./init.sh` o de `pytest`. El arnés existe para que los agentes demuestren su
   trabajo, no para que lo declaren.
2. **Un fichero por ejecución.** No sobrescribas el resultado de otro subagente.
3. **Corto.** Si un fichero de progreso pasa de ~100 líneas, resume: el objetivo
   es ahorrar contexto, no fabricar más.
4. **`current.md` no se edita: se regenera.** Al cerrar la feature, su resumen
   se añade a `history.md` y `current.md` refleja lo que quede abierto (idle si
   no queda nada). Los ficheros de subagente se pueden borrar cuando la feature
   está en `history.md`.

## Las otras dos memorias

`progress/` no es la única memoria del proyecto, y no se pisan:

| Dónde | Dueño | Plazo |
|-------|-------|-------|
| `progress/` | `harness` | La feature en curso y el histórico de features |
| `agents/workspace/memory/` | `memory` | Trayectorias de ejecución de agentes |
| `vault/` | `knowledge` | Conocimiento estable del proyecto y sus datos |

Un hallazgo duradero (por qué se eligió un modelo, qué significa una columna)
no vive aquí: pídele a `knowledge` que lo escriba en el vault. Esto es memoria
de trabajo, no de archivo.

## Buscable, no solo legible

Esta carpeta y `featureslist.json` entran en el índice semántico del proyecto,
así que el histórico se consulta en lenguaje natural en vez de releyéndolo:

```bash
make index-rag
uv run python -m agents --json run rag search --query "¿por qué elegimos K=4?"
uv run python -m agents --json run doc search --query "qué se decidió sobre las features"
```

Ejecuta `make index-rag` después de cerrar una feature — si no, el histórico
nuevo no está en el índice.

Si algún día esto se queda corto, lo único que cambia es dónde escribe el
agente `harness` (SQLite, DuckDB, un backend remoto compartido). El protocolo
de `AGENTS.md` no cambia: solo cambia el soporte.
