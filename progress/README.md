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
| `current.md` | `harness start` / `harness finish` | Feature en curso, criterios, bitácora y bloqueos |
| `history.md` | `harness finish` | Append-only: features cerradas con su evidencia |
| `<AGENTE>-<FEATURE-ID>.md` | `harness record` | Resultado de una ejecución concreta |

```bash
uv run python -m agents --json run harness record \
  --agent explorer --id DATA-001 --verdict ok --content "<informe>"
```

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
4. **`current.md` se vacía al cerrar la feature**, y su resumen se añade a
   `history.md`. Los ficheros de subagente se pueden borrar cuando la feature
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
