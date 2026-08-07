# harness — dueño del backlog y del progreso

Ejecuta la parte mecánica del arnés. No decide nada: los agentes markdown
(`lider`, `explorer`, `implementer`, `reviewer`) razonan, este escribe.

## Acciones

| Acción | Qué hace |
|--------|----------|
| `status` | Recuento del backlog + qué está in_progress + qué es elegible |
| `next` | La feature que toca (in_progress, o la primera con deps en done) |
| `start --id <ID> [--owner <quién>]` | Abre la feature y vuelca sus criterios en `progress/` |
| `gate [--quick true]` | Ejecuta `./init.sh` y devuelve el veredicto estructurado |
| `finish --id <ID> --evidence "<salida real>"` | Cierra la feature y escribe el histórico. Commitea automáticamente las rutas de `--changes` + los ficheros del cierre; si no queda nada que commitear, avisa y no commitea |
| `block --id <ID> --reason "<motivo>"` | Marca bloqueada |
| `record --agent <a> --id <ID> --content "<informe>"` | Guarda `progress/<a>-<ID>.md` |
| `add --id <ID> --title "<t>" --criteria "a;b;c"` | Añade feature al backlog |

```bash
uv run python -m agents --json run harness next
uv run python -m agents --json run harness start --id DATA-001
uv run python -m agents --json run harness gate
uv run python -m agents --json run harness finish --id DATA-001 --evidence "$(make test 2>&1 | tail -5)"
```

## Cierre con release (GIT-001)

`finish` encadena al cierre el flujo de release ligero en un único comando:

1. **Bump de versión patch** en `pyproject.toml` y `README.md` (badge `Version-X-green`
   y línea `**Versión:** X`) — delega en el agente `documentation`.
2. **Commit automático** Conventional Commits sin línea de co-autoría, con
   subject `cierra <ID>` — delega en el agente `git`.

Si no hay versión parseable o no hay cambios que resumir, se avisa en
`warnings` y la feature se cierra igualmente — el release ligero nunca bloquea
el cierre.

### Commit automático del cierre (ARNES-014)

Al cerrar una feature, `finish` commitea automáticamente —**sin flag**: se
intenta siempre; el único caso en que no se commitea es que no quede nada que
commitear, y entonces se avisa en `warnings` y la feature se cierra igual:

```bash
uv run python -m agents --json run harness finish --id DATA-001 \
  --evidence "$(make test 2>&1 | tail -5)" \
  --changes "climasafeai/features/x.py;climasafeai/models/y.py"
```

- Las rutas de `--changes` van separadas por `;`.
- El commit incluye **solo** esas rutas más los ficheros del propio cierre
  (`featureslist.json`, `progress/`, `pyproject.toml`, `README.md`), con el id
  de la feature como subject y sin co-autoría.
- Si `--changes` viene vacío o trae rutas que no existen → no se commitea nada
  y se avisa en `warnings`; la feature se cierra igualmente.
- Si el árbol trae cambios ajenos al ticket (trabajo de otro dueño o de otro
  ticket), no se para: se commitea solo lo del ticket, el resto se queda sin
  commitear (acumulado para el siguiente) y se avisa de qué se deja fuera.
- Si tras el filtrado no queda nada staged → no se commitea y se avisa.
- Los fallos del commit nunca bloquean el cierre: van a `warnings` y la feature
  queda `done` igualmente.

## Lo que rechaza

- **`finish` sin `./init.sh` en verde** → `success=false`. La regla del arnés es
  código, no un consejo.
- **`finish` sin `evidence`** → devuelve `needs`. Una afirmación no es evidencia.
- **`start` con otra feature abierta del mismo `--owner`** → una cosa a la vez
  por dueño. Dueños distintos sí trabajan en paralelo; sin `--owner` todos
  comparten el dueño legado y el candado es el de siempre. Detalles del formato
  de `progress/` (incluido `current-<dueño>.md`) en `progress/README.md`.
- **`start` con `depends_on` sin cerrar** → primero las dependencias.
- **`add` con `depends_on` inexistente** → el backlog no se corrompe.

## Por qué existe

Editar JSON a mano desde un prompt se rompe: comas, ids duplicados, estados
inventados, un `done` que nadie verificó. Este agente hace esas operaciones
de forma determinista y es el único dueño de `featureslist.json` y `progress/`.

Ver el ciclo completo: `skill harness_workflow`.

<!-- BEGIN AUTOGEN — lo regenera `make prompts-sync`; no lo edites a mano -->

## Acciones

| Acción | Argumentos |
|--------|------------|
| `run harness status` | — |
| `run harness next` | — |
| `run harness start` | `--id`, `--owner` |
| `run harness finish` | `--id`, `--evidence`, `--changes`, `--decisions`, `--pending` |
| `run harness block` | `--id`, `--reason` |
| `run harness record` | `--agent`, `--id`, `--content`, `--verdict` |
| `run harness gate` | `--quick` |
| `run harness add` | `--id`, `--title`, `--description`, `--criteria`, `--depends_on` |

## Límites

**Rol.** Dueño mecánico del arnés: mantiene el backlog y el progreso, y ejecuta la puerta init.sh.

**No hace:**
- decidir QUÉ feature toca ni cómo implementarla → eso lo razonan los agentes markdown del arnés (lider, explorer, implementer, reviewer)
- escribir código del producto → 'refactor' y el implementer
- ejecutar los tests por su cuenta → los ejecuta init.sh, o el agente 'test'
- cerrar una feature sin evidencia → devuelve needs, nunca la da por buena

**Necesita que le den:** el id de la feature; la evidencia real de verificación para cerrarla

**Escribe en (nadie más toca esto):** featureslist.json, progress/

**Se apoya en:** plan, test, review, audit

<!-- END AUTOGEN -->
