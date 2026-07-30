# Harness Workflow — Ciclo del arnés

Capa que gobierna a las demás: decide **qué** se hace y **cuándo está hecho**.
Los workflows de dominio (`data`, `ml`, `dev`...) dicen **cómo**.

## Pipeline
```
init.sh → progress/ → featureslist.json → explorer → implementer → reviewer → done
   │                                                                      │
   └────────────────── si init.sh falla: PARAR ───────────────────────────┘
```

## Paso a paso

| Paso | Comando | Quién | Verificación |
|------|---------|-------|-------------|
| Puerta | `run harness gate` (o `make init`) | `harness` | `success=true` = se puede trabajar |
| Elegir | `run harness next` | `harness` | retoma lo abierto o la primera con deps en `done` |
| Abrir | `run harness start --id <ID>` | `harness` | criterios volcados en `current.md` |
| Investigar | subagente `explorer` (solo lectura) | `explorer` | `run harness record --agent explorer` |
| Descomponer | `run plan brief --text "<feature>"` | `plan` | orden de trabajo con pasos y agentes |
| Implementar | subagente `implementer` | `implementer` | código + tests + `harness record` |
| Revisar | subagente `reviewer` | `reviewer` | `harness gate` en verde + criterios uno a uno |
| Cerrar | `run harness finish --id <ID> --evidence "..."` | `harness` | **rechaza** si la puerta falla o no hay evidencia |

El `lider` decide en qué orden pasa todo esto; `harness` lo ejecuta. Ningún
agente edita `featureslist.json` ni `progress/` a mano.

## Las tres reglas

1. **Nada se cierra sin `./init.sh` en verde.** Es la única definición de «hecho»
   — y la aplica `harness finish` en código, no un prompt.
2. **Evidencia, no afirmaciones.** Cada criterio se cierra con la salida real del
   comando que lo prueba. `finish` sin `--evidence` devuelve `needs`.
3. **Todo subagente registra con `harness record` antes de devolver el control.**
   Lo que solo vive en la ventana de contexto se pierde.

## Contexto: qué NO hacer

- No heredes el contexto del líder a los subagentes. Pásales el ID de la
  feature, sus criterios y las rutas. Nada más.
- No releas el repositorio en cada subagente. `progress/` existe justo para eso.
- No pases el contenido de un informe de subagente a otro: pasa la **ruta**.

## Cómo engancha con los agentes Python

El arnés decide y verifica; el trabajo determinista lo hacen los agentes:

| Necesidad del arnés | Agente | Acción |
|---------------------|--------|--------|
| Backlog y progreso (todo cambio de estado) | `harness` | `status`, `next`, `start`, `finish`, `block`, `record`, `add`, `gate` |
| Descomponer una feature en pasos | `plan` | `brief`, `answer`, `execute` |
| Comprobar que la suite pasa | `test` | `run_tests`, `coverage_summary` |
| Revisar el código de una feature | `review` | `review_package` |
| Diagnóstico antes de abrir trabajo | `doctor` | `python -m agents doctor` |
| Saber si el equipo va bien | `audit` | `suggest` |
| Contexto de sesiones anteriores | `audit` | `report`, `failures` |
| Buscar dónde está algo | `doc` | `search`, `graph_query` |

```bash
uv run python -m agents --json run plan brief --text "<descripción de la feature>"
uv run python -m agents --json run test run_tests
uv run python -m agents --json run review review_package
```

## Ficheros

| Fichero | Dueño | Qué es |
|---------|-------|--------|
| `AGENTS.md` | humano | Protocolo y reglas del juego |
| `init.sh` | humano + `reviewer` | La puerta. El reviewer puede endurecerla |
| `featureslist.json` | `harness` | Backlog con criterios de aceptación |
| `progress/current.md` | `harness` | Feature en curso |
| `progress/history.md` | `harness` | Append-only de lo cerrado |
| `progress/<agente>-<ID>.md` | `harness` (vía `record`) | Resultado de una ejecución |
| `.opencode/agents/*.md` | cada agente | Su propia definición (automejorable) |

## Automejora

Si el mismo fallo se cuela dos veces: si es automatizable va a `init.sh`; si es
regla del proyecto va a `AGENTS.md`; si es criterio de revisión va a
`.opencode/agents/reviewer.md`. Deja constancia en `progress/history.md`.

## Comandos

```bash
make init            # ./init.sh — verificación completa
make harness-check   # ./init.sh --quick — sin tests
make backlog         # estado de featureslist.json
./init.sh --json     # salida estructurada para agentes
uv run python -m agents.evals.runner --harness   # valida las piezas del arnés
```
