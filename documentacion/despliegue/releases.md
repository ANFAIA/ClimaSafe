# Releases: quién commitea, quién etiqueta (DEPLOY-003)

Cómo conviven el agente git (commits), `harness finish` (versión) y el release
automático en CI (tag + GitHub Release). Y por qué **no** usamos release-please.

## Reparto de papeles

| Papel | Quién | Dónde | Qué toca |
|-------|-------|-------|----------|
| Decidir cuándo se cierra una feature y se pushea | Humano (vía líder) | local | — |
| **Commits** | Agente `git` (`suggest_commit_message` / `commit_with_changelog`) | local | Historial entero, Conventional Commits, **sin coautoría** de ninguna herramienta |
| **Número de versión** | `harness finish` → `DocumentationAgent.bump_version` | local | `pyproject.toml` + badge/línea de `README.md` (patch +1 en cada cierre) |
| **Contenido del changelog** | Conventional Commits (los agrupa `scripts/release_notes.py`) | CI | Solo lectura del historial |
| **Tag + GitHub Release** | Workflow `release.yml` → `scripts/release_ci.sh` | CI en `ANFAIA/ClimaSafe` | Tag ligero `v<version>` + release con las notas en el cuerpo |

Regla de oro que se mantiene: **CI no commitea nada** (ni código ni
CHANGELOG.md) y ningún bot firma el historial.

## Fuente única de verdad

La versión vive **solo en `pyproject.toml`**. El README es un espejo que
actualiza el mismo `bump_version` en el mismo paso, así que nunca divergen.

Cada escritor tiene su fichero y cada lector su papel:

```
harness finish (local)  ──escribe──▶  pyproject.toml (0.0.NNN)
                        ──escribe──▶  README.md (espejo)
git agent (local)       ──escribe──▶  historial de commits
release.yml (CI ANFAIA) ──lee───────▶  pyproject.toml ──▶ tag v<version> + release
```

El tag **se deriva, no se propone**: si el push llega con la versión `0.0.116`
en `pyproject.toml`, el workflow crea `v0.0.116` sobre ese commit. Es
idempotente — si el tag ya existe (p. ej. porque alguien usó `make release`
en local antes de pushear), el workflow no hace nada.

## Evaluación: release-please vs el agente de release

Se evaluó `google-github-actions/release-please-action`. Cómo funciona:
parsea los Conventional Commits, calcula la siguiente versión semántica y abre
un *Release PR* que sube `pyproject.toml` + `CHANGELOG.md`; al mergear ese PR,
etiqueta y publica la release. Se descartó por tres choques directos:

1. **Dos escritores de la versión.** `harness finish` sube el patch en cada
   cierre; release-please calcularía la siguiente versión él solo (dos
   `feat:` desde el último tag → querría subir minor). Mismo campo, dos
   dueños, divergencia garantizada. El criterio «una única fuente de verdad»
   lo prohíbe.
2. **Commits con identidad de bot.** El Release PR lo firma
   `release-please[bot]`: viola la regla de este repo de que todo commit es
   del humano/local sin coautoría de herramientas.
3. **Flujo de Release PR innecesario.** Aquí se trabaja en main y se pushea;
   un PR intermedio por versión no aporta nada y añade un estado más que
   mantener.

Lo valioso de release-please — agrupar el changelog a partir de Conventional
Commits — ya lo cubre `scripts/release_notes.py` (stdlib puro, misma
agrupación que el GitAgent), así que no se pierde nada a cambio.

**Decisión:** el generador de tags/version es **el agente de release propio**
(`release.yml` + `release_ci.sh`). Guard en `tests/test_cicd.py`
(`test_no_release_please`) para que nadie lo reintroduzca por accidente.

## Cómo conviven commits del agente git y tag automático

No compiten porque cada uno escribe cosas distintas y el tag solo lee:

- Los **Conventional Commits** (que redacta el agente git) determinan el
  **contenido**: qué secciones salen en cada release (Añadido, Corrección de
  bugs…).
- La **política patch-por-cierre** de `harness finish` determina el
  **número**: una feature cerrada = un punto de versión, sin analizar tipos.
  Es deliberado: predecible y estable aunque un cierre agrupe varios `feat:`.

Si algún día se quiere semver estricto (breaking → minor/major), el cambio es
local y acotado: `_next_patch_version()` en `agents/agents/harness_agent.py`.
El pipeline de release no habría que tocarlo.

## Alcance: repo personal fuera

Los tags y releases viven **solo en `ANFAIA/ClimaSafe`**. El job de
`release.yml` lleva el guard `if: github.repository == 'ANFAIA/ClimaSafe'`:
si el workflow llegara a correr en otro repo (espejo, fork), se salta. El
pages personal (`cacelass/cacelass.github.io`) recibe únicamente docs y demo
vía `pages.yml` / `make pages-deploy`; jamás tags, versiones ni releases.

## Camino manual (opcional)

`make release` delega en el GitAgent (`tag_release`): bump + CHANGELOG.md +
tag en un único commit local, sin push. Es la vía alternativa al tag en CI;
la idempotencia hace que ambas convivan (el workflow ve el tag existente y no
repite).
