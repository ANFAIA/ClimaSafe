# Workflows de GitHub Actions — CI/CD y publicación

Tres workflows en `ANFAIA/ClimaSafe`:

| Workflow | Trigger | Qué hace |
|----------|---------|----------|
| `ci.yml` | PR + push a main | `make test` + lint de los ficheros Python cambiados |
| `release.yml` | push a main + dispatch | Tag `v0.0.x` + CHANGELOG + GitHub Release |
| `pages.yml` | push a main + dispatch | Publica docs y demo en `cacelass/cacelass.github.io` |

El `paper_scout.yml` preexistente no se toca.

## CI (`ci.yml`) y la decisión sobre `make lint`

`make lint` sobre todo el repo **falla por deuda preexistente** (23 errores de
`ruff check` + 66 ficheros que `ruff format` reformatearía), en ficheros ajenos
a este ticket: `data/weather_fetcher.py` y `models/ensemble.py` (E402, imports
diferidos a propósito), `models/bayes.py` (E701, one-liners compactos),
`models/recomendaciones.py`, `models/predict_model.py`, `data/grid_risk.py`,
`data/make_dataset.py` (F841, variables sin usar)…

Qué se hizo en DEPLOY-002:

1. **Autofix mecánico** de los 6 errores que `ruff check --fix` puede resolver
   solo: 4 ficheros sin newline final (`make_dataset.py`, `build_features.py`,
   `temporal_cv.py`, `train_model.py`) y 2 bindings de excepción sin usar
   (`telegram_bot.py`, `explicabilidad.py`). Limpieza incluida en este ticket.
2. **No se arreglan a mano los 23 restantes**: son decisiones de estilo/lógica
   en ficheros de otras sesiones, y reescribirlos en masa viola el principio de
   cambios quirúrgicos. Tampoco se formatea nada: `ruff format --check` falla
   repo-wide en 66 ficheros preexistentes.
3. Por eso `ci.yml` lintea **solo los ficheros Python cambiados** en el PR
   (`git diff` contra la base) o en el push a main (contra el commit anterior),
   y solo con `ruff check` (sin `ruff format --check`). Un PR que toque un `.py`
   con errores preexistentes fallará — es la idea: el autor arregla lo que toca.

El objetivo «y pasa» del criterio se cumple así: el CI no queda en un estado
que falle siempre. `make lint` completo sigue pendiente de una limpieza global
fuera del alcance de DEPLOY-002.

## Release (`release.yml`)

> Evaluación de release-please y modelo de convivencia commits↔tag:
> `documentacion/despliegue/releases.md`. El job solo corre en
> `ANFAIA/ClimaSafe` (guard `github.repository`).

- El bump de versión en `pyproject.toml` lo hace **`harness finish` en local**
  (DocumentationAgent/GIT-001) antes del push. El workflow lee esa versión y la
  publica: si no existe el tag `v<version>`, genera la sección de CHANGELOG
  (Conventional Commits, misma agrupación que el GitAgent del arnés), la
  prepende a `CHANGELOG.md`, crea el tag **anotado** sobre ese commit y una
  GitHub Release con las release notes agrupadas por tipo (feat/fix/docs/…).
- **Idempotente**: si el tag ya existe, no hace nada.
- **No commitea código**: solo `CHANGELOG.md` + tag + release.
- Lógica en `scripts/release_ci.sh` y `scripts/release_notes.py` (stdlib puro,
  sin depender del arnés).

## Publicación a Pages (`pages.yml` y `make pages-deploy`)

La publicación es **cross-repo**: `ANFAIA/ClimaSafe` → `cacelass/cacelass.github.io`.
Dos vías:

### CI — `pages.yml`

Necesita el secreto **`PAGES_DEPLOY_TOKEN`**: un PAT con scope `repo` sobre
`cacelass/cacelass.github.io`, creado en
github.com → Settings → Developer settings → Personal access tokens →
Fine-grained o classic (scope `repo`). Se añade en
`ANFAIA/ClimaSafe` → Settings → Secrets and variables → Actions.

Si el secreto no está definido, el job se **salta con un aviso** (no falla el
workflow). El job construye `site/` con MkDocs, clona el pages con el token,
copia `site/` → `climasafe/documentacion/` y `web/probar-ya/` →
`climasafe/probar-ya/`, y commitea+pushea solo esos paths.

### Local — `make pages-deploy`

Mismo script (`scripts/pages_deploy.sh`) contra un checkout local:

```bash
make pages-deploy          # copia + commit + push al repo local
make pages-deploy-dry      # copia + commit SIN push (PUSH=no)
```

Variables: `PAGES_DIR` (default `~/Documentos/migithub/cacelass.github.io`),
`PAGES_REMOTE` (default `origin`), `PAGES_REMOTE_URL` (clona si falta el
checkout). Con `PUSH=no` deja el commit hecho en local sin pushear y avisa.

## Validación

Los YAML se validan manualmente (revisión) y los scripts se prueban en local:
`make docs`, `make pages-deploy-dry` contra un checkout temporal, y
`bash scripts/release_ci.sh 0.0.x` en un repo de prueba con `PUSH=no`.
