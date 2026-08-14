#!/usr/bin/env bash
#
# release_ci.sh — release semántico en CI: tag v0.0.x + GitHub Release con
# release notes. Lo usa .github/workflows/release.yml al llegar a main.
#
# El bump de versión en pyproject.toml lo hace `harness finish` en local
# (DocumentationAgent/GIT-001) antes del push; este script lee la versión,
# y si no existe el tag v<version> la publica. Idempotente: si el tag ya
# existe, no hace nada.
#
# REGLA DE ORO: CI NO commitea NADA en el repo (ni CHANGELOG.md ni código).
# Los commits los hace el humano/local con su identidad. El tag es LIGERO
# (apunta al commit, sin objeto de tag ni autor), de modo que
# github-actions[bot] jamás aparece como autor ni contribuyente. El changelog
# de la versión se publica como cuerpo de la GitHub Release.
#
# Actualización OPCIONAL de CHANGELOG.md en LOCAL (solo modifica el fichero;
# el commit lo hace el humano):
#   UPDATE_CHANGELOG=yes bash scripts/release_ci.sh 0.0.71
#
# Uso:
#   bash scripts/release_ci.sh                 # versión de pyproject.toml
#   bash scripts/release_ci.sh 0.0.68          # versión explícita
#   PUSH=no bash scripts/release_ci.sh         # sin push (prueba local)
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VERSION="${1:-}"
if [[ -z "$VERSION" ]]; then
    VERSION="$(python3 -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])")"
fi
TAG="v$VERSION"
PUSH="${PUSH:-yes}"
UPDATE_CHANGELOG="${UPDATE_CHANGELOG:-no}"

echo "▶ Release: versión $VERSION → tag $TAG"

# Idempotencia: si el tag ya existe para esta versión, no hay nada que publicar.
if git rev-parse -q --verify "refs/tags/$TAG" >/dev/null 2>&1; then
    echo "El tag $TAG ya existe — release idempotente, nada que hacer."
    exit 0
fi

# Rango de commits desde el último tag (puede no haber ninguno: primer release).
LAST_TAG="$(git describe --tags --abbrev=0 2>/dev/null || true)"
if [[ -n "$LAST_TAG" ]]; then
    RANGE="${LAST_TAG}..HEAD"
    echo "▶ Commits desde el último tag: $LAST_TAG"
else
    RANGE="HEAD"
    echo "▶ Primer release — rango: inicio del repo"
fi

RELEASE_NOTES="$(python3 scripts/release_notes.py "$RANGE" "$VERSION" --release-notes)"
CHANGELOG_SECTION="$(python3 scripts/release_notes.py "$RANGE" "$VERSION")"

# CHANGELOG.md SOLO en local y SOLO modifica el fichero (sin commit): el
# commit del CHANGELOG lo hace el humano con su identidad, nunca el bot.
if [[ "$UPDATE_CHANGELOG" == "yes" ]]; then
    TMP="$(mktemp)"
    if [[ -f CHANGELOG.md ]]; then
        { printf '# Changelog\n\n%s' "$CHANGELOG_SECTION"; tail -n +2 CHANGELOG.md; } > "$TMP"
    else
        { printf '# Changelog\n\n%s' "$CHANGELOG_SECTION"; } > "$TMP"
    fi
    mv "$TMP" CHANGELOG.md
    echo "▶ CHANGELOG.md actualizado con la sección $TAG (sin commit — hazlo tú con tu identidad)."
fi

# Tag LIGERO (sin objeto de tag ni autor): github-actions[bot] nunca firma.
git tag "$TAG"

if [[ "$PUSH" == "yes" ]]; then
    echo "▶ Push del tag $TAG"
    git push origin "$TAG"
else
    echo "PUSH=no — tag creado en local (sin push)."
fi

# GitHub Release con las release notes agrupadas por tipo.
if command -v gh >/dev/null 2>&1 && [[ -n "${GITHUB_TOKEN:-}" ]]; then
    echo "▶ Creando GitHub Release $TAG"
    gh release create "$TAG" \
        --title "ClimaSafe $VERSION" \
        --notes "$RELEASE_NOTES" \
        --target "$(git rev-parse HEAD)"
else
    echo "gh/GITHUB_TOKEN no disponible — release notes generadas pero no publicadas como GitHub Release."
fi

echo "✓ Release $TAG listo"
