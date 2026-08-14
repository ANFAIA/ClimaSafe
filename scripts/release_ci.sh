#!/usr/bin/env bash
#
# release_ci.sh — release semántico en CI: tag v0.0.x + CHANGELOG + release
# notes. Lo usa .github/workflows/release.yml al llegar a main.
#
# El bump de versión en pyproject.toml ya lo hace `harness finish` en local
# (DocumentationAgent/GIT-001) antes del push: este script lee la versión,
# y si no existe el tag v<version> la publica. Idempotente: si el tag ya
# existe, no hace nada.
#
# CI solo toca CHANGELOG.md + tag + release: NO commitea código de producto.
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
BRANCH="${BRANCH:-${GITHUB_REF_NAME:-$(git rev-parse --abbrev-ref HEAD)}}"

# Identidad de git: en CI usa el bot de GitHub; en local, la del usuario.
GIT_NAME="${GIT_NAME:-github-actions[bot]}"
GIT_EMAIL="${GIT_EMAIL:-41898282+github-actions[bot]@users.noreply.github.com}"

echo "▶ Release: versión $VERSION → tag $TAG (rama $BRANCH)"

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

CHANGELOG_SECTION="$(python3 scripts/release_notes.py "$RANGE" "$VERSION")"
RELEASE_NOTES="$(python3 scripts/release_notes.py "$RANGE" "$VERSION" --release-notes)"

# Actualiza CHANGELOG.md — el único fichero del repo ANFAIA que toca CI.
TMP="$(mktemp)"
if [[ -f CHANGELOG.md ]]; then
    { printf '# Changelog\n\n%s' "$CHANGELOG_SECTION"; tail -n +2 CHANGELOG.md; } > "$TMP"
else
    { printf '# Changelog\n\n%s' "$CHANGELOG_SECTION"; } > "$TMP"
fi
mv "$TMP" CHANGELOG.md
echo "▶ CHANGELOG.md actualizado con la sección $TAG"

git add CHANGELOG.md
git -c "user.name=$GIT_NAME" -c "user.email=$GIT_EMAIL" \
    commit -m "docs(release): CHANGELOG para $TAG"

git -c "user.name=$GIT_NAME" -c "user.email=$GIT_EMAIL" \
    tag -a "$TAG" -m "Release $TAG"

if [[ "$PUSH" == "yes" ]]; then
    echo "▶ Push del commit y del tag $TAG a $BRANCH"
    git push origin "HEAD:$BRANCH"
    git push origin "$TAG"
else
    echo "PUSH=no — commit y tag creados en local (sin push)."
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
