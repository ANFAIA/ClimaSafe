#!/usr/bin/env bash
#
# pages_deploy.sh — publica la documentación (MkDocs → site/) en el GitHub Pages
# personal del humano (cacelass/cacelass.github.io), bajo
# projects/climasafe/documentation/.
#
# NO toca projects/climasafe/index.html (el home del proyecto vive en el pages
# personal y se mantiene aparte).
#
# Es la pieza compartida por dos vías:
#   - CI:  .github/workflows/pages.yml clona el repo destino con el secreto
#          PAGES_DEPLOY_TOKEN y ejecuta este script con PAGES_DIR en el runner.
#   - Local: make pages-deploy          (copia + commit + push)
#            make pages-deploy-dry      (copia + commit sin push, PUSH=no)
#
# Variables:
#   PAGES_DIR         checkout local del repo destino (default: ~/Documentos/migithub/cacelass.github.io)
#   PAGES_REMOTE      remote de PAGES_DIR a pushear (default: origin)
#   PAGES_REMOTE_URL  si se da y PAGES_DIR no existe, se clona desde aquí
#   PUSH              yes (default) | no  — con 'no' deja los cambios commit
#                     localmente y staged, sin push, y avisa.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAGES_DIR="${PAGES_DIR:-$HOME/Documentos/migithub/cacelass.github.io}"
PAGES_REMOTE="${PAGES_REMOTE:-origin}"
PUSH="${PUSH:-yes}"

echo "▶ Construyendo documentación (uv run mkdocs build → site/)"
(cd "$ROOT" && uv run mkdocs build)

if [[ ! -d "$PAGES_DIR/.git" ]]; then
    if [[ -n "${PAGES_REMOTE_URL:-}" ]]; then
        echo "▶ Clonando el repo destino cacelass/cacelass.github.io en $PAGES_DIR"
        git clone --depth 1 "$PAGES_REMOTE_URL" "$PAGES_DIR"
    else
        echo "✗ El repo destino no existe o no es un repo git: $PAGES_DIR" >&2
        echo "  Clónalo primero, p. ej.:" >&2
        echo "    git clone git@github.com:cacelass/cacelass.github.io.git ~/Documentos/migithub/cacelass.github.io" >&2
        echo "  o exporta PAGES_REMOTE_URL (https://...) para que este script lo clone." >&2
        exit 1
    fi
fi

DOC_DEST="$PAGES_DIR/projects/climasafe/documentation"
SRC_DEST="$PAGES_DIR/projects/climasafe-src"

echo "▶ Copiando site/ → projects/climasafe/documentation/ ($(du -sh "$ROOT/site" | cut -f1))"
rm -rf "$DOC_DEST"
mkdir -p "$DOC_DEST"
cp -R "$ROOT/site/." "$DOC_DEST/"

# .nojekyll: GitHub Pages ignora Jekyll y sirve index.html de directorios.
touch "$PAGES_DIR/.nojekyll"

echo "▶ Copiando fuente de docs (mkdocs.yml + docs_site/ + overrides/) → projects/climasafe-src/"
rm -rf "$SRC_DEST"
mkdir -p "$SRC_DEST"
cp "$ROOT/mkdocs.yml" "$SRC_DEST/"
cp -R "$ROOT/docs_site" "$SRC_DEST/docs_site"
cp -R "$ROOT/overrides" "$SRC_DEST/overrides"
cat > "$SRC_DEST/build.sh" <<'EOF'
#!/usr/bin/env bash
# Regenera projects/climasafe/documentation/ desde esta fuente, sin depender del repo ANFAIA.
# Requiere mkdocs y el theme material (pip install mkdocs mkdocs-material).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if ! command -v mkdocs >/dev/null 2>&1; then
    echo "mkdocs no instalado. Prueba: pip install mkdocs mkdocs-material" >&2
    exit 1
fi
mkdocs build -f mkdocs.yml -d /tmp/climasafe-docs-build
rm -rf ../climasafe/documentation
cp -R /tmp/climasafe-docs-build ../climasafe/documentation
echo "✓ Documentación regenerada en projects/climasafe/documentation/"
EOF
chmod +x "$SRC_DEST/build.sh"

cd "$PAGES_DIR"
git add .nojekyll projects/climasafe/documentation projects/climasafe-src

if git diff --cached --quiet; then
    echo "Sin cambios en projects/climasafe/documentation/ — nada que publicar."
    exit 0
fi

# El commit en el pages personal lo firma SIEMPRE un humano, nunca un bot:
# - En CI (GitHub Actions), el actor que disparó el push (GITHUB_ACTOR).
# - En local, la identidad git por defecto del usuario.
if [[ -n "${GITHUB_ACTOR:-}" ]]; then
    GIT_ID=( -c "user.name=$GITHUB_ACTOR" -c "user.email=${GITHUB_ACTOR_ID:-0}+${GITHUB_ACTOR}@users.noreply.github.com" )
else
    GIT_ID=()
fi
git "${GIT_ID[@]}" commit -m "deploy(climasafe): actualiza documentacion"

if [[ "$PUSH" == "yes" ]]; then
    echo "▶ Push a $PAGES_REMOTE"
    git push "$PAGES_REMOTE" HEAD
    echo "✓ Publicado en https://cacelass.github.io/projects/climasafe/documentation/"
else
    echo "PUSH=no — commit hecho en local y cambios staged en $PAGES_DIR (sin push)."
    git status --short | head -20
fi
