#!/usr/bin/env bash
#
# pages_deploy.sh — publica la documentación (MkDocs → site/) y la demo del
# navegador (web/probar-ya/) en el GitHub Pages personal del humano
# (cacelass/cacelass.github.io), bajo climasafe/documentacion/ y
# climasafe/probar-ya/ respectivamente.
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

DOC_DEST="$PAGES_DIR/climasafe/documentacion"
DOC_SRC_DEST="$PAGES_DIR/climasafe/documentacion-src"
DEMO_DEST="$PAGES_DIR/climasafe/probar-ya"

echo "▶ Copiando site/ → climasafe/documentacion/ ($(du -sh "$ROOT/site" | cut -f1))"
rm -rf "$DOC_DEST"
mkdir -p "$(dirname "$DOC_DEST")"
cp -R "$ROOT/site" "$DOC_DEST"

echo "▶ Copiando fuente de docs (mkdocs.yml + documentacion/ + overrides/) → climasafe/documentacion-src/"
# La fuente vive también en el pages personal: si el repo ANFAIA desaparece, la
# documentación se puede regenerar desde aquí (ver climasafe/documentacion-src/build.sh).
rm -rf "$DOC_SRC_DEST"
mkdir -p "$DOC_SRC_DEST"
cp "$ROOT/mkdocs.yml" "$DOC_SRC_DEST/"
cp -R "$ROOT/documentacion" "$DOC_SRC_DEST/documentacion"
cp -R "$ROOT/overrides" "$DOC_SRC_DEST/overrides"
cat > "$DOC_SRC_DEST/build.sh" <<'EOF'
#!/usr/bin/env bash
# Regenera climasafe/documentacion/ desde esta fuente, sin depender del repo ANFAIA.
# Requiere mkdocs y el theme material (pip install mkdocs mkdocs-material).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if ! command -v mkdocs >/dev/null 2>&1; then
    echo "mkdocs no instalado. Prueba: pip install mkdocs mkdocs-material" >&2
    exit 1
fi
mkdocs build -f mkdocs.yml -d /tmp/climasafe-docs-build
rm -rf ../documentacion
cp -R /tmp/climasafe-docs-build ../documentacion
echo "✓ Documentación regenerada en climasafe/documentacion/"
EOF
chmod +x "$DOC_SRC_DEST/build.sh"

echo "▶ Copiando web/probar-ya/ → climasafe/probar-ya/ ($(du -sh "$ROOT/web/probar-ya" | cut -f1))"
rm -rf "$DEMO_DEST"
mkdir -p "$(dirname "$DEMO_DEST")"
cp -R "$ROOT/web/probar-ya" "$DEMO_DEST"

cd "$PAGES_DIR"
git add climasafe/documentacion climasafe/documentacion-src climasafe/probar-ya

if git diff --cached --quiet; then
    echo "Sin cambios en climasafe/ — nada que publicar."
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
git "${GIT_ID[@]}" commit -m "deploy(climasafe): actualiza documentacion y demo probar-ya"

if [[ "$PUSH" == "yes" ]]; then
    echo "▶ Push a $PAGES_REMOTE"
    git push "$PAGES_REMOTE" HEAD
    echo "✓ Publicado en https://cacelass.github.io/climasafe/"
else
    echo "PUSH=no — commit hecho en local y cambios staged en $PAGES_DIR (sin push)."
    git status --short | head -20
fi
