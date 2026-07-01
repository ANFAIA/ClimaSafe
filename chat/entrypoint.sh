#!/usr/bin/env bash
# chat/entrypoint.sh — Entrypoint del contenedor ClimaSafeAI
# Generado por dskit (https://github.com/cacelass/dskit)
set -euo pipefail

# ── Colores ──────────────────────────────────────────────────────────────────
BOLD="\033[1m"
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
BLUE="\033[0;34m"
CYAN="\033[0;36m"
RESET="\033[0m"

clear 2>/dev/null || true   # no falla si no hay TTY

# ── Banner ASCII — DSKIT ──────────────────────────────────────────────────────
if command -v figlet &> /dev/null; then
    figlet -f big "DSKIT" 2>/dev/null || figlet "DSKIT"
    printf "${BOLD}${BLUE}"

echo '  _____    _____  _  __    ________     '
echo ' |  __ \  / ____|| |/ / (_)|__   __|'
echo ' | |  | || (___  | ´ /   _    | | '
echo ' | |  | | \___ \ |  |   | |   | |'
echo ' | |__| | ____) || . \  | |   | |'
echo ' |_____/ |_____/ |_|\_\ |_|   |_|'

printf "${RESET}"

printf "\n"
printf "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}\n"
printf "  ${BOLD}ClimaSafeAI${RESET}  ·  v0.0.1\n"
printf "  ML type : ${YELLOW}supervisado${RESET}\n"
printf "  Plantilla: ${BLUE}https://github.com/cacelass/dskit${RESET}\n"
printf "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}\n\n"

# ── Verificar modelos ─────────────────────────────────────────────────────────
MODEL_COUNT=0
if [ -d "/app/models" ]; then
    MODEL_COUNT=$(find /app/models -maxdepth 1 \( -name "*.joblib" -o -name "*.pt" \) 2>/dev/null | wc -l)
fi

if [ "${MODEL_COUNT}" -eq 0 ]; then
    printf "${YELLOW}⚠  No se encontraron modelos entrenados en models/${RESET}\n"

    # Buscar dataset: raiz del proyecto o data/raw/
    DATASET_PATH=""
    if [ -f "/app/dataset.csv" ]; then
        DATASET_PATH="/app/dataset.csv"
    elif ls /app/data/raw/*.csv 1>/dev/null 2>&1; then
        DATASET_PATH="$(ls /app/data/raw/*.csv | head -1)"
    fi

    if [ -n "${DATASET_PATH}" ]; then
        printf "${GREEN} Dataset encontrado: ${DATASET_PATH}${RESET}\n"
        printf "${GREEN}   Iniciando entrenamiento automatico...${RESET}\n\n"
        cd /app

        # Enviar "0" al prompt interactivo de main.py (pipeline completo)
        # timeout 3600s para evitar que el contenedor quede colgado
        if printf "0\n" | timeout 3600 python main.py; then
            printf "\n${GREEN} ✔ Entrenamiento completado con exito.${RESET}\n"
        else
            printf "\n${YELLOW}⚠  El entrenamiento finalizo con errores. Revisa los logs.${RESET}\n"
            printf "   Puedes volver a intentarlo desde dentro del chat con el comando 'train'.\n"
        fi
    else
        printf "${YELLOW}⚠  No se encontro dataset.csv${RESET}\n"
        printf "   Coloca tu dataset en la raiz del proyecto o en data/raw/ y reinicia,\n"
        printf "   o usa el comando 'train' desde la interfaz de chat.\n"
    fi
else
    printf "${GREEN} ✔ Modelos encontrados: ${MODEL_COUNT} archivo(s)${RESET}\n"
fi

printf "\n"
printf "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}\n"
printf "${BOLD} Iniciando interfaz web de chat en http://localhost:8080${RESET}\n"
printf "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}\n\n"

# ── Arrancar servidor FastAPI ─────────────────────────────────────────────────
cd /app
exec python -m uvicorn chat.app:app \
    --host 0.0.0.0 \
    --port 8080 \
    --log-level info