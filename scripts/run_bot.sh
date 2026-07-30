#!/usr/bin/env bash
# run_bot.sh — Lanza el bot Telegram con autoreinicio
#
# Uso:
#   ./scripts/run_bot.sh              # foreground (Ctrl+C)
#   ./scripts/run_bot.sh --daemon     # background, guarda PID
#   ./scripts/run_bot.sh --stop       # para el daemon
#   ./scripts/run_bot.sh --restart    # reinicia
#
# El bot se reinicia automáticamente si crashea (exit != 0).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PIDFILE="/tmp/climasafebot.pid"
LOGFILE="$ROOT/logs/bot.log"
mkdir -p "$ROOT/logs"

run_bot() {
  cd "$ROOT"
  set -a; [ -f .env ] && . .env; set +a
  while true; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Iniciando bot..." >> "$LOGFILE"
    uv run --no-sync python -m climasafeai.bot.telegram_bot >> "$LOGFILE" 2>&1
    EXIT=$?
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Bot terminó con exit code $EXIT. Reiniciando en 3s..." >> "$LOGFILE"
    sleep 3
  done
}

case "${1:-}" in
  --daemon)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "El bot ya está en ejecución (PID $(cat "$PIDFILE"))"
      exit 1
    fi
    nohup "$0" >/dev/null 2>&1 &
    BGPID=$!
    echo "$BGPID" > "$PIDFILE"
    echo "Bot iniciado en background (PID $BGPID). Log: $LOGFILE"
    echo "Para parar: $0 --stop"
    ;;
  --stop)
    if [ -f "$PIDFILE" ]; then
      kill "$(cat "$PIDFILE")" 2>/dev/null && echo "Bot detenido" || echo "El bot no estaba en ejecución"
      rm -f "$PIDFILE"
    fi
    pkill -f "climasafeai.bot.telegram_bot" 2>/dev/null || true
    ;;
  --restart)
    "$0" --stop
    sleep 2
    "$0" --daemon
    ;;
  *)
    run_bot
    ;;
esac
