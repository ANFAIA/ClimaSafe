"""
climasafeai.bot.programador_evaluaciones — Programador de evaluaciones como worker (MSG-004).

Sustituye al contenedor cron de MSG-002 por un proceso Python normal. Decide
CUÁNDO lanzar la evaluación (hoy / mañana / resumen diario) según
CRON_SCHEDULE y, cuando toca, encola el mensaje en la cola de MSG-003
(`ColaMensajes`); el mismo proceso corre el worker de la cola, que envía los
mensajes con el `MessageAdapter` (MSG-001). Cero contenedores:

    MSG_ADAPTER=telegram MSG_DESTINO=123456 CRON_TAREA=resumen \
    CRON_SCHEDULE="0 8 * * *" PERFIL_ID=1 \
        python -m climasafeai.bot.programador_evaluaciones

Configuración (variables de entorno):

- CRON_SCHEDULE — expresión cron de 5 campos (minuto hora día-del-mes mes
  día-de-la-semana), p.ej. "0 8 * * *" (defecto: mañanas a las 8:00). Cada
  campo admite `*`, un valor o una lista separada por comas; no admite
  rangos ("1-5") ni pasos ("*/5").
- CRON_TAREA   — ``resumen`` (defecto), ``hoy`` o ``manana``.
- MSG_ADAPTER / MSG_DESTINO / PERFIL_ID (o LAT/LON/...) — igual que en
  ``evaluacion_programada``: canal, destinos y ubicación de la evaluación.

El proceso corre dos tareas: el bucle de cron (comprueba la hora cada 20 s y
dispara como mucho una vez por minuto) y el worker de la cola
(`cola.procesar(esperar=True)`), que envía lo encolado.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime

from climasafeai.bot.cola_mensajes import ColaMensajes
from climasafeai.bot.evaluacion_programada import (
    TAREAS,
    crear_adapter,
    destinos_env,
    encolar_evaluacion,
    inicializar_db,
)

logger = logging.getLogger(__name__)

_INTERVALO_COMPROBACION = 20.0

# (mínimo, máximo) de cada campo cron, en orden: minuto hora dia mes semana.
_RANGOS_CRON = ((0, 59), (0, 23), (1, 31), (1, 12), (0, 7))


def _parsear_campo(campo: str, minimo: int, maximo: int) -> set[int] | None:
    """'*' → None (cualquier valor); '8' → {8}; '1,3' → {1,3}."""
    campo = campo.strip()
    if campo == "*":
        return None
    valores: set[int] = set()
    for parte in campo.split(","):
        parte = parte.strip()
        if not parte.isdigit():
            raise ValueError(f"campo cron no soportado: {parte!r} (usa *, un número o una lista)")
        n = int(parte)
        if not minimo <= n <= maximo:
            raise ValueError(f"valor {n} fuera de rango [{minimo}, {maximo}]")
        valores.add(n)
    return valores or None


def _parsear_cron(expresion: str) -> tuple[set[int] | None, ...]:
    """Valida y parsea una expresión cron de 5 campos."""
    campos = expresion.split()
    if len(campos) != 5:
        raise ValueError(
            f"CRON_SCHEDULE debe tener 5 campos (minuto hora dia mes semana): {expresion!r}"
        )
    return tuple(
        _parsear_campo(campo, minimo, maximo)
        for campo, (minimo, maximo) in zip(campos, _RANGOS_CRON)
    )


def coincide_cron(expresion: str, momento: datetime) -> bool:
    """¿Coincide `momento` con la expresión cron de 5 campos?

    El día de la semana se interpreta como en cron: 0 y 7 = domingo,
    1 = lunes ... 6 = sábado.
    """
    minuto, hora, dia, mes, semana = _parsear_cron(expresion)

    def _en(campo: set[int] | None, valor: int) -> bool:
        return campo is None or valor in campo

    dia_semana = momento.weekday() + 1  # 1=lunes ... 7=domingo
    return (
        _en(minuto, momento.minute)
        and _en(hora, momento.hour)
        and _en(dia, momento.day)
        and _en(mes, momento.month)
        and (_en(semana, dia_semana) or (dia_semana == 7 and _en(semana, 0)))
    )


def _leer_config() -> tuple[str, str, list[str]]:
    """(cron, tarea, destinos) desde env, abortando ante configuración inválida."""
    cron = os.getenv("CRON_SCHEDULE", "0 8 * * *")
    try:
        _parsear_cron(cron)
    except ValueError as e:
        raise SystemExit(f"CRON_SCHEDULE inválido: {e}")

    tarea = os.getenv("CRON_TAREA", "resumen").lower()
    if tarea not in TAREAS:
        raise SystemExit(f"CRON_TAREA desconocida: {tarea} (valores: {', '.join(TAREAS)})")

    return cron, tarea, destinos_env()


async def _bucle_cron(
    cola: ColaMensajes,
    tarea: str,
    cron: str,
    intervalo: float = _INTERVALO_COMPROBACION,
) -> None:
    """Comprueba el cron cada `intervalo` s y encola la evaluación cuando toca.

    Dispara como mucho una vez por minuto (guarda el último minuto que
    disparó), igual que un cron real: la comprobación es de granularidad
    minuto y el intervalo es menor que un minuto para no perderse el disparo.
    """
    ultimo_minuto: int | None = None
    while True:
        ahora = datetime.now()
        if coincide_cron(cron, ahora) and ahora.minute != ultimo_minuto:
            ultimo_minuto = ahora.minute
            await encolar_evaluacion(cola, tarea)
        await asyncio.sleep(intervalo)


async def main() -> None:
    # handler propio: importar `messaging` arrastra `telegram_bot`, que ya
    # configura el root logger y `basicConfig` no surte efecto.
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    inicializar_db()

    cron, tarea, destinos = _leer_config()
    adapter = crear_adapter()
    cola = ColaMensajes(adapter=adapter)
    logger.info(
        "Programador arrancado: %s cada '%s' para %s (%s)",
        tarea,
        cron,
        destinos,
        os.getenv("MSG_ADAPTER", "telegram"),
    )
    try:
        await asyncio.gather(
            _bucle_cron(cola, tarea, cron),
            cola.procesar(n_workers=1, esperar=True),
        )
    finally:
        close = getattr(adapter, "close", None)
        if close is not None:
            await close()


if __name__ == "__main__":
    asyncio.run(main())
