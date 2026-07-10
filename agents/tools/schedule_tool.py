"""
agents.tools.schedule_tool — Parseo y utilidades de cron.

Sin dependencias externas: parseo propio de expresiones cron estándar
(minuto hora día-del-mes mes día-de-la-semana). Soporta 5 campos estándar
y los alias @hourly, @daily, @weekly, @monthly, @yearly.
"""

from __future__ import annotations

import calendar
from datetime import date, datetime, timedelta
from typing import Any

from agents.tools.registry import register_tool


_ALIASES = {
    "@yearly": "0 0 1 1 *", "@annually": "0 0 1 1 *",
    "@monthly": "0 0 1 * *",
    "@weekly": "0 0 * * 0",
    "@daily": "0 0 * * *", "@midnight": "0 0 * * *",
    "@hourly": "0 * * * *",
}

_DAY_NAMES = {0: "domingo", 1: "lunes", 2: "martes", 3: "miércoles", 4: "jueves", 5: "viernes", 6: "sábado"}
_MONTH_NAMES = {1: "enero", 2: "febrero", 3: "marzo", 4: "abril", 5: "mayo", 6: "junio",
                7: "julio", 8: "agosto", 9: "septiembre", 10: "octubre", 11: "noviembre", 12: "diciembre"}


def _parse_field(field: str, lo: int, hi: int) -> set[int]:
    """Parsea un campo cron (minuto, hora, etc.) a conjunto de valores enteros."""
    if field == "*":
        return set(range(lo, hi + 1))
    values: set[int] = set()
    for part in field.split(","):
        if "/" in part:
            base, step = part.split("/")
            step = int(step)
            if base == "*":
                base_range = range(lo, hi + 1)
            elif "-" in base:
                a, b = base.split("-")
                base_range = range(int(a), int(b) + 1)
            else:
                base_range = range(int(base), hi + 1)
            values.update(base_range[::step])
        elif "-" in part:
            a, b = part.split("-")
            values.update(range(int(a), int(b) + 1))
        else:
            values.add(int(part))
    return {v for v in values if lo <= v <= hi}


def _expand(expr: str) -> tuple[set[int], set[int], set[int], set[int], set[int]]:
    expr = _ALIASES.get(expr.strip().lower(), expr.strip())
    parts = expr.split()
    if len(parts) != 5:
        raise ValueError(f"Expresión cron debe tener 5 campos, no {len(parts)}: '{expr}'")
    return (
        _parse_field(parts[0], 0, 59),
        _parse_field(parts[1], 0, 23),
        _parse_field(parts[2], 1, 31),
        _parse_field(parts[3], 1, 12),
        _parse_field(parts[4], 0, 6),
    )


@register_tool("schedule")
class ScheduleTool:
    @staticmethod
    def validate(expr: str) -> dict[str, bool | str]:
        """Valida una expresión cron y devuelve si es correcta."""
        try:
            _expand(expr)
            return {"valid": True, "error": ""}
        except (ValueError, IndexError) as e:
            return {"valid": False, "error": str(e)}

    @staticmethod
    def to_human(expr: str) -> str:
        """Convierte una expresión cron a descripción legible."""
        expr_in = expr.strip().lower()
        if expr_in in _ALIASES:
            alias_map = {
                "@yearly": "una vez al año (1 de enero a medianoche)",
                "@annually": "una vez al año (1 de enero a medianoche)",
                "@monthly": "una vez al mes (día 1 a medianoche)",
                "@weekly": "una vez a la semana (domingo a medianoche)",
                "@daily": "una vez al día (medianoche)",
                "@midnight": "una vez al día (medianoche)",
                "@hourly": "una vez por hora (minuto 0)",
            }
            return alias_map.get(expr_in, f"alias {expr}")
        try:
            minutes, hours, days, months, weekdays = _expand(expr)
        except ValueError:
            return f"expresión inválida: {expr}"
        parts = []
        if len(hours) == 24 and len(minutes) == 60:
            parts.append("cada minuto")
        elif len(hours) == 24:
            mins = sorted(minutes)
            parts.append(f"en los minutos {self._fmt_list(mins)} de cada hora")
        elif len(hours) == 1 and len(minutes) == 1:
            h = list(hours)[0]
            m = list(minutes)[0]
            parts.append(f"a las {h:02d}:{m:02d}")
        elif len(hours) == 1:
            h = list(hours)[0]
            parts.append(f"a la hora {h:02d} (minutos: {self._fmt_list(sorted(minutes))})")
        else:
            parts.append(f"horas {self._fmt_list(sorted(hours))}, minuto {self._fmt_list(sorted(minutes))}")
        if len(months) < 12:
            parts.append(f"en {self._fmt_list([_MONTH_NAMES[m] for m in sorted(months)])}")
        if len(days) < 31:
            parts.append(f"día {self._fmt_list(sorted(days))} del mes")
        if len(weekdays) < 7:
            parts.append(self._fmt_list([_DAY_NAMES[d] for d in sorted(weekdays)]))
        return ", ".join(parts) if parts else expr

    @staticmethod
    def next_run(expr: str, from_date: date | None = None) -> dict[str, Any]:
        """
        Calcula las próximas 5 ejecuciones de una expresión cron.

        from_date : fecha/hora desde la que calcular (default: ahora).
        Devuelve dict con 'next_runs' (lista de datetimes) y 'from'.
        """
        now = from_date or datetime.now()
        now = now if isinstance(now, datetime) else datetime.combine(now, datetime.min.time())
        try:
            minutes, hours, days, months, weekdays = _expand(expr)
        except ValueError:
            return {"next_runs": [], "from": now.isoformat()}
        runs: list[str] = []
        dt = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
        attempts = 0
        while len(runs) < 5 and attempts < 525600:
            if (dt.month in months and dt.day in days and
                dt.hour in hours and dt.minute in minutes and
                (dt.weekday() in weekdays if len(weekdays) < 7 else True)):
                runs.append(dt.isoformat())
            dt += timedelta(minutes=1)
            attempts += 1
        return {"next_runs": runs, "from": now.isoformat()}

    @staticmethod
    def _fmt_list(items: list) -> str:
        if len(items) <= 2:
            return ", ".join(str(x) for x in items)
        return ", ".join(str(x) for x in items[:-1]) + f" y {items[-1]}"

    @staticmethod
    def summary(expr: str) -> dict[str, Any]:
        """Resumen completo de una expresión cron."""
        valid = ScheduleTool.validate(expr)
        if not valid["valid"]:
            return {"valid": False, "error": valid["error"]}
        try:
            minutes, hours, days, months, weekdays = _expand(expr)
        except ValueError as e:
            return {"valid": False, "error": str(e)}
        return {
            "valid": True,
            "expression": expr,
            "human": ScheduleTool.to_human(expr),
            "minutes": sorted(minutes),
            "hours": sorted(hours),
            "days_of_month": sorted(days),
            "months": sorted(months),
            "weekdays": sorted(weekdays),
            "frequency": "cada minuto" if len(hours) == 24 and len(minutes) == 60
                        else f"cada hora" if len(hours) == 24
                        else f"cada día" if len(hours) == 1 and len(minutes) == 1 and len(weekdays) == 7 and len(days) == 31
                        else "personalizada",
        }
