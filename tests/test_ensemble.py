"""
test_ensemble.py — Tests para climasafeai/models/ensemble.py

Regresión DATA-003: el perfil horario debe construirse SOLO con el día
objetivo, no con el máximo por hora de todos los días del df_hora (que mezcla
14 días de histórico + el día objetivo para alimentar el LSTM).
"""
from datetime import date, datetime

import pandas as pd
import pytest

from climasafeai.models.ensemble import perfil_horario_desde_df


def _df_hora_dos_dias() -> pd.DataFrame:
    """df_hora con dos días: uno caluroso (HI 36 a las 16h) y otro templado (HI 26)."""
    caluroso = date(2026, 7, 20)
    templado = date(2026, 7, 21)
    filas = []
    for dia, hi_pico in ((caluroso, 36.0), (templado, 26.0)):
        for h in range(24):
            filas.append({
                "datetime": datetime(dia.year, dia.month, dia.day, h),
                "t2m_c": 20.0 + h * 0.5,
                "heat_index_c": hi_pico if h == 16 else 20.0,
            })
    return pd.DataFrame(filas)


def test_perfil_horario_solo_usa_el_dia_pedido():
    """Regresión DATA-003: con target_date=<templado>, el HI máx es ~26, no 36."""
    df = _df_hora_dos_dias()
    perfil = perfil_horario_desde_df(df, target_date=date(2026, 7, 21))

    assert perfil is not None
    max_hi = max(p["HI"] for p in perfil)
    assert max_hi == pytest.approx(26.0, abs=0.1)
    assert max_hi < 30  # el pico de 36 del otro día NO debe colarse

    h16 = next(p for p in perfil if p["hora"] == 16)
    assert h16["HI"] == pytest.approx(26.0, abs=0.1)


def test_perfil_horario_acepta_target_date_iso():
    """El target_date puede venir como ISO string (como en weather['target_date'])."""
    df = _df_hora_dos_dias()
    perfil = perfil_horario_desde_df(df, target_date="2026-07-21")

    assert max(p["HI"] for p in perfil) == pytest.approx(26.0, abs=0.1)


def test_perfil_horario_sin_target_date_usa_el_ultimo_dia():
    """Sin target_date se usa la última fecha del df (el día objetivo concatenado)."""
    df = _df_hora_dos_dias()
    perfil = perfil_horario_desde_df(df)

    assert max(p["HI"] for p in perfil) == pytest.approx(26.0, abs=0.1)


def test_perfil_horario_con_target_date_del_dia_caluroso():
    """Pidiendo explícitamente el día caluroso, el perfil sí refleja su pico."""
    df = _df_hora_dos_dias()
    perfil = perfil_horario_desde_df(df, target_date=date(2026, 7, 20))

    assert max(p["HI"] for p in perfil) == pytest.approx(36.0, abs=0.1)


# ─────────────────────────────────────────────────────────────────────────────
# DATA-004: perfil horario con resolución sub-horaria (res_min 5/15/30/60)
# ─────────────────────────────────────────────────────────────────────────────

def _df_hora_campana() -> pd.DataFrame:
    """24 h de un día con campana de HI (pico 41 a las 16h), como la gráfica MCP."""
    filas = []
    for h in range(24):
        hi = 20.0 + (41.0 - 20.0) * max(0.0, 1 - abs(h - 16) / 10)
        filas.append({
            "datetime": datetime(2026, 7, 21, h),
            "t2m_c": 15.0 + h * 0.4,
            "heat_index_c": round(hi, 1),
        })
    return pd.DataFrame(filas)


def test_perfil_horario_res_60_identico_al_historico():
    """DATA-004 criterio 5: res_min=60 es exactamente el perfil de un punto por hora."""
    df = _df_hora_campana()
    perfil = perfil_horario_desde_df(df, target_date=date(2026, 7, 21), res_min=60)

    esperado = [
        {"hora": h, "HI": round(20.0 + (41.0 - 20.0) * max(0.0, 1 - abs(h - 16) / 10), 1),
         "temp": round(15.0 + h * 0.4, 1)}
        for h in range(24)
    ]
    assert perfil == esperado
    assert all(isinstance(p["hora"], int) for p in perfil)


def test_perfil_horario_res_15_cuatro_puntos_por_hora():
    """DATA-004 criterio 1: con res_min=15 cada hora devuelve 4 puntos."""
    df = _df_hora_campana()
    perfil = perfil_horario_desde_df(df, target_date=date(2026, 7, 21), res_min=15)

    assert len(perfil) == 24 * 4
    hora10 = [p for p in perfil if 10 <= p["hora"] < 11]
    assert [p["hora"] for p in hora10] == [10.0, 10.25, 10.5, 10.75]
    # El ancla :00 conserva el máximo horario; los intermedios interpola lineal.
    h10 = next(p for p in perfil if p["hora"] == 10.0)
    h11 = next(p for p in perfil if p["hora"] == 11.0)
    h1030 = next(p for p in perfil if p["hora"] == 10.5)
    assert h1030["HI"] == pytest.approx(h10["HI"] + 0.5 * (h11["HI"] - h10["HI"]), abs=0.01)
    assert h1030["temp"] == pytest.approx(h10["temp"] + 0.5 * (h11["temp"] - h10["temp"]), abs=0.05)


def test_perfil_horario_res_5_y_30_validas():
    """DATA-004 (triage): 5 y 30 minutos también son resoluciones válidas."""
    df = _df_hora_campana()
    p5 = perfil_horario_desde_df(df, target_date=date(2026, 7, 21), res_min=5)
    p30 = perfil_horario_desde_df(df, target_date=date(2026, 7, 21), res_min=30)

    assert len(p5) == 24 * 12
    assert len(p30) == 24 * 2
    for p in (p5, p30):
        ancla10 = next(x for x in p if abs(x["hora"] - 10.0) < 1e-9)
        assert ancla10["HI"] == pytest.approx(28.4, abs=0.1)


def test_perfil_horario_res_invalida_raise():
    df = _df_hora_campana()
    with pytest.raises(ValueError):
        perfil_horario_desde_df(df, target_date=date(2026, 7, 21), res_min=10)


def test_perfil_horario_15min_desviacion_frente_al_horario_acotada():
    """DATA-004 criterio 2: los intermedios se desvían del dato horario de
    referencia solo dentro de la hora. Al re-agregar a máximo por hora, el error
    máximo es la subida del HI dentro de la hora (en 10:45 vs 10:00), nunca más
    de 3/4 de la subida entre horas consecutivas. En esta campana: 1.575°C."""
    df = _df_hora_campana()
    p60 = perfil_horario_desde_df(df, target_date=date(2026, 7, 21), res_min=60)
    p15 = perfil_horario_desde_df(df, target_date=date(2026, 7, 21), res_min=15)

    desv = []
    for h in range(24):
        hi_horario = next(p["HI"] for p in p60 if p["hora"] == h)
        hi_max15 = max(p["HI"] for p in p15 if int(p["hora"]) == h)
        desv.append(hi_max15 - hi_horario)
    assert max(desv) == pytest.approx(1.575, abs=0.01)
    assert max(desv) <= 1.6
