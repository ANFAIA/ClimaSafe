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
