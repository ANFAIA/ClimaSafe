"""Tests del programador de evaluaciones como worker (MSG-004).

El programador decide CUÁNDO lanzar (CRON_SCHEDULE), encola la evaluación en
la cola de MSG-003 y el worker de la cola la envía con el adapter (mockeado,
sin red). Se prueba el parseo cron y que el bucle encola y envía cuando toca.
"""

from __future__ import annotations

import asyncio
from datetime import datetime

import pytest

import climasafeai.bot.cola_mensajes as cm
import climasafeai.bot.evaluacion_programada as ep
import climasafeai.bot.programador_evaluaciones as pg
from climasafeai.bot.cola_mensajes import ColaMensajes


class FakeAdapter:
    """MessageAdapter de prueba: registra envíos, nunca toca la red."""

    def __init__(self):
        self.envios: list[tuple[str, str]] = []

    async def send(self, destino: str, texto: str, **kwargs):
        self.envios.append((destino, texto))


class TestCoincideCron:
    def test_minuto_y_hora(self):
        assert pg.coincide_cron("0 8 * * *", datetime(2026, 8, 14, 8, 0))
        assert not pg.coincide_cron("0 8 * * *", datetime(2026, 8, 14, 8, 1))
        assert not pg.coincide_cron("0 8 * * *", datetime(2026, 8, 14, 9, 0))

    def test_comodines_coinciden_siempre(self):
        assert pg.coincide_cron("* * * * *", datetime(2026, 8, 14, 23, 59))

    def test_listas(self):
        assert pg.coincide_cron("0 8,20 * * *", datetime(2026, 8, 14, 20, 0))
        assert not pg.coincide_cron("0 8,20 * * *", datetime(2026, 8, 14, 12, 0))

    def test_semana_domingo_es_cero_o_siete(self):
        domingo = datetime(2026, 8, 16, 8, 0)  # 2026-08-16 es domingo
        assert domingo.weekday() == 6
        assert pg.coincide_cron("0 8 * * 0", domingo)
        assert pg.coincide_cron("0 8 * * 7", domingo)
        assert not pg.coincide_cron("0 8 * * 1", domingo)

    def test_expresion_con_menos_campos_aborta(self):
        with pytest.raises(ValueError, match="5 campos"):
            pg.coincide_cron("0 8 * *", datetime.now())

    def test_campo_no_soportado_aborta(self):
        with pytest.raises(ValueError, match="no soportado"):
            pg.coincide_cron("*/5 8 * * *", datetime.now())

    def test_campo_fuera_de_rango_aborta(self):
        with pytest.raises(ValueError, match="fuera de rango"):
            pg.coincide_cron("0 25 * * *", datetime.now())


class TestLeerConfig:
    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("CRON_SCHEDULE", raising=False)
        monkeypatch.delenv("CRON_TAREA", raising=False)
        monkeypatch.setenv("MSG_DESTINO", "111")
        cron, tarea, destinos = pg._leer_config()
        assert cron == "0 8 * * *"
        assert tarea == "resumen"
        assert destinos == ["111"]

    def test_cron_invalido_aborta(self, monkeypatch):
        monkeypatch.setenv("CRON_SCHEDULE", "0 8 * *")
        monkeypatch.setenv("MSG_DESTINO", "111")
        with pytest.raises(SystemExit, match="CRON_SCHEDULE inválido"):
            pg._leer_config()

    def test_tarea_desconocida_aborta(self, monkeypatch):
        monkeypatch.setenv("CRON_SCHEDULE", "0 8 * * *")
        monkeypatch.setenv("CRON_TAREA", "pasado_mañana")
        monkeypatch.setenv("MSG_DESTINO", "111")
        with pytest.raises(SystemExit, match="CRON_TAREA desconocida"):
            pg._leer_config()


class TestBucleCron:
    @pytest.fixture
    def cola(self, tmp_path):
        return ColaMensajes(adapter=FakeAdapter(), db_path=tmp_path / "cola_pg.db")

    async def test_encola_y_envia_cuando_toca(self, monkeypatch, tmp_path, cola):
        monkeypatch.setattr(cm, "_POLL_INTERVALO", 0.02)
        # el cron siempre coincide: el bucle encola y el worker de la cola envía
        monkeypatch.setattr(pg, "coincide_cron", lambda cron, momento: True)
        monkeypatch.setenv("MSG_DESTINO", "111, 222")
        monkeypatch.delenv("PERFIL_ID", raising=False)
        monkeypatch.setenv("LAT", "40.41")
        monkeypatch.setenv("LON", "-3.70")

        async def _fake_calcular(*a, **k):
            return "texto de prueba"

        monkeypatch.setattr(ep, "_calcular_texto", _fake_calcular)

        bucle = asyncio.create_task(pg._bucle_cron(cola, "resumen", "0 8 * * *", intervalo=0.02))
        worker = asyncio.create_task(cola.procesar(n_workers=1, esperar=True))
        for _ in range(200):
            await asyncio.sleep(0.02)
            if cola.adapter.envios:
                break
        bucle.cancel()
        worker.cancel()
        with pytest.raises(asyncio.CancelledError):
            await bucle
        with pytest.raises(asyncio.CancelledError):
            await worker

        assert cola.adapter.envios == [("111", "texto de prueba"), ("222", "texto de prueba")]

    async def test_no_dispara_cuando_no_toca(self, monkeypatch, tmp_path, cola):
        monkeypatch.setattr(pg, "coincide_cron", lambda cron, momento: False)
        monkeypatch.setenv("MSG_DESTINO", "111")
        monkeypatch.delenv("PERFIL_ID", raising=False)
        monkeypatch.setenv("LAT", "40.41")
        monkeypatch.setenv("LON", "-3.70")

        bucle = asyncio.create_task(pg._bucle_cron(cola, "resumen", "0 8 * * *", intervalo=0.02))
        await asyncio.sleep(0.1)
        bucle.cancel()
        with pytest.raises(asyncio.CancelledError):
            await bucle

        assert cola._conteo_por_estado() == {}
