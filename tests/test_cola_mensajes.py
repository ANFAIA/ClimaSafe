"""Tests de la cola de mensajes (MSG-003).

La cola se prueba con sqlite en tmp_path y un adapter fake (transporte
mockeado, sin red), igual que test_messaging. La concurrencia se demuestra de
dos formas: (1) M mensajes con N workers → cada mensaje se envía exactamente
una vez; (2) un adapter con retraso → varios workers están enviando a la vez.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

import climasafeai.bot.cola_mensajes as cm
from climasafeai.bot.cola_mensajes import ColaMensajes


class FakeAdapter:
    """MessageAdapter de prueba: registra envíos, nunca toca la red."""

    def __init__(self, retraso: float = 0.0):
        self.retraso = retraso
        self.envios: list[tuple[str, str, dict]] = []
        self.activos = 0
        self.max_activos = 0

    async def send(self, destino: str, texto: str, **kwargs):
        self.activos += 1
        self.max_activos = max(self.max_activos, self.activos)
        if self.retraso:
            await asyncio.sleep(self.retraso)
        self.activos -= 1
        self.envios.append((destino, texto, kwargs))

    async def send_batch(self, mensajes: list[tuple[str, str]], **kwargs):
        for destino, texto in mensajes:
            await self.send(destino, texto, **kwargs)


class AdapterQueFalla:
    """Adapter que falla para un subconjunto de destinos."""

    def __init__(self, destinos_que_fallan: set[str]):
        self.destinos_que_fallan = destinos_que_fallan
        self.envios: list[tuple[str, str]] = []

    async def send(self, destino: str, texto: str, **kwargs):
        if destino in self.destinos_que_fallan:
            raise RuntimeError("boom")
        self.envios.append((destino, texto))

    async def send_batch(self, mensajes: list[tuple[str, str]], **kwargs):
        for destino, texto in mensajes:
            await self.send(destino, texto, **kwargs)


@pytest.fixture
def cola(tmp_path):
    return ColaMensajes(adapter=FakeAdapter(), db_path=tmp_path / "cola_test.db")


class TestEncolar:
    def test_encolar_devuelve_id_creciente(self, cola):
        assert cola.encolar("1", "hola") == 1
        assert cola.encolar("2", "adios") == 2

    def test_encolar_lote_inserta_todos(self, cola):
        assert cola.encolar_lote([("1", "a"), ("2", "b"), ("3", "c")]) == 3


class TestProcesar:
    async def test_envia_cada_mensaje_exactamente_una_vez(self, cola):
        cola.encolar_lote([(f"d{i}", f"m{i}") for i in range(20)])
        enviados = await cola.procesar(n_workers=4)

        assert sum(enviados) == 20
        assert len(cola.adapter.envios) == 20
        # ningún destino repetido: nadie envió un mensaje dos veces
        assert len({e[0] for e in cola.adapter.envios}) == 20
        assert cola._conteo_por_estado() == {"enviado": 20}

    async def test_kwargs_llegan_al_adapter(self, cola):
        cola.encolar("u1", "texto", prioridad="alta")
        await cola.procesar(n_workers=1)

        assert cola.adapter.envios == [("u1", "texto", {"prioridad": "alta"})]

    async def test_workers_envian_en_paralelo(self, tmp_path):
        adapter = FakeAdapter(retraso=0.05)
        cola = ColaMensajes(adapter=adapter, db_path=tmp_path / "cola_par.db")
        cola.encolar_lote([(f"d{i}", f"m{i}") for i in range(8)])

        await cola.procesar(n_workers=4)

        # con workers secuenciales nunca habría 2 envíos a la vez
        assert adapter.max_activos >= 2
        assert len(adapter.envios) == 8

    async def test_cola_vacia_no_envia_nada(self, cola):
        assert await cola.procesar(n_workers=3) == [0, 0, 0]
        assert cola.adapter.envios == []

    async def test_envio_fallido_marca_fallido_y_sigue(self, tmp_path):
        adapter = AdapterQueFalla({"d2"})
        cola = ColaMensajes(adapter=adapter, db_path=tmp_path / "cola_fail.db")
        cola.encolar_lote([("d1", "a"), ("d2", "b"), ("d3", "c")])

        await cola.procesar(n_workers=2)

        assert set(adapter.envios) == {("d1", "a"), ("d3", "c")}
        assert cola._conteo_por_estado() == {"enviado": 2, "fallido": 1}


class TestReservaAtomica:
    def test_reservar_desde_varios_hilos_no_duplica(self, tmp_path):
        """La reserva es atómica aunque los workers sean hilos reales.

        N hilos reclaman sobre la misma cola con conexiones distintas: cada
        mensaje se reclama exactamente una vez. Es la propiedad que garantiza
        que ningún mensaje se envía dos veces.
        """
        cola = ColaMensajes(adapter=FakeAdapter(), db_path=tmp_path / "cola_atomica.db")
        total = 40
        cola.encolar_lote([(f"d{i}", f"m{i}") for i in range(total)])

        reclamados: list[int] = []
        lock = threading.Lock()

        def _reclamar():
            while True:
                fila = cola._reservar()
                if fila is None:
                    return
                with lock:
                    reclamados.append(fila["id"])

        hilos = [threading.Thread(target=_reclamar) for _ in range(8)]
        for h in hilos:
            h.start()
        for h in hilos:
            h.join()

        assert len(reclamados) == total
        assert len(set(reclamados)) == total  # ningún mensaje reclamado dos veces


class TestModoServicio:
    async def test_espera_mensajes_y_se_puede_cancelar(self, cola):
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(cola.procesar(n_workers=1, esperar=True), timeout=0.3)

    async def test_recoge_mensajes_que_llegan_despues(self, cola, monkeypatch):
        monkeypatch.setattr(cm, "_POLL_INTERVALO", 0.05)
        cola.encolar("1", "primero")

        async def _productor():
            await asyncio.sleep(0.1)
            cola.encolar("2", "segundo")

        productor = asyncio.create_task(_productor())
        tarea = asyncio.create_task(cola.procesar(n_workers=1, esperar=True))
        for _ in range(200):
            await asyncio.sleep(0.02)
            if len(cola.adapter.envios) >= 2:
                break
        tarea.cancel()
        with pytest.raises(asyncio.CancelledError):
            await tarea
        await productor

        assert [e[1] for e in cola.adapter.envios] == ["primero", "segundo"]
