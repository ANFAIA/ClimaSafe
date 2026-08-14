"""Tests del cálculo y encolado de la evaluación programada (MSG-002/MSG-004).

El cálculo (predict_ensemble / prediccion_semanal) y el envío (MessageAdapter)
se mockean: nunca se toca la red ni se cargan modelos. Lo que se prueba es el
pegamento: selección de adaptador, perfil/destino desde env, formato del texto
y que el resultado se encola en la cola de MSG-003 para cada destino.
"""

from __future__ import annotations

import pytest

import climasafeai.bot.evaluacion_programada as ep
from climasafeai.bot.cola_mensajes import ColaMensajes
from climasafeai.bot.messaging import HermesAdapter, TelegramAdapter, WebhookAdapter


class TestCrearAdapter:
    def test_default_es_telegram(self, monkeypatch):
        monkeypatch.delenv("MSG_ADAPTER", raising=False)
        assert isinstance(ep.crear_adapter(), TelegramAdapter)

    def test_elige_por_env(self, monkeypatch):
        monkeypatch.setenv("MSG_ADAPTER", "webhook")
        assert isinstance(ep.crear_adapter(), WebhookAdapter)

    def test_hermes_desde_nombre(self):
        assert isinstance(ep.crear_adapter("hermes"), HermesAdapter)

    def test_desconocido_aborta(self, monkeypatch):
        monkeypatch.setenv("MSG_ADAPTER", "carrier_pigeon")
        with pytest.raises(SystemExit, match="MSG_ADAPTER desconocido"):
            ep.crear_adapter()


class TestProbDominante:
    def test_max_de_calor_y_frio(self):
        resultado = {
            "perfil": {
                "calor": {"prob_personalizada": 0.2},
                "frio": {"prob_personalizada": 0.55},
            }
        }
        assert ep._prob_dominante(resultado) == 0.55

    def test_sin_probabilidades_devuelve_cero(self):
        assert ep._prob_dominante({"perfil": {}}) == 0.0


class TestFormatearDia:
    def test_incluye_clase_prob_y_recomendacion(self, monkeypatch):
        monkeypatch.setattr(ep, "recomendacion_resumen", lambda r: "hidrátate")
        texto = ep._formatear_dia(
            "Riesgo mañana",
            {"clase_final_label": "PRECAUCION", "perfil": {"calor": {"prob_personalizada": 0.35}}},
        )
        assert "Riesgo mañana" in texto
        assert "PRECAUCION" in texto
        assert "35%" in texto
        assert "hidrátate" in texto


class TestFormatearResumen:
    def test_hoy_y_manana_con_confianza(self):
        from datetime import date, timedelta

        hoy = date.today()
        manana = hoy + timedelta(days=1)
        serie = {
            "dias": [
                {
                    "fecha": hoy.isoformat(),
                    "clase": "SEGURO",
                    "prob": 0.12,
                    "confianza_conformal": "alta",
                },
                {
                    "fecha": manana.isoformat(),
                    "clase": "PELIGRO",
                    "prob": 0.62,
                    "confianza_conformal": "media",
                },
            ]
        }
        texto = ep._formatear_resumen(serie, "Madrid")
        assert "Resumen diario" in texto
        assert f"Hoy ({hoy.isoformat()}): SEGURO (12%), confianza alta" in texto
        assert f"Mañana ({manana.isoformat()}): PELIGRO (62%), confianza media" in texto


class TestCargarUbicacionYPerfil:
    def test_desde_perfil_de_la_db(self, monkeypatch):
        monkeypatch.setenv("PERFIL_ID", "7")
        monkeypatch.setattr(
            ep._db,
            "obtener_perfil",
            lambda pid: {
                "lat": 40.41,
                "lon": -3.70,
                "provincia": "Madrid",
                "edad": 70,
                "sexo": "hombre",
                "comorbilidades": ["cardiovascular"],
                "farmacos": [],
                "situacion_social": ["vive_solo"],
            },
        )
        perfil, lat, lon, provincia = ep._cargar_ubicacion_y_perfil()
        assert (lat, lon, provincia) == (40.41, -3.70, "Madrid")
        assert perfil["edad"] == 70
        assert perfil["comorbilidades"] == {"cardiovascular"}
        assert perfil["situacion_social"] == {"vive_solo"}

    def test_perfil_inexistente_aborta(self, monkeypatch):
        monkeypatch.setenv("PERFIL_ID", "999")
        monkeypatch.setattr(ep._db, "obtener_perfil", lambda pid: None)
        with pytest.raises(SystemExit, match="no existe"):
            ep._cargar_ubicacion_y_perfil()

    def test_desde_env_sin_perfil(self, monkeypatch):
        monkeypatch.delenv("PERFIL_ID", raising=False)
        monkeypatch.setenv("LAT", "41.38")
        monkeypatch.setenv("LON", "2.17")
        monkeypatch.setenv("PROVINCIA", "Barcelona")
        monkeypatch.setenv("EDAD", "65")
        perfil, lat, lon, provincia = ep._cargar_ubicacion_y_perfil()
        assert (lat, lon, provincia) == (41.38, 2.17, "Barcelona")
        assert perfil["edad"] == 65

    def test_sin_ubicacion_aborta(self, monkeypatch):
        monkeypatch.delenv("PERFIL_ID", raising=False)
        monkeypatch.delenv("LAT", raising=False)
        monkeypatch.delenv("LON", raising=False)
        with pytest.raises(SystemExit, match="Falta PERFIL_ID o LAT/LON"):
            ep._cargar_ubicacion_y_perfil()


class TestCalcularTexto:
    async def test_resumen_usa_prediccion_semanal(self, monkeypatch):
        monkeypatch.setattr(
            ep,
            "prediccion_semanal",
            lambda **kw: {
                "dias": [
                    {
                        "fecha": "2026-08-13",
                        "clase": "SEGURO",
                        "prob": 0.1,
                        "confianza_conformal": "alta",
                    },
                    {
                        "fecha": "2026-08-14",
                        "clase": "SEGURO",
                        "prob": 0.1,
                        "confianza_conformal": "alta",
                    },
                ]
            },
        )
        texto = await ep._calcular_texto("resumen", 40.41, -3.70, "Madrid", {})
        assert texto.startswith("📋 *Resumen diario*")

    async def test_manana_usa_predict_ensemble(self, monkeypatch):
        llamado: dict = {}

        def _fake_predict(**kwargs):
            llamado.update(kwargs)
            return {
                "clase_final_label": "PELIGRO",
                "perfil": {"calor": {"prob_personalizada": 0.7}},
            }

        monkeypatch.setattr(ep, "predict_ensemble", _fake_predict)
        monkeypatch.setattr(ep, "recomendacion_resumen", lambda r: "evita la calle")
        texto = await ep._calcular_texto("manana", 40.41, -3.70, "Madrid", {})
        assert llamado["target_date"] is not None
        assert llamado["provincia"] == "Madrid"
        assert "PELIGRO" in texto and "70%" in texto


class FakeAdapter:
    """MessageAdapter de prueba: registra los mensajes, no toca la red."""

    def __init__(self):
        self.mensajes: list[tuple[str, str]] = []
        self.cerrado = False

    async def send(self, destino: str, texto: str, **kwargs):
        self.mensajes.append((destino, texto))

    async def send_batch(self, mensajes: list[tuple[str, str]], **kwargs):
        self.mensajes.extend(mensajes)

    async def close(self):
        self.cerrado = True


class TestEncolarEvaluacion:
    @pytest.fixture
    def cola(self, tmp_path):
        return ColaMensajes(adapter=FakeAdapter(), db_path=tmp_path / "cola_ep.db")

    @pytest.fixture
    def env_ok(self, monkeypatch):
        monkeypatch.setenv("MSG_DESTINO", "111, 222")
        monkeypatch.delenv("PERFIL_ID", raising=False)
        monkeypatch.setenv("LAT", "40.41")
        monkeypatch.setenv("LON", "-3.70")

    async def test_encola_el_texto_calculado_para_cada_destino(self, monkeypatch, cola, env_ok):
        async def _fake_calcular(*a, **k):
            return "texto del resumen"

        monkeypatch.setattr(ep, "_calcular_texto", _fake_calcular)

        encolados = await ep.encolar_evaluacion(cola)

        assert encolados == 2
        # el envío lo hace el worker de la cola con el adapter de la cola
        await cola.procesar(n_workers=1)
        assert cola.adapter.mensajes == [("111", "texto del resumen"), ("222", "texto del resumen")]
        assert cola._conteo_por_estado() == {"enviado": 2}

    async def test_encola_la_tarea_pasada(self, monkeypatch, cola, env_ok):
        llamado: dict = {}

        async def _fake_calcular(*a, **k):
            llamado["tarea"] = a[0]
            return "texto"

        monkeypatch.setattr(ep, "_calcular_texto", _fake_calcular)

        await ep.encolar_evaluacion(cola, "manana")

        assert llamado["tarea"] == "manana"

    async def test_sin_destino_aborta(self, monkeypatch, cola):
        monkeypatch.setenv("MSG_DESTINO", "")
        monkeypatch.delenv("PERFIL_ID", raising=False)
        monkeypatch.setenv("LAT", "40.41")
        monkeypatch.setenv("LON", "-3.70")
        with pytest.raises(SystemExit, match="MSG_DESTINO es obligatorio"):
            await ep.encolar_evaluacion(cola)

    async def test_tarea_desconocida_aborta(self, monkeypatch, cola, env_ok):
        with pytest.raises(SystemExit, match="CRON_TAREA desconocida"):
            await ep.encolar_evaluacion(cola, "pasado_mañana")
