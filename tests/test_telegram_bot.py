"""Tests del formulario determinista del bot de Telegram (BOT-002)."""

from __future__ import annotations

from datetime import date

import pytest

from climasafeai.bot.telegram_bot import (
    Estado,
    FIELD_LABELS,
    _conversaciones,
    _edad_desde_fecha_nacimiento,
    _parsear_fecha_nacimiento,
    procesar_mensaje,
    procesar_callback,
    _siguiente,
    _format_template,
)


class _HoyFijo(date):
    """Fija 'hoy' en 2026-08-03 para que la edad calculada sea determinista."""

    @classmethod
    def today(cls):
        return cls(2026, 8, 3)


class _FakeDBSinPerfil:
    """DB sin perfiles guardados: el bot no debe tocar el SQLite real en tests."""

    def buscar_por_telegram(self, chat_id: str):
        return None


def _conv_limpia(chat_id: int = 1):
    """Resetea la conversación para el test."""
    _conversaciones.clear()
    _conversaciones[chat_id] = {"estado": Estado.SEXO, "data": {}}
    return _conversaciones[chat_id]


@pytest.fixture(autouse=True)
def limpiar():
    _conversaciones.clear()
    yield
    _conversaciones.clear()


class TestOrdenCampos:
    def test_estados_en_orden(self):
        """Verifica que todos los estados están en FIELD_LABELS y en orden."""
        order = list(Estado)
        # IDLE y DONE no tienen campo (DONE es el cierre del formulario)
        no_campo = {Estado.IDLE, Estado.DONE}
        etiquetados = [e for e in order if e not in no_campo]
        for est in etiquetados:
            assert est in FIELD_LABELS, f"Falta FIELD_LABELS para {est}"
        assert len(etiquetados) == len(FIELD_LABELS)
        assert etiquetados == list(FIELD_LABELS.keys())

    def test_siguiente_flujo_completo(self):
        """Recorre todos los estados desde SEXO hasta DONE."""
        order = list(Estado)
        idx_sexo = order.index(Estado.SEXO)
        idx_done = order.index(Estado.DONE)
        for i in range(idx_sexo, idx_done):
            actual = order[i]
            sig = _siguiente(actual)
            assert sig == order[i + 1], f"{actual} → {sig}, esperado {order[i + 1]}"


class TestValidacionNumerica:
    @pytest.mark.asyncio
    async def test_duracion_valida(self):
        _conv_limpia()
        _conversaciones[1]["estado"] = Estado.DURACION
        r = await procesar_mensaje(1, "3.5")
        assert r is None
        assert _conversaciones[1]["data"]["duracion_h"] == 3.5

    @pytest.mark.asyncio
    async def test_duracion_fuera_rango(self):
        _conv_limpia()
        _conversaciones[1]["estado"] = Estado.DURACION
        r = await procesar_mensaje(1, "30")
        assert r is not None
        assert "24" in r

    @pytest.mark.asyncio
    async def test_hora_inicio_valida(self):
        _conv_limpia()
        _conversaciones[1]["estado"] = Estado.HORA_INICIO
        r = await procesar_mensaje(1, "8")
        assert r is None
        assert _conversaciones[1]["data"]["hora_inicio"] == 8

    @pytest.mark.asyncio
    async def test_hora_inicio_con_minutos(self):
        _conv_limpia()
        _conversaciones[1]["estado"] = Estado.HORA_INICIO
        r = await procesar_mensaje(1, "8:30")
        assert r is None
        assert _conversaciones[1]["data"]["hora_inicio"] == 8

    @pytest.mark.asyncio
    async def test_hora_inicio_fuera_rango(self):
        _conv_limpia()
        _conversaciones[1]["estado"] = Estado.HORA_INICIO
        r = await procesar_mensaje(1, "25")
        assert r is not None
        assert "0 y 23" in r or "24" in r

    @pytest.mark.asyncio
    async def test_grasa_opcional_saltar(self):
        _conv_limpia()
        _conversaciones[1]["estado"] = Estado.GRASA
        r = await procesar_mensaje(1, "saltar")
        assert r is None
        assert _conversaciones[1]["data"]["porcentaje_grasa"] is None
        # Debe avanzar al siguiente estado (FOTOTIPO)
        assert _conversaciones[1]["estado"] == Estado.FOTOTIPO

    @pytest.mark.asyncio
    async def test_grasa_acepta_coma_y_porcentaje(self):
        _conv_limpia()
        _conversaciones[1]["estado"] = Estado.GRASA
        r = await procesar_mensaje(1, "20,5%")
        assert r is None
        assert _conversaciones[1]["data"]["porcentaje_grasa"] == 20.5

    @pytest.mark.asyncio
    async def test_grasa_fuera_de_rango_no_avanza(self):
        _conv_limpia()
        _conversaciones[1]["estado"] = Estado.GRASA
        r = await procesar_mensaje(1, "120")
        assert r is not None and "entre 3 y 65" in r
        assert _conversaciones[1]["estado"] == Estado.GRASA


class TestBotonesCallback:
    @pytest.mark.asyncio
    async def test_sexo_hombre(self):
        _conv_limpia()
        _conversaciones[1]["estado"] = Estado.SEXO
        texto, es_final = await procesar_callback(1, "hombre")
        assert texto is None
        assert not es_final
        assert _conversaciones[1]["data"]["sexo"] == "hombre"
        # Debe avanzar a EDAD
        assert _conversaciones[1]["estado"] == Estado.EDAD

    @pytest.mark.asyncio
    async def test_sexo_mujer(self):
        _conv_limpia()
        _conversaciones[1]["estado"] = Estado.SEXO
        await procesar_callback(1, "mujer")
        assert _conversaciones[1]["data"]["sexo"] == "mujer"

    @pytest.mark.asyncio
    async def test_aclimatado_si(self):
        _conv_limpia()
        _conversaciones[1]["estado"] = Estado.ACLIMATADO
        await procesar_callback(1, "si")
        assert _conversaciones[1]["data"]["aclimatado"] is True

    @pytest.mark.asyncio
    async def test_aclimatado_no(self):
        _conv_limpia()
        _conversaciones[1]["estado"] = Estado.ACLIMATADO
        await procesar_callback(1, "no")
        assert _conversaciones[1]["data"]["aclimatado"] is False

    @pytest.mark.asyncio
    async def test_actividad_intensa(self):
        _conv_limpia()
        _conversaciones[1]["estado"] = Estado.ACTIVIDAD
        await procesar_callback(1, "intensa")
        assert _conversaciones[1]["data"]["nivel_actividad"] == "intensa"

    @pytest.mark.asyncio
    async def test_rama_trabajo_pregunta_el_tipo(self):
        _conv_limpia()
        _conversaciones[1]["estado"] = Estado.TRABAJO
        await procesar_callback(1, "trabajo")
        assert _conversaciones[1]["estado"] == Estado.TIPO_TRABAJO
        await procesar_callback(1, "campo")
        assert _conversaciones[1]["data"]["ocupacion"] == "campo"
        # No se le pregunta qué deporte hace: va a trabajar
        assert _conversaciones[1]["estado"] == Estado.COMORBILIDADES

    @pytest.mark.asyncio
    async def test_rama_propia_pregunta_la_actividad_y_no_lleva_ocupacion(self):
        """Trabajar en el campo no cuenta si ESTA salida es un paseo."""
        _conv_limpia()
        _conversaciones[1]["estado"] = Estado.TRABAJO
        await procesar_callback(1, "propia")
        assert _conversaciones[1]["data"]["ocupacion"] is None
        # Se le pregunta qué va a hacer, saltándose el tipo de trabajo
        assert _conversaciones[1]["estado"] == Estado.DEPORTE
        await procesar_mensaje(1, "senderismo")
        assert _conversaciones[1]["data"]["deporte"] == "senderismo"

    @pytest.mark.asyncio
    async def test_estado_previo_multiselect(self):
        """Fiesta pesa x1.8, la mala noche x1.2 y la enfermedad reciente x1.3."""
        _conv_limpia()
        _conversaciones[1]["estado"] = Estado.ESTADO_PREVIO
        await procesar_callback(1, "fiesta")
        await procesar_callback(1, "falta_sueno")
        assert _conversaciones[1]["data"]["estado_previo"] == {"fiesta", "falta_sueno"}

    @pytest.mark.asyncio
    async def test_medicacion_multiselect(self):
        _conv_limpia()
        _conversaciones[1]["estado"] = Estado.MEDICACION
        await procesar_callback(1, "diureticos_asa")
        assert _conversaciones[1]["data"]["farmacos"] == {"diureticos_asa"}
        # volver a pulsar lo quita
        await procesar_callback(1, "diureticos_asa")
        assert _conversaciones[1]["data"]["farmacos"] == set()

    @pytest.mark.asyncio
    async def test_comorbilidades_multiselect(self):
        _conv_limpia()
        _conversaciones[1]["estado"] = Estado.COMORBILIDADES
        # Añadir cardiovascular
        await procesar_callback(1, "cardiovascular")
        assert "cardiovascular" in _conversaciones[1]["data"]["comorbilidades"]
        assert _conversaciones[1]["estado"] == Estado.COMORBILIDADES  # no avanza
        # Añadir diabetes
        await procesar_callback(1, "diabetes")
        assert len(_conversaciones[1]["data"]["comorbilidades"]) == 2
        # Terminar
        texto, es_final = await procesar_callback(1, "__done__")
        assert not es_final
        assert _conversaciones[1]["estado"] != Estado.COMORBILIDADES  # avanza

    @pytest.mark.asyncio
    async def test_ubicacion_escrita_se_geocodifica(self, monkeypatch):
        """El nombre lo resuelve Nominatim, nunca un LLM."""
        import climasafeai.bot.telegram_bot as mod
        monkeypatch.setattr(mod, "buscar_lugar", lambda n: {
            "lat": 42.29, "lon": -8.81, "provincia": "Pontevedra", "nombre": "Aldán, Pontevedra",
        })
        _conv_limpia()
        _conversaciones[1]["estado"] = Estado.UBICACION
        r = await procesar_mensaje(1, "Aldán")
        assert r is None
        assert _conversaciones[1]["data"]["lat"] == 42.29
        assert _conversaciones[1]["data"]["provincia"] == "Pontevedra"
        assert _conversaciones[1]["estado"] == Estado.GUARDAR_PERFIL

    @pytest.mark.asyncio
    async def test_ubicacion_no_encontrada_no_avanza(self, monkeypatch):
        """Si no se encuentra, se vuelve a preguntar. Jamás se inventan coordenadas."""
        import climasafeai.bot.telegram_bot as mod
        monkeypatch.setattr(mod, "buscar_lugar", lambda n: None)
        _conv_limpia()
        _conversaciones[1]["estado"] = Estado.UBICACION
        r = await procesar_mensaje(1, "Zzzqqq")
        assert r is not None and "no he encontrado" in r.lower()
        assert _conversaciones[1]["estado"] == Estado.UBICACION
        assert "lat" not in _conversaciones[1]["data"]


class TestTemplate:
    def test_template_siempre_funciona(self):
        """La plantilla no requiere LLM ni API key."""
        result = {
            "clase_final": 2,
            "clase_final_label": "PELIGRO",
            "perfil": {"calor": {"prob_personalizada": 0.72}},
            "perfil_usuario": {"hora_inicio": 15, "duracion_actividad_h": 1},
            "weather": {
                "provincia": "Sevilla",
                "current": {"t2m_c": 38.5, "rh": 45},
                "uv_index": 8,
                "perfil_horario": [{"hora": 15, "HI": 39.0, "temp": 38.0}],
            },
            "recomendaciones": [
                "No te expongas al sol",
                "Bebe agua",
                "Usa protección solar",
                "Evita esfuerzos",
            ],
        }
        texto = _format_template(result, "Sevilla")
        # BOT-013: la clase va anclada en su escala y el % como frecuencia
        assert "Sevilla — PELIGRO, el nivel más alto de tres" in texto
        assert "en unos 72 el calor te pasaría factura" in texto
        # BOT-020: el parte abre con la clasificación y la probabilidad en %
        assert texto.startswith(
            "Clasificación: PELIGRO — probabilidad de riesgo personalizada por calor: 72% (0.7200)."
        )
        assert "🌡️ Temperatura prevista: 38.0 °C" in texto
        assert "☀️ Índice UV (media): 8" in texto
        assert "Recomendaciones de la herramienta (nivel PELIGRO, no las suavizo):" in texto


class TestParteFinal:
    """BOT-005: el parte incluye riesgo con %, temperatura y UV, no solo la clase."""

    def _resultado(self, lugar="Moaña, Pontevedra"):
        return {
            "clase_final": 1,
            "clase_final_label": "PRECAUCIÓN",
            "perfil": {"calor": {"prob_personalizada": 0.2}},
            "perfil_usuario": {"hora_inicio": 8, "duracion_actividad_h": 2},
            "weather": {
                "provincia": "Pontevedra",
                "current": {"t2m_c": 30.0, "rh": 60},
                "uv_index": 6,
                "perfil_horario": [
                    {"hora": 8, "HI": 25.0, "temp": 20.0},
                    {"hora": 9, "HI": 26.0, "temp": 22.0},
                    {"hora": 10, "HI": 28.0, "temp": 24.0},
                ],
            },
            "recomendaciones": ["Mantente hidratado"],
        }

    def test_parte_incluye_riesgo_temperatura_uv_y_ubicacion(self):
        texto = _format_template(self._resultado(), "Moaña, Pontevedra")
        assert "Moaña, Pontevedra — PRECAUCIÓN, el nivel intermedio de tres" in texto
        assert "en unos 20 el calor te pasaría factura" in texto
        # Temperatura: media en las horas de actividad (8-9 → (20+22)/2)
        assert "🌡️ Temperatura prevista: 21.0 °C" in texto
        assert "☀️ Índice UV (media): 6" in texto

    def test_sin_perfil_horario_cae_a_temperatura_actual_y_uv_nd(self):
        result = self._resultado()
        result["weather"].pop("perfil_horario")
        result["weather"]["current"] = {"t2m_c": 22.0, "rh": 70}
        result["weather"]["uv_index"] = None
        texto = _format_template(result)
        assert "🌡️ Temperatura prevista: 22.0 °C" in texto
        assert "☀️ Índice UV (media): n/d" in texto


class TestStartConPerfil:
    """BOT-005: con perfil previo, /start avisa y pregunta la intensidad."""

    @pytest.mark.asyncio
    async def test_start_con_perfil_avisa_y_pregunta_intensidad(self, monkeypatch):
        import climasafeai.bot.telegram_bot as mod

        class _FakeDB:
            def buscar_por_telegram(self, chat_id: str):
                return {"id": 7, "alias": "Aldán"} if chat_id == "1" else None

            def obtener_perfil(self, _pid: int):
                return {
                    "alias": "Aldán", "sexo": "hombre", "edad": 57,
                    "comorbilidades": [], "farmacos": [], "situacion_social": [],
                }

        enviados: list[str] = []

        async def _fake_tg(method: str, **kwargs):
            if method == "sendMessage":
                enviados.append(kwargs["text"])
            return {"ok": True, "result": {}}

        monkeypatch.setattr(mod, "_db", _FakeDB())
        monkeypatch.setattr(mod, "_tg", _fake_tg)
        monkeypatch.setattr(mod, "_modelo_por_defecto", lambda: mod.MODELO_DETERMINISTA)

        await mod.procesar_update({"message": {"chat": {"id": 1}, "text": "/start"}})

        assert len(enviados) == 2, enviados
        assert "se cargaron tus datos previos" in enviados[0].lower(), enviados[0]
        assert "¿Qué intensidad tendrá la actividad?" in enviados[1], enviados
        assert _conversaciones[1]["estado"] == Estado.ACTIVIDAD


class TestRecomendacionContexto:
    """BOT-005: la recomendación se adapta al contexto, no es SPF 30+ siempre."""

    @staticmethod
    def _resultado(t=None, uv=None, wc=None, hi=None, clase=1):
        return {
            "clase_final": clase,
            "weather": {
                "current": {"t2m_c": t},
                "uv_index": uv,
            },
            "modelos": {
                "Formula": {
                    "frio": {"wind_chill_c": wc},
                    "calor": {"heat_index_c": hi},
                },
            },
        }

    def test_frio_recomienda_abrigo_sin_spf(self):
        from climasafeai.models.recomendaciones import recomendacion_resumen
        rec = recomendacion_resumen(self._resultado(t=5, uv=1, wc=-2, hi=5))
        assert "abrígate" in rec
        assert "SPF" not in rec

    def test_calor_con_uv_recomienda_spf_y_evitar_horas_centrales(self):
        from climasafeai.models.recomendaciones import recomendacion_resumen
        rec = recomendacion_resumen(self._resultado(t=34, uv=7, wc=20, hi=36))
        assert "Mantente hidratado" in rec
        assert "SPF 30+" in rec
        assert "evita la exposición prolongada entre las horas de mayor calor" in rec

    def test_tiempo_suave_sin_uv_no_impone_spf(self):
        from climasafeai.models.recomendaciones import recomendacion_resumen
        rec = recomendacion_resumen(self._resultado(t=22, uv=2, wc=15, hi=22, clase=0))
        assert "SPF" not in rec
        assert "hidratado" in rec

    def test_peligro_recomienda_no_hacer_actividad(self):
        from climasafeai.models.recomendaciones import recomendacion_resumen
        rec = recomendacion_resumen(self._resultado(t=38, uv=9, wc=25, hi=42, clase=2))
        assert "evita la actividad física" in rec


class TestRecomendacionCanalDominante:
    """BOT-011: la recomendación sigue al canal dominante y no mezcla canales.

    El canal que manda es el de mayor `prob_personalizada` (o el único que
    supera el umbral). El canal que queda por debajo de 0.15 NO aporta
    recomendaciones, aunque el clima físico apunte a él.
    """

    @staticmethod
    def _resultado(prob_calor=0.0, prob_frio=0.0, t=20, uv=None, wc=10, hi=25, clase=1):
        return {
            "clase_final": clase,
            "perfil": {
                "calor": {"prob_personalizada": prob_calor, "factores": ["no_aclimatado"]},
                "frio": {"prob_personalizada": prob_frio, "factores": ["edad"]},
            },
            "weather": {
                "current": {"t2m_c": t},
                "uv_index": uv,
                "provincia": "Pontevedra",
            },
            "modelos": {
                "Formula": {
                    "frio": {"wind_chill_c": wc},
                    "calor": {"heat_index_c": hi},
                },
            },
        }

    def test_recomendacion_canal_dominante_calor_no_mezcla_frio(self):
        from climasafeai.models.recomendaciones import recomendacion_resumen

        # O Casal, Pontevedra: PRECAUCIÓN 21%, 35.3 °C, UV 7.6 → manda calor
        rec = recomendacion_resumen(self._resultado(
            prob_calor=0.21, prob_frio=0.02, t=35.3, uv=7.6, wc=5, hi=38, clase=1,
        ))
        assert "evita la exposición prolongada entre las horas de mayor calor" in rec
        assert "SPF 30+" in rec
        assert "abrígate" not in rec  # el canal frío no manda: nada de abrigo

    def test_recomendacion_canal_dominante_frio_no_mezcla_calor(self):
        from climasafeai.models.recomendaciones import recomendacion_resumen

        # WC muy negativo, prob_frio alta → manda frío
        rec = recomendacion_resumen(self._resultado(
            prob_calor=0.03, prob_frio=0.45, t=2, uv=1, wc=-18, hi=4, clase=1,
        ))
        assert "abrígate" in rec
        assert "evita la exposición prolongada" not in rec  # nada de calor
        assert "SPF" not in rec

    def test_recomendacion_canal_bajo_umbral_no_aporta_aunque_el_clima_apunte(self):
        from climasafeai.models.recomendaciones import recomendacion_resumen

        # prob_frio=0.10 < 0.15: aunque haga frío físico (wc=-3, t=5), el canal
        # frío no aporta recomendaciones; tampoco el calor (prob 0.05)
        rec = recomendacion_resumen(self._resultado(
            prob_calor=0.05, prob_frio=0.10, t=5, uv=2, wc=-3, hi=8, clase=0,
        ))
        assert "abrígate" not in rec
        assert "evita la exposición prolongada" not in rec
        assert "hidratado" in rec

    def test_recomendacion_canal_dominante_ambos_activos_gana_el_mayor(self):
        from climasafeai.models.recomendaciones import recomendacion_resumen

        rec = recomendacion_resumen(self._resultado(
            prob_calor=0.30, prob_frio=0.55, t=12, uv=3, wc=-5, hi=18, clase=1,
        ))
        assert "abrígate" in rec
        assert "evita la exposición prolongada" not in rec

    def test_recomendacion_sin_canales_degrada_a_clima_fisico(self):
        from climasafeai.models.recomendaciones import recomendacion_resumen

        # Sin `perfil` (dict mínimo): se mantiene la lógica previa por clima físico
        rec = recomendacion_resumen(TestRecomendacionContexto._resultado(t=5, uv=1, wc=-2, hi=5))
        assert "abrígate" in rec


class TestFlujoCompleto:
    @pytest.mark.asyncio
    async def test_flujo_simulado(self, monkeypatch):
        """Simula una conversación completa desde /start hasta DONE."""
        import climasafeai.bot.telegram_bot as mod
        monkeypatch.setattr(mod, "buscar_lugar", lambda n: {
            "lat": 42.29, "lon": -8.81, "provincia": "Pontevedra", "nombre": "Aldán, Pontevedra",
        })
        monkeypatch.setattr(mod, "date", _HoyFijo)
        _conv_limpia()
        _conversaciones[1]["estado"] = Estado.SEXO

        await procesar_callback(1, "hombre")
        assert _conversaciones[1]["estado"] == Estado.EDAD

        await procesar_mensaje(1, "15/03/1990")
        assert _conversaciones[1]["data"]["edad"] == 36
        assert _conversaciones[1]["data"]["fecha_nacimiento"] == "15/03/1990"
        assert _conversaciones[1]["estado"] == Estado.GRASA

        await procesar_mensaje(1, "saltar")
        assert _conversaciones[1]["estado"] == Estado.FOTOTIPO, f"Esperado FOTOTIPO, es {_conversaciones[1]['estado']}"

        # fototipo: elige tipo 3 (o saltar como texto)
        await procesar_callback(1, "3")
        assert _conversaciones[1]["data"]["fototipo"] == "3"
        assert _conversaciones[1]["estado"] == Estado.ACLIMATADO

        await procesar_callback(1, "no")
        assert _conversaciones[1]["data"]["aclimatado"] is False

        await procesar_callback(1, "intensa")
        assert _conversaciones[1]["data"]["nivel_actividad"] == "intensa"

        await procesar_callback(1, "no")          # entrenado
        assert _conversaciones[1]["data"]["entrenado"] is False

        await procesar_mensaje(1, "8")            # duración
        assert _conversaciones[1]["data"]["duracion_h"] == 8

        await procesar_mensaje(1, "8")            # hora de inicio
        assert _conversaciones[1]["data"]["hora_inicio"] == 8

        await procesar_callback(1, "trabajo")     # sale a trabajar
        assert _conversaciones[1]["estado"] == Estado.TIPO_TRABAJO
        await procesar_callback(1, "campo")
        assert _conversaciones[1]["data"]["ocupacion"] == "campo"
        # Va a trabajar: no se le pregunta qué deporte hace
        assert _conversaciones[1]["estado"] == Estado.COMORBILIDADES

        await procesar_callback(1, "cardiovascular")
        await procesar_callback(1, "__done__")    # comorbilidades

        await procesar_callback(1, "diureticos_asa")
        await procesar_callback(1, "__done__")    # medicación

        await procesar_callback(1, "fiesta")
        await procesar_callback(1, "__done__")    # cómo llega a la salida
        assert _conversaciones[1]["estado"] == Estado.SITUACION_SOCIAL

        # situación social: skip
        await procesar_callback(1, "__done__")
        assert _conversaciones[1]["estado"] == Estado.UBICACION

        await procesar_mensaje(1, "Aldán")
        assert _conversaciones[1]["estado"] == Estado.GUARDAR_PERFIL

        # Rechazar guardar perfil → DONE
        await procesar_callback(1, "guardar_no")
        assert _conversaciones[1]["estado"] == Estado.DONE

        data = _conversaciones[1]["data"]
        assert data["sexo"] == "hombre"
        assert data["comorbilidades"] == {"cardiovascular"}
        assert data["farmacos"] == {"diureticos_asa"}
        assert data["estado_previo"] == {"fiesta"}
        assert data["lat"] == 42.29

    @pytest.mark.asyncio
    async def test_perfil_usa_las_claves_que_lee_el_modelo(self, monkeypatch):
        """El fallo histórico de este proyecto: una clave mal escrita no da error.

        `medicacion` en vez de `farmacos`, o `grasa_corporal` en vez de
        `porcentaje_grasa`, hacen que el factor se salte en silencio y que el
        riesgo salga por debajo del que el propio modelo calcula.
        """
        import climasafeai.bot.telegram_bot as mod

        capturado = {}

        def _fake_predict(**kwargs):
            capturado.update(kwargs)
            return {
                "clase_final_label": "SEGURO",
                "weather": {"provincia": "Pontevedra", "current": {"t2m_c": 22.0, "rh": 70}},
                "perfil": {"calor": {"prob_personalizada": 0.1, "factores": []}},
                "recomendaciones": [],
            }

        def _sin_llm(_perfil, _result, _config=None):
            return None

        monkeypatch.setattr(mod, "predict_ensemble", _fake_predict)
        monkeypatch.setattr("climasafeai.llm.rag_qwen.ask_con_perfil", _sin_llm)

        _conversaciones.clear()
        _conversaciones[1] = {"estado": Estado.DONE, "data": {
            "sexo": "hombre", "edad": 57, "porcentaje_grasa": 20.5,
            "fototipo": "3",
            "aclimatado": False, "nivel_actividad": "intensa", "entrenado": True,
            "duracion_h": 8, "hora_inicio": 8, "ocupacion": "campo",
            "comorbilidades": {"cardiovascular"}, "farmacos": {"diureticos_asa"},
            "estado_previo": {"fiesta", "falta_sueno"},
            "situacion_social": {"vive_solo"},
            "lat": 42.29, "lon": -8.81, "provincia": "Pontevedra", "lugar": "Aldán",
        }}

        texto = await mod.ejecutar_prediccion(1)

        perfil = capturado["perfil"]
        assert perfil["farmacos"] == {"diureticos_asa"}
        assert perfil["porcentaje_grasa"] == 20.5
        assert perfil["fototipo"] == "3"
        assert perfil["situacion_social"] == {"vive_solo"}
        assert perfil["ocupacion"] == "campo"
        assert perfil["entrenado"] is True
        assert perfil["fiesta"] is True and perfil["falta_sueno"] is True
        assert "enfermedad_reciente" not in perfil
        assert capturado["lat"] == 42.29 and capturado["lon"] == -8.81
        # Campos que no lee nadie: no deben viajar
        for muerto in ("peso", "altura", "tipo_actividad", "medicacion", "grasa_corporal"):
            assert muerto not in perfil
        # Sin LLM, responde igual y dice dónde
        assert "Aldán" in texto


class TestGeocodificacion:
    """Funciones puras del geocodificador: sin red."""

    @pytest.mark.parametrize("address, esperado", [
        ({"province": "Provincia de Pontevedra"}, "Pontevedra"),
        ({"province": "Pontevedra"}, "Pontevedra"),
        ({"state": "Madrid"}, "Madrid"),
        ({"county": "O Morrazo"}, "O Morrazo"),
        ({"city": "Vigo"}, None),
        ({}, None),
    ])
    def test_extraer_provincia(self, address, esperado):
        from climasafeai.bot.geocoding import _extraer_provincia
        assert _extraer_provincia(address) == esperado

    def test_nombre_prefiere_lo_que_busco_el_usuario(self):
        """Quien busca 'Aldán' quiere ver 'Aldán', no el municipio que lo contiene."""
        from climasafeai.bot.geocoding import _extraer_nombre
        item = {"name": "Aldán",
                "address": {"village": "Cangas de Morrazo", "province": "Pontevedra"}}
        assert _extraer_nombre(item) == "Aldán, Pontevedra"

    def test_buscar_lugar_vacio_no_llama_a_la_red(self):
        from climasafeai.bot.geocoding import buscar_lugar
        assert buscar_lugar("") is None
        assert buscar_lugar("   ") is None


class TestDeporteMET:
    """El deporte no tiene coeficiente propio: fija la intensidad por su MET.

    Fuente de los MET: 2024 Adult Compendium of Physical Activities
    (doi:10.1016/j.jshs.2023.10.010).
    """

    @pytest.mark.parametrize("deporte, nivel", [
        ("pasear", "moderada"),          # 3.5 MET
        ("senderismo", "intensa"),       # 6.0
        ("futbol", "intensa"),           # 7.0
        ("tenis", "muy_intensa"),        # 8.0
        ("correr", "muy_intensa"),       # 10.5
    ])
    def test_met_fija_la_intensidad(self, deporte, nivel):
        from climasafeai.features.personalizacion import nivel_actividad_de_deporte
        assert nivel_actividad_de_deporte(deporte) == nivel

    def test_deporte_desconocido_no_inventa_intensidad(self):
        """El pádel no está en el Compendium: mejor None que un número inventado."""
        from climasafeai.features.personalizacion import nivel_actividad_de_deporte
        assert nivel_actividad_de_deporte("padel") is None
        assert nivel_actividad_de_deporte(None) is None

    def test_solo_se_ofrecen_deportes_con_met(self):
        from climasafeai.bot.telegram_bot import DEPORTES
        from climasafeai.features.personalizacion import DEPORTE_MET
        assert set(DEPORTES) <= set(DEPORTE_MET), "hay deportes en el menú sin MET medido"

    @pytest.mark.asyncio
    async def test_elegir_deporte_sobrescribe_la_intensidad(self):
        _conv_limpia()
        _conversaciones[1]["estado"] = Estado.DEPORTE
        _conversaciones[1]["data"]["nivel_actividad"] = "moderada"   # lo que dijo él
        await procesar_callback(1, "tenis")                          # 8 MET
        assert _conversaciones[1]["data"]["nivel_actividad"] == "muy_intensa"
        assert _conversaciones[1]["data"]["_nivel_desde_deporte"] is True

    @pytest.mark.asyncio
    async def test_deporte_escrito_a_mano_respeta_la_intensidad(self):
        _conv_limpia()
        _conversaciones[1]["estado"] = Estado.DEPORTE
        _conversaciones[1]["data"]["nivel_actividad"] = "ligera"
        _conversaciones[1]["data"]["_texto_deporte"] = True
        await procesar_mensaje(1, "padel")
        assert _conversaciones[1]["data"]["deporte"] == "padel"
        assert _conversaciones[1]["data"]["nivel_actividad"] == "ligera"


class TestBienvenida:
    """BOT-003 + CHAT-003: el primer contacto y /help explican /start, sin /chat."""

    @staticmethod
    def _sin_comandos_de_depuracion(texto: str) -> None:
        for cmd in ("/model", "/qwen", "/api", "/determinista"):
            assert cmd not in texto, f"{cmd} no lo necesita el usuario final"

    @pytest.mark.asyncio
    async def test_help_explica_start_como_unico_camino(self, monkeypatch):
        import climasafeai.bot.telegram_bot as mod
        monkeypatch.setattr(mod, "_modelo_por_defecto", lambda: mod.MODELO_DETERMINISTA)
        _conversaciones[1] = {"estado": Estado.IDLE, "data": {}}

        r = await procesar_mensaje(1, "/help")

        assert "/start" in r and "/chat" not in r, r
        assert "Cuestionario" in r, r
        self._sin_comandos_de_depuracion(r)

    @pytest.mark.asyncio
    async def test_help_invita_a_preguntar_dudas_tras_el_parte(self, monkeypatch):
        """CHAT-003: tras el parte se pueden preguntar dudas (p. ej. SPF)."""
        import climasafeai.bot.telegram_bot as mod
        monkeypatch.setattr(mod, "_modelo_por_defecto", lambda: mod.MODELO_DETERMINISTA)
        _conversaciones[1] = {"estado": Estado.IDLE, "data": {}}

        r = await procesar_mensaje(1, "/help")

        assert "/chat" not in r, r
        assert "pregunt" in r.lower(), r
        assert "SPF" in r, r

    @pytest.mark.asyncio
    async def test_primer_contacto_sin_comando_muestra_la_bienvenida(self, monkeypatch):
        """Antes contestaba 'Envía /start para comenzar.' y nada más."""
        import climasafeai.bot.telegram_bot as mod
        monkeypatch.setattr(mod, "_modelo_por_defecto", lambda: mod.MODELO_DETERMINISTA)

        r = await procesar_mensaje(1, "hola")

        assert r == mod.BIENVENIDA, r

    @pytest.mark.asyncio
    async def test_start_sin_perfil_manda_bienvenida_y_luego_la_primera_pregunta(self, monkeypatch):
        import climasafeai.bot.telegram_bot as mod

        class _FakeDB:
            def buscar_por_telegram(self, chat_id: str):
                return None

        enviados: list[str] = []

        async def _fake_tg(method: str, **kwargs):
            if method == "sendMessage":
                enviados.append(kwargs["text"])
            return {"ok": True, "result": {}}

        monkeypatch.setattr(mod, "_db", _FakeDB())
        monkeypatch.setattr(mod, "_tg", _fake_tg)
        monkeypatch.setattr(mod, "_modelo_por_defecto", lambda: mod.MODELO_DETERMINISTA)

        await mod.procesar_update({"message": {"chat": {"id": 1}, "text": "/start"}})

        assert len(enviados) == 2, enviados
        assert enviados[0] == mod.BIENVENIDA, enviados[0]
        assert "¿Cuál es tu sexo?" in enviados[1], enviados

    def test_bienvenida_no_menciona_el_chat(self):
        """Criterio 1: /chat desaparece de la bienvenida y del bot."""
        import climasafeai.bot.telegram_bot as mod
        assert "/chat" not in mod.BIENVENIDA
        assert "/start" in mod.BIENVENIDA


class TestLogging:
    """BOT-004: el token no puede acabar en el log, y las lineas no se duplican."""

    def _preparar(self, tmp_path, monkeypatch, token="123456789:AAFAKE_token_de_prueba_xxxxxxxxxxxx"):
        import logging
        from climasafeai.bot.telegram_bot import _setup_logging
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", token)
        monkeypatch.chdir(tmp_path)
        raiz = logging.getLogger()
        previos = list(raiz.handlers)
        for h in previos:
            raiz.removeHandler(h)
        yield_data = (raiz, previos, token, _setup_logging)
        return yield_data

    def test_el_token_no_aparece_en_el_log(self, tmp_path, monkeypatch):
        import logging
        raiz, previos, token, setup = self._preparar(tmp_path, monkeypatch)
        try:
            setup()
            logging.getLogger("httpx").setLevel(logging.INFO)
            logging.getLogger("httpx").info(
                "HTTP Request: POST https://api.telegram.org/bot%s/sendMessage" % token
            )
            for h in raiz.handlers:
                h.flush()
            texto = (tmp_path / "logs" / "bot.log").read_text()
            assert token not in texto
            assert "TOKEN_OCULTO" in texto
        finally:
            for h in list(raiz.handlers):
                raiz.removeHandler(h)
            for h in previos:
                raiz.addHandler(h)

    def test_todos_los_handlers_llevan_el_filtro_del_token(self, tmp_path, monkeypatch):
        """El _OcultarToken va en el fichero y en la consola (si existe): no puede
        haber una ruta de escritura con el token en claro."""
        import logging
        from climasafeai.bot.telegram_bot import _OcultarToken
        raiz, previos, token, setup = self._preparar(tmp_path, monkeypatch)
        try:
            setup()
            assert raiz.handlers, "debe haber al menos el handler de fichero"
            for h in raiz.handlers:
                assert any(isinstance(f, _OcultarToken) for f in h.filters), \
                    f"{type(h).__name__} no filtra el token"
        finally:
            for h in list(raiz.handlers):
                raiz.removeHandler(h)
            for h in previos:
                raiz.addHandler(h)

    def test_consola_solo_si_stdout_es_un_terminal(self, tmp_path, monkeypatch):
        """Produccion redirige stdout al fichero (no tty): sin console handler.
        En un terminal real (tty): consola + fichero, y ambos filtran el token."""
        import logging
        import sys
        from climasafeai.bot.telegram_bot import _OcultarToken, _setup_logging
        raiz, previos, token, setup = self._preparar(tmp_path, monkeypatch)
        try:
            class _FakeOut:
                def __init__(self, tty):
                    self._tty = tty
                def isatty(self):
                    return self._tty
                def write(self, _m):
                    pass
                def flush(self):
                    pass

            # RotatingFileHandler hereda de StreamHandler: la consola es el que
            # escribe a un stream que NO es un fichero
            def es_consola(h):
                return isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)

            # stdout redirigido a fichero (run_bot.sh): solo el handler de fichero
            monkeypatch.setattr(sys, "stdout", _FakeOut(tty=False))
            _setup_logging()
            assert len(raiz.handlers) == 1, "sin consola queda solo el fichero"
            assert not es_consola(raiz.handlers[0]), \
                "stdout redirigido a fichero: el console handler duplicaria las lineas"
            assert any(isinstance(f, _OcultarToken) for f in raiz.handlers[0].filters)

            # terminal real: consola + fichero, y ambos filtran el token
            for h in list(raiz.handlers):
                raiz.removeHandler(h)
            monkeypatch.setattr(sys, "stdout", _FakeOut(tty=True))
            _setup_logging()
            consolas = [h for h in raiz.handlers if es_consola(h)]
            assert len(consolas) == 1, "con un terminal real la consola debe estar"
            for h in raiz.handlers:
                assert any(isinstance(f, _OcultarToken) for f in h.filters), \
                    f"{type(h).__name__} no filtra el token"
        finally:
            for h in list(raiz.handlers):
                raiz.removeHandler(h)
            for h in previos:
                raiz.addHandler(h)

    def test_varios_arranques_no_duplican_las_lineas(self, tmp_path, monkeypatch):
        """run_bot.sh reinicia en bucle: cada arranque anadia otro par de handlers."""
        import logging
        raiz, previos, token, setup = self._preparar(tmp_path, monkeypatch)
        try:
            setup()
            n_handlers = len(raiz.handlers)
            setup()
            setup()
            assert len(raiz.handlers) == n_handlers, "un arranque no puede anadir handlers de mas"
            logging.getLogger("prueba").info("una sola vez")
            for h in raiz.handlers:
                h.flush()
            lineas = [ln for ln in (tmp_path / "logs" / "bot.log").read_text().splitlines() if ln.strip()]
            assert len(lineas) == 1, f"la linea se ha escrito {len(lineas)} veces"
        finally:
            for h in list(raiz.handlers):
                raiz.removeHandler(h)
            for h in previos:
                raiz.addHandler(h)


# ── CHAT-003: tras /start con LLM, chat abierto de preguntas ───────────────

_ALDAN = {"lat": 42.29, "lon": -8.81, "provincia": "Pontevedra", "nombre": "Aldán, Pontevedra"}

_RESULTADO_PELIGRO = {
    "clase_final": 2,
    "clase_final_label": "PELIGRO",
    "perfil": {"calor": {"prob_personalizada": 0.72, "factores": []}},
    "perfil_usuario": {"hora_inicio": 17, "duracion_actividad_h": 2},
    "weather": {
        "provincia": "Pontevedra",
        "current": {"t2m_c": 36.0, "rh": 50},
        "uv_index": 7,
        "perfil_horario": [{"hora": 17, "HI": 39.0, "temp": 36.0},
                           {"hora": 18, "HI": 38.0, "temp": 35.0}],
    },
    "modelos": {"Formula": {"frio": {"wind_chill_c": 30}, "calor": {"heat_index_c": 39}}},
    "recomendaciones": ["Evita la actividad"],
}

_RESPONSE_LLM = "Aldán — Riesgo PELIGRO (72%). Evita la actividad."


class TestChatAbiertoTrasStart:
    """CHAT-003: al terminar /start con LLM el chat queda abierto para dudas.

    La recogida conversacional desaparece: /start con botones es el único
    camino. Al llegar a DONE, si el modelo NO es determinista el parte lo
    redacta `ask_con_perfil` y el chat queda abierto para preguntas libres con
    RAG sobre el parte; si es determinista, plantilla y cierre como antes.
    """

    @staticmethod
    def _data_completa():
        return {
            "sexo": "hombre", "edad": 57, "porcentaje_grasa": 20.5,
            "fototipo": "3", "aclimatado": False, "nivel_actividad": "moderada",
            "entrenado": True, "duracion_h": 2, "hora_inicio": 17,
            "comorbilidades": {"cardiovascular"}, "farmacos": {"diureticos_asa"},
            "estado_previo": {"fiesta"}, "situacion_social": {"vive_solo"},
            "lat": 42.29, "lon": -8.81, "provincia": "Pontevedra", "lugar": "Aldán",
        }

    @staticmethod
    def _capturar_enviados(monkeypatch):
        import climasafeai.bot.telegram_bot as mod
        enviados: list[str] = []

        async def _fake_enviar(_cid, texto, kb=None, reply_markup=None):
            enviados.append(texto)

        monkeypatch.setattr(mod, "enviar_mensaje", _fake_enviar)
        return mod, enviados

    def test_cierre_invita_a_preguntar_y_a_volver_a_start(self):
        """Criterio 4: el cierre del parte invita a dudas (SPF) y a /start."""
        import climasafeai.bot.telegram_bot as mod
        assert "SPF" in mod.CHAT_CIERRE
        assert "/start" in mod.CHAT_CIERRE
        assert "duda" in mod.CHAT_CIERRE.lower()

    @pytest.mark.asyncio
    async def test_con_llm_el_parte_abre_el_chat_y_guarda_el_contexto(self, monkeypatch):
        """Criterio 2: con LLM, la respuesta la redacta ask_con_perfil y el
        chat queda abierto para preguntas."""
        import climasafeai.bot.telegram_bot as mod
        monkeypatch.setattr(mod, "predict_ensemble", lambda **kw: _RESULTADO_PELIGRO)
        monkeypatch.setattr(
            mod, "ask_con_perfil",
            lambda _p, _r, _c=None, _l=None: _RESPONSE_LLM,
        )
        _, enviados = self._capturar_enviados(monkeypatch)
        _conversaciones[1] = {
            "estado": Estado.DONE, "modelo": "ollama/qwen2.5:7b",
            "data": self._data_completa(),
        }

        await mod._finalizar_parte(1)

        assert enviados == [f"{_RESPONSE_LLM}\n\n{mod.CHAT_CIERRE}"]
        assert _conversaciones[1]["data"]["_prediccion_hecha"] is True
        assert _conversaciones[1]["ultima_prediccion"] == _RESPONSE_LLM
        assert _conversaciones[1]["estado"] == Estado.DONE  # sigue abierta

    @pytest.mark.asyncio
    async def test_determinista_responde_con_plantilla_y_cierra(self, monkeypatch):
        """Criterio 3: sin LLM, plantilla y cierre normal, sin chat abierto."""
        import climasafeai.bot.telegram_bot as mod
        monkeypatch.setattr(mod, "predict_ensemble", lambda **kw: _RESULTADO_PELIGRO)
        _, enviados = self._capturar_enviados(monkeypatch)
        _conversaciones[1] = {
            "estado": Estado.DONE, "modelo": mod.MODELO_DETERMINISTA,
            "data": self._data_completa(),
        }

        await mod._finalizar_parte(1)

        assert 1 not in _conversaciones          # cerrada
        assert len(enviados) == 1
        assert "PELIGRO" in enviados[0]
        assert mod.CHAT_CIERRE not in enviados[0]

    @pytest.mark.asyncio
    async def test_pregunta_tras_el_parte_va_al_rag_con_el_parte_de_contexto(self, monkeypatch):
        """Criterio 7: preguntar por SPF responde con info útil, no rechazo."""
        import climasafeai.bot.telegram_bot as mod
        recibido: dict = {}

        def _fake_rag(q, k1, k2, c, ctx=None, perfil=None):
            recibido["pregunta"] = q
            recibido["contexto"] = ctx
            recibido["perfil"] = perfil
            return {"answer": "El SPF es el factor de protección solar."}

        monkeypatch.setattr(mod, "ask_with_rag", _fake_rag)
        _conversaciones[1] = {
            "estado": Estado.DONE, "modelo": "ollama/qwen2.5:7b",
            "ultima_prediccion": _RESPONSE_LLM,
            "data": {"_prediccion_hecha": True},
        }

        r = await procesar_mensaje(1, "¿qué es SPF?")

        assert r == "El SPF es el factor de protección solar."
        assert recibido["pregunta"] == "¿qué es SPF?"
        assert "PELIGRO" in recibido["contexto"]
        assert recibido["contexto"].startswith("Parte que le acabas de dar")

    @pytest.mark.parametrize("comando", ["salir", "exit", "/salir"])
    @pytest.mark.asyncio
    async def test_salir_cierra_el_chat_y_vuelve_al_inicio(self, monkeypatch, comando):
        """Criterio 6: /salir (o equivalente) cierra el chat de preguntas."""
        import climasafeai.bot.telegram_bot as mod
        monkeypatch.setattr(mod, "ask_with_rag", lambda *a, **k: {"answer": "x"})
        _conversaciones[1] = {
            "estado": Estado.DONE, "modelo": "ollama/qwen2.5:7b",
            "ultima_prediccion": _RESPONSE_LLM,
            "data": {"_prediccion_hecha": True},
        }

        r = await procesar_mensaje(1, comando)

        assert "Saliendo" in r and "/start" in r, r
        assert _conversaciones[1]["estado"] == Estado.IDLE
        assert not _conversaciones[1]["data"].get("_prediccion_hecha")
        # De vuelta al inicio: un mensaje suelto muestra la bienvenida
        r2 = await procesar_mensaje(1, "hola")
        assert r2 == mod.BIENVENIDA

    @pytest.mark.asyncio
    async def test_start_nuevo_sobreescribe_el_parte_anterior(self, monkeypatch):
        """Criterio 5: /start nuevo resetea el parte y arranca el formulario."""
        import climasafeai.bot.telegram_bot as mod
        monkeypatch.setattr(mod, "_db", _FakeDBSinPerfil())
        monkeypatch.setattr(mod, "_modelo_por_defecto", lambda: mod.MODELO_DETERMINISTA)
        _conversaciones[1] = {
            "estado": Estado.DONE, "modelo": "ollama/qwen2.5:7b",
            "ultima_prediccion": _RESPONSE_LLM,
            "ultimo_resultado": _RESULTADO_PELIGRO,
            "data": {"_prediccion_hecha": True, "lat": 42.29, "lon": -8.81},
        }

        r = await procesar_mensaje(1, "/start")

        assert r == mod.BIENVENIDA
        assert _conversaciones[1]["estado"] == Estado.SEXO
        assert "ultima_prediccion" not in _conversaciones[1]
        assert "ultimo_resultado" not in _conversaciones[1]
        assert "_prediccion_hecha" not in _conversaciones[1]["data"]

    @pytest.mark.asyncio
    async def test_flujo_completo_con_llm_abre_el_chat_y_responde_spf(self, monkeypatch):
        """De punta a punta: /start con LLM → parte + chat abierto → SPF."""
        import climasafeai.bot.telegram_bot as mod
        monkeypatch.setattr(mod, "predict_ensemble", lambda **kw: _RESULTADO_PELIGRO)
        monkeypatch.setattr(
            mod, "ask_con_perfil",
            lambda _p, _r, _c=None, _l=None: _RESPONSE_LLM,
        )
        monkeypatch.setattr(
            mod, "ask_with_rag",
            lambda q, k1, k2, c, ctx=None, perfil=None: {"answer": "Usa SPF 30+ y renueva cada 2 horas."},
        )
        enviados: list[str] = []

        async def _fake_tg(method: str, **kwargs):
            if method == "sendMessage":
                enviados.append(kwargs["text"])
            return {"ok": True, "result": {}}

        monkeypatch.setattr(mod, "_tg", _fake_tg)
        _conversaciones[1] = {
            "estado": Estado.GUARDAR_PERFIL, "modelo": "ollama/qwen2.5:7b",
            "data": self._data_completa(),
        }

        # El usuario pulsa "No" a guardar perfil → DONE → parte con LLM + chat abierto
        await mod.procesar_update({"callback_query": {
            "id": "q1", "data": "guardar_no",
            "message": {"chat": {"id": 1}, "message_id": 1},
        }})

        assert len(enviados) == 1, enviados
        assert _RESPONSE_LLM in enviados[0]
        assert mod.CHAT_CIERRE in enviados[0]
        assert _conversaciones[1]["estado"] == Estado.DONE
        assert _conversaciones[1]["data"]["_prediccion_hecha"]

        # Pregunta libre con RAG sobre el parte
        await mod.procesar_update({"message": {"chat": {"id": 1}, "text": "¿qué es SPF?"}})
        assert len(enviados) == 2, enviados
        assert "SPF 30+" in enviados[1]
        assert _conversaciones[1]["estado"] == Estado.DONE  # sigue abierta


class TestEnviarMensajeMarkdown:
    """Un asterisco suelto del LLM tumbaba el mensaje entero con un 400."""

    @pytest.mark.asyncio
    async def test_si_telegram_rechaza_el_formato_se_reenvia_en_plano(self, monkeypatch):
        import httpx
        import climasafeai.bot.telegram_bot as mod

        intentos: list[dict] = []

        async def _fake_tg(method: str, **kwargs):
            intentos.append(kwargs)
            if "parse_mode" in kwargs:
                raise httpx.HTTPStatusError(
                    "bad entities", request=httpx.Request("POST", "http://x"),
                    response=httpx.Response(400),
                )
            return {"ok": True}

        monkeypatch.setattr(mod, "_tg", _fake_tg)

        await mod.enviar_mensaje(1, "hidratación *importante_ y suelto")

        assert len(intentos) == 2
        assert "parse_mode" not in intentos[1]
        assert intentos[1]["text"] == "hidratación *importante_ y suelto"


# ── BOT-010: fecha de nacimiento en vez de edad ────────────────────────────


class TestFechaNacimiento:
    """Edad desde fecha de nacimiento y validación de la fecha (BOT-010)."""

    HOY = date(2026, 8, 3)

    def test_fecha_nacimiento_edad_cumpleanos_pasado(self):
        """Nacido el 15/03/1990: a 03/08/2026 ya cumplió los 36."""
        assert _edad_desde_fecha_nacimiento(date(1990, 3, 15), self.HOY) == 36

    def test_fecha_nacimiento_edad_cumpleanos_futuro(self):
        """Nacido el 15/11/1990: aún no ha cumplido los 36, tiene 35."""
        assert _edad_desde_fecha_nacimiento(date(1990, 11, 15), self.HOY) == 35

    def test_fecha_nacimiento_29_de_febrero(self):
        """El 29/02 solo cuenta en bisiestos: el 28/02 aún no ha cumplido."""
        assert _edad_desde_fecha_nacimiento(date(2000, 2, 29), date(2027, 2, 28)) == 26
        assert _edad_desde_fecha_nacimiento(date(2000, 2, 29), date(2027, 3, 1)) == 27

    def test_fecha_nacimiento_solo_anio_es_1_enero(self):
        ok, edad = _parsear_fecha_nacimiento("1965", self.HOY)
        assert ok is True
        assert edad == _edad_desde_fecha_nacimiento(date(1965, 1, 1), self.HOY)

    def test_fecha_nacimiento_acepta_guiones(self):
        ok, edad = _parsear_fecha_nacimiento("15-03-1990", self.HOY)
        assert ok is True and edad == 36

    def test_fecha_nacimiento_mes_o_dia_de_un_digito(self):
        ok, edad = _parsear_fecha_nacimiento("15/3/1990", self.HOY)
        assert ok is True and edad == 36

    def test_fecha_nacimiento_futura_error(self):
        ok, msg = _parsear_fecha_nacimiento("15/03/2030", self.HOY)
        assert ok is False and "futuro" in msg

    def test_fecha_nacimiento_anterior_a_1900_error(self):
        ok, msg = _parsear_fecha_nacimiento("01/01/1899", self.HOY)
        assert ok is False and "1900" in msg

    def test_fecha_nacimiento_anio_mayor_que_hoy_error(self):
        ok, msg = _parsear_fecha_nacimiento("2030", self.HOY)
        assert ok is False and "futuro" in msg

    def test_fecha_nacimiento_formato_malo_error_claro(self):
        """Formato malo o fecha imposible: mensaje claro, nunca excepción."""
        for mal in ("abc", "15/03", "31/02/1990", "15/13/1990", "15-03-1990-junk", ""):
            ok, msg = _parsear_fecha_nacimiento(mal, self.HOY)
            assert ok is False, f"{mal!r} no debería validar"
            assert "DD/MM/AAAA" in msg, f"{mal!r} → {msg}"

    @pytest.mark.asyncio
    async def test_fecha_nacimiento_flujo_guarda_edad_calculada(self, monkeypatch):
        """El formulario guarda data['edad'] como entero desde la fecha."""
        import climasafeai.bot.telegram_bot as mod
        monkeypatch.setattr(mod, "date", _HoyFijo)
        _conv_limpia()
        _conversaciones[1]["estado"] = Estado.EDAD
        r = await procesar_mensaje(1, "15/03/1990")
        assert r is None
        assert _conversaciones[1]["data"]["edad"] == 36
        assert _conversaciones[1]["data"]["fecha_nacimiento"] == "15/03/1990"
        assert _conversaciones[1]["estado"] == Estado.GRASA

    @pytest.mark.asyncio
    async def test_fecha_nacimiento_flujo_formato_malo_no_avanza(self, monkeypatch):
        import climasafeai.bot.telegram_bot as mod
        monkeypatch.setattr(mod, "date", _HoyFijo)
        _conv_limpia()
        _conversaciones[1]["estado"] = Estado.EDAD
        r = await procesar_mensaje(1, "abc")
        assert r is not None and "DD/MM/AAAA" in r
        assert "edad" not in _conversaciones[1]["data"]
        assert _conversaciones[1]["estado"] == Estado.EDAD

    @pytest.mark.asyncio
    async def test_fecha_nacimiento_flujo_fecha_futura_no_avanza(self, monkeypatch):
        import climasafeai.bot.telegram_bot as mod
        monkeypatch.setattr(mod, "date", _HoyFijo)
        _conv_limpia()
        _conversaciones[1]["estado"] = Estado.EDAD
        r = await procesar_mensaje(1, "15/03/2030")
        assert r is not None and "futuro" in r
        assert "edad" not in _conversaciones[1]["data"]
        assert _conversaciones[1]["estado"] == Estado.EDAD


class TestPartePorcentaje:
    """BOT-010: el parte explica el % de riesgo en lenguaje llano.

    BOT-013: ese "lenguaje llano" era otro porcentaje ("21% de probabilidad de
    riesgo térmico"). Ahora es frecuencia natural, y las dos vías del parte
    —plantilla y LLM— dicen exactamente lo mismo porque comparten
    `lineas_parte`.
    """

    def _resultado(self, prob=0.21, clase="PRECAUCIÓN"):
        return {
            "clase_final": 1,
            "clase_final_label": clase,
            "perfil": {"calor": {"prob_personalizada": prob}},
            "perfil_usuario": {"hora_inicio": 8, "duracion_actividad_h": 2},
            "weather": {
                "provincia": "Pontevedra",
                "current": {"t2m_c": 30.0, "rh": 60},
                "uv_index": 6,
                "perfil_horario": [{"hora": 8, "HI": 25.0, "temp": 20.0}],
            },
            "modelos": {
                "Formula": {"frio": {"wind_chill_c": 25.0}, "calor": {"heat_index_c": 36.0}},
            },
            "recomendaciones": ["Mantente hidratado"],
        }

    def test_parte_porcentaje_plantilla_explica_el_porcentaje(self):
        texto = _format_template(self._resultado(prob=0.21), "Moaña, Pontevedra")
        assert "en unos 21 el calor te pasaría factura" in texto
        # BOT-020: la cifra aparece en la cabecera, con su etiqueta, no suelta
        # como "Riesgo PRECAUCIÓN (21%)" que parecía explicar la clase.
        assert "probabilidad de riesgo personalizada por calor: 21% (0.2100)" in texto
        assert "Riesgo PRECAUCIÓN (21%)" not in texto
        assert "mayor cuanto más se acerque a 100%" not in texto

    def test_plantilla_separa_la_clase_del_porcentaje(self):
        """Criterio 2 de BOT-013 en la vía determinista, sin depender del LLM."""
        texto = _format_template(self._resultado(prob=0.21), "Moaña, Pontevedra")
        assert "no es lo que decide tu nivel" in texto
        assert "umbrales de tu provincia" in texto
        assert "una cifra baja puede venir con un nivel alto" in texto

    def test_plantilla_avisa_de_la_confianza_baja(self):
        """Criterio 4: con confianza baja el parte no da la cifra igual de seco."""
        result = self._resultado(prob=0.21)
        result["modelos"]["XGBoost_calor"] = {"conformal_confianza": "baja"}
        texto = _format_template(result, "Moaña, Pontevedra")
        assert "hoy el modelo tiene poca confianza" in texto
        assert "ve por el lado seguro" in texto

    def test_plantilla_sin_conformal_no_se_inventa_la_confianza(self):
        """`conformal_confianza` es None si falta el joblib: se calla, no revienta."""
        result = self._resultado(prob=0.21)
        result["modelos"]["XGBoost_calor"] = {"conformal_confianza": None}
        texto = _format_template(result, "Moaña, Pontevedra")
        assert "confianza" not in texto.lower()
        assert "en unos 21 el calor te pasaría factura" in texto

    def test_parte_porcentaje_prompt_llm_instruye_explicar(self, monkeypatch):
        import climasafeai.llm.rag_qwen as rag

        capturado: dict = {}

        def _fake_chat(messages, config):
            # LLM-003 antepone SYSTEM_PARTE como messages[0]; los datos del
            # parte van en el mensaje con role=user.
            user_msgs = [m for m in messages if m["role"] == "user"]
            capturado["prompt"] = user_msgs[0]["content"]
            return ("Moaña — Riesgo PRECAUCIÓN: 21% de probabilidad de riesgo térmico "
                    "durante la actividad. Mantente hidratado.")

        monkeypatch.setattr(rag, "_chat_litellm", _fake_chat)
        texto = rag.ask_con_perfil(
            {"hora_inicio": 8},
            self._resultado(prob=0.21),
            config=rag.LLMConfig(),
            lugar="Moaña, Pontevedra",
        )

        # BOT-013: al LLM se le da la frase ya redactada, no un porcentaje que
        # tenga que explicar él
        assert "en unos 21 el calor te pasaría factura" in capturado["prompt"]
        assert "de probabilidad de riesgo térmico durante la actividad" not in capturado["prompt"]
        # LLM-005: la coletilla del % ya no está en el prompt ni en el ejemplo
        assert "mayor cuanto más se acerque a 100%" not in capturado["prompt"]
        # El parte devuelto conserva lo que escribió el LLM
        assert texto.startswith("Moaña — Riesgo PRECAUCIÓN")


class TestParteBOT020:
    """BOT-020: el parte tras la predicción se entiende de un vistazo.

    Abre con la clasificación y la probabilidad en %, resume la jornada, lista
    los factores con su multiplicador de mayor a menor, compara con la salida
    anterior si la hay y cierra con la tabla horaria y las recomendaciones de
    la herramienta sin suavizar.
    """

    def _resultado(self, prob=0.69, clase=2):
        return {
            "clase_final": clase,
            "clase_final_label": "PELIGRO" if clase >= 2 else "PRECAUCIÓN",
            "perfil": {"calor": {"prob_personalizada": prob, "factores": [
                {"nombre": "trabajo Construcción / albañilería (carga pesada, PPE, sol directo)",
                 "categoria": "ocupacional", "factor": 2.2},
                {"nombre": "duración 8.0 h", "categoria": "fisiologico", "factor": 1.4},
                {"nombre": "hora inicio 8:00 (solapa pico calor)", "categoria": "fisiologico", "factor": 1.2},
                {"nombre": "falta de sueño / mala noche", "categoria": "fisiologico", "factor": 1.2},
            ]}},
            "perfil_usuario": {
                "hora_inicio": 8, "duracion_actividad_h": 8,
                "nivel_actividad": "moderada", "aclimatado": False,
                "ocupacion": "construccion", "falta_sueno": True,
            },
            "weather": {
                "provincia": "Madrid",
                "current": {"t2m_c": 36.0, "rh": 40},
                "uv_index": 8,
                # Campana: el HI sube hasta las 13 y baja, para que el pico de
                # la curva quede en medio de la ventana y haya inicio/pico/fin.
                "perfil_horario": [
                    {"hora": h, "HI": {8: 25, 9: 28, 10: 31, 11: 33, 12: 35, 13: 37, 14: 34, 15: 31, 16: 29}[h],
                     "temp": 20 + h}
                    for h in range(8, 17)
                ],
            },
            "recomendaciones": [
                "Evita la actividad al aire libre entre las 12:00 y las 17:00",
                "Mantente hidratado y en un lugar fresco",
            ],
        }

    def test_cabecera_abre_con_clasificacion_y_probabilidad(self):
        """Criterio 1: el parte abre con la clase y la probabilidad en %."""
        texto = _format_template(self._resultado(prob=0.6909), "Madrid")
        assert texto.startswith(
            "Clasificación: PELIGRO — probabilidad de riesgo personalizada por calor: 69% (0.6909)."
        )

    def test_resumen_jornada_en_una_linea(self):
        """Criterio 2: actividad, horario, duración, intensidad, aclimatado, sueño."""
        texto = _format_template(self._resultado(), "Madrid")
        assert (
            "Con esta jornada (trabajo de construcción, 8:00-16:00, 8h, "
            "actividad moderada, no aclimatado, con falta de sueño)" in texto
        )

    def test_factores_con_multiplicador_ordenados_de_mayor_a_menor(self):
        """Criterio 3: cada factor con su x y ordenados por peso."""
        texto = _format_template(self._resultado(), "Madrid")
        x22 = texto.index("• trabajo de construcción al aire libre (factor x2.2)")
        x14 = texto.index("• la duración de 8h (factor x1.4)")
        x12 = texto.index("• el horario que solapa con el pico de calor (factor x1.2)")
        sueno = texto.index("• la falta de sueño (factor x1.2)")
        assert x22 < x14 < x12 < sueno

    def test_tabla_horaria_inicio_pico_fin_con_heat_index(self):
        """Criterio 4: tabla Hora / Riesgo / Heat Index con inicio, pico y fin."""
        texto = _format_template(self._resultado(), "Madrid")
        assert "Hora | Riesgo | Heat Index" in texto
        assert "8:00 (inicio) |" in texto
        assert "13:00 (pico) |" in texto
        assert "15:00 (fin) |" in texto
        assert "°C" in texto

    def test_recomendaciones_tal_cual_sin_suavizar(self):
        """Criterio 4: las recomendaciones de la herramienta, línea por línea."""
        texto = _format_template(self._resultado(), "Madrid")
        assert "Recomendaciones de la herramienta (nivel PELIGRO, no las suavizo):" in texto
        assert "1. Evita la actividad al aire libre entre las 12:00 y las 17:00" in texto
        assert "2. Mantente hidratado y en un lugar fresco" in texto

    def test_comparacion_con_salida_anterior(self):
        """Criterio 5: si hay salida previa y el nivel sube, se dice."""
        texto = _format_template(
            self._resultado(), "Madrid",
            salida_anterior={"clase_final": 0, "actividad": "correr por la tarde"},
        )
        assert "Es un nivel más alto que la simulación anterior de correr por la tarde." in texto

    def test_sin_salida_anterior_no_se_inventa_comparacion(self):
        """Criterio 5: hoy no hay salida guardada (BOT-017): no se compara."""
        texto = _format_template(self._resultado(), "Madrid")
        assert "simulación anterior" not in texto


class TestFranjasBOT012:
    """BOT-012: el parte dice la franja de mayor riesgo del día y la recomendada.

    Criterio 1: ambas franjas salen en el parte. Criterio 2: los valores son
    los que ya calculan riesgo_horario_acumulado, pico_riesgo_actividad y
    recomendar_horario (se comparan contra esas funciones, no contra un
    número a mano). Criterio 3: sin perfil horario el parte lo dice.
    """

    def _resultado(self):
        return TestParteBOT020()._resultado()

    def test_parte_dice_franja_de_mayor_riesgo_y_recomendada(self):
        """Criterio 1: el parte indica ambas franjas, con la de riesgo primero."""
        texto = _format_template(self._resultado(), "Madrid")
        assert "Franja de mayor riesgo del día: en torno a las 13:00" in texto, texto
        assert "Franja recomendada para la actividad:" in texto, texto
        # La franja de riesgo aparece antes que la recomendada.
        assert texto.index("Franja de mayor riesgo") < texto.index("Franja recomendada")

    def test_franja_recomendada_coincide_con_recomendar_horario(self):
        """Criterio 2: la franja recomendada es la de recomendar_horario, no se recalcula."""
        from climasafeai.features.personalizacion import recomendar_horario

        result = self._resultado()
        ph = result["weather"]["perfil_horario"]
        pu = result["perfil_usuario"]
        rec = recomendar_horario(ph, pu)
        assert rec is not None and rec.get("hora_inicio") is not None
        texto = _format_template(result, "Madrid")
        esperado = (
            f"Franja recomendada para la actividad: "
            f"{rec['hora_inicio']:.0f}:00-{rec['hora_fin']:.0f}:00"
        )
        assert esperado in texto, texto

    def test_pico_de_riesgo_viene_de_pico_riesgo_actividad(self):
        """Criterio 2: la cifra de riesgo de la franja sale de pico_riesgo_actividad."""
        from climasafeai.features.personalizacion import pico_riesgo_actividad, riesgo_horario_acumulado

        result = self._resultado()
        ph = result["weather"]["perfil_horario"]
        pu = result["perfil_usuario"]
        curva = riesgo_horario_acumulado(ph, pu)
        pico = pico_riesgo_actividad(curva, pu)
        assert pico is not None
        texto = _format_template(result, "Madrid")
        assert f"riesgo {pico:.2f} de 1" in texto, texto

    def test_sin_perfil_horario_lo_dice_en_vez_de_inventar(self):
        """Criterio 3: sin perfil horario el parte avisa, no inventa ninguna franja."""
        result = self._resultado()
        result["weather"].pop("perfil_horario")
        texto = _format_template(result, "Madrid")
        assert "No hay datos horarios para hoy" in texto, texto
        assert "Franja de mayor riesgo" not in texto
        assert "Franja recomendada" not in texto


class TestChatParteConcisa:
    """BOT-011: el chat abierto explica el parte con datos reales, en 2-3 frases.

    La respuesta no puede ser un texto genérico: el contexto que recibe el LLM
    lleva la probabilidad, los factores y la ubicación de ESA predicción.
    """

    @staticmethod
    def _conv():
        result = TestRecomendacionCanalDominante._resultado(
            prob_calor=0.21, prob_frio=0.02, t=35.3, uv=7.6, wc=5, hi=38, clase=1,
        )
        return {
            "modelo": "ollama/qwen2.5:7b",
            "estado": Estado.DONE,
            "ultima_prediccion": "O Casal, Pontevedra — Riesgo PRECAUCIÓN (21%).",
            "ultimo_resultado": result,
            "data": {"_prediccion_hecha": True},
        }

    def test_chat_parte_concisa_contexto_lleva_datos_reales(self):
        from climasafeai.bot.telegram_bot import _contexto_parte_conversacion

        ctx = _contexto_parte_conversacion(self._conv())

        assert ctx.startswith("Parte que le acabas de dar al usuario")
        assert "O Casal" in ctx                      # el parte entregado
        assert "21%" in ctx                          # probabilidad real
        assert "Pontevedra" in ctx                   # ubicación real
        assert "no_aclimatado" in ctx                # factor real de esa predicción
        assert "2-3 frases" in ctx                   # instrucción de concisión
        assert "sin textos genéricos" in ctx

    def test_chat_parte_concisa_contexto_lleva_multiplicadores_y_ocupacion(self):
        from climasafeai.bot.telegram_bot import _contexto_parte_conversacion

        conv = self._conv()
        # Factores como los del pipeline real: dicts {nombre, factor}
        conv["ultimo_resultado"]["perfil"]["calor"]["factores"] = [
            {"nombre": "trabajo Construcción / albañilería (carga pesada, PPE, sol directo)", "factor": 2.2},
            {"nombre": "no aclimatado", "factor": 1.3},
        ]
        conv["ultimo_resultado"]["perfil_usuario"] = {
            "hora_inicio": 8, "duracion_actividad_h": 2, "ocupacion": "construccion",
        }

        ctx = _contexto_parte_conversacion(conv)

        # LLM-005: el contexto lleva los coeficientes (xN), no solo el nombre
        assert "trabajo Construcción / albañilería" in ctx
        assert "x2.2" in ctx
        assert "no aclimatado (x1.3)" in ctx
        # La ocupación se pasa con su etiqueta y coeficiente
        assert "Ocupación:" in ctx
        assert "Construcción / albañilería (carga pesada, PPE, sol directo) (x2.2)" in ctx
        # Prohibido el consejo genérico del RAG sin filtrar
        assert "reduce la exposición en interiores" in ctx

    @pytest.mark.asyncio
    async def test_chat_parte_concisa_flujo_pasa_el_contexto_al_rag(self, monkeypatch):
        import climasafeai.bot.telegram_bot as mod
        recibido: dict = {}

        def _fake_rag(q, k1, k2, c, ctx=None, perfil=None):
            recibido["pregunta"] = q
            recibido["contexto"] = ctx
            recibido["perfil"] = perfil
            return {"answer": "Tu riesgo de calor es del 21% por no estar aclimatado."}

        monkeypatch.setattr(mod, "ask_with_rag", _fake_rag)
        _conversaciones[1] = self._conv()

        r = await procesar_mensaje(1, "¿qué es la probabilidad personalizada?")

        assert r == "Tu riesgo de calor es del 21% por no estar aclimatado."
        assert recibido["pregunta"] == "¿qué es la probabilidad personalizada?"
        assert "21%" in recibido["contexto"]
        assert "Pontevedra" in recibido["contexto"]
        assert "2-3 frases" in recibido["contexto"]


class TestChatBOT014:
    """BOT-014: el contexto del chat solo lleva el canal dominante y los
    factores con su coeficiente, de mayor a menor.

    Criterio 1: un parte de calor no mete la probabilidad ni los factores de
    frío en el contexto (y al revés). Criterio 2: los factores se mandan
    ordenados por su coeficiente y con el coeficiente (xN), para que el LLM
    pueda decir cuál pesa más. Criterio 4: la dominancia se decide con
    `_canal_dominante` de recomendaciones, no a mano.
    """

    @staticmethod
    def _conv(prob_calor=0.21, prob_frio=0.02, factores_calor=None, factores_frio=None):
        result = TestRecomendacionCanalDominante._resultado(
            prob_calor=prob_calor, prob_frio=prob_frio, t=35.3, uv=7.6, wc=5, hi=38, clase=1,
        )
        result["perfil"]["calor"]["factores"] = factores_calor or [
            {"nombre": "trabajo", "factor": 2.2},
            {"nombre": "no aclimatado", "factor": 1.3},
        ]
        result["perfil"]["frio"]["factores"] = factores_frio or [
            {"nombre": "edad", "factor": 1.1},
        ]
        return {
            "modelo": "ollama/qwen2.5:7b",
            "estado": Estado.DONE,
            "ultima_prediccion": "O Casal, Pontevedra — Riesgo PRECAUCIÓN (21%).",
            "ultimo_resultado": result,
            "data": {"_prediccion_hecha": True},
        }

    def test_contexto_calor_no_muestra_el_canal_frio(self):
        from climasafeai.bot.telegram_bot import _contexto_parte_conversacion

        ctx = _contexto_parte_conversacion(self._conv())

        # Solo el canal dominante: la probabilidad de calor y sus factores...
        assert "Probabilidad personalizada (calor): 21%" in ctx
        assert "trabajo (x2.2)" in ctx
        assert "no aclimatado (x1.3)" in ctx
        # ...y ni la probabilidad ni los factores del canal frío.
        assert "Probabilidad personalizada (frio" not in ctx
        assert "Probabilidad personalizada (frío" not in ctx
        assert "edad" not in ctx

    def test_contexto_frio_no_muestra_el_canal_calor(self):
        from climasafeai.bot.telegram_bot import _contexto_parte_conversacion

        ctx = _contexto_parte_conversacion(self._conv(prob_calor=0.03, prob_frio=0.45))

        assert "Probabilidad personalizada (frio): 45%" in ctx
        assert "Probabilidad personalizada (calor" not in ctx
        assert "no aclimatado" not in ctx

    def test_factores_ordenados_por_coeficiente_con_su_peso(self):
        from climasafeai.bot.telegram_bot import _contexto_parte_conversacion

        conv = self._conv(factores_calor=[
            {"nombre": "menor", "factor": 1.1},
            {"nombre": "mayor", "factor": 2.2},
            {"nombre": "medio", "factor": 1.5},
        ])
        ctx = _contexto_parte_conversacion(conv)

        assert "mayor (x2.2)" in ctx
        assert "medio (x1.5)" in ctx
        assert "menor (x1.1)" in ctx
        # De mayor a menor coeficiente, no en el orden del diccionario.
        assert ctx.index("mayor (x2.2)") < ctx.index("medio (x1.5)") < ctx.index("menor (x1.1)")
