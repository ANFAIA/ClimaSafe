"""Tests del formulario determinista del bot de Telegram (BOT-002)."""

from __future__ import annotations

import pytest

from climasafeai.bot.telegram_bot import (
    Estado,
    FIELD_LABELS,
    _conversaciones,
    procesar_mensaje,
    procesar_callback,
    _siguiente,
    _format_template,
)


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
        # IDLE y DONE no tienen campo
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
    async def test_edad_valida(self):
        _conv_limpia()
        _conversaciones[1]["estado"] = Estado.EDAD
        r = await procesar_mensaje(1, "25")
        assert r is None, f"Esperado None, got {r}"
        assert _conversaciones[1]["data"]["edad"] == 25

    @pytest.mark.asyncio
    async def test_edad_fuera_rango(self):
        _conv_limpia()
        _conversaciones[1]["estado"] = Estado.EDAD
        r = await procesar_mensaje(1, "150")
        assert r is not None
        assert "120" in r

    @pytest.mark.asyncio
    async def test_edad_invalida(self):
        _conv_limpia()
        _conversaciones[1]["estado"] = Estado.EDAD
        r = await procesar_mensaje(1, "abc")
        assert r is not None
        assert "inválido" in r or "número" in r

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
            "clase_final_label": "PELIGRO",
            "perfil": {"calor": {"prob_personalizada": 0.72}},
            "weather": {
                "provincia": "Sevilla",
                "current": {"t2m_c": 38.5, "rh": 45},
                "perfil_horario": [{"hora": 15, "HI": 39.0}],
            },
            "recomendaciones": [
                "No te expongas al sol",
                "Bebe agua",
                "Usa protección solar",
                "Evita esfuerzos",
            ],
        }
        texto = _format_template(result)
        assert "PELIGRO" in texto
        assert "Sevilla" in texto
        assert "72%" in texto or "0.72" in texto
        assert "No te expongas" in texto


class TestFlujoCompleto:
    @pytest.mark.asyncio
    async def test_flujo_simulado(self, monkeypatch):
        """Simula una conversación completa desde /start hasta DONE."""
        import climasafeai.bot.telegram_bot as mod
        monkeypatch.setattr(mod, "buscar_lugar", lambda n: {
            "lat": 42.29, "lon": -8.81, "provincia": "Pontevedra", "nombre": "Aldán, Pontevedra",
        })
        _conv_limpia()
        _conversaciones[1]["estado"] = Estado.SEXO

        await procesar_callback(1, "hombre")
        assert _conversaciones[1]["estado"] == Estado.EDAD

        await procesar_mensaje(1, "57")
        assert _conversaciones[1]["data"]["edad"] == 57
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

        async def _sin_llm(_result):
            return None

        monkeypatch.setattr(mod, "predict_ensemble", _fake_predict)
        monkeypatch.setattr(mod, "_format_with_llm", _sin_llm)

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
