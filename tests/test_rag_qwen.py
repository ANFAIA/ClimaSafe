"""Tests del módulo climasafeai.llm.rag_qwen — RAG + LLM unificado (LiteLLM)."""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock, ANY

import pytest

from climasafeai.llm.rag_qwen import (
    LLMConfig,
    MODELO_FINE_TUNED,
    MODELO_LOCAL_CPU,
    UMBRAL_DISTANCIA,
    _format_factores,
    _format_docs,
    ask_with_rag,
    ask_raw,
    check_ollama,
)


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def sample_factores():
    return [
        {
            "tipo": "calor",
            "categoria": "ambientales",
            "clave": "humedad",
            "texto": "Alta humedad reduce la eficiencia del sudor",
            "distance": 0.15,
        },
    ]


@pytest.fixture
def sample_docs():
    return [
        {
            "titulo": "Arquitectura",
            "seccion": "Modelo de riesgo",
            "texto": "El ensemble combina XGBoost, LSTM y fórmula",
            "distance": 0.22,
        },
    ]


# ── Helpers ─────────────────────────────────────────────────────────────


class TestFormat:
    def test_format_factores_vacio(self):
        assert "(ninguno)" in _format_factores([])

    def test_format_factores_con_resultados(self, sample_factores):
        res = _format_factores(sample_factores)
        assert "calor" in res
        assert "humedad" in res
        assert "0.150" in res

    def test_format_docs_vacio(self):
        assert "(ninguno)" in _format_docs([])

    def test_format_docs_con_resultados(self, sample_docs):
        res = _format_docs(sample_docs)
        assert "Arquitectura" in res
        assert "Modelo de riesgo" in res
        assert "0.220" in res


# ── Config ──────────────────────────────────────────────────────────────


class TestLLMConfig:
    def test_default_model(self):
        cfg = LLMConfig()
        assert cfg.model == "ollama/qwen2.5:1.5b"

    def test_max_tokens_default_suficiente(self):
        """BOT-022: 1024 tokens cortaban la respuesta a mitad; el default sube."""
        assert LLMConfig().max_tokens >= 2048

    def test_custom_model(self):
        cfg = LLMConfig(model="ollama/qwen2.5:7b")
        assert cfg.model == "ollama/qwen2.5:7b"

    def test_desde_modelo(self):
        cfg = LLMConfig.desde_modelo("groq/llama-3.3-70b-versatile")
        assert cfg.model == "groq/llama-3.3-70b-versatile"


class TestMejorDisponible:
    """LLM-003: el default local lo decide el benchmark, no el tamaño.

    Si alguien revierte el orden a [FINE_TUNED, GPU, CPU], estos tests fallan.
    """

    def test_fine_tuned_es_qwen3_climasafe(self):
        """LLM-014: el fine-tuned por defecto es el qwen3 entrenado en LLM-013,
        no el qwen2.5:climasafe del ciclo anterior."""
        assert MODELO_FINE_TUNED == "ollama/qwen3:climasafe"
        assert "qwen2.5:climasafe" not in MODELO_FINE_TUNED

    @patch("climasafeai.llm.rag_qwen._modelos_ollama")
    def test_solo_qwen3_disponible_lo_elige(self, mock_list):
        mock_list.return_value = ["qwen3:1.7b"]
        cfg = LLMConfig.mejor_disponible()
        assert cfg.model == "ollama/qwen3:1.7b"

    @patch("climasafeai.llm.rag_qwen._modelos_ollama")
    def test_fine_tuned_primero_si_existe(self, mock_list):
        # El orden [FINE_TUNED, GPU, QWEN3, CPU] pone el fine-tuned primero:
        # aunque qwen3 base también esté, gana el fine-tuned.
        mock_list.return_value = ["qwen3:1.7b", "qwen3:climasafe"]
        cfg = LLMConfig.mejor_disponible()
        assert cfg.model == MODELO_FINE_TUNED

    @patch("climasafeai.llm.rag_qwen._modelos_ollama")
    def test_vacio_cae_al_fallback_cpu(self, mock_list):
        mock_list.return_value = []
        cfg = LLMConfig.mejor_disponible()
        assert cfg.model == MODELO_LOCAL_CPU


# ── check_ollama ────────────────────────────────────────────────────────


class TestCheckOllama:
    @patch("climasafeai.llm.rag_qwen._modelos_ollama")
    def test_ollama_disponible(self, mock_list):
        mock_list.return_value = ["qwen2.5:1.5b", "qwen2.5:7b"]
        res = check_ollama()
        assert res["available"] is True
        assert "qwen2.5:7b" in res["models"]

    @patch("climasafeai.llm.rag_qwen._modelos_ollama")
    def test_ollama_no_disponible(self, mock_list):
        mock_list.return_value = []
        res = check_ollama()
        assert res["available"] is False

    @patch("climasafeai.llm.rag_qwen._modelos_ollama")
    def test_best_model_prioriza_qwen3_climasafe(self, mock_list):
        """LLM-014: con el fine-tuned qwen3:climasafe en Ollama, check_ollama
        lo detecta y lo devuelve como mejor modelo."""
        mock_list.return_value = ["qwen2.5:1.5b", "qwen3:climasafe", "qwen3:1.7b"]
        res = check_ollama()
        assert res["best_model"] == MODELO_FINE_TUNED


# ── ask_raw ────────────────────────────────────────────────────────────


class TestAskRaw:
    @patch("climasafeai.llm.rag_qwen._chat_litellm")
    def test_respuesta_ok(self, mock_chat):
        mock_chat.return_value = "El calor extremo es peligroso para la salud."
        res = ask_raw("¿Qué es el calor extremo?", config=LLMConfig())
        assert res["answer"] == "El calor extremo es peligroso para la salud."
        assert res["error"] is None
        assert res["model"] == "ollama/qwen2.5:1.5b"

    @patch("climasafeai.llm.rag_qwen._chat_litellm")
    def test_respuesta_fallo(self, mock_chat):
        mock_chat.return_value = None
        res = ask_raw("pregunta", config=LLMConfig())
        assert res["answer"] is None
        assert "No se pudo obtener respuesta" in res["error"]


# ── ask_with_rag ────────────────────────────────────────────────────────


class TestAskWithRag:
    @patch("climasafeai.llm.rag_qwen.DBManager")
    @patch("climasafeai.llm.rag_qwen._chat_litellm")
    def test_con_contexto(self, mock_chat, mock_db):
        mock_instance = MagicMock()
        mock_instance.search_factores.return_value = [
            {
                "tipo": "calor",
                "categoria": "ambientales",
                "clave": "humedad",
                "texto": "humedad relativa",
                "distance": 0.15,
            },
        ]
        mock_instance.search_documentos.return_value = [
            {
                "titulo": "Arquitectura",
                "seccion": "Modelo",
                "texto": "ensemble de modelos",
                "distance": 0.22,
            },
        ]
        mock_db.return_value = mock_instance

        mock_chat.return_value = (
            "La humedad alta empeora el calor. [Fuente: calor/ambientales/humedad]"
        )

        res = ask_with_rag("¿Cómo afecta la humedad?", config=LLMConfig())
        assert res["answer"] is not None
        assert len(res["sources_factores"]) == 1
        assert len(res["sources_docs"]) == 1
        assert res["error"] is None
        # Verificar que el prompt incluye el contexto
        call_args = mock_chat.call_args[0][0]  # messages
        system_msgs = [m for m in call_args if m["role"] == "system"]
        user_msgs = [m for m in call_args if m["role"] == "user"]
        assert len(system_msgs) == 1
        assert len(user_msgs) == 1
        assert "humedad" in user_msgs[0]["content"]
        assert "Arquitectura" in user_msgs[0]["content"]

    @patch("climasafeai.llm.rag_qwen.DBManager")
    @patch("climasafeai.llm.rag_qwen._chat_litellm")
    def test_sin_contexto_cae_a_raw(self, mock_chat, mock_db):
        mock_instance = MagicMock()
        mock_instance.search_factores.return_value = []
        mock_instance.search_documentos.return_value = []
        mock_db.return_value = mock_instance

        mock_chat.return_value = "Respuesta general"
        res = ask_with_rag("pregunta genérica", config=LLMConfig())
        # Sin contexto debería llamar a raw (system RAW, no RAG)
        call_args = mock_chat.call_args[0][0]
        system_content = [m["content"] for m in call_args if m["role"] == "system"][0]
        assert "RAG" not in system_content  # no es el system RAG

    @patch("climasafeai.llm.rag_qwen.DBManager")
    @patch("climasafeai.llm.rag_qwen._chat_litellm")
    def test_resultados_por_encima_del_umbral_caen_a_raw(self, mock_chat, mock_db):
        """RAG-003: lo que no supera UMBRAL_DISTANCIA no se inyecta.

        La pregunta de SPF recuperaba factores de frío a distancias 0.547+
        (vive solo, encamado, no sale de casa — medidas reales sobre
        data/climasafe.db). Con el umbral, nada pasa y la pregunta cae a
        ask_raw, sin contexto.
        """
        mock_instance = MagicMock()
        mock_instance.search_factores.return_value = [
            {
                "tipo": "frio",
                "categoria": "situacional",
                "clave": "vive_solo",
                "texto": "vive solo. tipo: frio. coeficiente: 1.3",
                "distance": 0.547,
            },
            {
                "tipo": "frio",
                "categoria": "situacional",
                "clave": "encamado",
                "texto": "encamado. tipo: frio. coeficiente: 1.5",
                "distance": 0.561,
            },
        ]
        mock_instance.search_documentos.return_value = [
            {
                "titulo": "PRD",
                "seccion": "Problema",
                "texto": "notas internas del proyecto",
                "distance": 0.553,
            },
        ]
        mock_db.return_value = mock_instance

        mock_chat.return_value = "Respuesta sin contexto"
        res = ask_with_rag("¿qué es SPF?", config=LLMConfig())
        assert res["sources_factores"] == []
        assert res["sources_docs"] == []
        # Cae a raw: system RAW, no el de RAG
        call_args = mock_chat.call_args[0][0]
        system_content = [m["content"] for m in call_args if m["role"] == "system"][0]
        assert "RAG" not in system_content

    @patch("climasafeai.llm.rag_qwen.DBManager")
    @patch("climasafeai.llm.rag_qwen._chat_litellm")
    def test_mezcla_filtra_lo_que_no_supera_el_umbral(self, mock_chat, mock_db):
        """RAG-003: por debajo del umbral se inyecta, por encima no, y cada
        fuente se filtra por separado.

        Distancias reales sobre data/climasafe.db: el factor diabetes entra a
        0.419 (por debajo del umbral); el documento interno de ML
        (Contrafactuales) no se recupera por colección y, aunque llegara a la
        búsqueda, su distancia real para esta pregunta es 0.509 (por encima).
        """
        mock_instance = MagicMock()
        mock_instance.search_factores.return_value = [
            {
                "tipo": "calor",
                "categoria": "comorbilidades",
                "clave": "diabetes",
                "texto": "diabetes. tipo: calor. coeficiente: 1.2",
                "distance": 0.419,
            },
        ]
        mock_instance.search_documentos.return_value = [
            {
                "titulo": "Contrafactuales",
                "seccion": "ML",
                "texto": "notas internas de ML",
                "distance": 0.509,
            },
        ]
        mock_db.return_value = mock_instance

        mock_chat.return_value = "La diabetes sube el riesgo"
        res = ask_with_rag(
            "tengo diabetes, cuanto me sube el riesgo con 38 grados", config=LLMConfig()
        )
        # El factor relevante (0.419 <= umbral) entra; el doc interno no
        assert len(res["sources_factores"]) == 1
        assert res["sources_docs"] == []
        call_args = mock_chat.call_args[0][0]
        user_content = [m["content"] for m in call_args if m["role"] == "user"][0]
        assert "diabetes" in user_content
        assert "Contrafactuales" not in user_content

    @patch("climasafeai.llm.rag_qwen._chat_litellm")
    def test_error_llm(self, mock_chat, monkeypatch):
        mock_db_instance = MagicMock()
        mock_db_instance.search_factores.return_value = [
            {"tipo": "calor", "categoria": "x", "clave": "y", "texto": "z", "distance": 0.1},
        ]
        mock_db_instance.search_documentos.return_value = []
        monkeypatch.setattr("climasafeai.llm.rag_qwen.DBManager", lambda: mock_db_instance)

        mock_chat.return_value = None
        res = ask_with_rag("pregunta", config=LLMConfig())
        assert res["answer"] is None
        assert res["error"] is not None

    @patch("climasafeai.llm.rag_qwen.DBManager")
    @patch("climasafeai.llm.rag_qwen._chat_litellm")
    def test_con_perfil_anade_adaptacion_contextual(self, mock_chat, mock_db):
        """LLM-005 criterio 5: ask_with_rag con perfil adapta al contexto
        (ocupación + factores con multiplicador) y prohíbe el consejo genérico."""
        mock_instance = MagicMock()
        mock_instance.search_factores.return_value = [
            {
                "tipo": "calor",
                "categoria": "ambientales",
                "clave": "humedad",
                "texto": "humedad relativa",
                "distance": 0.15,
            },
        ]
        mock_instance.search_documentos.return_value = [
            {
                "titulo": "Arquitectura",
                "seccion": "Modelo",
                "texto": "ensemble de modelos",
                "distance": 0.22,
            },
        ]
        mock_db.return_value = mock_instance

        mock_chat.return_value = "Lleva agua y pañuelo para el polvo."
        res = ask_with_rag(
            "¿qué hago con este calor en la obra?",
            config=LLMConfig(),
            perfil={
                "ocupacion": "construccion",
                "perfil": {
                    "calor": {
                        "factores": [
                            {
                                "nombre": "trabajo Construcción / albañilería (carga pesada, PPE, sol directo)",
                                "factor": 2.2,
                            },
                        ]
                    },
                    "frio": {"factores": []},
                },
            },
        )
        assert res["answer"] is not None
        call_args = mock_chat.call_args[0][0]  # messages
        user_msgs = [m for m in call_args if m["role"] == "user"]
        system_msgs = [m for m in call_args if m["role"] == "system"]
        user_content = user_msgs[0]["content"]
        system_content = system_msgs[0]["content"]
        # Adaptación contextual con ocupación y factor con multiplicador
        assert "ADAPTACIÓN CONTEXTUAL" in user_content
        assert "Ocupación:" in user_content
        assert "Construcción / albañilería (carga pesada, PPE, sol directo) (x2.2)" in user_content
        assert (
            "trabajo Construcción / albañilería (carga pesada, PPE, sol directo) (x2.2)"
            in user_content
        )
        # Consejo genérico prohibido por el system prompt
        assert "reduce la exposición en interiores" in system_content


# ── ask_con_perfil ─────────────────────────────────────────────────────


class TestAskConPerfil:
    """BOT-005: la explicación del modelo lleva la ubicación, UV y contexto.

    LLM-005: además lleva los factores con su coeficiente (xN), la ocupación
    y las franjas horarias reales; y ya no incluye la coletilla del %.
    """

    def _resultado(self):
        return {
            "clase_final": 1,
            "clase_final_label": "PRECAUCIÓN",
            "perfil": {"calor": {"prob_personalizada": 0.2, "factores": []}},
            "weather": {
                "provincia": "Pontevedra",
                "current": {"t2m_c": 34.0, "rh": 45},
                "uv_index": 7,
                "perfil_horario": [{"hora": 15, "HI": 36.0, "temp": 34.0}],
            },
            "modelos": {
                "Formula": {
                    "frio": {"wind_chill_c": 25.0},
                    "calor": {"heat_index_c": 36.0},
                },
            },
            "recomendaciones": ["Mantente hidratado"],
        }

    def _resultado_con_factores_y_ocupacion(self):
        """Resultado con factores del pipeline real (dicts {nombre, factor})
        y perfil con ocupación construcción (x2.2 en _OCUPACION_NIVELES)."""
        resultado = self._resultado()
        resultado["perfil"]["calor"]["factores"] = [
            {
                "nombre": "trabajo Construcción / albañilería (carga pesada, PPE, sol directo)",
                "factor": 2.2,
            },
            {"nombre": "no aclimatado", "factor": 1.3},
        ]
        resultado["weather"]["perfil_horario"] = [
            {"hora": h, "HI": 25.0 + (7 if h == 15 else 0), "temp": 24.0 + (7 if h == 15 else 0)}
            for h in range(24)
        ]
        return resultado

    @patch("climasafeai.llm.rag_qwen._chat_litellm")
    def test_prompt_incluye_ubicacion_uv_y_recomendacion_contextual(self, mock_chat):
        from climasafeai.llm.rag_qwen import ask_con_perfil

        mock_chat.return_value = "Moaña, Pontevedra — Riesgo PRECAUCIÓN (20%). Mantente hidratado."
        texto = ask_con_perfil(
            {"hora_inicio": 15, "duracion_actividad_h": 1},
            self._resultado(),
            config=LLMConfig(),
            lugar="Moaña, Pontevedra",
        )
        # BOT-013: el texto del LLM se devuelve tal cual, salvo que se le
        # repone la frase que separa el nivel del porcentaje si se la comió.
        assert texto.startswith(mock_chat.return_value)
        assert "no es lo que decide tu nivel" in texto

        mensajes = mock_chat.call_args[0][0]

        # El system prompt fija el idioma: qwen3:1.7b redactaba el parte de
        # Pontevedra en portugués cuando esta llamada iba sin system.
        assert mensajes[0]["role"] == "system"
        assert "español" in mensajes[0]["content"].lower()
        assert "portugués" in mensajes[0]["content"].lower()

        prompt = mensajes[-1]["content"]
        assert "Ubicación: Moaña, Pontevedra" in prompt
        assert "Índice UV: 7" in prompt
        assert "Recomendación contextual:" in prompt
        assert "SPF 30+" in prompt
        # BOT-013: el % ya no se pide como "20% de probabilidad de riesgo
        # térmico"; va como frecuencia natural en las frases obligatorias.
        assert "de probabilidad de riesgo térmico durante la actividad" not in prompt
        assert "en unos 20 el calor te pasaría factura" in prompt
        # LLM-005: la coletilla ya no está ni en el prompt ni en el ejemplo
        assert "mayor cuanto más se acerque a 100%" not in prompt
        # La franja horaria sale de los datos reales, no la inventa el LLM
        assert "Franja recomendada" in prompt
        assert "Pico de calor" in prompt

    @patch("climasafeai.llm.rag_qwen._chat_litellm")
    def test_prompt_lleva_coeficientes_ocupacion_y_franjas_reales(self, mock_chat):
        from climasafeai.llm.rag_qwen import ask_con_perfil

        mock_chat.return_value = "Parte."
        ask_con_perfil(
            {"hora_inicio": 10, "duracion_actividad_h": 2, "ocupacion": "construccion"},
            self._resultado_con_factores_y_ocupacion(),
            config=LLMConfig(),
            lugar="Madrid",
        )
        mensajes = mock_chat.call_args[0][0]
        prompt = mensajes[-1]["content"]

        # Factores con su multiplicador, no solo el nombre
        assert "Factores de riesgo:" in prompt
        assert "trabajo Construcción / albañilería" in prompt
        assert "(x2.2)" in prompt
        assert "no aclimatado (x1.3)" in prompt
        # Ocupación del perfil con etiqueta y coeficiente
        assert "Ocupación:" in prompt
        assert "Construcción / albañilería (carga pesada, PPE, sol directo) (x2.2)" in prompt
        # Franjas reales: la recomendada (recomendar_horario) y el pico de HI
        assert "Franja recomendada (menor riesgo):" in prompt
        assert "Pico de calor (evitar si puedes): 15:00 (HI 32.0°C)" in prompt
        # La coletilla no está ni en el prompt ni en el ejemplo
        assert "mayor cuanto más" not in prompt

    @patch("climasafeai.llm.rag_qwen._chat_litellm")
    def test_system_parte_explica_terminos_y_adapta_al_contexto(self, mock_chat):
        from climasafeai.llm.rag_qwen import ask_con_perfil

        mock_chat.return_value = "Parte."
        ask_con_perfil(
            {"hora_inicio": 10, "duracion_actividad_h": 2, "ocupacion": "construccion"},
            self._resultado_con_factores_y_ocupacion(),
            config=LLMConfig(),
            lugar="Madrid",
        )
        mensajes = mock_chat.call_args[0][0]
        system = mensajes[0]["content"]

        # LLM-005: normas del SYSTEM_PARTE
        assert "lenguaje" in system and "llano" in system  # términos técnicos (SPF, HI, WC)
        assert "reduce la exposición en interiores" in system  # consejo prohibido
        assert "No inventes horas" in system  # horas reales, no inventadas
        # BOT-013: el system ya no pide "multiplica el riesgo por N" — lo
        # prohíbe. El factor dominante va traducido a llano desde Python.
        assert "FRASES OBLIGATORIAS" in system
        assert "no le dice nada a quien lee" in system


# ── BOT-013: el parte se entiende sin saber ML ─────────────────────────


class TestParteLlano:
    """BOT-013: 'PRECAUCION: 19%' no lo entiende nadie que no sepa ML.

    Todo lo que el parte debe decir sí o sí se redacta en Python y el LLM solo
    lo copia: el ancla de la clase, la frecuencia natural, la separación entre
    clase y porcentaje, el factor dominante con línea base y la confianza.
    """

    # ── helpers deterministas ──────────────────────────────────────────

    def test_clase_llana_ancla_el_nivel_en_su_escala(self):
        from climasafeai.llm.rag_qwen import _clase_llana

        # El ensemble escribe "PRECAUCION" (CLASES) y el bot "PRECAUCIÓN"
        assert _clase_llana("PRECAUCION") == _clase_llana("PRECAUCIÓN")
        assert "nivel intermedio de tres" in _clase_llana("PRECAUCION")
        assert "seguro / precaución / peligro" in _clase_llana("PRECAUCION")
        assert "nivel más bajo de tres" in _clase_llana("SEGURO")
        assert "nivel más alto de tres" in _clase_llana("PELIGRO")
        # Una clase desconocida no revienta ni se inventa una escala
        assert _clase_llana(None) == "?"
        assert _clase_llana("DESCONOCIDO") == "DESCONOCIDO"

    def test_frecuencia_natural_traduce_el_porcentaje(self):
        from climasafeai.llm.rag_qwen import _frecuencia_natural

        assert "en unos 19 el calor te pasaría factura" in _frecuencia_natural(0.19)
        assert "De cada 100 días" in _frecuencia_natural(0.19)
        assert "en unos 72" in _frecuencia_natural(0.7156)
        assert "en 1 " in _frecuencia_natural(0.01)
        assert "en menos de 1" in _frecuencia_natural(0.002)
        # Entradas degeneradas: 0, None y fuera de rango no revientan
        assert "en menos de 1" in _frecuencia_natural(0)
        assert "en menos de 1" in _frecuencia_natural(None)
        assert "en unos 100" in _frecuencia_natural(3.0)

    def test_coeficiente_llano_traduce_el_multiplicador(self):
        from climasafeai.llm.rag_qwen import _coeficiente_llano

        assert _coeficiente_llano(2.2) == "algo más del doble"  # construcción
        assert _coeficiente_llano(2.7) == "casi el triple"  # campo
        assert _coeficiente_llano(2.0) == "el doble"
        assert _coeficiente_llano(3.0) == "el triple"
        assert _coeficiente_llano(1.3) == "un 30% más alto"  # no aclimatado
        assert _coeficiente_llano(1.0) == "prácticamente el mismo"
        assert _coeficiente_llano(4.5) == "más de 4 veces mayor"

    def test_factor_dominante_lleva_linea_base(self):
        from climasafeai.llm.rag_qwen import _linea_factor_dominante

        linea = _linea_factor_dominante(
            [
                {"nombre": "no aclimatado", "factor": 1.3},
                {
                    "nombre": "trabajo Construcción / albañilería (carga pesada, PPE, sol directo)",
                    "factor": 2.2,
                },
            ]
        )
        # El x2.2 no significa nada sin un "comparado con quién"
        assert linea == (
            "Lo que más pesa en tu caso no es el tiempo, es tu trabajo: en "
            "construcción tu riesgo es algo más del doble que el de alguien "
            "como tú a cubierto."
        )

    def test_factor_dominante_no_ocupacional_y_casos_vacios(self):
        from climasafeai.llm.rag_qwen import _linea_factor_dominante

        linea = _linea_factor_dominante([{"nombre": "no aclimatado", "factor": 1.3}])
        assert "es no aclimatado" in linea
        assert "un 30% más alto que el de alguien como tú sin él" in linea
        # Sin factores, o con factores que no suben nada, no hay línea
        assert _linea_factor_dominante([]) is None
        assert _linea_factor_dominante(["texto suelto"]) is None
        assert _linea_factor_dominante([{"nombre": "x", "factor": 1.0}]) is None

    def test_confianza_conformal_sale_del_ensemble_y_tolera_none(self):
        from climasafeai.llm.rag_qwen import _confianza_conformal

        assert (
            _confianza_conformal({"modelos": {"XGBoost_calor": {"conformal_confianza": "baja"}}})
            == "baja"
        )
        # Sin artefacto conformal el ensemble deja None: no se inventa nada
        assert (
            _confianza_conformal({"modelos": {"XGBoost_calor": {"conformal_confianza": None}}})
            is None
        )
        assert _confianza_conformal({}) is None
        # Si el canal de calor no la trae, vale la del canal de frío
        assert (
            _confianza_conformal(
                {"modelos": {"RandomForest_frio": {"conformal_confianza": "alta"}}}
            )
            == "alta"
        )

    # ── el prompt del parte ────────────────────────────────────────────

    def _resultado(self, prob=0.19, confianza="alta"):
        return {
            "clase_final": 1,
            "clase_final_label": "PRECAUCION",
            "perfil": {
                "calor": {
                    "prob_personalizada": prob,
                    "factores": [
                        {
                            "nombre": "trabajo Construcción / albañilería (carga pesada, PPE, sol directo)",
                            "factor": 2.2,
                        },
                        {"nombre": "no aclimatado", "factor": 1.3},
                    ],
                }
            },
            "weather": {
                "provincia": "Pontevedra",
                "current": {"t2m_c": 28.0, "rh": 70},
                "uv_index": 6,
                "perfil_horario": [
                    {"hora": h, "HI": 25.0 + (5 if h == 15 else 0), "temp": 24.0} for h in range(24)
                ],
            },
            "modelos": {
                "XGBoost_calor": {"conformal_confianza": confianza},
                "Formula": {"frio": {"wind_chill_c": 20.0}, "calor": {"heat_index_c": 30.0}},
            },
            "recomendaciones": ["Mantente hidratado"],
        }

    def _prompt(self, mock_chat, resultado):
        from climasafeai.llm.rag_qwen import ask_con_perfil

        mock_chat.return_value = "Parte."
        ask_con_perfil(
            {"hora_inicio": 10, "duracion_actividad_h": 6, "ocupacion": "construccion"},
            resultado,
            config=LLMConfig(),
            lugar="Vigo, Pontevedra",
        )
        return mock_chat.call_args[0][0][-1]["content"]

    @patch("climasafeai.llm.rag_qwen._chat_litellm")
    def test_prompt_lleva_las_frases_obligatorias_ya_redactadas(self, mock_chat):
        prompt = self._prompt(mock_chat, self._resultado())

        assert "FRASES OBLIGATORIAS" in prompt
        # 0. BOT-020: el parte abre con la clasificación y la probabilidad en %
        assert (
            "Clasificación: PRECAUCIÓN — probabilidad de riesgo personalizada por "
            "calor: 19% (0.1900)." in prompt
        )
        # 1. la clase anclada en su escala
        assert "Vigo, Pontevedra — PRECAUCIÓN, el nivel intermedio de tres" in prompt
        # 2. el porcentaje como frecuencia natural
        assert "en unos 19 el calor te pasaría factura" in prompt
        # 3. el factor dominante en llano y con línea base
        assert (
            "en construcción tu riesgo es algo más del doble que el de alguien como tú a cubierto"
            in prompt
        )
        # el multiplicador crudo ya no se le pide al LLM como frase
        assert "multiplica el riesgo por 2.2" not in prompt

    @patch("climasafeai.llm.rag_qwen._chat_litellm")
    def test_prompt_separa_la_clase_del_porcentaje(self, mock_chat):
        prompt = self._prompt(mock_chat, self._resultado())

        # Criterio 2: son dos caminos de decisión distintos y el parte lo dice
        assert "Esa cifra no es lo que decide tu nivel" in prompt
        assert "umbrales de tu provincia" in prompt
        assert "una cifra baja puede venir con un nivel alto" in prompt
        # y el bloque de datos también lo deja claro para el LLM
        assert "NO la probabilidad de abajo" in prompt

    @patch("climasafeai.llm.rag_qwen._chat_litellm")
    def test_prompt_lleva_la_confianza_conformal_en_llano(self, mock_chat):
        prompt = self._prompt(mock_chat, self._resultado(confianza="alta"))
        assert "Confianza del modelo: alta" in prompt
        assert "El modelo está seguro de este nivel" in prompt

        prompt = self._prompt(mock_chat, self._resultado(confianza="media"))
        assert "duda entre este nivel y el de al lado" in prompt

    @patch("climasafeai.llm.rag_qwen._chat_litellm")
    def test_confianza_baja_avisa_en_el_prompt(self, mock_chat):
        prompt = self._prompt(mock_chat, self._resultado(confianza="baja"))
        assert "Confianza del modelo: baja" in prompt
        assert "hoy el modelo tiene poca confianza" in prompt
        assert "ve por el lado seguro" in prompt

    @patch("climasafeai.llm.rag_qwen._chat_litellm")
    def test_sin_conformal_el_parte_no_se_inventa_la_confianza(self, mock_chat):
        prompt = self._prompt(mock_chat, self._resultado(confianza=None))
        assert "Confianza del modelo: no medida" in prompt
        assert "El modelo está seguro" not in prompt
        assert "poca confianza" not in prompt
        # el resto del parte sigue completo
        assert "en unos 19 el calor te pasaría factura" in prompt

    @patch("climasafeai.llm.rag_qwen._chat_litellm")
    def test_se_repone_la_separacion_clase_porcentaje_si_el_llm_la_olvida(self, mock_chat):
        from climasafeai.llm.rag_qwen import ask_con_perfil

        # Caso real: qwen2.5:1.5b copió 4 de las 5 frases obligatorias y se
        # comió justo la que explica que el nivel no sale del porcentaje.
        mock_chat.return_value = (
            "Vigo, Pontevedra — PRECAUCIÓN, el nivel intermedio de tres. "
            "De cada 100 días como el de hoy y con la misma salida, en unos 19 "
            "el calor te pasaría factura. Hidrátate y usa SPF 30+."
        )
        texto = ask_con_perfil(
            {"ocupacion": "construccion"},
            self._resultado(confianza="alta"),
            config=LLMConfig(),
            lugar="Vigo, Pontevedra",
        )
        assert "no es lo que decide tu nivel" in texto
        assert "una cifra baja puede venir con un nivel alto" in texto

    @patch("climasafeai.llm.rag_qwen._chat_litellm")
    def test_no_se_repone_lo_que_el_llm_ya_copio(self, mock_chat):
        from climasafeai.llm.rag_qwen import LINEA_CLASE_VS_PORCENTAJE, ask_con_perfil

        mock_chat.return_value = f"Vigo — PRECAUCIÓN. {LINEA_CLASE_VS_PORCENTAJE} Hidrátate."
        texto = ask_con_perfil(
            {"ocupacion": "construccion"},
            self._resultado(confianza="alta"),
            config=LLMConfig(),
            lugar="Vigo, Pontevedra",
        )
        assert texto.count("no es lo que decide tu nivel") == 1

    @patch("climasafeai.llm.rag_qwen._chat_litellm")
    def test_el_aviso_de_confianza_baja_se_anade_si_el_llm_lo_olvida(self, mock_chat):
        from climasafeai.llm.rag_qwen import ask_con_perfil

        # Un 1.5B se salta frases del prompt: el aviso es de seguridad y no
        # puede quedar a su criterio.
        mock_chat.return_value = "Vigo — PRECAUCIÓN. Hidrátate."
        texto = ask_con_perfil(
            {"ocupacion": "construccion"},
            self._resultado(confianza="baja"),
            config=LLMConfig(),
            lugar="Vigo, Pontevedra",
        )
        assert "hoy el modelo tiene poca confianza" in texto

    @patch("climasafeai.llm.rag_qwen._chat_litellm")
    def test_el_aviso_no_se_duplica_si_el_llm_ya_lo_copio(self, mock_chat):
        from climasafeai.llm.rag_qwen import _CONFIANZA_LLANA, ask_con_perfil

        mock_chat.return_value = f"Vigo — PRECAUCIÓN. {_CONFIANZA_LLANA['baja']}"
        texto = ask_con_perfil(
            {"ocupacion": "construccion"},
            self._resultado(confianza="baja"),
            config=LLMConfig(),
            lugar="Vigo, Pontevedra",
        )
        assert texto.count("poca confianza") == 1

    @patch("climasafeai.llm.rag_qwen._chat_litellm")
    def test_se_repone_la_frecuencia_si_el_llm_se_la_come(self, mock_chat):
        from climasafeai.llm.rag_qwen import ask_con_perfil

        # Caso real de qwen2.5:1.5b: en una tirada se comió la separación
        # clase/porcentaje y en la siguiente la frecuencia. Sin ella el parte
        # se queda sin ninguna cifra.
        mock_chat.return_value = (
            "Vigo, Pontevedra — PRECAUCIÓN, el nivel intermedio de tres. "
            "Lo que más pesa en tu caso no es el tiempo, es tu trabajo. "
            "El modelo está seguro de este nivel. Hidrátate."
        )
        texto = ask_con_perfil(
            {"ocupacion": "construccion"},
            self._resultado(confianza="alta"),
            config=LLMConfig(),
            lugar="Vigo, Pontevedra",
        )
        assert "en unos 19 el calor te pasaría factura" in texto
        assert "no es lo que decide tu nivel" in texto
        # Lo que sí copió no se duplica
        assert texto.count("Lo que más pesa") == 1
        assert texto.count("El modelo está seguro") == 1


# ── ARNES-003: modo debug del payload hacia el LLM ─────────────────────


class TestTruncadoYThink:
    """BOT-022: el corte por max_tokens se avisa y el <think> de qwen3 no llega.

    Reproduce el fallo del 13-08: completion=1024 tokens exactos
    (max_tokens) → respuesta truncada a mitad y larga, que además tumbaba el
    envío a Telegram. El default de max_tokens sube y el corte que quede se
    marca; el bloque de razonamiento de qwen3 no contamina la respuesta.
    """

    @staticmethod
    def _mock_completion(
        respuesta: str, finish: str = "stop", completion_tokens: int | None = None
    ) -> MagicMock:
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = respuesta
        mock_resp.choices[0].finish_reason = finish
        if completion_tokens is None:
            mock_resp.usage = None
        else:
            mock_resp.usage.completion_tokens = completion_tokens
        return mock_resp

    def test_respuesta_truncada_por_length_lleva_aviso(self):
        from climasafeai.llm.rag_qwen import AVISO_TRUNCADO, ask_raw

        with patch("litellm.completion") as mock_completion:
            mock_completion.return_value = self._mock_completion(
                "Bebe agua y evita el sol", finish="length"
            )
            res = ask_raw("pregunta", config=LLMConfig())
        assert AVISO_TRUNCADO in res["answer"]

    def test_respuesta_completa_sin_aviso(self):
        from climasafeai.llm.rag_qwen import AVISO_TRUNCADO, ask_raw

        with patch("litellm.completion") as mock_completion:
            mock_completion.return_value = self._mock_completion("Bebe agua")
            res = ask_raw("pregunta", config=LLMConfig())
        assert AVISO_TRUNCADO not in res["answer"]

    def test_ollama_finish_stop_con_tokens_agotados_lleva_aviso(self):
        """BOT-022: Ollama corta por max_tokens pero reporta finish_reason='stop'
        (verificado con qwen3:climasafe real); el aviso no puede depender solo
        de finish_reason='length'."""
        from climasafeai.llm.rag_qwen import AVISO_TRUNCADO, ask_raw

        with patch("litellm.completion") as mock_completion:
            mock_completion.return_value = self._mock_completion(
                "Respuesta que se corta", finish="stop", completion_tokens=2048
            )
            res = ask_raw("pregunta", config=LLMConfig())
        assert AVISO_TRUNCADO in res["answer"]

    def test_limpiar_bloque_think_multilinea(self):
        from climasafeai.llm.rag_qwen import _limpiar_bloque_think

        limpio = _limpiar_bloque_think(
            "<think>\nVoy a calcular el riesgo paso a paso...\n</think>\nBebe agua."
        )
        assert limpio == "Bebe agua."

    def test_limpiar_think_cierre_suelto(self):
        """BOT-022: qwen3:climasafe emitió solo el cierre </think> sin apertura
        (conversación real del 13-08); no debe llegar al usuario."""
        from climasafeai.llm.rag_qwen import _limpiar_bloque_think

        limpio = _limpiar_bloque_think("</think>\n\nRIESGO: SEGURO\nÍndice personalizado: 0.84")
        assert limpio.startswith("RIESGO: SEGURO")
        assert "</think>" not in limpio

    def test_qwen3_con_thinking_no_contamina_la_respuesta(self):
        from climasafeai.llm.rag_qwen import ask_raw

        with patch("litellm.completion") as mock_completion:
            mock_completion.return_value = self._mock_completion(
                "<think>razonamiento interno</think>Bebe agua y evita el sol."
            )
            res = ask_raw("pregunta", config=LLMConfig(model="ollama/qwen3:1.7b"))
        assert "<think>" not in res["answer"]
        assert "razonamiento interno" not in res["answer"]

    def test_qwen25_sin_bloque_no_se_toca(self):
        from climasafeai.llm.rag_qwen import ask_raw

        with patch("litellm.completion") as mock_completion:
            mock_completion.return_value = self._mock_completion(
                "Texto con <think> mencionado pero sin bloque"
            )
            res = ask_raw("pregunta", config=LLMConfig(model="ollama/qwen2.5:1.5b"))
        assert res["answer"] == "Texto con <think> mencionado pero sin bloque"


class TestDebugLLM:
    """CLIMASAFE_DEBUG_LLM=1 registra el payload exacto y los tokens por parte.

    Apagado (default) no cambia nada: ni se registra ni se toca el payload.
    Los tests parchean `litellm.completion` (no `_chat_litellm`) para que el
    hook de debug, que vive dentro de `_chat_litellm`, se ejecute de verdad.
    """

    @staticmethod
    def _mock_completion(respuesta: str) -> MagicMock:
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = respuesta
        return mock_resp

    def test_apagado_no_registra_ni_cambia_la_respuesta(self, caplog, monkeypatch):
        from climasafeai.llm.rag_qwen import ask_raw

        monkeypatch.delenv("CLIMASAFE_DEBUG_LLM", raising=False)
        with patch("litellm.completion") as mock_completion:
            mock_completion.return_value = self._mock_completion("Respuesta normal.")
            with caplog.at_level("INFO"):
                res = ask_raw("¿qué es SPF?", config=LLMConfig())
        assert res["answer"] == "Respuesta normal."
        assert "[CLIMASAFE_DEBUG_LLM]" not in caplog.text

    @patch("climasafeai.llm.rag_qwen.DBManager")
    @patch("litellm.completion")
    def test_activado_registra_payload_completo_y_desglose(
        self, mock_completion, mock_db, caplog, monkeypatch
    ):
        from climasafeai.llm.rag_qwen import SYSTEM_RAG

        monkeypatch.setenv("CLIMASAFE_DEBUG_LLM", "1")
        mock_completion.return_value = self._mock_completion("La humedad alta empeora el calor.")
        mock_instance = MagicMock()
        mock_instance.search_factores.return_value = [
            {
                "tipo": "calor",
                "categoria": "ambientales",
                "clave": "humedad",
                "texto": "humedad relativa",
                "distance": 0.15,
            },
        ]
        mock_instance.search_documentos.return_value = [
            {
                "titulo": "Arquitectura",
                "seccion": "Modelo",
                "texto": "ensemble de modelos",
                "distance": 0.22,
            },
        ]
        mock_db.return_value = mock_instance

        with caplog.at_level("INFO"):
            ask_with_rag("¿cómo afecta la humedad?", config=LLMConfig())

        # Criterio 1: el system prompt completo y el contexto RAG sin resumir
        assert "[CLIMASAFE_DEBUG_LLM]" in caplog.text
        assert "== PAYLOAD HACIA EL LLM" in caplog.text
        assert "--- mensaje 0 [system] (" in caplog.text
        assert SYSTEM_RAG in caplog.text
        assert "humedad relativa" in caplog.text
        assert "ensemble de modelos" in caplog.text
        # Criterio 2: desglose de tokens por parte
        assert "== DESGLOSE DE TOKENS" in caplog.text
        assert "system prompt:" in caplog.text
        assert "contexto RAG:" in caplog.text
        assert "pregunta / contexto:" in caplog.text
        assert "TOTAL:" in caplog.text
        # El desglose numérico se puede leer: contexto RAG > 0
        linea_rag = next(x for x in caplog.text.splitlines() if "contexto RAG:" in x)
        assert int(linea_rag.split(":")[-1].strip()) > 0

    @patch("climasafeai.llm.rag_qwen.DBManager")
    @patch("litellm.completion")
    def test_caso_spf_cae_a_ask_raw_sin_contexto(
        self, mock_completion, mock_db, caplog, monkeypatch
    ):
        """Caso conocido: '¿qué es SPF?' recuperaba factores de frío como contexto.

        Reproduce lo que devuelve el DB real (ver data/climasafe.db): 4 de los
        5 primeros factores son de frío con distancias 0.547+ (vive_solo 0.547,
        encamado 0.561, no_sale 0.595). Con el umbral de distancia (RAG-003)
        nada supera 0.50 y la pregunta cae a ask_raw, sin inyectar contexto
        irrelevante.
        """
        monkeypatch.setenv("CLIMASAFE_DEBUG_LLM", "1")
        mock_completion.return_value = self._mock_completion(
            "El SPF es el factor de protección solar."
        )
        mock_instance = MagicMock()
        mock_instance.search_factores.return_value = [
            {
                "tipo": "frio",
                "categoria": "situacional",
                "clave": "vive_solo",
                "texto": "vive solo. tipo: frio. coeficiente: 1.3",
                "distance": 0.547,
            },
            {
                "tipo": "frio",
                "categoria": "situacional",
                "clave": "encamado",
                "texto": "encamado. tipo: frio. coeficiente: 1.5",
                "distance": 0.561,
            },
            {
                "tipo": "frio",
                "categoria": "situacional",
                "clave": "no_sale",
                "texto": "no sale de casa. tipo: frio. coeficiente: 1.3",
                "distance": 0.595,
            },
        ]
        mock_instance.search_documentos.return_value = []
        mock_db.return_value = mock_instance

        with caplog.at_level("INFO"):
            res = ask_with_rag("¿qué es SPF?", config=LLMConfig())

        assert res["error"] is None
        # Nada supera el umbral: las fuentes quedan vacías y no se inyecta
        # contexto (la llamada es ask_raw, con el system RAW).
        assert res["sources_factores"] == []
        assert res["sources_docs"] == []
        # El payload que se registra es el de ask_raw: sin bloque RAG ni los
        # factores de frío que antes se colaban en la pregunta.
        assert "[CLIMASAFE_DEBUG_LLM]" in caplog.text
        assert "=== FACTORES DE RIESGO RELEVANTES ===" not in caplog.text
        assert "vive_solo" not in caplog.text
        assert "encamado" not in caplog.text
        assert "no_sale" not in caplog.text
        assert "=== PREGUNTA ===" not in caplog.text
        assert "¿qué es SPF?" in caplog.text
