"""Tests del módulo climasafeai.llm.rag_qwen — RAG + LLM unificado (LiteLLM)."""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock, ANY

import pytest

from climasafeai.llm.rag_qwen import (
    LLMConfig,
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
        {"tipo": "calor", "categoria": "ambientales", "clave": "humedad",
         "texto": "Alta humedad reduce la eficiencia del sudor", "distance": 0.15},
    ]


@pytest.fixture
def sample_docs():
    return [
        {"titulo": "Arquitectura", "seccion": "Modelo de riesgo",
         "texto": "El ensemble combina XGBoost, LSTM y fórmula", "distance": 0.22},
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

    def test_custom_model(self):
        cfg = LLMConfig(model="ollama/qwen2.5:7b")
        assert cfg.model == "ollama/qwen2.5:7b"

    def test_desde_modelo(self):
        cfg = LLMConfig.desde_modelo("groq/llama-3.3-70b-versatile")
        assert cfg.model == "groq/llama-3.3-70b-versatile"


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
            {"tipo": "calor", "categoria": "ambientales", "clave": "humedad",
             "texto": "humedad relativa", "distance": 0.15},
        ]
        mock_instance.search_documentos.return_value = [
            {"titulo": "Arquitectura", "seccion": "Modelo",
             "texto": "ensemble de modelos", "distance": 0.22},
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


# ── ask_con_perfil ─────────────────────────────────────────────────────


class TestAskConPerfil:
    """BOT-005: la explicación del modelo lleva la ubicación, UV y contexto."""

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

    @patch("climasafeai.llm.rag_qwen._chat_litellm")
    def test_prompt_incluye_ubicacion_uv_y_recomendacion_contextual(self, mock_chat):
        from climasafeai.llm.rag_qwen import ask_con_perfil

        mock_chat.return_value = (
            "Moaña, Pontevedra — Riesgo PRECAUCIÓN (20%). Mantente hidratado."
        )
        texto = ask_con_perfil(
            {"hora_inicio": 15, "duracion_actividad_h": 1},
            self._resultado(),
            config=LLMConfig(),
            lugar="Moaña, Pontevedra",
        )
        assert texto == mock_chat.return_value

        prompt = mock_chat.call_args[0][0][0]["content"]
        assert "Ubicación: Moaña, Pontevedra" in prompt
        assert "Índice UV: 7" in prompt
        assert "Recomendación contextual:" in prompt
        assert "SPF 30+" in prompt
        # El ejemplo del prompt usa el formato "ubicación — Riesgo CLASE (XX%)"
        assert "Moaña, Pontevedra — Riesgo PRECAUCIÓN (20%)" in prompt
