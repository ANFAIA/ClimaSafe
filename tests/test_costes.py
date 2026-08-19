"""Tests del contador de tokens y coste por llamada al LLM (ARNES-004).

Cubre los cuatro bloques de la feature:
1. Lectura del `usage` que devuelve LiteLLM (prompt/completion tokens).
2. Cálculo de coste desde la tabla PRECIOS_MODELOS (un solo sitio).
3. Coste cero en modelos locales (Ollama) con tokens y latencia medidos.
4. Acumulado por sesión consultable.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from climasafeai.llm.costes import (
    PRECIOS_MODELOS,
    _ACUMULADOS,
    coste_llamada,
    es_local,
    precios_de,
    registrar_llamada,
    resumen_sesion,
)
from climasafeai.llm.rag_qwen import LLMConfig, _chat_litellm, ask_raw


@pytest.fixture(autouse=True)
def _acumulados_limpios():
    """El acumulado es estado global de módulo: cada test empieza de cero."""
    _ACUMULADOS.clear()
    yield
    _ACUMULADOS.clear()


def _respuesta_litellm(prompt_tokens: int, completion_tokens: int) -> MagicMock:
    """Respuesta fake con la forma del objeto que devuelve litellm.completion:
    `resp.usage` con prompt_tokens/completion_tokens y `choices[0].message`."""
    resp = MagicMock()
    resp.usage.prompt_tokens = prompt_tokens
    resp.usage.completion_tokens = completion_tokens
    resp.choices[0].message.content = "respuesta de prueba"
    resp.choices[0].finish_reason = "stop"
    return resp


# ── 1. Lectura del usage de LiteLLM ────────────────────────────────────


class TestLecturaUsage:
    def test_chat_litellm_registra_tokens_desde_usage(self):
        """Cada llamada al LLM registra prompt/completion del `usage` real."""
        with patch("litellm.completion", return_value=_respuesta_litellm(120, 45)) as mock_comp:
            out = _chat_litellm(
                [{"role": "user", "content": "hola"}],
                LLMConfig(model="groq/openai/gpt-oss-20b"),
            )
        assert out == "respuesta de prueba"
        assert mock_comp.call_count == 1
        resumen = resumen_sesion()
        assert resumen["llamadas"] == 1
        assert resumen["prompt_tokens"] == 120
        assert resumen["completion_tokens"] == 45

    def test_usage_sin_tokens_registra_cero(self):
        """Si el proveedor no devuelve usage, se registra igual con ceros."""
        with patch("litellm.completion", return_value=_respuesta_litellm(0, 0)):
            _chat_litellm([{"role": "user", "content": "hola"}], LLMConfig())
        assert resumen_sesion()["llamadas"] == 1
        assert resumen_sesion()["prompt_tokens"] == 0

    def test_ask_raw_propaga_la_sesion(self):
        """La sesión llega hasta el registro a través de la ruta pública."""
        with patch("litellm.completion", return_value=_respuesta_litellm(30, 10)):
            res = ask_raw("pregunta", config=LLMConfig(), sesion_id="chat-42")
        assert res["answer"] == "respuesta de prueba"
        assert resumen_sesion("chat-42")["llamadas"] == 1
        # La sesión por defecto no se contamina.
        assert resumen_sesion()["llamadas"] == 0


# ── 2. Cálculo de coste desde la tabla de precios ──────────────────────


class TestCosteDesdeTabla:
    def test_tabla_es_un_solo_sitio(self):
        """PRECIOS_MODELOS existe y tiene el modelo remoto por defecto."""
        assert "gpt-oss-20b" in PRECIOS_MODELOS

    def test_coste_modelo_remoto(self):
        """0.20 $/M prompt + 0.80 $/M completion, con 1.000 tokens de cada."""
        coste = coste_llamada("groq/openai/gpt-oss-20b", 1_000, 1_000)
        assert coste == pytest.approx(0.00020 + 0.00080)
        # 1M tokens de prompt = 0.20 $ exactos.
        assert coste_llamada("groq/openai/gpt-oss-20b", 1_000_000, 0) == pytest.approx(0.20)

    def test_precios_de_quita_prefijos_anidados(self):
        """'groq/openai/gpt-oss-20b' resuelve contra la entrada 'gpt-oss-20b'."""
        prompt, completion = precios_de("groq/openai/gpt-oss-20b")
        assert prompt == pytest.approx(0.20 / 1_000_000)
        assert completion == pytest.approx(0.80 / 1_000_000)


# ── 3. Modelos locales: coste cero, tokens y latencia sí ───────────────


class TestModelosLocales:
    def test_ollama_es_local(self):
        assert es_local("ollama/qwen2.5:1.5b")
        assert not es_local("groq/openai/gpt-oss-20b")

    def test_coste_cero_en_local_con_tokens(self):
        """En Ollama el coste es cero pero los tokens cuentan."""
        assert coste_llamada("ollama/qwen2.5:1.5b", 50_000, 10_000) == 0.0
        assert coste_llamada("ollama/qwen3:climasafe", 1, 1) == 0.0

    def test_registro_local_mide_tokens_y_latencia(self):
        """La latencia y los tokens se registran aunque el coste sea cero."""
        detalle = registrar_llamada("ollama/qwen3:1.7b", 240, 80, 3.456, sesion_id="local")
        assert detalle["coste"] == 0.0
        assert detalle["latencia_s"] == 3.456
        assert detalle["prompt_tokens"] == 240
        assert detalle["completion_tokens"] == 80
        resumen = resumen_sesion("local")
        assert resumen["coste"] == 0.0
        assert resumen["latencia_s"] == pytest.approx(3.456)
        assert resumen["prompt_tokens"] == 240

    def test_chat_litellm_local_cero_y_medido(self):
        """La vía real (LiteLLM + Ollama) también registra cero/medido."""
        with patch("litellm.completion", return_value=_respuesta_litellm(90, 25)):
            _chat_litellm(
                [{"role": "user", "content": "hola"}],
                LLMConfig(model="ollama/qwen2.5:1.5b"),
            )
        resumen = resumen_sesion()
        assert resumen["coste"] == 0.0
        assert resumen["prompt_tokens"] == 90
        assert resumen["completion_tokens"] == 25
        assert resumen["latencia_s"] > 0


# ── 4. Acumulado por sesión consultable ────────────────────────────────


class TestAcumuladoSesion:
    def test_acumula_varias_llamadas(self):
        registrar_llamada("groq/openai/gpt-oss-20b", 100, 50, 1.0, sesion_id="s1")
        registrar_llamada("groq/openai/gpt-oss-20b", 200, 60, 2.0, sesion_id="s1")
        resumen = resumen_sesion("s1")
        assert resumen["llamadas"] == 2
        assert resumen["prompt_tokens"] == 300
        assert resumen["completion_tokens"] == 110
        assert resumen["latencia_s"] == pytest.approx(3.0)
        # 0.20 $/M * 300 + 0.80 $/M * 110
        assert resumen["coste"] == pytest.approx(0.20 * 300 / 1e6 + 0.80 * 110 / 1e6)

    def test_sesiones_independientes(self):
        registrar_llamada("groq/openai/gpt-oss-20b", 10, 5, 1.0, sesion_id="chat-1")
        registrar_llamada("groq/openai/gpt-oss-20b", 20, 8, 2.0, sesion_id="chat-2")
        assert resumen_sesion("chat-1")["prompt_tokens"] == 10
        assert resumen_sesion("chat-2")["prompt_tokens"] == 20

    def test_sesion_sin_llamadas_devuelve_ceros(self):
        resumen = resumen_sesion("no-existe")
        assert resumen["llamadas"] == 0
        assert resumen["coste"] == 0.0
        assert resumen["latencia_s"] == 0.0

    def test_conversacion_completa_da_resumen_consultable(self):
        """Una conversación de varios turnos deja un acumulado consultable:
        es la salida que se pega como evidencia de una conversación real."""
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = _respuesta_litellm(110, 35)
            _chat_litellm([{"role": "user", "content": "turno 1"}], LLMConfig(), sesion_id="conv")
            mock_comp.return_value = _respuesta_litellm(95, 28)
            _chat_litellm([{"role": "user", "content": "turno 2"}], LLMConfig(), sesion_id="conv")
            mock_comp.return_value = _respuesta_litellm(150, 60)
            _chat_litellm([{"role": "user", "content": "turno 3"}], LLMConfig(), sesion_id="conv")

        resumen = resumen_sesion("conv")
        assert resumen["llamadas"] == 3
        assert resumen["prompt_tokens"] == 355
        assert resumen["completion_tokens"] == 123
        assert resumen["latencia_s"] > 0
