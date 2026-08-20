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
    TOPE_TOKENS_POR_PETICION,
    _ACUMULADOS,
    PresupuestoExcedidoError,
    comprobar_presupuesto,
    coste_llamada,
    es_local,
    precios_de,
    registrar_llamada,
    resumen_sesion,
    tope_tokens_peticion,
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


# ── 5. Tope de presupuesto de tokens por petición (ARNES-010) ──────────


class TestTopePresupuesto:
    def test_tope_por_defecto_por_encima_de_la_referencia(self):
        """El tope por defecto (10.000) deja pasar la referencia que motivó la
        feature — spacebot gasta ~7.900 tokens por mensaje — y corta los picos."""
        assert TOPE_TOKENS_POR_PETICION == 10_000
        assert TOPE_TOKENS_POR_PETICION > 7_900

    def test_tope_se_ajusta_por_env(self, monkeypatch):
        """El tope es configurable sin tocar código: CLIMASAFE_MAX_TOKENS_PETICION."""
        monkeypatch.setenv("CLIMASAFE_MAX_TOKENS_PETICION", "500")
        assert tope_tokens_peticion() == 500

    def test_tope_invalido_cae_al_default(self, monkeypatch):
        """Un valor no entero no puede romper el arranque ni dejar el tope en 0."""
        monkeypatch.setenv("CLIMASAFE_MAX_TOKENS_PETICION", "abc")
        assert tope_tokens_peticion() == TOPE_TOKENS_POR_PETICION
        monkeypatch.setenv("CLIMASAFE_MAX_TOKENS_PETICION", "0")
        assert tope_tokens_peticion() == TOPE_TOKENS_POR_PETICION

    def test_comprobar_presupuesto_lanza_con_etiqueta_clara(self):
        with pytest.raises(PresupuestoExcedidoError, match="payload estimado"):
            comprobar_presupuesto(101, etiqueta="payload estimado", tope=100)
        # Bajo el tope no lanza.
        comprobar_presupuesto(99, etiqueta="payload estimado", tope=100)

    def test_peticion_bajo_tope_pasa(self, monkeypatch):
        monkeypatch.setenv("CLIMASAFE_MAX_TOKENS_PETICION", "100000")
        with patch("litellm.completion", return_value=_respuesta_litellm(10, 5)) as mock_comp:
            out = _chat_litellm([{"role": "user", "content": "hola"}], LLMConfig())
        assert out == "respuesta de prueba"
        assert mock_comp.call_count == 1

    def test_peticion_que_supera_tope_se_corta_sin_llamar(self, monkeypatch):
        """Tope 50, payload estimado ~100: no se llama al proveedor ni se gasta."""
        monkeypatch.setenv("CLIMASAFE_MAX_TOKENS_PETICION", "50")
        with patch("litellm.completion") as mock_comp:
            with pytest.raises(PresupuestoExcedidoError, match="Presupuesto de tokens superado"):
                _chat_litellm(
                    [{"role": "user", "content": "x" * 400}],
                    LLMConfig(),
                )
        mock_comp.assert_not_called()

    def test_traza_clara_de_la_peticion_cortada(self, monkeypatch, caplog):
        """El corte no es silencioso: la traza dice la cifra, el tope y la petición."""
        monkeypatch.setenv("CLIMASAFE_MAX_TOKENS_PETICION", "50")
        with patch("litellm.completion") as mock_comp:
            with pytest.raises(PresupuestoExcedidoError):
                _chat_litellm(
                    [{"role": "user", "content": "x" * 400}],
                    LLMConfig(model="groq/openai/gpt-oss-20b"),
                    sesion_id="chat-99",
                )
        trazas = [r.message for r in caplog.records if "cortada por presupuesto" in r.message]
        assert len(trazas) == 1
        assert "payload estimado" in trazas[0]
        assert "50" in trazas[0]
        assert "groq/openai/gpt-oss-20b" in trazas[0]
        assert "chat-99" in trazas[0]

    def test_corte_por_usage_real_tras_la_llamada(self, monkeypatch):
        """La estimación puede subestimar: el usage real (prompt+completion)
        también cuenta y corta aunque el payload estimado pasara."""
        monkeypatch.setenv("CLIMASAFE_MAX_TOKENS_PETICION", "100")
        with patch("litellm.completion", return_value=_respuesta_litellm(80, 40)):
            with pytest.raises(PresupuestoExcedidoError, match="usage real"):
                _chat_litellm([{"role": "user", "content": "hola"}], LLMConfig())

    def test_ask_raw_devuelve_error_claro_por_presupuesto(self, monkeypatch):
        """La ruta pública devuelve el motivo en `error`, no un fallo genérico."""
        monkeypatch.setenv("CLIMASAFE_MAX_TOKENS_PETICION", "50")
        with patch("litellm.completion") as mock_comp:
            res = ask_raw("x" * 400, config=LLMConfig())
        mock_comp.assert_not_called()
        assert res["answer"] is None
        assert "Presupuesto de tokens superado" in res["error"]

    def test_ask_con_perfil_captura_el_corte_y_degrada(self, monkeypatch, caplog):
        """ask_con_perfil devuelve None (el bot degrada a la plantilla) y la
        traza del corte queda en el log, no en silencio."""
        with patch(
            "climasafeai.llm.rag_qwen._chat_litellm",
            side_effect=PresupuestoExcedidoError("tope 50 superado"),
        ):
            from climasafeai.llm.rag_qwen import ask_con_perfil

            res = ask_con_perfil({}, {"weather": {}})
        assert res is None
        assert any("cortado por presupuesto" in r.message for r in caplog.records)
