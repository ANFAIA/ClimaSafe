"""Tests del módulo climasafeai.llm.benchmark — benchmark sin juez LLM (LLM-003).

Son funciones puras y deterministas: no tocan Ollama ni litellm, así que se
importan y se prueban sin red. Las pruebas de LLM-019 (proveedores gratuitos)
mockean litellm.completion igual que hace tests/test_costes.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from climasafeai.llm.benchmark import (
    _formato_ok,
    _numeros,
    evaluar_modelo,
    evaluar_respuesta,
)


class TestFormatoOk:
    """El bot espera tres líneas: RIESGO, índice personalizado y factor total."""

    def test_acepta_formato_completo(self):
        texto = (
            "RIESGO: PRECAUCION\n"
            "Índice personalizado: 0.24\n"
            "Índice poblacional: 0.10\n"
            "Factor total aplicado: ×0.24\n"
            "Factores activados:\n- humedad"
        )
        assert _formato_ok(texto) is True

    def test_rechaza_texto_libre(self):
        assert _formato_ok("Hoy hace mucho calor, ten cuidado y bebe agua") is False


class TestNumeros:
    """La normalización es para comparar: '0,240' y '0.24' son el mismo número."""

    def test_coma_y_punto_normalizan_a_lo_mismo(self):
        assert _numeros("0,240 y 0.24") == {"0.24"}

    def test_conjunto_de_varios(self):
        assert _numeros("32 grados, 0,5 y 1.75") == {"32", "0.5", "1.75"}


class TestEvaluarRespuesta:
    """Compara una respuesta con su referencia: todo determinista."""

    def _ejemplo(self) -> dict:
        return {
            "instruction": "Predice el riesgo térmico",
            "input": "Temperatura 32 grados. Índice personalizado: 0.24.",
            "output": (
                "RIESGO: SEGURO\n"
                "Índice personalizado: 0.24\n"
                "Factor total aplicado: ×0.24"
            ),
        }

    def test_cifra_inventada_se_detecta(self):
        res = evaluar_respuesta(
            "RIESGO: SEGURO\n"
            "Índice personalizado: 0.24\n"
            "Factor total aplicado: ×0.24\n"
            "Humedad relativa: 85%",
            self._ejemplo(),
        )
        assert res["n_inventadas"] > 0
        assert "85" in res["inventadas"]

    def test_respuesta_correcta_acierta_clase(self):
        res = evaluar_respuesta(
            "RIESGO: SEGURO\n"
            "Índice personalizado: 0.24\n"
            "Factor total aplicado: ×0.24",
            self._ejemplo(),
        )
        assert res["clase_ok"] is True
        assert res["n_inventadas"] == 0
        assert res["formato_ok"] is True
        assert res["err_indice"] == 0.0


# ── LLM-019: proveedores gratuitos (Groq/Gemini/OpenRouter) con coste ──


def _respuesta_litellm(texto: str, prompt_tokens: int, completion_tokens: int) -> MagicMock:
    """Respuesta fake con la forma del objeto que devuelve litellm.completion."""
    resp = MagicMock()
    resp.usage.prompt_tokens = prompt_tokens
    resp.usage.completion_tokens = completion_tokens
    resp.choices[0].message.content = texto
    resp.choices[0].finish_reason = "stop"
    return resp


RESPUESTA_OK = (
    "RIESGO: SEGURO\n"
    "Índice personalizado: 0.24\n"
    "Índice poblacional: 0.10\n"
    "Factor total aplicado: ×0.24\n"
    "\n"
    "Factores activados:\n- ninguno\n"
    "\n"
    "Recomendaciones:\n- hidratarse"
)


class TestOpenRouterSinClave:
    """Sin OPENROUTER_API_KEY el modelo se reporta 'sin clave', nunca inventado."""

    def _ejemplos(self) -> list[dict]:
        return [{
            "instruction": "Predice",
            "input": "Temperatura 32 grados. Índice personalizado: 0.24.",
            "output": RESPUESTA_OK,
        }]

    def test_sin_clave_devuelve_error_claro_sin_llamar(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with patch("litellm.completion") as mock_comp:
            informe = evaluar_modelo(
                "openrouter/deepseek/deepseek-chat-v3-0324:free", self._ejemplos()
            )
        mock_comp.assert_not_called()
        assert informe["error"].startswith("sin clave")
        assert "OPENROUTER_API_KEY" in informe["error"]

    def test_con_clave_intenta_llamar(self, monkeypatch):
        """La vía openrouter/ es un camino de código listo: con clave evalúa normal."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key-para-test")
        with patch(
            "litellm.completion",
            return_value=_respuesta_litellm(RESPUESTA_OK, 100, 50),
        ):
            informe = evaluar_modelo(
                "openrouter/deepseek/deepseek-chat-v3-0324:free", self._ejemplos(), verbose=False
            )
        assert "error" not in informe
        assert informe["clase_acc"] == 1.0

    def test_no_openrouter_pasa_sin_comprobar_clave(self, monkeypatch):
        """La puerta solo aplica a prefijos openrouter/: groq no se ve afectada."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with patch(
            "litellm.completion",
            return_value=_respuesta_litellm(RESPUESTA_OK, 100, 50),
        ):
            informe = evaluar_modelo("groq/openai/gpt-oss-20b", self._ejemplos(), verbose=False)
        assert "error" not in informe


class TestCostePorPeticion:
    """El coste por petición sale de la tabla PRECIOS_MODELOS (ARNES-004),
    acumulado en una sesión dedicada por modelo para no contaminar 'default'."""

    def _ejemplos(self) -> list[dict]:
        return [
            {
                "instruction": "Predice",
                "input": "Temperatura 32 grados. Índice personalizado: 0.24.",
                "output": RESPUESTA_OK,
            },
        ] * 2

    def test_coste_remoto_desde_tabla_de_precios(self, monkeypatch):
        from climasafeai.llm.costes import _ACUMULADOS

        _ACUMULADOS.clear()
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with patch(
            "litellm.completion",
            return_value=_respuesta_litellm(RESPUESTA_OK, 4_000, 500),
        ):
            informe = evaluar_modelo("groq/openai/gpt-oss-20b", self._ejemplos(), verbose=False)
        # gpt-oss-20b: 0.20 $/M prompt + 0.80 $/M completion
        # → (4000×0.20 + 500×0.80)/1e6 = $0.0012 por petición.
        assert informe["coste_por_peticion"] == pytest.approx(0.0012)
        assert informe["tok_por_peticion"] == 4_500
        _ACUMULADOS.clear()

    def test_local_y_free_cuestan_cero(self, monkeypatch):
        from climasafeai.llm.costes import _ACUMULADOS

        _ACUMULADOS.clear()
        with patch(
            "litellm.completion",
            return_value=_respuesta_litellm(RESPUESTA_OK, 500, 100),
        ):
            informe = evaluar_modelo("ollama/qwen3:1.7b", self._ejemplos(), verbose=False)
        assert informe["coste_por_peticion"] == 0.0
        assert informe["tok_por_peticion"] == 600
        _ACUMULADOS.clear()

    def test_sesion_default_no_se_contamina(self, monkeypatch):
        from climasafeai.llm.costes import _ACUMULADOS, resumen_sesion

        _ACUMULADOS.clear()
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with patch(
            "litellm.completion",
            return_value=_respuesta_litellm(RESPUESTA_OK, 100, 50),
        ):
            evaluar_modelo("groq/openai/gpt-oss-20b", self._ejemplos(), verbose=False)
        assert resumen_sesion()["llamadas"] == 0
        _ACUMULADOS.clear()
