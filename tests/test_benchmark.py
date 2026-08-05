"""Tests del módulo climasafeai.llm.benchmark — benchmark sin juez LLM (LLM-003).

Son funciones puras y deterministas: no tocan Ollama ni litellm, así que se
importan y se prueban sin red.
"""

from __future__ import annotations

from climasafeai.llm.benchmark import (
    _formato_ok,
    _numeros,
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
