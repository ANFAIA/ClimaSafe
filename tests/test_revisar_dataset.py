"""Tests de climasafeai.llm.revisar_dataset (LLM-015).

El QC del dataset sintético debe señalar, en un mini-dataset, cada problema que
un humano revisaría: pares sin el parte meteorológico, perfiles imposibles,
duplicados casi idénticos, desequilibrio de clases y respuestas cuya clase no
coincide con el pipeline.

El detector de clase es el único que toca `predict_ensemble`; en los tests se
mockea (los modelos entrenados no son un requisito de los tests del QC).
"""

from __future__ import annotations

import json

import pytest

import climasafeai.llm.revisar_dataset as qc


# ── Fixtures ────────────────────────────────────────────────────────────────


def _ejemplo(
    edad: int = 45,
    sexo: str = "hombre",
    comorbilidades: str = "",
    parte: str = "Tiempo en esa franja: 25.0 °C de media, máx 25.0 °C, "
                 "humedad 50 %, viento 10.0 km/h, UV 3.0",
    clase: str = "SEGURO",
) -> dict:
    extra = f" Comorbilidades: {comorbilidades}." if comorbilidades else ""
    return {
        "instruction": "Predice el riesgo térmico para este perfil y da recomendaciones.",
        "input": (
            f"Edad: {edad}. Sexo: {sexo}. Grasa corporal: 25%. Aclimatado: sí. "
            f"Fototipo: III. Actividad: reposo. Duración: 1.0h. Desde las: 10:00. "
            f"Ubicación: Bilbao.{extra} {parte}"
        ),
        "output": f"RIESGO: {clase}\n\nÍndice personalizado: 0.10",
    }


def _mini_dataset() -> list[dict]:
    """12 pares con los 5 problemas, uno de cada tipo + un par sano."""
    return [
        _ejemplo(),  # 0: sano
        _ejemplo(parte=""),  # 1: sin 'Tiempo en esa franja'
        _ejemplo(parte="Tiempo en esa franja: 25.0 °C de media, "
                       "humedad 50 %, viento 10.0 km/h"),  # 2: parte sin máx/UV
        _ejemplo(edad=200, sexo="alien", comorbilidades="cancer"),  # 3: imposible
        _ejemplo(parte="Tiempo en esa franja: 41.5 °C de media, máx 41.5 °C, "
                       "humedad 30 %, viento 13.0 km/h, UV 6.0"),  # 4: duplicado de 5
        _ejemplo(parte="Tiempo en esa franja: 42.1 °C de media, máx 42.1 °C, "
                       "humedad 30 %, viento 13.0 km/h, UV 6.0"),  # 5: duplicado de 4
        _ejemplo(clase="SEGURO"),  # 6: la respuesta afirma SEGURO...
        _ejemplo(clase="SEGURO"),
        _ejemplo(clase="SEGURO"),
        _ejemplo(clase="SEGURO"),
        _ejemplo(clase="SEGURO"),
        _ejemplo(clase="SEGURO"),
        # PELIGRO 0/12 → desequilibrio (< 10 %)
    ]


def _escribir(tmp_path, ejemplos: list[dict]) -> str:
    ruta = tmp_path / "train.jsonl"
    with open(ruta, "w") as f:
        for e in ejemplos:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    return str(ruta)


# ── Detectores puros ────────────────────────────────────────────────────────


def test_sin_marca():
    ejemplos = [_ejemplo(), _ejemplo(parte="")]
    assert qc.detectar_sin_marca(ejemplos) == [1]


def test_parte_incompleto():
    ejemplos = [_ejemplo(), _ejemplo(parte="Tiempo en esa franja: 25.0 °C de media, "
                                            "humedad 50 %, viento 10.0 km/h")]
    assert qc.detectar_parte_incompleto(ejemplos) == [1]


def test_parte_completo_con_misma_max_media_no_marca_incompleto():
    # La máxima SIEMPRE va aunque coincida con la media (LLM-004); si falta, el
    # QC lo señala aunque la media esté.
    ejemplos = [_ejemplo(parte="Tiempo en esa franja: 25.0 °C de media, "
                               "humedad 50 %, viento 10.0 km/h, UV 3.0")]
    assert qc.detectar_parte_incompleto(ejemplos) == [0]


def test_perfil_imposible():
    ejemplos = [_ejemplo(), _ejemplo(edad=200, sexo="alien", comorbilidades="cancer")]
    hallazgos = qc.detectar_perfiles_imposibles(ejemplos)
    assert len(hallazgos) == 1
    assert hallazgos[0]["indice"] == 1
    problemas = " ".join(hallazgos[0]["problemas"])
    assert "edad 200" in problemas
    assert "sexo 'alien'" in problemas
    assert "cancer" in problemas


def test_perfil_valido_no_marca_imposible():
    # Las claves que el pipeline sí reconoce (ocupaciones de personalizacion.py,
    # comorbilidades de factores_riesgo.json) no son un perfil imposible.
    e = _ejemplo(comorbilidades="diabetes,cardiovascular")
    e["input"] = e["input"].replace(
        " Ubicación: Bilbao.", " Ubicación: Bilbao. Ocupación: reparto."
    )
    assert qc.detectar_perfiles_imposibles([e]) == []


def test_duplicados_casi_identicos():
    # Los tres comparten perfil (los números del parte están anonimizados): el
    # par 1-2 difiere solo en la cifra de temperatura, y 0 también comparte el
    # perfil. El detector los señala todos.
    ejemplos = [
        _ejemplo(),
        _ejemplo(parte="Tiempo en esa franja: 41.5 °C de media, máx 41.5 °C, "
                       "humedad 30 %, viento 13.0 km/h, UV 6.0"),
        _ejemplo(parte="Tiempo en esa franja: 42.1 °C de media, máx 42.1 °C, "
                       "humedad 30 %, viento 13.0 km/h, UV 6.0"),
    ]
    pares = qc.detectar_duplicados(ejemplos)
    assert len(pares) == 3
    par_12 = [p for p in pares if {p["i"], p["j"]} == {1, 2}]
    assert len(par_12) == 1
    assert par_12[0]["similitud"] == 1.0


def test_duplicados_distinto_perfil_no_cuenta():
    # Con la normalización por estructura los números no distinguen (edad 45 vs
    # 85 es 'Edad: #'), pero un campo extra sí: comorbilidades + ocupación
    # añaden tokens que bajan la similitud por debajo del umbral.
    distinto = _ejemplo(comorbilidades="diabetes")
    distinto["input"] = distinto["input"].replace(
        " Ubicación: Bilbao.", " Ubicación: Bilbao. Ocupación: reparto."
    )
    assert qc.detectar_duplicados([_ejemplo(), distinto]) == []


def test_desequilibrio_bajo_umbral():
    ejemplos = [_ejemplo(clase="SEGURO") for _ in range(8)]
    ejemplos += [_ejemplo(clase="PRECAUCION") for _ in range(3)]
    ejemplos += [_ejemplo(clase="PELIGRO")]  # 1/12 = 8.3 % < 10 %
    res = qc.detectar_desequilibrio(ejemplos)
    assert res["desequilibrio"] is True
    assert res["clases_bajo_umbral"][0]["clase"] == "PELIGRO"


def test_desequilibrio_equilibrado():
    ejemplos = [_ejemplo(clase="SEGURO") for _ in range(4)]
    ejemplos += [_ejemplo(clase="PRECAUCION") for _ in range(3)]
    ejemplos += [_ejemplo(clase="PELIGRO") for _ in range(3)]
    assert qc.detectar_desequilibrio(ejemplos)["desequilibrio"] is False


# ── Detector de clase (pipeline mockeado) ───────────────────────────────────


def _mock_predict_ensemble(monkeypatch, clase: str):
    def _fake(lat=None, lon=None, provincia="Madrid", perfil=None,
              target_date=None, weather=None, resolucion=60):
        return {"clase_final_label": clase}
    monkeypatch.setattr("climasafeai.models.ensemble.predict_ensemble", _fake)


def test_clase_incoherente_es_hallazgo(monkeypatch):
    _mock_predict_ensemble(monkeypatch, "PELIGRO")
    ejemplo = _ejemplo(clase="SEGURO")
    hallazgo = qc._verificar_par(ejemplo)
    assert hallazgo is not None
    assert hallazgo["verificable"] is True
    assert hallazgo["clase_afirmada"] == "SEGURO"
    assert hallazgo["clase_pipeline"] == "PELIGRO"
    assert hallazgo["gravedad"] == "critica"


def test_clase_coherente_no_es_hallazgo(monkeypatch):
    _mock_predict_ensemble(monkeypatch, "SEGURO")
    assert qc._verificar_par(_ejemplo(clase="SEGURO")) is None


def test_parte_incompleto_no_verificable_por_clase(monkeypatch):
    _mock_predict_ensemble(monkeypatch, "PELIGRO")
    ejemplo = _ejemplo(parte="Tiempo en esa franja: 25.0 °C de media, "
                             "humedad 50 %, viento 10.0 km/h")
    hallazgo = qc._verificar_par(ejemplo)
    assert hallazgo is not None
    assert hallazgo["verificable"] is False
    assert "parte incompleto" in hallazgo["razon"]


def test_verificar_clase_modo_degradado(monkeypatch):
    monkeypatch.setattr(qc, "_pipeline_disponible", lambda: False)
    res = qc.verificar_clase([_ejemplo()] * 5, muestra=5)
    assert res["disponible"] is False
    assert res["no_verificables"] == 5
    assert "no verificables" in res["aviso"]


def test_verificar_clase_muestra_cero_no_ejecuta_pipeline(monkeypatch):
    res = qc.verificar_clase([_ejemplo()] * 5, muestra=0)
    assert res["n_verificados"] == 0


# ── Integración: el script señala todos los problemas ───────────────────────


def test_revisar_integracion(tmp_path):
    ruta = _escribir(tmp_path, _mini_dataset())
    val_ruta = tmp_path / "val.jsonl"
    with open(val_ruta, "w") as f:
        f.write(json.dumps(_ejemplo(), ensure_ascii=False) + "\n")

    resultado = qc.revisar(ruta, str(val_ruta), muestra=0)
    train = resultado["por_fichero"]["train"]

    assert train["sin_marca"]["n"] == 1
    assert train["parte_incompleto"]["n"] == 1
    assert train["perfiles_imposibles"]["n"] == 1
    # Los pares 4 y 5 (mismo perfil, 41.5 vs 42.1 °C) son el duplicado buscado;
    # con la normalización por estructura también comparten perfil con otros.
    assert train["duplicados"]["n_pares"] >= 1
    pares_45 = [p for p in train["duplicados"]["pares"]
                if {p["i"], p["j"]} == {4, 5}]
    assert len(pares_45) == 1
    assert train["desequilibrio"]["desequilibrio"] is True
    # El detector de clase con muestra 0 no toca el pipeline.
    assert train["clase_pipeline"]["n_verificados"] == 0

    resumen = resultado["resumen"]["train"]
    assert resumen["sin_marca"] == 1
    assert resumen["parte_incompleto"] == 1
    assert resumen["perfiles_imposibles"] == 1
    assert resumen["duplicados"] >= 1
    assert resumen["desequilibrio"] is True


def test_main_escribe_informe_json(tmp_path, monkeypatch, capsys):
    ruta = _escribir(tmp_path, _mini_dataset())
    val_ruta = tmp_path / "val.jsonl"
    with open(val_ruta, "w") as f:
        f.write(json.dumps(_ejemplo(), ensure_ascii=False) + "\n")
    out = tmp_path / "informe.json"
    monkeypatch.setattr(
        "sys.argv",
        ["revisar_dataset.py", "--train", ruta, "--val", str(val_ruta),
         "--out", str(out), "--muestra", "0"],
    )
    qc.main()
    informe = json.loads(out.read_text())
    assert informe["por_fichero"]["train"]["duplicados"]["n_pares"] >= 1
    salida = capsys.readouterr().out
    assert "perfiles imposibles" in salida.lower()
