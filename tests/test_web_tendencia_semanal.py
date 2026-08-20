"""Tests de la tendencia semanal en pantalla (FORECAST-004).

La gráfica semanal vive en la SPA de un fichero (chat/static/index.html), igual
que el resto del flujo: se testea el HTML/JS estático (patrón de
test_web_navegacion.py) y el contrato del endpoint en test_web_predict.py.
FORECAST-004 pide que la banda de confianza conformal se explique en lenguaje
llano, que se muestre `banda_origen` cuando exista y que un horizonte corto se
avise explícitamente sin extrapolar en silencio.
"""

import pathlib

_INDEX = pathlib.Path(__file__).resolve().parents[1] / "chat" / "static" / "index.html"
_HTML = _INDEX.read_text(encoding="utf-8")

_LEGENDA_LLANA = "franja donde puede caer el riesgo real"


def _seccion(desde, hasta):
    return _HTML[_HTML.index(desde):_HTML.rindex(hasta)]


# ── La banda se explica en pantalla (criterio 1) ───────────────────────────


def test_existe_elemento_de_explicacion_de_la_banda():
    assert 'id="weekly-banda-explica"' in _HTML


def test_la_explicacion_dice_que_es_el_intervalo_conformal():
    fn = _seccion("async function verTendenciaSemanal()", "function _pintarSemanal(")
    assert "La franja azul es la banda de confianza" in fn
    assert "intervalo de la" in fn
    assert "predicción conformal (α=0.1)" in fn
    assert "puede caer el riesgo real de cada día" in fn


def test_la_explicacion_muestra_banda_origen_cuando_existe():
    fn = _seccion("async function verTendenciaSemanal()", "function _pintarSemanal(")
    assert "data.banda_origen ?" in fn
    assert "Origen: ' + data.banda_origen" in fn
    # si no existe, no se inventa: la frase llana se queda sola
    assert ": '');" in fn


# ── Horizonte corto: se avisa sin extrapolar (criterio 2) ──────────────────


def test_horizonte_corto_avisa_hasta_donde_llega_y_que_no_se_dibuja():
    fn = _seccion("async function verTendenciaSemanal()", "function _pintarSemanal(")
    assert "el forecast meteorológico llega hasta ' + data.forecast_hasta" in fn
    assert "El resto de días no se dibujan" in fn
    assert "no se extrapola en silencio" in fn
    # la serie que se pinta es solo la que devuelve el backend: no se rellenan
    # días inventados para completar la semana
    assert "_pintarSemanal(data);" in fn
    pintar = _seccion("function _pintarSemanal(data)", "function mostrarOverride")
    assert "const dias = data.dias;" in pintar
    assert "labels = dias.map" in pintar


# ── Leyenda en lenguaje llano (criterio 3) ─────────────────────────────────


def test_la_leyenda_usa_lenguaje_llano():
    pintar = _seccion("function _pintarSemanal(data)", "function mostrarOverride")
    assert "banda conformal" not in pintar
    assert "franja donde puede caer el riesgo real" in pintar


def test_la_leyenda_sigue_explicando_la_confianza_por_dia():
    pintar = _seccion("function _pintarSemanal(data)", "function mostrarOverride")
    assert "confianza alta" in pintar
    assert "confianza media" in pintar
    assert "confianza baja" in pintar
