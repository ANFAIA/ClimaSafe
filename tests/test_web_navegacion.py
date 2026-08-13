"""Tests del flujo de navegación de la web (WEB-007).

El chat deja de ser un panel dentro del paso 1 (con el mapa y el formulario
de predicción delante) y pasa a ser una vista `.step` dedicada, al mismo
nivel que step1, step2 y step-admin. El selector Individual / Grupo / Chat
sigue siendo la navegación: elegir Chat navega a la vista del chat.

Se testea el HTML estático (chat/static/index.html), igual que hace
test_web_rutinas.py con el escapado: la SPA es un único fichero y la
navegación vive en él.
"""

import pathlib

_INDEX = pathlib.Path(__file__).resolve().parents[1] / "chat" / "static" / "index.html"
_HTML = _INDEX.read_text(encoding="utf-8")


def _seccion(desde, hasta):
    return _HTML[_HTML.index(desde):_HTML.rindex(hasta)]


def _paso_chat():
    return _seccion('<div class="step" id="step-chat">', '<div class="step" id="step2">')


def _paso1():
    return _seccion('<div class="step visible" id="step1">', '<div class="step" id="step-chat">')


# ── El chat es una vista dedicada, no un panel dentro del paso 1 ──────────


def test_chat_es_un_paso_dedicado():
    assert '<div class="step" id="step-chat">' in _HTML
    # No queda el panel embebido del paso 1
    assert "step1-chat" not in _HTML


def test_el_chat_no_convive_con_mapa_ni_formulario():
    """Dentro de la vista chat solo está el asistente: sin mapa, sin selector
    de perfil, sin formulario de predicción."""
    chat = _paso_chat()
    for id_ajeno in ("id=\"map\"", "id=\"perfil-select\"", "id=\"provincia\"",
                     "id=\"step1-individual\"", "id=\"step1-grupo\""):
        assert id_ajeno not in chat, f"{id_ajeno} aparece en la vista chat"
    for id_chat in ("id=\"chat-box\"", "id=\"chat-opciones\"", "id=\"chat-input\""):
        assert id_chat in chat, f"falta {id_chat} en la vista chat"


def test_el_chat_esta_fuera_del_paso1():
    """El contenido del chat vive en #step-chat, no dentro de #step1."""
    paso1 = _paso1()
    for id_chat in ("id=\"chat-box\"", "id=\"chat-opciones\"", "id=\"chat-input\""):
        assert id_chat not in paso1, f"{id_chat} sigue dentro del paso 1"


# ── Navegación: el selector lleva a la vista chat y se vuelve ─────────────


def test_selector_tiene_chat_al_mismo_nivel_que_individual_y_grupo():
    paso1 = _paso1()
    for onclick in ("setModoPaso1('individual')", "setModoPaso1('grupo')",
                    "setModoPaso1('chat')"):
        assert onclick in paso1, f"falta el botón {onclick} en el selector"


def test_set_modo_chat_navega_a_la_vista_dedicada():
    fn = _seccion("function setModoPaso1(modo) {", "function volverDesdeChat()")
    assert "document.getElementById('step1').classList.remove('visible')" in fn
    assert "document.getElementById('step-chat').classList.add('visible')" in fn
    assert "if (!chatAbierto) abrirChat();" in fn
    # No vuelve a haber lógica del panel embebido
    assert "step1-chat" not in fn


def test_volver_desde_chat_restaura_el_paso1():
    fn = _seccion("function volverDesdeChat()", "function setSubModoPaso2")
    assert "document.getElementById('step-chat').classList.remove('visible')" in fn
    assert "document.getElementById('step1').classList.add('visible')" in fn
    assert "setModoPaso1(currentMode || 'individual')" in fn


def test_la_vista_chat_tiene_boton_para_volver():
    chat = _paso_chat()
    assert "volverDesdeChat()" in chat


# ── El chat mantiene su funcionalidad (incluido ver detalle) ──────────────


def test_ver_detalle_completo_sigue_presente():
    # El chip final del chat sigue llamando al mismo flujo: resultados + paso 2.
    assert "Ver detalle completo" in _HTML
    fn = _seccion("cont.appendChild(chip('Ver detalle completo'",
                  "cont.appendChild(chip('Nueva consulta'")
    assert "mostrarResultados(data.resultado)" in fn
    assert "irPaso2();" in fn


def test_ir_paso2_oculta_la_vista_chat():
    """Al ver el detalle, la vista chat se quita del medio (no se queda
    delante de los resultados)."""
    fn = _seccion("function irPaso2() {", "function irPaso1()")
    assert "document.getElementById('step-chat').classList.remove('visible')" in fn


# ── El resto de vistas siguen intactas ─────────────────────────────────────


def test_las_demas_vistas_siguen_existiendo():
    for marcador in ('<div class="step visible" id="step1">',
                     '<div class="step" id="step2">',
                     '<div class="step" id="step-admin">',
                     '<div id="results">',
                     'id="map"',
                     'id="grupo-form"'):
        assert marcador in _HTML, f"falta {marcador}"


def test_domcontentloaded_sigue_arrancando_individual():
    fn = _seccion("document.addEventListener('DOMContentLoaded'", "</script>")
    assert "setModoPaso1('individual')" in fn
