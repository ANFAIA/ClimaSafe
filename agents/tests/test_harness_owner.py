"""
ARNES-013 — El candado de `harness start` es por dueño, no global.

Dos asistentes en paralelo (opencode y Claude Code) tenían que poder abrir una
feature cada uno; con el candado global el segundo no podía trabajar. La regla
nueva: un dueño, una feature abierta. Y `progress/current.md` pasa a ser estado
derivado del backlog, para que cerrar una feature no borre el current del otro.
"""

from __future__ import annotations

import json
from pathlib import Path

from agents.agents.harness_agent import HarnessAgent
from agents.config import ProjectConfig
from agents.context import SharedContext


def _write_backlog(root: Path, *extra: dict) -> None:
    features = [
        {
            "id": "A-001",
            "title": "Primera",
            "description": "la primera",
            "acceptance_criteria": ["que funcione"],
            "status": "pending",
        },
        {
            "id": "A-002",
            "title": "Segunda",
            "description": "la segunda",
            "acceptance_criteria": ["que también funcione"],
            "status": "pending",
        },
        {
            "id": "A-003",
            "title": "Tercera",
            "description": "la tercera",
            "acceptance_criteria": ["y la tercera"],
            "status": "pending",
        },
    ]
    features.extend(extra)
    (root / "featureslist.json").write_text(
        json.dumps({"features": features}, ensure_ascii=False), encoding="utf-8"
    )


def _agent(root: Path) -> HarnessAgent:
    return HarnessAgent(context=SharedContext(root=root, config=ProjectConfig(project_slug="mi_paquete")))


def _feature(root: Path, fid: str) -> dict:
    doc = json.loads((root / "featureslist.json").read_text(encoding="utf-8"))
    return next(f for f in doc["features"] if f["id"] == fid)


# -- criterio 2: dueños distintos conviven -----------------------------------
def test_dos_duenos_distintos_abren_a_la_vez(project_root):
    _write_backlog(project_root)
    agent = _agent(project_root)

    primero = agent.start(id="A-001", owner="opencode")
    segundo = agent.start(id="A-002", owner="claude")

    assert primero.success, primero.message
    assert segundo.success, segundo.message
    assert _feature(project_root, "A-001")["status"] == "in_progress"
    assert _feature(project_root, "A-002")["status"] == "in_progress"
    assert _feature(project_root, "A-001")["owner"] == "opencode"
    assert _feature(project_root, "A-002")["owner"] == "claude"


# -- criterio 3: el mismo dueño sigue rechazado ------------------------------
def test_mismo_dueno_no_puede_abrir_dos(project_root):
    _write_backlog(project_root)
    agent = _agent(project_root)
    assert agent.start(id="A-001", owner="claude").success

    result = agent.start(id="A-002", owner="claude")

    assert not result.success
    assert "A-001" in result.message  # nombra la que ya está abierta
    assert "claude" in result.message
    assert _feature(project_root, "A-002")["status"] == "pending"


def test_el_dueno_se_normaliza_antes_de_comparar(project_root):
    """'Claude' y '  claude ' son el mismo dueño: no se cuela una segunda tarea."""
    _write_backlog(project_root)
    agent = _agent(project_root)
    assert agent.start(id="A-001", owner="Claude").success

    assert not agent.start(id="A-002", owner="  claude ").success


# -- criterio 6: compatibilidad hacia atrás ----------------------------------
def test_sin_owner_sigue_admitiendo_una_sola(project_root):
    _write_backlog(project_root)
    agent = _agent(project_root)
    assert agent.start(id="A-001").success

    result = agent.start(id="A-002")

    assert not result.success
    assert "A-001" in result.message
    # sin --owner la feature no gana el campo: el backlog es el de siempre
    assert "owner" not in _feature(project_root, "A-001")


def test_sin_owner_current_conserva_el_formato_de_hoy(project_root):
    _write_backlog(project_root)
    assert _agent(project_root).start(id="A-001").success

    current = (project_root / "progress" / "current.md").read_text(encoding="utf-8")
    assert current.startswith("# Tarea actual")
    assert "**Feature:** A-001" in current
    assert "**Estado:** in_progress" in current
    assert "**Responsable:** implementer" in current
    assert "- [ ] que funcione" in current


def test_features_legadas_sin_owner_no_rompen(project_root):
    """Una feature abierta antes de ARNES-013 (sin campo owner) sigue siendo el legado."""
    _write_backlog(project_root, {
        "id": "LEGACY-001",
        "title": "Abierta antes",
        "description": "sin campo owner",
        "acceptance_criteria": ["nada"],
        "status": "in_progress",
        "started": "2026-08-01",
    })
    agent = _agent(project_root)

    # otro dueño puede abrir la suya al lado de la legada...
    assert agent.start(id="A-001", owner="claude").success
    # ...pero quien no se identifica choca con ella, como antes
    assert not agent.start(id="A-002").success


def test_reabrir_la_misma_feature_sigue_permitido(project_root):
    _write_backlog(project_root)
    agent = _agent(project_root)
    assert agent.start(id="A-001", owner="claude").success

    assert agent.start(id="A-001", owner="claude").success
    assert agent.start(id="A-001").success  # y sin dueño tampoco se autobloquea


# -- criterio 4: current.md derivado -----------------------------------------
def test_con_dos_en_curso_current_es_un_indice(project_root):
    _write_backlog(project_root)
    agent = _agent(project_root)
    agent.start(id="A-001", owner="opencode")
    agent.start(id="A-002", owner="claude")

    current = (project_root / "progress" / "current.md").read_text(encoding="utf-8")
    assert current.startswith("# Tareas actuales")
    assert "A-001" in current and "A-002" in current
    assert "progress/current-opencode.md" in current
    assert "progress/current-claude.md" in current

    # y cada dueño tiene su fichero de detalle con la plantilla completa
    for owner, fid in (("opencode", "A-001"), ("claude", "A-002")):
        detalle = (project_root / "progress" / f"current-{owner}.md").read_text(encoding="utf-8")
        assert f"**Feature:** {fid}" in detalle
        assert f"**Responsable:** {owner}" in detalle


def test_en_el_indice_tambien_tiene_ficha_la_feature_sin_dueno(project_root):
    """Una feature legada (sin owner) no puede quedarse sin detalle al activarse el índice."""
    _write_backlog(project_root, {
        "id": "DATA-004",
        "title": "Abierta antes",
        "description": "la que abrió el otro asistente",
        "acceptance_criteria": ["que no se pierda su objetivo"],
        "status": "in_progress",
        "started": "2026-08-07",
    })
    assert _agent(project_root).start(id="A-001", owner="claude").success

    current = (project_root / "progress" / "current.md").read_text(encoding="utf-8")
    assert current.startswith("# Tareas actuales")
    assert "progress/current-claude.md" in current
    assert "progress/current-data-004.md" in current
    # ninguna fila del índice se queda sin fichero de detalle
    filas = [ln for ln in current.splitlines() if ln.startswith("| ") and "|---" not in ln]
    assert all("| — |" not in fila for fila in filas[1:]), filas
    # y la columna Dueño sigue diciendo la verdad
    assert "_(sin dueño)_" in current


def test_la_ficha_de_la_feature_sin_dueno_conserva_objetivo_y_criterios(project_root):
    _write_backlog(project_root, {
        "id": "DATA-004",
        "title": "Abierta antes",
        "description": "la que abrió el otro asistente",
        "acceptance_criteria": ["que no se pierda su objetivo", "ni sus criterios"],
        "status": "in_progress",
        "started": "2026-08-07",
    })
    _agent(project_root).start(id="A-001", owner="claude")

    ficha = (project_root / "progress" / "current-data-004.md").read_text(encoding="utf-8")
    assert "**Feature:** DATA-004" in ficha
    assert "**Iniciada:** 2026-08-07" in ficha
    assert "la que abrió el otro asistente" in ficha
    assert "- [ ] que no se pierda su objetivo" in ficha
    assert "- [ ] ni sus criterios" in ficha


def test_cerrar_una_no_borra_la_ficha_de_la_otra_tenga_dueno_o_no(project_root):
    _write_backlog(project_root, {
        "id": "DATA-004",
        "title": "Abierta antes",
        "description": "la que abrió el otro asistente",
        "acceptance_criteria": ["que no se pierda"],
        "status": "in_progress",
        "started": "2026-08-07",
    })
    progress = project_root / "progress"
    agent = _agent(project_root)
    agent.start(id="A-001", owner="claude")
    assert (progress / "current-data-004.md").exists()

    # bloquear la que tiene dueño no toca la ficha de la que no lo tiene...
    assert agent.block(id="A-001", reason="se aparca").success
    assert not (progress / "current-claude.md").exists()
    assert (progress / "current-data-004.md").exists()

    # ...ni al revés
    agent.start(id="A-002", owner="claude")
    assert agent.block(id="DATA-004", reason="la cierra su dueño").success
    assert not (progress / "current-data-004.md").exists()
    assert (progress / "current-claude.md").exists()


def test_finish_de_un_dueno_no_pisa_el_current_del_otro(project_root):
    """El clobber que arregla ARNES-013: cerrar lo mío no borraba lo tuyo, lo borraba."""
    _write_backlog(project_root)
    (project_root / "init.sh").write_text(
        '#!/usr/bin/env bash\n'
        'echo \'{"ready": true, "checks": [{"check": "pytest", "status": "ok", "detail": "3 passed"}]}\'\n',
        encoding="utf-8",
    )
    agent = _agent(project_root)
    agent.start(id="A-001", owner="opencode")
    agent.start(id="A-002", owner="claude")

    assert agent.finish(id="A-001", evidence="salida real").success

    progress = project_root / "progress"
    assert not (progress / "current-opencode.md").exists()   # el dueño que cerró
    detalle = (progress / "current-claude.md").read_text(encoding="utf-8")
    assert "**Feature:** A-002" in detalle                   # el otro, intacto
    # con una sola en curso, current.md vuelve al formato de siempre
    current = (progress / "current.md").read_text(encoding="utf-8")
    assert current.startswith("# Tarea actual")
    assert "**Feature:** A-002" in current


def test_block_libera_solo_el_fichero_de_su_dueno(project_root):
    _write_backlog(project_root)
    agent = _agent(project_root)
    agent.start(id="A-001", owner="opencode")
    agent.start(id="A-002", owner="claude")

    assert agent.block(id="A-002", reason="falta el dataset").success

    progress = project_root / "progress"
    assert not (progress / "current-claude.md").exists()
    assert (progress / "current-opencode.md").exists()
    assert "**Feature:** A-001" in (progress / "current.md").read_text(encoding="utf-8")


def test_sin_nada_en_curso_current_vuelve_a_idle(project_root):
    _write_backlog(project_root)
    agent = _agent(project_root)
    agent.start(id="A-001", owner="claude")

    assert agent.block(id="A-001", reason="se aparca").success

    current = (project_root / "progress" / "current.md").read_text(encoding="utf-8")
    assert "**Estado:** idle" in current
    assert not (project_root / "progress" / "current-claude.md").exists()


def test_status_solo_avisa_si_un_dueno_tiene_dos(project_root):
    _write_backlog(project_root)
    agent = _agent(project_root)
    agent.start(id="A-001", owner="opencode")
    agent.start(id="A-002", owner="claude")

    result = agent.status()

    assert result.success
    assert sorted(result.data["in_progress"]) == ["A-001", "A-002"]
    assert not result.warnings


def test_el_nombre_del_dueno_se_sanea_para_el_fichero(project_root):
    _write_backlog(project_root)
    assert _agent(project_root).start(id="A-001", owner="Claude Code/1").success

    assert (project_root / "progress" / "current-claude-code-1.md").exists()
