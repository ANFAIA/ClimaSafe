from __future__ import annotations

import sys

import pytest

from agents.agents.api_agent import APIAgent

fastapi = pytest.importorskip("fastapi", reason="fastapi es un extra opcional (use_api) del template")


def _write_synthetic_api(root, *, with_undocumented_endpoint: bool = False):
    api_dir = root / "api"
    api_dir.mkdir(exist_ok=True)
    (api_dir / "__init__.py").write_text("")
    extra_route = '\n\n@app.get("/info")\ndef info():\n    return {"model": "x"}\n' if with_undocumented_endpoint else ""
    (api_dir / "main.py").write_text(
        '"""\n'
        "api/main.py — API de prueba.\n\n"
        "Endpoints:\n"
        "    GET  /health    → estado del servicio\n"
        "    POST /predict   → predicción\n"
        '"""\n'
        "from fastapi import FastAPI\n\n"
        "app = FastAPI()\n\n"
        '@app.get("/health")\n'
        "def health():\n"
        '    return {"status": "ok"}\n\n'
        '@app.post("/predict")\n'
        "def predict():\n"
        '    return {"prediction": 1}\n'
        f"{extra_route}"
    )


@pytest.fixture(autouse=True)
def _cleanup_api_module_cache():
    yield
    sys.modules.pop("api.main", None)
    sys.modules.pop("api", None)


def _make_importable(root, monkeypatch):
    monkeypatch.syspath_prepend(str(root))
    # Asegura un paquete 'api' fresco, no uno cacheado de un test anterior con otra raíz.
    sys.modules.pop("api", None)
    sys.modules.pop("api.main", None)


def test_check_endpoints_documented_all_in_sync(context):
    _write_synthetic_api(context.root)
    agent = APIAgent(context=context)
    result = agent.check_endpoints_documented()
    assert result.success
    assert result.warnings == []


def test_check_endpoints_documented_flags_undocumented_endpoint(context):
    _write_synthetic_api(context.root, with_undocumented_endpoint=True)
    agent = APIAgent(context=context)
    result = agent.check_endpoints_documented()
    assert not result.success
    assert any("/info" in w for w in result.warnings)


def test_check_endpoints_documented_missing_file(context):
    agent = APIAgent(context=context)
    result = agent.check_endpoints_documented()
    assert not result.success


def test_smoke_test_hits_real_health_endpoint(context, monkeypatch):
    _write_synthetic_api(context.root)
    _make_importable(context.root, monkeypatch)

    agent = APIAgent(context=context)
    result = agent.smoke_test()
    assert result.success
    assert result.data["status_code"] == 200
    assert result.data["body"] == {"status": "ok"}


def test_smoke_test_reports_missing_app_attribute(context, monkeypatch):
    api_dir = context.root / "api"
    api_dir.mkdir()
    (api_dir / "__init__.py").write_text("")
    (api_dir / "main.py").write_text("# sin variable 'app'\nx = 1\n")
    _make_importable(context.root, monkeypatch)

    agent = APIAgent(context=context)
    result = agent.smoke_test()
    assert not result.success
